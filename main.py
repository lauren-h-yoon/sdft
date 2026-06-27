from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    # crack-coconut patch: LoRA + max_steps so we can match scale across baselines
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank (0 = full-param training, default).")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (default: 2 * lora_rank).")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="LoRA target_modules: 'all-linear' or comma-separated list.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps; -1 uses num_train_epochs.")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="HF Trainer report_to (wandb|none|tensorboard).")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # crack-coconut patch: optionally wrap student in LoRA so trainable params
    # match the other baselines (crack_coconut / SDPO) for fair compute parity.
    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
        tm_arg = args.lora_target_modules
        lora_targets = tm_arg if tm_arg == "all-linear" else [s.strip() for s in tm_arg.split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=lora_targets,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[sdft-cc-patch] wrapped student in LoRA (r={args.lora_rank}, alpha={lora_alpha})")
        model.print_trainable_parameters()
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = 1024,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        num_iterations = 1,
        num_generations = 1,
        save_steps = 100,
        max_grad_norm = 1,
        report_to = args.report_to,
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        # crack-coconut patch: ref-model sync (EMA toward student) is incompatible
        # with LoRA — the per-param shape mismatch (rank vs hidden_size) crashes
        # _sync_param. With LoRA, freeze the reference at base weights instead.
        sync_ref_model = (args.lora_rank == 0),
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
