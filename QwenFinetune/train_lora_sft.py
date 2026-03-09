"""
SFT (Supervised Fine-Tuning) with LoRA for Qwen3.5-35B-A3B.

Goal: fine-tune Qwen to directly generate "good" image prompts from LP fields,
      replacing the GPT5 two-step pipeline at inference time.

Training data format (sft_train.jsonl / sft_eval.jsonl):
  Each line is a JSON object with a "messages" list in ChatML format:
  {
    "id": "...",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."},
      {"role": "assistant", "content": "<Prompt1>...</Prompt1>..."}
    ]
  }

Requirements:
  pip install torch transformers peft accelerate bitsandbytes datasets trl

Usage:
  # Single GPU (e.g. A100 80G)
  python train_lora_sft.py --model_name_or_path Qwen/Qwen2.5-35B-Instruct \
      --train_file data/sft_train.jsonl \
      --eval_file  data/sft_eval.jsonl  \
      --output_dir checkpoints/qwen35_sft_lora

  # Multi-GPU via accelerate
  accelerate launch --num_processes 4 train_lora_sft.py ...
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_LENGTH = 2048   # max tokens per sample (prompt + response)
IGNORE_INDEX = -100         # label mask value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Tokenisation: apply_chat_template + mask prompt tokens
# ---------------------------------------------------------------------------

def tokenize_sample(
    sample: dict,
    tokenizer,
    max_length: int,
) -> dict:
    """
    Apply the model's chat template, then mask out all non-assistant tokens
    so the loss is only computed on the generated prompts.
    """
    messages = sample["messages"]

    # Build the full text with chat template (adds special tokens)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Also build prefix (everything up to but not including assistant turn)
    prefix_messages = [m for m in messages if m["role"] != "assistant"]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the prompt marker for assistant
    )

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    prefix_ids = tokenizer(
        prefix_text,
        truncation=False,
        return_attention_mask=False,
    )

    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]
    labels = list(input_ids)  # copy

    # Mask everything up to where the assistant response starts
    prefix_len = len(prefix_ids["input_ids"])
    for i in range(min(prefix_len, len(labels))):
        labels[i] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_dataset(
    records: list[dict],
    tokenizer,
    max_length: int,
    num_proc: int = 4,
) -> Dataset:
    ds = Dataset.from_list(records)
    ds = ds.map(
        lambda ex: tokenize_sample(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Tokenising",
    )
    # Remove samples that are too long after truncation still have no labels
    ds = ds.filter(lambda x: any(l != IGNORE_INDEX for l in x["labels"]))
    return ds


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(args) -> LoraConfig:
    # Target the attention and MLP projection layers
    # Qwen2.5 uses: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    # Model
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-35B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--load_in_4bit", action="store_true", default=True,
                   help="Use 4-bit quantisation (QLoRA) to reduce VRAM usage")
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="",
                   help="Comma-separated list; leave empty for default attention+MLP layers")
    # Training
    p.add_argument("--output_dir", default="checkpoints/qwen35_sft_lora")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_strategy", default="steps")
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--report_to", default="none", help="'wandb', 'tensorboard', or 'none'")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading model from {args.model_name_or_path} ...")
    bnb_config = None
    if args.load_in_4bit and not args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model.config.use_cache = False  # required when using gradient checkpointing
    model.enable_input_require_grads()  # required for PEFT with gradient checkpointing

    # -----------------------------------------------------------------------
    # Apply LoRA
    # -----------------------------------------------------------------------
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Prepare datasets
    # -----------------------------------------------------------------------
    print("Loading and tokenising train data ...")
    train_records = load_jsonl(args.train_file)
    eval_records = load_jsonl(args.eval_file)

    train_dataset = preprocess_dataset(train_records, tokenizer, args.max_length)
    eval_dataset = preprocess_dataset(eval_records, tokenizer, args.max_length)

    print(f"  Train samples after filtering: {len(train_dataset)}")
    print(f"  Eval  samples after filtering: {len(eval_dataset)}")

    # -----------------------------------------------------------------------
    # Data collator
    # -----------------------------------------------------------------------
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    # -----------------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.eval_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("\nStarting SFT training ...")
    trainer.train()

    # -----------------------------------------------------------------------
    # Save LoRA adapter
    # -----------------------------------------------------------------------
    adapter_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
