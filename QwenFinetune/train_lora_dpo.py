"""
DPO (Direct Preference Optimization) with LoRA for Qwen3.5-35B-A3B.

Goal: Use good/bad prompt pairs to directly align the model toward producing
      "good" image prompts via preference learning, without requiring a reward model.

This script can be used:
  (a) Standalone — train from the base Qwen model with DPO
  (b) After SFT — continue from an SFT checkpoint for further alignment

Training data format (dpo_train.jsonl / dpo_eval.jsonl):
  {
    "id": "...",
    "prompt": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."}
    ],
    "chosen":   [{"role": "assistant", "content": "<Prompt1>...</Prompt1>..."}],
    "rejected": [{"role": "assistant", "content": "<Prompt1>...</Prompt1>..."}]
  }

Requirements:
  pip install torch transformers peft accelerate bitsandbytes datasets trl

Usage:
  # From base model
  python train_lora_dpo.py \
      --model_name_or_path Qwen/Qwen2.5-35B-Instruct \
      --train_file data/dpo_train.jsonl \
      --eval_file  data/dpo_eval.jsonl  \
      --output_dir checkpoints/qwen35_dpo_lora

  # From SFT checkpoint (recommended: better starting point)
  python train_lora_dpo.py \
      --model_name_or_path checkpoints/qwen35_sft_lora/lora_adapter \
      --load_from_sft_adapter \
      --train_file data/dpo_train.jsonl \
      --eval_file  data/dpo_eval.jsonl  \
      --output_dir checkpoints/qwen35_dpo_lora
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


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


def messages_to_text(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def preprocess_dpo_dataset(records: list[dict], tokenizer) -> Dataset:
    """
    Convert to TRL DPOTrainer format:
      - "prompt": the formatted prompt string (system + user)
      - "chosen": assistant response string (good)
      - "rejected": assistant response string (bad)
    """
    processed = []
    for r in records:
        prompt_text = messages_to_text(tokenizer, r["prompt"], add_generation_prompt=True)
        chosen_text = r["chosen"][0]["content"]
        rejected_text = r["rejected"][0]["content"]
        processed.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
    return Dataset.from_list(processed)


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(args) -> LoraConfig:
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
    p.add_argument("--max_length", type=int, default=2048,
                   help="Max total tokens (prompt + chosen/rejected)")
    p.add_argument("--max_prompt_length", type=int, default=1536,
                   help="Max tokens for the prompt portion")
    # Model
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-35B-Instruct")
    p.add_argument("--load_from_sft_adapter", action="store_true", default=False,
                   help="If True, model_name_or_path points to a PEFT adapter directory")
    p.add_argument("--base_model_for_adapter", default="",
                   help="Base model path when loading from SFT adapter")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")
    # DPO
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta: KL divergence penalty weight (0.1 is typical)")
    p.add_argument("--loss_type", default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo", "kto_pair"],
                   help="DPO loss variant")
    # Training
    p.add_argument("--output_dir", default="checkpoints/qwen35_dpo_lora")
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
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
    p.add_argument("--report_to", default="none")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------
    model_path = args.base_model_for_adapter if args.load_from_sft_adapter else args.model_name_or_path
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",   # DPO typically uses left-padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
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

    print(f"Loading base model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    if args.load_from_sft_adapter:
        # Load existing SFT LoRA weights, then re-wrap for DPO training
        print(f"Loading SFT LoRA adapter from {args.model_name_or_path} ...")
        model = PeftModel.from_pretrained(model, args.model_name_or_path, is_trainable=True)
    else:
        # Fresh LoRA for DPO
        lora_config = build_lora_config(args)
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Prepare datasets
    # -----------------------------------------------------------------------
    print("Loading DPO data ...")
    train_records = load_jsonl(args.train_file)
    eval_records = load_jsonl(args.eval_file)

    train_dataset = preprocess_dpo_dataset(train_records, tokenizer)
    eval_dataset = preprocess_dpo_dataset(eval_records, tokenizer)

    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # -----------------------------------------------------------------------
    # DPO Training config
    # -----------------------------------------------------------------------
    dpo_config = DPOConfig(
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
        gradient_checkpointing=True,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    # -----------------------------------------------------------------------
    # DPO Trainer
    # -----------------------------------------------------------------------
    # The reference model (for KL penalty) can be the same frozen base model.
    # DPOTrainer handles this automatically when ref_model=None with PEFT.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # auto-managed when model is a PEFT model
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting DPO training ...")
    trainer.train()

    # -----------------------------------------------------------------------
    # Save adapter
    # -----------------------------------------------------------------------
    adapter_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
