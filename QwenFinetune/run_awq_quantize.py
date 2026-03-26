"""
AWQ INT4 quantization using llm-compressor (vLLM official recommended tool).
Supports MoE architectures including Qwen3-MoE.

Install:
  pip install llm-compressor

Usage:
  python run_awq_quantize.py \
      --model_path  /path/to/merged_model \
      --calib_data  data/calib_data.jsonl \
      --output_dir  /path/to/merged_model_awq_int4
"""

import argparse
import json
from pathlib import Path


def load_calib_dataset(jsonl_path: str, n: int, seqlen: int, tokenizer):
    """
    Convert messages JSONL into tokenized dataset for llm-compressor calibration.
    Returns a list of tokenized input_ids tensors.
    """
    import torch

    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = []
            system = rec.get("system", "")
            if system:
                messages.append({"role": "system", "content": system})
            for msg in rec.get("messages", []):
                messages.append(msg)
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # fallback: plain concatenation
                text = "\n\n".join(m.get("content", "") for m in messages)
            texts.append(text)
            if len(texts) >= n:
                break

    print(f"[INFO] Tokenizing {len(texts)} calibration samples (max_len={seqlen}) ...")
    tokenized = []
    for text in texts:
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seqlen,
        )["input_ids"]
        tokenized.append(ids)

    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--calib_data",    required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--quant_bits",    type=int, default=4)
    parser.add_argument("--group_size",    type=int, default=128)
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--calib_seqlen",  type=int, default=1024)
    args = parser.parse_args()

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    except ImportError:
        raise SystemExit(
            "[ERROR] llm-compressor not installed.\n"
            "        Run: pip install llm-compressor"
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"[INFO] Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print(f"[INFO] Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading calibration data: {args.calib_data}")
    calib_data = load_calib_dataset(
        args.calib_data, args.calib_samples, args.calib_seqlen, tokenizer
    )
    print(f"[INFO] Calibration samples loaded: {len(calib_data)}")

    # GPTQ INT4 with group_size — well-supported for MoE by llm-compressor
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",           # weights 4-bit, activations 16-bit
        ignore=["lm_head"],       # keep lm_head in full precision
        group_size=args.group_size,
    )

    print(f"[INFO] Running GPTQ INT4 quantization (W4A16, group_size={args.group_size}) ...")
    print(f"       This may take 30-90 min depending on GPU count.")

    out_path = str(Path(args.output_dir))
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_data,
        recipe=recipe,
        max_seq_length=args.calib_seqlen,
        num_calibration_samples=args.calib_samples,
        output_dir=out_path,
    )

    print(f"\n[INFO] Done: {out_path}")
    print(f"       Load with vLLM: --quantization gptq")


if __name__ == "__main__":
    main()
