"""
AWQ INT4 quantization using autoawq directly (no swift dependency).

Usage:
  python run_awq_quantize.py \
      --model_path  /path/to/merged_model \
      --calib_data  data/calib_data.jsonl \
      --output_dir  /path/to/merged_model_awq_int4
"""

import argparse
import json
from pathlib import Path


def load_calib_texts(jsonl_path: str, n: int, seqlen: int) -> list[str]:
    """
    Convert messages JSONL into flat text strings for AWQ calibration.
    AWQ needs plain text, not chat-formatted dicts.
    """
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            # Flatten to a single string: system + user (+ assistant if present)
            parts = []
            system = rec.get("system", "")
            if system:
                parts.append(system)
            for msg in messages:
                parts.append(msg.get("content", ""))
            text = "\n\n".join(p for p in parts if p)
            # Truncate to approx seqlen chars (rough proxy for token count)
            texts.append(text[: seqlen * 4])
            if len(texts) >= n:
                break
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--calib_data",    required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--quant_bits",    type=int,   default=4)
    parser.add_argument("--group_size",    type=int,   default=128)
    parser.add_argument("--zero_point",    type=str,   default="true")
    parser.add_argument("--calib_samples", type=int,   default=64)
    parser.add_argument("--calib_seqlen",  type=int,   default=1024)
    args = parser.parse_args()

    zero_point = args.zero_point.lower() == "true"

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "[ERROR] autoawq not installed.\n"
            "        Run: pip install autoawq"
        )

    print(f"[INFO] Loading model: {args.model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.quant_bits,
        "version": "GEMM",
    }
    print(f"[INFO] Quant config: {quant_config}")

    print(f"[INFO] Loading calibration data: {args.calib_data}")
    calib_texts = load_calib_texts(args.calib_data, args.calib_samples, args.calib_seqlen)
    print(f"[INFO] Calibration samples: {len(calib_texts)}")

    print("[INFO] Running AWQ quantization (this may take 30-60 min) ...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving quantized model to: {out_path}")
    model.save_quantized(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print(f"[INFO] Done: {out_path}")


if __name__ == "__main__":
    main()
