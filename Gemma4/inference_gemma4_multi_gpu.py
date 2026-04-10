"""
Multi-GPU data-parallel launcher for Gemma 4 inference.

Splits eval data into N shards, launches N independent inference_gemma4.py
processes (one per GPU), then merges results.

Usage:
  python Gemma4/inference_gemma4_multi_gpu.py \
      --model_id ./gemma-4-26B-A4B-it \
      --input_file QwenFinetune/data/sft_eval_cot.jsonl \
      --output_file Gemma4/results/gemma4_zeroshot_eval.jsonl \
      --num_gpus 8 \
      --max_new_tokens 2048
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_data(records: list[dict], num_shards: int) -> list[list[dict]]:
    """Split records into num_shards roughly equal parts."""
    shards = [[] for _ in range(num_shards)]
    for i, r in enumerate(records):
        shards[i % num_shards].append(r)
    return shards


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Gemma 4 inference launcher")
    parser.add_argument("--model_id", required=True, help="Model path")
    parser.add_argument("--adapter_path", default="", help="Optional LoRA adapter path")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", default="Gemma4/results/gemma4_zeroshot_eval.jsonl",
                        help="Final merged output file")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--no_think", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    args = parser.parse_args()

    # Load and split data
    records = load_jsonl(args.input_file)
    total = len(records)
    num_gpus = min(args.num_gpus, total)
    shards = split_data(records, num_gpus)

    print(f"Total samples: {total}")
    print(f"Using {num_gpus} GPUs, shard sizes: {[len(s) for s in shards]}")

    # Create temp dir for shard files
    tmpdir = tempfile.mkdtemp(prefix="gemma4_shards_")
    shard_inputs = []
    shard_outputs = []

    for i, shard in enumerate(shards):
        inp = os.path.join(tmpdir, f"shard_{i}_input.jsonl")
        out = os.path.join(tmpdir, f"shard_{i}_output.jsonl")
        write_jsonl(shard, inp)
        shard_inputs.append(inp)
        shard_outputs.append(out)

    # Launch processes
    script_dir = Path(__file__).resolve().parent
    inference_script = str(script_dir / "inference_gemma4.py")

    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, inference_script,
            "--model_id", args.model_id,
            "--input_file", shard_inputs[gpu_id],
            "--output_file", shard_outputs[gpu_id],
            "--max_new_tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--top_k", str(args.top_k),
            "--batch_size", "1",
        ]
        if args.adapter_path:
            cmd.extend(["--adapter_path", args.adapter_path])
        if args.no_think:
            cmd.append("--no_think")
        if args.load_in_4bit:
            cmd.append("--load_in_4bit")
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

        print(f"[GPU {gpu_id}] Launching with {len(shards[gpu_id])} samples ...")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu_id, proc))

    # Wait for all processes and stream output
    failed = []
    for gpu_id, proc in processes:
        stdout, _ = proc.communicate()
        output = stdout.decode("utf-8", errors="replace")

        # Print last few lines of each GPU's output
        lines = output.strip().split("\n")
        summary_lines = [l for l in lines if "compliance" in l.lower() or "processed" in l.lower()
                         or "total inference" in l.lower() or "saved" in l.lower()
                         or "format" in l.lower() or "tags" in l.lower()]
        if summary_lines:
            print(f"\n[GPU {gpu_id}] Summary:")
            for l in summary_lines[-5:]:
                print(f"  {l}")

        if proc.returncode != 0:
            failed.append(gpu_id)
            print(f"\n[GPU {gpu_id}] FAILED (exit code {proc.returncode})")
            # Print last 20 lines for debugging
            for l in lines[-20:]:
                print(f"  {l}")

    elapsed = time.time() - start_time

    if failed:
        print(f"\nERROR: GPUs {failed} failed. Check output above.")
        sys.exit(1)

    # Merge results in original order
    # Each shard was round-robin assigned, so we need to interleave back
    shard_results = []
    for i in range(num_gpus):
        if os.path.exists(shard_outputs[i]):
            shard_results.append(load_jsonl(shard_outputs[i]))
        else:
            print(f"WARNING: shard {i} output file missing: {shard_outputs[i]}")
            shard_results.append([])

    # Reconstruct original order (reverse of round-robin split)
    merged = [None] * total
    for shard_idx, shard_data in enumerate(shard_results):
        for j, record in enumerate(shard_data):
            original_idx = j * num_gpus + shard_idx
            if original_idx < total:
                merged[original_idx] = record

    # Filter out any None entries (shouldn't happen if all shards succeeded)
    merged = [r for r in merged if r is not None]

    write_jsonl(merged, args.output_file)

    # Print summary stats
    n_compliant = sum(1 for r in merged if r.get("format_compliant", False))
    n_tags = sum(1 for r in merged if len(r.get("generated_prompts", [])) == 5
                 and r.get("generated_prompts", [""])[0] != r.get("raw_output", "").strip())

    print(f"\n{'='*60}")
    print(f"Multi-GPU Inference Complete")
    print(f"{'='*60}")
    print(f"GPUs used:                {num_gpus}")
    print(f"Total samples:            {len(merged)}")
    print(f"Total time:               {elapsed:.1f}s ({elapsed/total:.1f}s/sample effective)")
    print(f"Format compliance (full): {n_compliant}/{len(merged)} ({100*n_compliant/len(merged):.1f}%)")
    print(f"All 5 tags present:       {n_tags}/{len(merged)} ({100*n_tags/len(merged):.1f}%)")
    print(f"{'='*60}")
    print(f"\nOutput: {args.output_file}")
    print(f"Temp shards: {tmpdir}")

    # Cleanup temp files
    for f in shard_inputs + shard_outputs:
        if os.path.exists(f):
            os.remove(f)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
