"""Batch ZImage generation from T2I_RAI_prompts.tsv.

Usage:
    python batch_zimage.py [--output OUTPUT_DIR] [--start START_ID] [--end END_ID] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import concurrent.futures
from pathlib import Path

# Add zimage_client to path
sys.path.insert(0, r"C:\Users\jinjinchen\OneDrive - Microsoft\code\GEMS\agent")
from zimage_client import call_zimage

WIDTH = 1344
HEIGHT = 768


def generate_one(prompt_id: str, prompt: str, output_dir: Path) -> str:
    """Generate one image. Returns status string."""
    out_file = output_dir / f"{prompt_id}.png"
    if out_file.exists():
        return f"[SKIP] {prompt_id} - already exists"
    try:
        png_bytes = call_zimage(prompt, width=WIDTH, height=HEIGHT, timeout_s=180)
        out_file.write_bytes(png_bytes)
        return f"[OK]   {prompt_id} - {len(png_bytes)} bytes"
    except Exception as e:
        return f"[FAIL] {prompt_id} - {e}"


def main():
    parser = argparse.ArgumentParser(description="Batch ZImage generation")
    parser.add_argument("--input", default=r"C:\Users\jinjinchen\Downloads\T2I_RAI_prompts.tsv")
    parser.add_argument("--output", default=r"C:\Users\jinjinchen\Downloads\T2I_RAI_images")
    parser.add_argument("--start", type=int, default=None, help="Start prompt ID (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End prompt ID (inclusive)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read TSV
    prompts = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            pid, prompt = row[0].strip(), row[1].strip()
            if not pid or not prompt:
                continue
            try:
                pid_int = int(pid)
            except ValueError:
                continue
            if args.start and pid_int < args.start:
                continue
            if args.end and pid_int > args.end:
                continue
            prompts.append((pid, prompt))

    print(f"Total prompts to process: {len(prompts)}")
    print(f"Output dir: {output_dir}")
    print(f"Size: {WIDTH}x{HEIGHT}")
    print(f"Workers: {args.workers}")
    print()

    done = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, pid, prompt, output_dir): pid
            for pid, prompt in prompts
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            if "[OK]" in result:
                done += 1
            elif "[SKIP]" in result:
                skipped += 1
            else:
                failed += 1

    elapsed = time.time() - t0
    print(f"\nDone: {done} | Skipped: {skipped} | Failed: {failed} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
