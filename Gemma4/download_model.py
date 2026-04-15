"""
Download Gemma 4 26B-A4B-it from HuggingFace and move to shared storage.

Usage:
  python Gemma4/download_model.py
  python Gemma4/download_model.py --token hf_xxx        # specify token
  python Gemma4/download_model.py --local-only           # download only, don't move
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "google/gemma-4-26B-A4B-it"
CKPT_ROOT = "/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT"


def main():
    parser = argparse.ArgumentParser(description="Download Gemma 4 model")
    parser.add_argument("--repo_id", default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--local-only", action="store_true", help="Don't move to shared storage")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] Need HF token. Use --token or set HF_TOKEN env var.")
        print("       Gemma 4 is a gated model, login required.")
        return

    model_name = args.repo_id.split("/")[-1]
    local_dir = f"./{model_name}"
    target_dir = f"{CKPT_ROOT}/{model_name}"

    print(f"Repo:       {args.repo_id}")
    print(f"Local dir:  {local_dir}")
    print(f"Target dir: {target_dir}")
    print()

    # Step 1: Download
    print("[Step 1] Downloading from HuggingFace ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[OK] Downloaded to {local_dir}")

    # Step 2: Move to shared storage
    if args.local_only:
        print("[SKIP] --local-only specified, not moving.")
        print(f"Done! Files at: {local_dir}")
        return

    if not Path(CKPT_ROOT).exists():
        print(f"[WARN] Target root {CKPT_ROOT} not found. Keeping files in {local_dir}")
        return

    if Path(target_dir).exists():
        print(f"[WARN] {target_dir} already exists. Skipping move.")
        print(f"       Delete it first if you want to re-download.")
    else:
        print(f"[Step 2] Moving to {target_dir} ...")
        shutil.move(local_dir, target_dir)
        print(f"[OK] Moved to {target_dir}")

    print()
    print(f"Done! Model path for training/inference:")
    print(f"  {target_dir}")


if __name__ == "__main__":
    main()
