"""
Download calibration datasets from ModelScope to a local cache directory.
Run this on a machine with stable internet access, then transfer to the server.

Usage:
  python download_calib_datasets.py --output_dir ./data/modelscope_cache

Datasets downloaded:
  - AI-ModelScope/alpaca-gpt4-data-zh  (Chinese instruction data)
  - AI-ModelScope/alpaca-gpt4-data-en  (English instruction data)

After download, transfer to server and set:
  export MODELSCOPE_CACHE=/path/to/modelscope_cache
Or pass --dataset_cache_dir in the swift export command.
"""

import argparse
import json
from pathlib import Path


def download_modelscope(dataset_id: str, output_dir: Path):
    """Download a ModelScope dataset to local directory."""
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise SystemExit(
            "[ERROR] modelscope not installed.\n"
            "        Run: pip install modelscope"
        )

    print(f"[INFO] Downloading {dataset_id} ...")
    ds = MsDataset.load(dataset_id, split="train")

    save_path = output_dir / dataset_id.replace("/", "__")
    save_path.mkdir(parents=True, exist_ok=True)

    records = []
    for item in ds:
        records.append(item)

    out_file = save_path / "train.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved {len(records)} records -> {out_file}")
    return out_file


def download_hf(dataset_id: str, output_dir: Path):
    """Fallback: download via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "[ERROR] datasets not installed.\n"
            "        Run: pip install datasets"
        )

    print(f"[INFO] Downloading {dataset_id} via HuggingFace ...")
    ds = load_dataset(dataset_id, split="train")

    save_path = output_dir / dataset_id.replace("/", "__")
    save_path.mkdir(parents=True, exist_ok=True)

    out_file = save_path / "train.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")

    print(f"[INFO] Saved {len(ds)} records -> {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="./data/modelscope_cache",
        help="Local directory to save downloaded datasets",
    )
    parser.add_argument(
        "--backend",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Download backend (default: modelscope)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "AI-ModelScope/alpaca-gpt4-data-zh",
        "AI-ModelScope/alpaca-gpt4-data-en",
    ]

    saved_files = []
    for ds_id in datasets:
        try:
            if args.backend == "modelscope":
                f = download_modelscope(ds_id, out_dir)
            else:
                f = download_hf(ds_id, out_dir)
            saved_files.append((ds_id, f))
        except Exception as e:
            print(f"[ERROR] Failed to download {ds_id}: {e}")

    print("\n============================================")
    print("Downloaded files:")
    for ds_id, f in saved_files:
        print(f"  {ds_id} -> {f}")
    print("\nUsage in quantize_model.sh:")
    for ds_id, f in saved_files:
        print(f"  --dataset {f} \\")
    print("============================================")


if __name__ == "__main__":
    main()
