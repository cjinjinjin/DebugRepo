"""
Standalone YOLO human detection script.
Replaces: python -m unitorch_microsoft.omnipixel.models.yolo

Reads a TSV file, fetches images from a zip image server (HTTP),
runs YOLO detection for persons (class 0), and appends results to the TSV.

Usage:
    python yolo_detect.py \
        --data_file input.tsv \
        --output_file output.tsv \
        --names "col1,col2,col3" \
        --image_col image \
        --model yolo12n.pt \
        --image_server http://0.0.0.0:11230
"""

import argparse
import os
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from ultralytics import YOLO


def fetch_image(image_server: str, filename: str) -> Image.Image:
    """Fetch an image from the zip image server."""
    url = f"{image_server}/?file={filename}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def detect_humans(model, image: Image.Image, conf: float = 0.25):
    """
    Run YOLO detection for person class (class 0).

    Returns:
        human_count: int
        max_area_ratio: float  (largest person box area / image area)
        max_box: str  (x1,y1,x2,y2 of largest person box, or empty)
    """
    results = model(image, classes=[0], conf=conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return 0, 0.0, ""

    img_w, img_h = image.size
    img_area = img_w * img_h

    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    human_count = len(xyxy)

    # Compute area of each box
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights

    max_idx = np.argmax(areas)
    max_area_ratio = float(areas[max_idx] / img_area) if img_area > 0 else 0.0
    max_box = ",".join(f"{v:.1f}" for v in xyxy[max_idx])

    return human_count, max_area_ratio, max_box


def main():
    parser = argparse.ArgumentParser(description="Standalone YOLO human detection")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Input TSV file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output TSV file with detection results appended")
    parser.add_argument("--names", type=str, required=True,
                        help="Comma-separated column names for the TSV")
    parser.add_argument("--image_col", type=str, default="image",
                        help="Column name containing image filenames (default: image)")
    parser.add_argument("--model", type=str, default="yolo12n.pt",
                        help="YOLO model path or name (default: yolo12n.pt)")
    parser.add_argument("--image_server", type=str, default="http://0.0.0.0:11230",
                        help="Zip image server URL (default: http://0.0.0.0:11230)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Not used, kept for CLI compatibility")
    args = parser.parse_args()

    # Parse column names
    col_names = [c.strip() for c in args.names.split(",")]

    # Load TSV
    print(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file, sep="\t", header=None, names=col_names)
    print(f"  Rows: {len(df)}, Columns: {col_names}")

    if args.image_col not in df.columns:
        print(f"ERROR: image column '{args.image_col}' not found in columns {col_names}")
        import sys
        sys.exit(1)

    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # Run detection
    human_counts = []
    max_area_ratios = []
    max_boxes = []

    total = len(df)
    for i, row in df.iterrows():
        filename = row[args.image_col]
        try:
            image = fetch_image(args.image_server, filename)
            hc, mar, mb = detect_humans(model, image, conf=args.conf)
        except Exception as e:
            print(f"  Warning: failed on {filename}: {e}")
            hc, mar, mb = 0, 0.0, ""

        human_counts.append(hc)
        max_area_ratios.append(mar)
        max_boxes.append(mb)

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total}")

    df["human_count"] = human_counts
    df["max_area_ratio"] = max_area_ratios
    df["max_box"] = max_boxes

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_csv(args.output_file, sep="\t", index=False, header=False)
    print(f"Results saved to {args.output_file}")
    print(f"  Added columns: human_count, max_area_ratio, max_box")


if __name__ == "__main__":
    main()
