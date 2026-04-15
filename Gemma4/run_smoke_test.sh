#!/bin/bash
# Quick single-sample test to verify Gemma 4 loads and generates correctly
# before running full batch eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_ID="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it"

echo "============================================"
echo "Gemma 4 Quick Smoke Test"
echo "============================================"
echo "Model: $MODEL_ID"
echo ""

python "${SCRIPT_DIR}/inference_gemma4.py" \
    --model_id "$MODEL_ID" \
    --url "https://www.example.com/premium-wireless-headphones" \
    --title "Premium Wireless Headphones - Crystal Clear Sound" \
    --heading "Experience Music Like Never Before" \
    --content "Our premium wireless headphones deliver studio-quality sound with active noise cancellation. Featuring 40-hour battery life, comfortable over-ear design, and seamless Bluetooth 5.3 connectivity. Perfect for commuters, remote workers, and audiophiles." \
    --max_new_tokens 2048

echo ""
echo "Smoke test complete."
