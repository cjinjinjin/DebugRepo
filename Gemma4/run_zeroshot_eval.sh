#!/bin/bash
# Gemma 4 26B-A4B-it zero-shot evaluation
# Run on training machine with 8x A100-80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model
MODEL_ID="google/gemma-4-26B-A4B-it"

# Data — use SFT eval set for fair comparison with Qwen3 baselines
EVAL_DATA="${ROOT_DIR}/QwenFinetune/data/sft_eval_cot.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/gemma4_zeroshot_eval.jsonl"
REPORT_FILE="${OUTPUT_DIR}/gemma4_zeroshot_report.json"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Gemma 4 26B-A4B-it Zero-shot Evaluation"
echo "============================================"
echo "Model:     $MODEL_ID"
echo "Eval data: $EVAL_DATA"
echo "Output:    $OUTPUT_FILE"
echo ""

# Step 1: Run inference
echo "[Step 1] Running zero-shot inference ..."
python "${SCRIPT_DIR}/inference_gemma4.py" \
    --model_id "$MODEL_ID" \
    --input_file "$EVAL_DATA" \
    --output_file "$OUTPUT_FILE" \
    --max_new_tokens 2048 \
    --batch_size 1

echo ""

# Step 2: Run evaluation using existing evaluate.py
echo "[Step 2] Running evaluation ..."
python "${ROOT_DIR}/QwenFinetune/evaluate.py" \
    --generated_file "$OUTPUT_FILE" \
    --report_file "$REPORT_FILE"

echo ""
echo "============================================"
echo "Done. Report saved to: $REPORT_FILE"
echo "============================================"
