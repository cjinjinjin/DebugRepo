#!/bin/bash
# Evaluate a Gemma 4 checkpoint (SFT or DPO)
# Usage: bash eval_checkpoint.sh /path/to/checkpoint_or_merged_model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT_PATH="${1:-}"
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: bash eval_checkpoint.sh /path/to/checkpoint_or_merged_model [eval_name]"
    exit 1
fi

EVAL_NAME="${2:-gemma4_eval}"
MODEL_ID="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-26B-A4B-it"

# Data
EVAL_DATA="${ROOT_DIR}/QwenFinetune/data/sft_eval_cot.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/${EVAL_NAME}.jsonl"
REPORT_FILE="${OUTPUT_DIR}/${EVAL_NAME}_report.json"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Gemma 4 Checkpoint Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Eval data:  $EVAL_DATA"
echo "Output:     $OUTPUT_FILE"
echo ""

# Check if it's a LoRA adapter or merged model
if [ -f "${CHECKPOINT_PATH}/adapter_config.json" ]; then
    echo "Detected LoRA adapter, loading with base model..."
    ADAPTER_ARG="--adapter_path ${CHECKPOINT_PATH}"
    MODEL_ARG="--model_id ${MODEL_ID}"
else
    echo "Detected merged model, loading directly..."
    ADAPTER_ARG=""
    MODEL_ARG="--model_id ${CHECKPOINT_PATH}"
fi

# Step 1: Inference
echo "[Step 1] Running inference ..."
python "${SCRIPT_DIR}/inference_gemma4.py" \
    ${MODEL_ARG} \
    ${ADAPTER_ARG} \
    --input_file "$EVAL_DATA" \
    --output_file "$OUTPUT_FILE" \
    --max_new_tokens 2048 \
    --batch_size 1

echo ""

# Step 2: Evaluate
echo "[Step 2] Running evaluation ..."
python "${ROOT_DIR}/QwenFinetune/evaluate.py" \
    --generated_file "$OUTPUT_FILE" \
    --report_file "$REPORT_FILE"

echo ""
echo "============================================"
echo "Done. Report: $REPORT_FILE"
echo "============================================"
