#!/bin/bash
# Evaluate the base Qwen3-30B-A3B model (official checkpoint, no fine-tuning)
# on the 190-sample validation set.
#
# Usage:
#   bash eval_base_model.sh [model_path]
#
# Example:
#   bash eval_base_model.sh /path/to/Qwen3-30B-A3B

set -euo pipefail

MODEL_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
EVAL_DATA="${DATA_DIR}/dpo_combined_eval_cot.jsonl"
RESULTS_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_grpo_experiments/base_model_eval"
RESULT_FILE="${RESULTS_DIR}/eval_swift_output.jsonl"
REPORT_FILE="${RESULTS_DIR}/eval_report.json"

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Base Model Evaluation (no fine-tuning)"
echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Val data : ${EVAL_DATA}"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: batch inference (no merge needed for base model) ────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 -m swift.cli.infer \
    --model                        "${MODEL_PATH}" \
    --val_dataset                  "${EVAL_DATA}" \
    --max_length                   8192 \
    --max_new_tokens               4096 \
    --infer_backend                vllm \
    --max_batch_size               32 \
    --vllm_tensor_parallel_size    8 \
    --result_path                  "${RESULT_FILE}"

echo ""
echo "Inference done. Running evaluate.py ..."

# ── Locate the actual output file ───────────────────────────────────────────
if [ ! -f "${RESULT_FILE}" ]; then
    echo "[WARN] ${RESULT_FILE} not found. Searching for swift output ..."
    LATEST=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${EVAL_DATA}" \
        -not -path "*/data/*" \
        -not -name "sft_*.jsonl" \
        -not -name "dpo_*.jsonl" \
        2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    if [ -n "${LATEST}" ]; then
        echo "[INFO] Using: ${LATEST}"
        RESULT_FILE="${LATEST}"
    else
        echo "[ERROR] No swift output file found. Check swift infer logs above."
        exit 1
    fi
fi

# ── Step 2: evaluation ─────────────────────────────────────────────────────
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 "${SCRIPT_DIR}/evaluate.py" \
    --generated_file "${RESULT_FILE}" \
    --report_file "${REPORT_FILE}" \
    --gt_file "${EVAL_DATA}"

echo ""
echo "Report saved to ${REPORT_FILE}"

# ── Step 3: extract prompts for t2i model ──────────────────────────────────
T2I_FILE="${RESULTS_DIR}/prompts_for_t2i.txt"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 "${SCRIPT_DIR}/extract_prompts_for_t2i.py" \
    --infer_file  "${RESULT_FILE}" \
    --gt_file     "${EVAL_DATA}" \
    --output_file "${T2I_FILE}"

echo "T2I prompts saved to ${T2I_FILE}"
