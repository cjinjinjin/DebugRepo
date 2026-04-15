#!/bin/bash
# Evaluate with vLLM constrained decoding (regex-guided generation).
#
# Usage:
#   # COT + constrained
#   bash eval_vllm_constrained.sh [checkpoint_dir]
#
#   # No-think + constrained (faster)
#   NO_THINK=1 bash eval_vllm_constrained.sh [checkpoint_dir]
#
#   # Unconstrained baseline
#   CONSTRAINED=0 bash eval_vllm_constrained.sh [checkpoint_dir]

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
ADAPTER_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v11-20260405-094235/checkpoint-10}"
MERGED_MODEL_PATH="${ADAPTER_PATH}/merged_model"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
EVAL_DATA="${DATA_DIR}/dpo_combined_eval_cot.jsonl"
RESULTS_DIR="${ADAPTER_PATH}/eval_results"

# Defaults: constrained=on, no_think=off
CONSTRAINED="${CONSTRAINED:-1}"
NO_THINK="${NO_THINK:-0}"

# Build output filename based on mode
if [ "${CONSTRAINED}" = "1" ] && [ "${NO_THINK}" = "1" ]; then
    MODE="constrained_no_think"
elif [ "${CONSTRAINED}" = "1" ]; then
    MODE="constrained_cot"
else
    MODE="unconstrained"
fi

RESULT_FILE="${RESULTS_DIR}/eval_vllm_${MODE}.jsonl"
REPORT_FILE="${RESULTS_DIR}/eval_report_${MODE}.json"

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Adapter  : ${ADAPTER_PATH}"
echo "Merged   : ${MERGED_MODEL_PATH}"
echo "Mode     : ${MODE}"
echo "Val data : ${EVAL_DATA}"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: merge LoRA adapter into base model ───────────────────────────────
if [ -d "${MERGED_MODEL_PATH}" ]; then
    echo "[INFO] Merged model already exists at ${MERGED_MODEL_PATH}, skipping merge."
else
    echo "[INFO] Merging DPO LoRA adapter into base model ..."
    /home/aiscuser/.conda/envs/swift_train/bin/python3.10 -m swift.cli.export \
        --model        "${MODEL_PATH}" \
        --adapters     "${ADAPTER_PATH}" \
        --merge_lora   true \
        --output_dir   "${MERGED_MODEL_PATH}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Merge failed. Exiting."
        exit 1
    fi
    echo "[INFO] Merge done: ${MERGED_MODEL_PATH}"
fi

# ── Step 2: vLLM constrained inference ──────────────────────────────────────
INFER_ARGS="--model_path ${MERGED_MODEL_PATH} --input_file ${EVAL_DATA} --output_file ${RESULT_FILE}"

if [ "${CONSTRAINED}" = "1" ]; then
    INFER_ARGS="${INFER_ARGS} --constrained"
fi
if [ "${NO_THINK}" = "1" ]; then
    INFER_ARGS="${INFER_ARGS} --no_think"
fi

echo "[INFO] Running vLLM inference (${MODE}) ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 \
    "${SCRIPT_DIR}/vllm_constrained_infer.py" ${INFER_ARGS}

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference failed. Exiting."
    exit 1
fi

echo ""
echo "Inference done. Running evaluate.py ..."

# ── Step 3: evaluation ───────────────────────────────────────────────────────
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 \
    "${SCRIPT_DIR}/evaluate.py" \
    --generated_file "${RESULT_FILE}" \
    --report_file "${REPORT_FILE}" \
    --gt_file "${EVAL_DATA}"

echo ""
echo "Report saved to ${REPORT_FILE}"

# ── Step 4: extract prompts for t2i model ────────────────────────────────────
T2I_FILE="${RESULTS_DIR}/prompts_for_t2i_${MODE}.txt"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 \
    "${SCRIPT_DIR}/extract_prompts_for_t2i.py" \
    --infer_file  "${RESULT_FILE}" \
    --gt_file     "${EVAL_DATA}" \
    --output_file "${T2I_FILE}"

echo "T2I prompts saved to ${T2I_FILE}"
