#!/bin/bash
# Evaluate the DPO fine-tuned model using swift infer (batch inference)
# then run evaluate.py on the results.
#
# Usage:
#   bash eval_swift_dpo.sh [checkpoint_dir]
#
# Example:
#   bash eval_swift_dpo.sh /vc_data/.../checkpoint-5
#
# If no checkpoint_dir is given, ADAPTER_PATH below is used (best checkpoint from training log).

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
# NOTE: training log shows best_model_checkpoint = checkpoint-5, use that by default
ADAPTER_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v3-20260320-155846/checkpoint-50}"
MERGED_MODEL_PATH="${ADAPTER_PATH}/merged_model"
DATA_DIR="./data"
EVAL_DATA="${DATA_DIR}/dpo_refine_eval_cot.jsonl"
RESULTS_DIR="${ADAPTER_PATH}/eval_results"
RESULT_FILE="${RESULTS_DIR}/eval_swift_output.jsonl"
REPORT_FILE="${RESULTS_DIR}/eval_report.json"

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Adapter  : ${ADAPTER_PATH}"
echo "Merged   : ${MERGED_MODEL_PATH}"
echo "Val data : ${EVAL_DATA}"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: merge LoRA adapter into base model ───────────────────────────────
# DPO trains on top of the SFT adapter; the merged model includes both.
if [ -d "${MERGED_MODEL_PATH}" ]; then
    echo "[INFO] Merged model already exists at ${MERGED_MODEL_PATH}, skipping merge."
else
    echo "[INFO] Merging DPO LoRA adapter into base model ..."
    /home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 -m swift.cli.export \
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

# ── Step 2: batch inference ──────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 -m swift.cli.infer \
    --model                        "${MERGED_MODEL_PATH}" \
    --val_dataset                  "${EVAL_DATA}" \
    --max_length                   8192 \
    --infer_backend                vllm \
    --max_batch_size               32 \
    --vllm_tensor_parallel_size    8 \
    --result_path                  "${RESULT_FILE}"

echo ""
echo "Inference done. Running evaluate.py ..."

# ── Locate the actual output file ────────────────────────────────────────────
if [ ! -f "${RESULT_FILE}" ]; then
    echo "[WARN] ${RESULT_FILE} not found. Searching for swift output ..."
    FOUND=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${EVAL_DATA}" \
        -not -path "*/data/*" \
        -not -name "sft_*.jsonl" \
        -not -name "dpo_*.jsonl" \
        2>/dev/null | head -5)
    echo "Candidate files:"
    echo "${FOUND}"
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

# ── Step 3: evaluation ───────────────────────────────────────────────────────
EVAL_ARGS="--generated_file ${RESULT_FILE} --report_file ${REPORT_FILE} --gt_file ${EVAL_DATA}"

# Uncomment to enable LLM-as-Judge (requires OPENAI_API_KEY):
# export OPENAI_API_KEY="your-key-here"
# export OPENAI_API_BASE="https://api.openai.com/v1"
# EVAL_ARGS="${EVAL_ARGS} --llm_judge --llm_model gpt-4o"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 evaluate.py ${EVAL_ARGS}

echo ""
echo "Report saved to ${REPORT_FILE}"

# ── Step 4: extract prompts for t2i model ────────────────────────────────────
T2I_FILE="${RESULTS_DIR}/prompts_for_t2i.txt"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 extract_prompts_for_t2i.py \
    --infer_file  "${RESULT_FILE}" \
    --gt_file     "${EVAL_DATA}" \
    --output_file "${T2I_FILE}"

echo "T2I prompts saved to ${T2I_FILE}"
