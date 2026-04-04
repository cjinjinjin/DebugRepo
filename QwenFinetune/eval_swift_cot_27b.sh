#!/bin/bash
# Evaluate the fine-tuned Qwen3.5-27B CoT model using swift infer (batch inference)
# then run evaluate.py on the results.
#
# Usage:
#   bash eval_swift_cot_27b.sh [checkpoint_dir]

MODEL_PATH="${MODEL_PATH:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3.5-27B}"
ADAPTER_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen35_27b_sft_lora_cot_v1/best_checkpoint}"
MERGED_MODEL_PATH="${ADAPTER_PATH}/merged_model"
DATA_DIR="./data"
RESULTS_DIR="${ADAPTER_PATH}/eval_results"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="${RESULTS_DIR}/eval_swift_output_${RUN_TS}.jsonl"
REPORT_FILE="${RESULTS_DIR}/eval_report_${RUN_TS}.json"

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Adapter  : ${ADAPTER_PATH}"
echo "Merged   : ${MERGED_MODEL_PATH}"
echo "Val data : ${DATA_DIR}/dpo_refine_eval_cot.jsonl"
echo "Run TS   : ${RUN_TS}"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: merge LoRA adapter into base model ───────────────────────────────
if [ -d "${MERGED_MODEL_PATH}" ]; then
    echo "[INFO] Merged model already exists at ${MERGED_MODEL_PATH}, skipping merge."
else
    echo "[INFO] Merging LoRA adapter into base model ..."
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
    --val_dataset                  "${DATA_DIR}/dpo_refine_eval_cot.jsonl" \
    --max_length                   8192 \
    --infer_backend                vllm \
    --max_batch_size               32 \
    --vllm_tensor_parallel_size    8 \
    --result_path                  "${RESULT_FILE}"

echo ""
echo "Inference done. Running evaluate.py ..."

# ── Locate the actual output file (Step 2 cont.) ────────────────────────────
if [ ! -f "${RESULT_FILE}" ]; then
    echo "[WARN] ${RESULT_FILE} not found. Searching for swift output ..."
    FOUND=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${DATA_DIR}/dpo_refine_eval_cot.jsonl" \
        -not -path "*/data/*" \
        -not -name "sft_*.jsonl" \
        2>/dev/null | head -5)
    echo "Candidate files:"
    echo "${FOUND}"
    LATEST=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${DATA_DIR}/dpo_refine_eval_cot.jsonl" \
        -not -path "*/data/*" \
        -not -name "sft_*.jsonl" \
        2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    if [ -n "${LATEST}" ]; then
        echo "[INFO] Renaming ${LATEST} -> ${RESULT_FILE}"
        mv "${LATEST}" "${RESULT_FILE}"
    else
        echo "[ERROR] No swift output file found. Check swift infer logs above."
        exit 1
    fi
fi

# ── Step 3: evaluation ───────────────────────────────────────────────────────
EVAL_ARGS="--generated_file ${RESULT_FILE} --report_file ${REPORT_FILE} --gt_file ${DATA_DIR}/dpo_refine_eval_cot.jsonl"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 evaluate.py ${EVAL_ARGS}

echo ""
echo "Report saved to ${REPORT_FILE}"

# ── Step 4: extract prompts for t2i model ────────────────────────────────────
T2I_FILE="${RESULTS_DIR}/prompts_for_t2i_${RUN_TS}.txt"

/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 extract_prompts_for_t2i.py \
    --infer_file  "${RESULT_FILE}" \
    --gt_file     "${DATA_DIR}/dpo_refine_eval_cot.jsonl" \
    --output_file "${T2I_FILE}"

echo "T2I prompts saved to ${T2I_FILE}"
