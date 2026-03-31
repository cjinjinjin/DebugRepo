#!/bin/bash
# Inference pipeline for new raw TSV data using the DPO fine-tuned checkpoint.
#
# Steps:
#   1. Preprocess TSV  -> JSONL (swift infer format)
#   2. Merge LoRA adapter into base model (skipped if already done)
#   3. swift infer (vLLM, 8 GPUs)
#   4. Extract T2I prompts
#
# Usage:
#   bash infer_new_data.sh [tsv_file] [checkpoint_dir] [think|nothink]
#
# Examples:
#   bash infer_new_data.sh RawData/UHRS2K_SD_Random200_0324.tsv
#   bash infer_new_data.sh RawData/UHRS2K_SD_Random200_0324.tsv /vc_data/.../checkpoint-50 nothink

# Resolve INPUT_TSV to absolute path before cd, so relative paths from the
# caller's working directory are handled correctly regardless of where this
# script lives.
INPUT_TSV="${1:-RawData/UHRS2K_SD_Random200_0324.tsv}"
if [[ "${INPUT_TSV}" != /* ]]; then
    INPUT_TSV="$(pwd)/${INPUT_TSV}"
fi

# Always run relative to the directory containing this script so that
# prepare_infer_input.py and extract_prompts_for_t2i.py can be found.
cd "$(dirname "$0")" || exit 1

# Thinking mode: "think" (default, slower, CoT) or "nothink" (faster, no <think> block)
THINK_MODE="${3:-think}"
# Temperature: default 0.3 for prompt diversity; set 0 for greedy (most stable)
TEMPERATURE="${4:-0.3}"

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
ADAPTER_PATH="${2:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v3-20260320-155846/checkpoint-50}"
MERGED_MODEL_PATH="${ADAPTER_PATH}/merged_model"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
# Derive a short name from the TSV filename for output naming
TSV_STEM=$(basename "${INPUT_TSV}" .tsv)
RESULTS_DIR="${ADAPTER_PATH}/infer_results/${TSV_STEM}_${RUN_TS}_${THINK_MODE}_t${TEMPERATURE}"
INFER_INPUT="${RESULTS_DIR}/infer_input.jsonl"
RESULT_FILE="${RESULTS_DIR}/infer_output.jsonl"
T2I_FILE="${RESULTS_DIR}/prompts_for_t2i.txt"

mkdir -p "${RESULTS_DIR}"

# Save a copy of this script to the results dir for reproducibility
cp "$0" "${RESULTS_DIR}/infer_new_data.sh"

echo "============================================"
echo "Model    : ${MODEL_PATH}"
echo "Adapter  : ${ADAPTER_PATH}"
echo "Merged   : ${MERGED_MODEL_PATH}"
echo "Input TSV: ${INPUT_TSV}"
echo "Run TS   : ${RUN_TS}"
echo "Think    : ${THINK_MODE}"
echo "Output   : ${RESULT_FILE}"
echo "============================================"

# ── Step 1: preprocess TSV -> JSONL ──────────────────────────────────────────
echo "[INFO] Preprocessing ${INPUT_TSV} -> ${INFER_INPUT} ..."
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 prepare_infer_input.py \
    --input_tsv   "${INPUT_TSV}" \
    --output_jsonl "${INFER_INPUT}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Preprocessing failed. Exiting."
    exit 1
fi
echo "[INFO] Preprocessing done: $(wc -l < "${INFER_INPUT}") records"

# ── Step 2: merge LoRA adapter into base model ───────────────────────────────
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

# ── Step 3: batch inference ──────────────────────────────────────────────────
# Set system prompt based on THINK_MODE
if [ "${THINK_MODE}" = "nothink" ]; then
    SYSTEM_ARG="--system /no_think"
else
    SYSTEM_ARG=""
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 -m swift.cli.infer \
    --model                        "${MERGED_MODEL_PATH}" \
    --val_dataset                  "${INFER_INPUT}" \
    --max_new_tokens               2048 \
    --vllm_max_model_len           4096 \
    --infer_backend                vllm \
    --max_batch_size               32 \
    --vllm_tensor_parallel_size    8 \
    --temperature                  "${TEMPERATURE}" \
    --top_p                        0.9 \
    ${SYSTEM_ARG} \
    --result_path                  "${RESULT_FILE}"

echo ""
echo "Inference done."

# ── Locate the actual output file (swift may ignore --result_path) ────────────
if [ ! -f "${RESULT_FILE}" ]; then
    echo "[WARN] ${RESULT_FILE} not found. Searching for swift output ..."
    LATEST=$(find . /tmp ~/ms-image-quality-filters-aether-module-main \
        -name "*.jsonl" -newer "${INFER_INPUT}" \
        -not -path "*/data/*" \
        -not -name "sft_*.jsonl" \
        -not -name "dpo_*.jsonl" \
        -not -name "infer_input.jsonl" \
        2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    if [ -n "${LATEST}" ]; then
        echo "[INFO] Renaming ${LATEST} -> ${RESULT_FILE}"
        mv "${LATEST}" "${RESULT_FILE}"
    else
        echo "[ERROR] No swift output file found. Check swift infer logs above."
        exit 1
    fi
fi

# ── Step 4: extract T2I prompts ───────────────────────────────────────────────
/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10 extract_prompts_for_t2i.py \
    --infer_file  "${RESULT_FILE}" \
    --gt_file     "${INFER_INPUT}" \
    --output_file "${T2I_FILE}"

echo "T2I prompts saved to ${T2I_FILE}"
echo "Results dir: ${RESULTS_DIR}"
