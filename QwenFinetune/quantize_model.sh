#!/bin/bash
# Quantize the merged DPO model to AWQ INT4 with a mixed calibration dataset.
#
# Steps:
#   1. Build calibration data (private TSV + public wikitext-2)
#   2. AWQ INT4 quantization with calibration
#   3. Quick smoke test (10 samples) to verify output quality
#
# Usage:
#   bash quantize_model.sh [merged_model_path]
#
# Example:
#   bash quantize_model.sh /vc_data/.../checkpoint-50/merged_model

MERGED_MODEL_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v3-20260320-155846/checkpoint-50/merged_model}"
QUANTIZED_MODEL_PATH="${MERGED_MODEL_PATH}_awq_int4"
DATA_DIR="./data"
CALIB_DATA="${DATA_DIR}/calib_data.jsonl"
PRIVATE_TSV="RawData/UHRS2K_SD_Random200_0324.tsv"
SMOKE_INPUT="${DATA_DIR}/calib_smoke_10.jsonl"
SMOKE_OUTPUT="${QUANTIZED_MODEL_PATH}/smoke_output.jsonl"
PYTHON="/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10"

echo "============================================"
echo "Merged model : ${MERGED_MODEL_PATH}"
echo "Quantized to : ${QUANTIZED_MODEL_PATH}"
echo "Calib data   : ${CALIB_DATA}"
echo "============================================"

# ── Step 1: build calibration dataset ────────────────────────────────────────
if [ -f "${CALIB_DATA}" ]; then
    echo "[INFO] Calibration data already exists at ${CALIB_DATA}, skipping."
else
    echo "[INFO] Building calibration dataset ..."
    ${PYTHON} prepare_calib_data.py \
        --private_tsv   "${PRIVATE_TSV}" \
        --n_private     64 \
        --n_public      64 \
        --output_jsonl  "${CALIB_DATA}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Calibration data preparation failed. Exiting."
        exit 1
    fi
fi
echo "[INFO] Calibration data: $(wc -l < "${CALIB_DATA}") records"

# ── Step 2: AWQ INT4 quantization ────────────────────────────────────────────
if [ -d "${QUANTIZED_MODEL_PATH}" ]; then
    echo "[INFO] Quantized model already exists at ${QUANTIZED_MODEL_PATH}, skipping."
else
    echo "[INFO] Running AWQ INT4 quantization ..."
    ${PYTHON} -m swift.cli.export \
        --model          "${MERGED_MODEL_PATH}" \
        --quant_bits     4 \
        --quant_method   awq \
        --calib_dataset  "${CALIB_DATA}" \
        --output_dir     "${QUANTIZED_MODEL_PATH}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Quantization failed. Exiting."
        exit 1
    fi
    echo "[INFO] Quantization done: ${QUANTIZED_MODEL_PATH}"
fi

# ── Step 3: smoke test ────────────────────────────────────────────────────────
echo ""
echo "[INFO] Running smoke test (10 samples) on quantized model ..."
head -10 "${CALIB_DATA}" > "${SMOKE_INPUT}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
${PYTHON} -m swift.cli.infer \
    --model                        "${QUANTIZED_MODEL_PATH}" \
    --val_dataset                  "${SMOKE_INPUT}" \
    --max_length                   8192 \
    --infer_backend                vllm \
    --max_batch_size               4 \
    --vllm_tensor_parallel_size    4 \
    --quantization                 awq \
    --result_path                  "${SMOKE_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Smoke test failed. Check quantized model."
    exit 1
fi

echo ""
echo "============================================"
echo "Quantized model : ${QUANTIZED_MODEL_PATH}"
echo "Smoke output    : ${SMOKE_OUTPUT}"
echo "============================================"
echo "Next step — start serving:"
echo "  bash serve_model.sh ${QUANTIZED_MODEL_PATH}"
