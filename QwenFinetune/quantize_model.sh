#!/bin/bash
# Quantize the merged DPO model to GPTQ INT4 with a mixed calibration dataset.
#
# Steps:
#   1. Build calibration data (private TSV + public wikitext-2)
#   2. GPTQ INT4 quantization via swift export (uses swift_train env)
#   3. Quick smoke test (10 samples) via vLLM (uses vllm_infer env)
#
# Usage:
#   bash quantize_model.sh [merged_model_path]
#
# Example:
#   bash quantize_model.sh /vc_data/.../checkpoint-50/merged_model

MERGED_MODEL_PATH="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v3-20260320-155846/checkpoint-50/merged_model}"
QUANTIZED_MODEL_PATH="${MERGED_MODEL_PATH}_gptq_int4"
DATA_DIR="./data"
CALIB_DATA="${DATA_DIR}/calib_data.jsonl"
PRIVATE_TSV="RawData/UHRS2K_SD_Random200_0324.tsv"
SMOKE_INPUT="${DATA_DIR}/calib_smoke_10.jsonl"
SMOKE_OUTPUT="${QUANTIZED_MODEL_PATH}/smoke_output.jsonl"
PYTHON_INFER="/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10"
PYTHON_TRAIN="/home/aiscuser/.conda/envs/swift_train/bin/python"

echo "============================================"
echo "Merged model : ${MERGED_MODEL_PATH}"
echo "Quantized to : ${QUANTIZED_MODEL_PATH}"
echo "Calib data   : ${CALIB_DATA}"
echo "Quant env    : swift_train (swift 4.0.1)"
echo "Infer env    : vllm_infer  (vllm 0.8.5)"
echo "============================================"

# ── Step 1: build calibration dataset ────────────────────────────────────────
# 私有数据（GT eval + TSV）全部采入，再补充等量 alpaca 公开数据混合
# 最终 calib_data.jsonl 里每一条都会被用到（quant_n_samples = 文件总行数）
if [ -f "${CALIB_DATA}" ]; then
    echo "[INFO] Calibration data already exists at ${CALIB_DATA}, skipping."
else
    echo "[INFO] Building calibration dataset (private full + alpaca public) ..."
    ${PYTHON_INFER} prepare_calib_data.py \
        --private_tsv   "${PRIVATE_TSV}" \
        --n_private     256 \
        --n_public      0 \
        --output_jsonl  "${CALIB_DATA}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Calibration data preparation failed. Exiting."
        exit 1
    fi
fi
N_CALIB=$(wc -l < "${CALIB_DATA}")
echo "[INFO] Calibration data: ${N_CALIB} records (all will be used)"

# ── Step 2: GPTQ INT4 quantization via swift export (swift_train env) ─────────
if [ -d "${QUANTIZED_MODEL_PATH}" ]; then
    echo "[INFO] Quantized model already exists at ${QUANTIZED_MODEL_PATH}, skipping."
else
    echo "[INFO] Running GPTQ INT4 quantization via swift export ..."
    OMP_NUM_THREADS=14 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ${PYTHON_TRAIN} -m swift.cli.export \
        --model           "${MERGED_MODEL_PATH}" \
        --quant_bits      4 \
        --quant_method    gptq_v2 \
        --dataset         "${CALIB_DATA}" \
                          'AI-ModelScope/alpaca-gpt4-data-zh#128' \
                          'AI-ModelScope/alpaca-gpt4-data-en#128' \
        --quant_n_samples $((N_CALIB + 256)) \
        --quant_batch_size 1 \
        --max_length      2048 \
        --output_dir      "${QUANTIZED_MODEL_PATH}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Quantization failed. Exiting."
        exit 1
    fi
    echo "[INFO] Quantization done: ${QUANTIZED_MODEL_PATH}"
fi

# ── Step 3: smoke test via vLLM (vllm_infer env) ─────────────────────────────
echo ""
echo "[INFO] Running smoke test (10 samples) on quantized model ..."
head -10 "${CALIB_DATA}" > "${SMOKE_INPUT}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
${PYTHON_INFER} run_vllm_smoke_test.py \
    --model          "${QUANTIZED_MODEL_PATH}" \
    --input_jsonl    "${SMOKE_INPUT}" \
    --output_jsonl   "${SMOKE_OUTPUT}" \
    --tp             4 \
    --enable_reasoning

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
