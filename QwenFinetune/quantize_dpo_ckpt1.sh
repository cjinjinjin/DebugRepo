#!/bin/bash
# Quantize DPO v12 checkpoint-1 (best DPO checkpoint, 47.9% fully compliant)
# Pipeline: merge LoRA → build calib data → GPTQ INT4 quantize → smoke test → eval
#
# Usage:
#   bash quantize_dpo_ckpt1.sh [checkpoint_dir]
#
# Example:
#   bash quantize_dpo_ckpt1.sh /vc_data/.../v12-20260407-052859/checkpoint-1

BASE_MODEL="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
DPO_CKPT="${1:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_dpo_lora_cot_refine/v12-20260407-052859/checkpoint-1}"
MERGED_MODEL_PATH="${DPO_CKPT}/merged_model"
QUANTIZED_MODEL_PATH="${MERGED_MODEL_PATH}_gptq_int4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
CALIB_DATA="${DATA_DIR}/calib_dpo_ckpt1.jsonl"
GT_JSONL="${DATA_DIR}/dpo_combined_train_cot.jsonl"
EVAL_DATA="${DATA_DIR}/dpo_combined_eval_cot.jsonl"
PRIVATE_TSV="RawData/UHRS2K_SD_Random200_0324.tsv"
SMOKE_INPUT="${QUANTIZED_MODEL_PATH}/smoke_input_10.jsonl"
SMOKE_OUTPUT="${QUANTIZED_MODEL_PATH}/smoke_output.jsonl"
EVAL_RESULTS_DIR="${QUANTIZED_MODEL_PATH}/eval_results"

PYTHON_INFER="/home/aiscuser/.conda/envs/vllm_infer/bin/python3.10"
PYTHON_TRAIN="/home/aiscuser/.conda/envs/swift_train/bin/python3.10"

echo "============================================"
echo "Base model   : ${BASE_MODEL}"
echo "DPO adapter  : ${DPO_CKPT}"
echo "Merged model : ${MERGED_MODEL_PATH}"
echo "Quantized to : ${QUANTIZED_MODEL_PATH}"
echo "Calib data   : ${CALIB_DATA}"
echo "GT JSONL     : ${GT_JSONL}"
echo "============================================"

# ── Step 0: merge LoRA adapter into base model ─────────────────────────────
if [ -d "${MERGED_MODEL_PATH}" ]; then
    echo "[INFO] Merged model already exists at ${MERGED_MODEL_PATH}, skipping merge."
else
    echo "[INFO] Merging DPO LoRA adapter into base model ..."
    ${PYTHON_TRAIN} -m swift.cli.export \
        --model        "${BASE_MODEL}" \
        --adapters     "${DPO_CKPT}" \
        --merge_lora   true \
        --output_dir   "${MERGED_MODEL_PATH}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Merge failed. Exiting."
        exit 1
    fi
    echo "[INFO] Merge done: ${MERGED_MODEL_PATH}"
fi

# ── Step 1: build calibration dataset ──────────────────────────────────────
# Prefer GT JSONL (real chosen responses from DPO training) for higher quality
# calibration. Falls back to TSV + dummy assistant if GT not available.
if [ -f "${CALIB_DATA}" ]; then
    echo "[INFO] Calibration data already exists at ${CALIB_DATA}, skipping."
else
    if [ -f "${GT_JSONL}" ]; then
        echo "[INFO] Building calibration dataset from GT JSONL (real assistant turns) ..."
        ${PYTHON_INFER} "${SCRIPT_DIR}/prepare_calib_data.py" \
            --gt_jsonl      "${GT_JSONL}" \
            --private_tsv   "${PRIVATE_TSV}" \
            --n_private     512 \
            --n_public      0 \
            --output_jsonl  "${CALIB_DATA}"
    else
        echo "[WARN] GT JSONL not found at ${GT_JSONL}, falling back to TSV ..."
        ${PYTHON_INFER} "${SCRIPT_DIR}/prepare_calib_data.py" \
            --private_tsv   "${PRIVATE_TSV}" \
            --n_private     256 \
            --n_public      0 \
            --output_jsonl  "${CALIB_DATA}"
    fi

    if [ $? -ne 0 ]; then
        echo "[ERROR] Calibration data preparation failed. Exiting."
        exit 1
    fi
fi
N_CALIB=$(wc -l < "${CALIB_DATA}")
echo "[INFO] Calibration data: ${N_CALIB} records"

# ── Step 2: GPTQ INT4 quantization via swift export ────────────────────────
# max_length=4096: covers P95 of input+output tokens (avg_input=1303, avg_output=946)
# Only use task-domain calibration data (no alpaca mix) for better activation coverage
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
        --quant_n_samples ${N_CALIB} \
        --quant_batch_size 4 \
        --max_length      4096 \
        --output_dir      "${QUANTIZED_MODEL_PATH}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Quantization failed. Exiting."
        exit 1
    fi
    echo "[INFO] Quantization done: ${QUANTIZED_MODEL_PATH}"
fi

# ── Step 3: smoke test via vLLM (gptq + gptq_marlin) ──────────────────────
echo ""
echo "[INFO] Running smoke test (10 samples) on quantized model ..."
mkdir -p "$(dirname "${SMOKE_INPUT}")"
head -10 "${CALIB_DATA}" > "${SMOKE_INPUT}"

# 3a. Test with standard gptq kernel
echo "[INFO] Smoke test: quantization=gptq ..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
${PYTHON_INFER} "${SCRIPT_DIR}/run_vllm_smoke_test.py" \
    --model          "${QUANTIZED_MODEL_PATH}" \
    --input_jsonl    "${SMOKE_INPUT}" \
    --output_jsonl   "${SMOKE_OUTPUT}" \
    --tp             4 \
    --enable_reasoning

if [ $? -ne 0 ]; then
    echo "[ERROR] Smoke test (gptq) failed. Check quantized model."
    exit 1
fi

# 3b. Test with gptq_marlin kernel (2-3x faster inference if compatible)
SMOKE_OUTPUT_MARLIN="${QUANTIZED_MODEL_PATH}/smoke_output_marlin.jsonl"
echo "[INFO] Smoke test: quantization=gptq_marlin ..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
${PYTHON_INFER} "${SCRIPT_DIR}/run_vllm_smoke_test.py" \
    --model          "${QUANTIZED_MODEL_PATH}" \
    --input_jsonl    "${SMOKE_INPUT}" \
    --output_jsonl   "${SMOKE_OUTPUT_MARLIN}" \
    --tp             4 \
    --quantization   gptq_marlin \
    --enable_reasoning

if [ $? -ne 0 ]; then
    echo "[WARN] Smoke test (gptq_marlin) failed. Marlin kernel may not be compatible."
    echo "       Fall back to --quantization gptq for serving."
else
    echo "[INFO] gptq_marlin compatible! Use --quantization gptq_marlin for faster serving."
fi

# ── Step 4: full eval on quantized model (190 samples) ─────────────────────
if [ -f "${EVAL_DATA}" ]; then
    echo ""
    echo "[INFO] Running full evaluation (190 samples) on quantized model ..."
    mkdir -p "${EVAL_RESULTS_DIR}"
    EVAL_OUTPUT="${EVAL_RESULTS_DIR}/eval_swift_output.jsonl"
    EVAL_REPORT="${EVAL_RESULTS_DIR}/eval_report.json"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ${PYTHON_INFER} -m swift.cli.infer \
        --model                        "${QUANTIZED_MODEL_PATH}" \
        --val_dataset                  "${EVAL_DATA}" \
        --max_length                   8192 \
        --max_new_tokens               4096 \
        --infer_backend                vllm \
        --max_batch_size               32 \
        --vllm_tensor_parallel_size    8 \
        --result_path                  "${EVAL_OUTPUT}"

    if [ $? -eq 0 ] && [ -f "${EVAL_OUTPUT}" ]; then
        echo "[INFO] Running evaluate.py on quantized model output ..."
        ${PYTHON_INFER} "${SCRIPT_DIR}/evaluate.py" \
            --generated_file "${EVAL_OUTPUT}" \
            --report_file    "${EVAL_REPORT}" \
            --gt_file        "${EVAL_DATA}"
        echo "[INFO] Eval report: ${EVAL_REPORT}"
    else
        echo "[WARN] Inference failed or output not found. Skipping evaluation."
    fi
else
    echo "[WARN] Eval data not found at ${EVAL_DATA}, skipping full evaluation."
fi

echo ""
echo "============================================"
echo "Quantized model : ${QUANTIZED_MODEL_PATH}"
echo "Smoke output    : ${SMOKE_OUTPUT}"
echo "Eval report     : ${EVAL_RESULTS_DIR}/eval_report.json"
echo "============================================"
echo "Next steps:"
echo "  1. Compare eval report with pre-quant baseline (DPO ckpt-1: 47.9% fully compliant)"
echo "  2. Serve: bash serve_model.sh ${QUANTIZED_MODEL_PATH}"
