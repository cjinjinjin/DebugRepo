#!/bin/bash
# Benchmark Gemma 4 E4B-it speed (Dense, 4.5B effective / 8B total)
#
# Tests: TTFT (prefill) + decode speed, with LP length sweep
# Reuses the same benchmark_speed.py as 26B-A4B-it
#
# Usage:
#   conda activate gemma4
#   bash Gemma4/benchmark_speed_e4b.sh

set -e

MODEL_ID="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-E4B-it"
INPUT_FILE="QwenFinetune/data/dpo_combined_eval_cot.jsonl"
SAVE_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Gemma4_results"

echo "=========================================="
echo "Gemma 4 E4B-it Speed Benchmark"
echo "Model: ${MODEL_ID}"
echo "=========================================="

# Single run: 10 samples + 2 warmup, no-CoT, no-think, single GPU
CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
    --model_id "${MODEL_ID}" \
    --input_file "${INPUT_FILE}" \
    --num_samples 10 \
    --warmup 2 \
    --no_think \
    --save_outputs "${SAVE_DIR}/gemma4_e4b_benchmark.jsonl"
