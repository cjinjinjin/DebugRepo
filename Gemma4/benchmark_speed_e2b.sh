#!/bin/bash
# Benchmark Gemma 4 E2B-it speed (Dense, 2B effective)
#
# Usage:
#   conda activate gemma4
#   bash Gemma4/benchmark_speed_e2b.sh

set -e

MODEL_ID="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/gemma-4-E2B-it"
INPUT_FILE="QwenFinetune/data/dpo_combined_eval_cot.jsonl"
SAVE_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/Gemma4_results"

echo "=========================================="
echo "Gemma 4 E2B-it Speed Benchmark"
echo "Model: ${MODEL_ID}"
echo "=========================================="

# Single run: 10 samples + 2 warmup, no-CoT, no-think, single GPU
CUDA_VISIBLE_DEVICES=0 python Gemma4/benchmark_speed.py \
    --model_id "${MODEL_ID}" \
    --input_file "${INPUT_FILE}" \
    --num_samples 10 \
    --warmup 2 \
    --no_think \
    --save_outputs "${SAVE_DIR}/gemma4_e2b_benchmark.jsonl"
