#!/bin/bash

SFT_ADAPTER="${SFT_ADAPTER:-/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_14b_sft_lora_cot_v1/best_checkpoint}"
MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-14B"
DATA_DIR="./data"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_14b_dpo_lora_cot_v1"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_TIMEOUT=7200 \
NCCL_DEBUG=WARN \
TORCH_NCCL_BLOCKING_WAIT=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
swift rlhf \
    --rlhf_type     dpo \
    --model         ${MODEL_PATH} \
    --adapters      ${SFT_ADAPTER} \
    --dataset       ${DATA_DIR}/dpo_refine_train_cot.jsonl \
    --val_dataset   ${DATA_DIR}/dpo_refine_eval_cot.jsonl \
    --train_type    lora \
    --lora_rank     64 \
    --lora_alpha    128 \
    --num_train_epochs            50 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate               5e-5 \
    --lr_scheduler_type           cosine \
    --warmup_ratio                0.05 \
    --max_length                  8192 \
    --output_dir                  ${OUTPUT_DIR} \
    --bf16                        true \
    --gradient_checkpointing      true \
    --deepspeed                   zero2 \
    --save_steps                  5 \
    --eval_steps                  5 \
    --logging_steps               5 \
    --beta                        0.1
