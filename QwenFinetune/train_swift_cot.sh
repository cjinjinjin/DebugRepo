#!/bin/bash

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
DATA_DIR="./data"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/qwen3_sft_lora_cot_8192"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NCCL_TIMEOUT=7200 \
NCCL_DEBUG=WARN \
TORCH_NCCL_BLOCKING_WAIT=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
swift sft \
    --model        ${MODEL_PATH} \
    --dataset      ${DATA_DIR}/sft_train_cot.jsonl \
    --val_dataset  ${DATA_DIR}/sft_eval_cot.jsonl \
    --train_type   lora \
    --lora_rank    64 \
    --lora_alpha   128 \
    --num_train_epochs            10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate               1e-4 \
    --lr_scheduler_type           cosine \
    --warmup_ratio                0.05 \
    --max_length                  8192 \
    --output_dir                  ${OUTPUT_DIR} \
    --bf16                        true \
    --gradient_checkpointing      true \
    --deepspeed                   ./ds_zero3.json \
    --save_steps                  10 \
    --eval_steps                  10 \
    --logging_steps               10
