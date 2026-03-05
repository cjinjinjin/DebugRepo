#!/bin/bash

MODEL_PATH="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/pretrained_models/Qwen3-30B-A3B"
DATA_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/QwenFinetune/data"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT/checkpoints/qwen3_sft_lora_cot"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model        ${MODEL_PATH} \
    --dataset      ${DATA_DIR}/sft_train_cot.jsonl \
    --val_dataset  ${DATA_DIR}/sft_eval_cot.jsonl \
    --train_type   lora \
    --lora_rank    64 \
    --lora_alpha   128 \
    --lora_target_modules all-linear \
    --num_train_epochs            3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate               1e-4 \
    --lr_scheduler_type           cosine \
    --warmup_ratio                0.05 \
    --max_length                  2048 \
    --output_dir                  ${OUTPUT_DIR} \
    --bf16                        true \
    --gradient_checkpointing      true \
    --save_steps                  100 \
    --eval_steps                  100 \
    --logging_steps               10 \
    --nproc_per_node              8
