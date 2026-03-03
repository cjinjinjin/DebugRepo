#!/bin/bash
# =============================================================
# 数据转换脚本：CSV -> SFT JSON
# 运行方式：bash prepare_sft_data.sh
# =============================================================

INPUT_CSV="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223_processed_results.csv"
IMAGE_FOLDER="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/UHRS_Task_BIC_evaluation_label_list_1223/"
OUTPUT_DIR="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/Data/AutoGen/PromptFollowing/sft_data/"

python PromptFollowingEvalution/prepare_sft_data.py \
    --input_csv    "$INPUT_CSV" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir   "$OUTPUT_DIR" \
    --val_ratio    0.1
