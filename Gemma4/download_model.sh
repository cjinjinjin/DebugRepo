#!/bin/bash
# Download Gemma 4 26B-A4B-it and move to shared CKPT storage
# Usage:
#   bash Gemma4/download_model.sh
#   HF_TOKEN=hf_xxx bash Gemma4/download_model.sh
set -euo pipefail

REPO_ID="google/gemma-4-26B-A4B-it"
MODEL_NAME="gemma-4-26B-A4B-it"
LOCAL_DIR="./${MODEL_NAME}"
CKPT_ROOT="/vc_data/shares/bingads.algo.prod.adsplus/ProdAdsPlusShare/Team/RichAds/AIGC/CKPT"
TARGET_DIR="${CKPT_ROOT}/${MODEL_NAME}"

# Check HF token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN not set. Gemma 4 is a gated model."
    echo "  export HF_TOKEN=hf_xxx"
    echo "  bash Gemma4/download_model.sh"
    exit 1
fi

echo "============================================"
echo "Download Gemma 4 26B-A4B-it"
echo "============================================"
echo "Repo:       ${REPO_ID}"
echo "Local dir:  ${LOCAL_DIR}"
echo "Target dir: ${TARGET_DIR}"
echo ""

# Step 1: Download
echo "[Step 1] Downloading from HuggingFace ..."
huggingface-cli download "${REPO_ID}" \
    --local-dir "${LOCAL_DIR}" \
    --local-dir-use-symlinks False \
    --token "${HF_TOKEN}"

echo "[OK] Downloaded to ${LOCAL_DIR}"

# Step 2: Move to shared storage
if [ ! -d "${CKPT_ROOT}" ]; then
    echo "[WARN] ${CKPT_ROOT} not found. Keeping files in ${LOCAL_DIR}"
    echo "Done! Model path: ${LOCAL_DIR}"
    exit 0
fi

if [ -d "${TARGET_DIR}" ]; then
    echo "[WARN] ${TARGET_DIR} already exists. Skipping move."
    echo "       Delete it first if you want to re-download."
else
    echo "[Step 2] Copying to ${TARGET_DIR} ..."
    cp -r "${LOCAL_DIR}" "${TARGET_DIR}"
    echo "[OK] Copied to ${TARGET_DIR}"
    echo "[INFO] You can remove the local copy: rm -rf ${LOCAL_DIR}"
fi

echo ""
echo "============================================"
echo "Done! Model path for training/inference:"
echo "  ${TARGET_DIR}"
echo "============================================"
