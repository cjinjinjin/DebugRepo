#!/bin/bash
# Gemma 4 一键部署：环境配置 → 模型下载 → smoke test → zero-shot 评估
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash Gemma4/run_all.sh
#
# 前提：
#   1. 新机器已安装 conda 和 CUDA driver
#   2. 已在 https://huggingface.co/google/gemma-4-26B-A4B-it 接受许可协议
#   3. HF_TOKEN 已设置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Pre-checks ────────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN not set."
    echo "  export HF_TOKEN=hf_xxx"
    echo "  bash Gemma4/run_all.sh"
    exit 1
fi

if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Please install conda first."
    exit 1
fi

echo "============================================"
echo "Gemma 4 Full Pipeline"
echo "============================================"
echo ""

# ── Step 1: Setup environment ─────────────────────────────────────────────────
echo "########################################"
echo "# Step 1/4: Setup conda environment"
echo "########################################"
bash "${SCRIPT_DIR}/setup_env.sh"

# Activate env — find pip/python paths
if command -v conda &>/dev/null; then
    CONDA_ENVS_ROOT="$(conda info --json 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['envs_dirs'][0])" 2>/dev/null)"
fi
if [ -z "${CONDA_ENVS_ROOT:-}" ] || [ ! -d "${CONDA_ENVS_ROOT}" ]; then
    if   [ -d "${HOME}/.conda/envs" ];         then CONDA_ENVS_ROOT="${HOME}/.conda/envs"
    elif [ -d "/home/aiscuser/.conda/envs" ];  then CONDA_ENVS_ROOT="/home/aiscuser/.conda/envs"
    elif [ -d "/opt/conda/envs" ];             then CONDA_ENVS_ROOT="/opt/conda/envs"
    fi
fi
export PATH="${CONDA_ENVS_ROOT}/gemma4/bin:${PATH}"

echo ""

# ── Step 2: Download model ───────────────────────────────────────────────────
echo "########################################"
echo "# Step 2/4: Download model"
echo "########################################"
bash "${SCRIPT_DIR}/download_model.sh"

echo ""

# ── Step 3: Smoke test ───────────────────────────────────────────────────────
echo "########################################"
echo "# Step 3/4: Smoke test (single sample)"
echo "########################################"
bash "${SCRIPT_DIR}/run_smoke_test.sh"

echo ""

# ── Step 4: Zero-shot evaluation ─────────────────────────────────────────────
echo "########################################"
echo "# Step 4/4: Zero-shot evaluation"
echo "########################################"
bash "${SCRIPT_DIR}/run_zeroshot_eval.sh"

echo ""
echo "============================================"
echo "All done!"
echo ""
echo "Results:"
echo "  Report: ${SCRIPT_DIR}/results/gemma4_zeroshot_report.json"
echo "  Output: ${SCRIPT_DIR}/results/gemma4_zeroshot_eval.jsonl"
echo ""
echo "Next steps (if zero-shot results unsatisfactory):"
echo "  bash Gemma4/train_sft.sh"
echo "  bash Gemma4/train_dpo.sh /path/to/sft_merged_model"
echo "============================================"
