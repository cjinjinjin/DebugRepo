#!/bin/bash
# Gemma 4 环境配置脚本（新机器一键部署）
#
# 创建 conda 环境 gemma4，包含推理和训练所有依赖。
# Gemma 4 26B-A4B-it 是 gated model，需要先在 HuggingFace 上接受许可协议。
#
# Usage:
#   bash Gemma4/setup_env.sh                    # 完整安装
#   bash Gemma4/setup_env.sh --inference-only    # 只装推理依赖（不含 deepspeed/trl）
#
# After setup:
#   conda activate gemma4
#   huggingface-cli login          # 输入 HF token（Gemma 4 是 gated model）
#   bash Gemma4/run_smoke_test.sh  # 验证模型能正常加载和推理
set -euo pipefail

ENV_NAME="gemma4"
INFERENCE_ONLY=false

if [ "${1:-}" = "--inference-only" ]; then
    INFERENCE_ONLY=true
fi

# ── Detect conda ─────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    CONDA_ENVS_ROOT="$(conda info --json 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['envs_dirs'][0])" 2>/dev/null)"
fi
if [ -z "${CONDA_ENVS_ROOT:-}" ] || [ ! -d "${CONDA_ENVS_ROOT}" ]; then
    if   [ -d "${HOME}/.conda/envs" ];         then CONDA_ENVS_ROOT="${HOME}/.conda/envs"
    elif [ -d "/home/aiscuser/.conda/envs" ];  then CONDA_ENVS_ROOT="/home/aiscuser/.conda/envs"
    elif [ -d "/opt/conda/envs" ];             then CONDA_ENVS_ROOT="/opt/conda/envs"
    else
        echo "[ERROR] Cannot find conda envs directory. Is conda installed?"
        exit 1
    fi
fi

PIP="${CONDA_ENVS_ROOT}/${ENV_NAME}/bin/pip"
PYTHON="${CONDA_ENVS_ROOT}/${ENV_NAME}/bin/python3.10"

echo "============================================"
echo "Gemma 4 Environment Setup"
echo "============================================"
echo "Env name:       ${ENV_NAME}"
echo "Conda root:     ${CONDA_ENVS_ROOT}"
echo "Inference only: ${INFERENCE_ONLY}"
echo ""

# ── Create conda env ─────────────────────────────────────────────────────────
if [ -d "${CONDA_ENVS_ROOT}/${ENV_NAME}" ]; then
    echo "[WARN] Env '${ENV_NAME}' already exists. Skipping conda create."
    echo "       To recreate: conda remove -y -n ${ENV_NAME} --all && rerun this script."
else
    echo "[INFO] Creating conda env: ${ENV_NAME} (python 3.10) ..."
    conda create -y -n "${ENV_NAME}" python=3.10
fi

# ── PyTorch (cu126, 必须与 torchvision 同 CUDA 版本) ────────────────────────
echo "[INFO] Installing PyTorch + torchvision + torchaudio (cu126) ..."
${PIP} install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# ── Core ML deps ─────────────────────────────────────────────────────────────
# transformers 必须用最新版，4.57.6 不支持 Gemma 4 的 Gemma4Processor
echo "[INFO] Installing transformers (latest, Gemma 4 support) + accelerate + peft ..."
${PIP} install -U transformers "accelerate>=1.0.0" "peft>=0.13.0"

# ── Gemma 4 specific: AutoProcessor needs pillow + sentencepiece ─────────────
echo "[INFO] Installing Gemma 4 processor dependencies ..."
${PIP} install pillow sentencepiece protobuf

# ── Quantization ─────────────────────────────────────────────────────────────
echo "[INFO] Installing bitsandbytes (4bit/8bit quantization) ..."
${PIP} install bitsandbytes

# ── HuggingFace CLI (for gated model login) ──────────────────────────────────
echo "[INFO] Installing huggingface_hub CLI ..."
${PIP} install "huggingface_hub[cli]"

# ── Training deps (skip if --inference-only) ──────────────────────────────────
if [ "${INFERENCE_ONLY}" = false ]; then
    echo ""
    echo "[INFO] Installing training dependencies ..."

    echo "[INFO] Installing ms-swift 4.1.0.dev0 from GitHub main ..."
    ${PIP} install "git+https://github.com/modelscope/ms-swift.git"

    echo "[INFO] Installing DeepSpeed ..."
    ${PIP} install deepspeed

    echo "[INFO] Installing trl ..."
    ${PIP} install "trl==0.28.0"

    echo "[INFO] Installing vLLM 0.19.0 ..."
    ${PIP} install "vllm==0.19.0"

    echo "[INFO] Installing datasets ..."
    ${PIP} install "datasets>=2.18"

    # Re-pin transformers to latest (ms-swift may downgrade)
    echo "[INFO] Re-pinning transformers to latest ..."
    ${PIP} install -U transformers
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying installation ..."
if [ "${INFERENCE_ONLY}" = true ]; then
    ${PYTHON} -c "
import torch, transformers, accelerate, peft, PIL
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU 0:', torch.cuda.get_device_name(0))
"
else
    ${PYTHON} -c "
import torch, transformers, accelerate, peft, PIL, swift, deepspeed, trl, vllm
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('ms-swift:', swift.__version__)
print('deepspeed:', deepspeed.__version__)
print('trl:', trl.__version__)
print('vllm:', vllm.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU 0:', torch.cuda.get_device_name(0))
"
fi

echo ""
echo "============================================"
echo "[OK] Environment '${ENV_NAME}' is ready."
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. huggingface-cli login     # paste your HF token (Gemma 4 is gated)"
echo "  3. bash Gemma4/run_smoke_test.sh"
echo ""
if [ "${INFERENCE_ONLY}" = false ]; then
    echo "Training:"
    echo "  bash Gemma4/train_sft.sh"
    echo "  bash Gemma4/train_dpo.sh /path/to/sft_merged_model"
fi
echo "============================================"
