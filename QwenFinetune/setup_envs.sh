#!/bin/bash
# One-shot setup for both conda environments on a fresh machine.
#
# Creates two envs:
#   vllm_infer  — for batch inference  (torch 2.5.1 + vllm 0.8.5 + ms-swift)
#   swift_train — for GRPO/LoRA training (torch 2.8.0+cu126 + vllm 0.10.2 + deepspeed + ms-swift)
#
# Usage:
#   bash setup_envs.sh              # setup both envs
#   bash setup_envs.sh vllm_infer   # setup inference env only
#   bash setup_envs.sh swift_train  # setup training env only
#
# After setup:
#   conda activate vllm_infer  && bash eval_swift_cot.sh
#   conda activate swift_train && bash train_swift_cot.sh

set -e

# ── Detect conda prefix ───────────────────────────────────────────────────────
# Machines may have conda at /opt/conda or /home/aiscuser/.conda; detect both.
# Prefer the envs directory that conda actually uses (from conda info).
# Fall back to well-known paths only if conda is not on PATH.
if command -v conda &>/dev/null; then
    CONDA_ENVS_ROOT="$(conda info --json 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['envs_dirs'][0])" 2>/dev/null)"
fi
if [ -z "${CONDA_ENVS_ROOT}" ] || [ ! -d "${CONDA_ENVS_ROOT}" ]; then
    if   [ -d "${HOME}/.conda/envs" ];         then CONDA_ENVS_ROOT="${HOME}/.conda/envs"
    elif [ -d "/home/aiscuser/.conda/envs" ];  then CONDA_ENVS_ROOT="/home/aiscuser/.conda/envs"
    elif [ -d "/opt/conda/envs" ];             then CONDA_ENVS_ROOT="/opt/conda/envs"
    else
        echo "[ERROR] Cannot find conda envs directory. Is conda installed?"
        exit 1
    fi
fi
echo "[INFO] Using conda envs root: ${CONDA_ENVS_ROOT}"

# ── Helper ────────────────────────────────────────────────────────────────────
pip_for() { echo "${CONDA_ENVS_ROOT}/$1/bin/pip"; }

setup_vllm_infer() {
    local ENV="vllm_infer"
    local PIP; PIP=$(pip_for "${ENV}")

    echo ""
    echo "============================================"
    echo "Setting up: ${ENV}"
    echo "  torch 2.5.1 (cu124) | vllm 0.8.5 | ms-swift[llm]>=4.0"
    echo "============================================"

    conda create -y -n "${ENV}" python=3.10

    echo "[INFO] Installing PyTorch 2.5.1 (cu124) ..."
    ${PIP} install torch==2.5.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

    echo "[INFO] Installing ms-swift and dependencies ..."
    ${PIP} install "ms-swift[llm]==4.0.2" "scipy>=1.11" "datasets>=2.18" "autoawq"

    echo "[INFO] Installing quantization dependencies ..."
    ${PIP} install "optimum==1.23.3" "auto-gptq==0.7.1" "bitsandbytes"

    echo "[INFO] Pinning transformers ..."
    ${PIP} install "transformers==4.57.6"

    echo "[INFO] Installing vLLM 0.8.5 (pinned after ms-swift to avoid override) ..."
    ${PIP} install "vllm==0.8.5"

    echo "[INFO] Installing outlines for constrained decoding ..."
    ${PIP} install outlines

    echo "[INFO] Verifying ..."
    ${CONDA_ENVS_ROOT}/${ENV}/bin/python3.10 -c \
        "import torch, vllm, swift; print('torch:', torch.__version__, '| vllm:', vllm.__version__, '| swift:', swift.__version__)"

    echo "[OK] ${ENV} ready."
}

setup_swift_train() {
    local ENV="swift_train"
    local PIP; PIP=$(pip_for "${ENV}")
    local PYTHON="${CONDA_ENVS_ROOT}/${ENV}/bin/python3.10"

    echo ""
    echo "============================================"
    echo "Setting up: ${ENV}"
    echo "  torch 2.8.0 (cu126) | vllm 0.10.2 | deepspeed | ms-swift 4.1.0.dev0 (GitHub) | trl 0.28.0"
    echo "============================================"

    conda create -y -n "${ENV}" python=3.10

    echo "[INFO] Installing PyTorch 2.8.0 (cu126, compatible with CUDA driver 12080) ..."
    ${PIP} install torch==2.8.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

    echo "[INFO] Installing ms-swift 4.1.0.dev0 from GitHub main ..."
    ${PIP} install "git+https://github.com/modelscope/ms-swift.git"

    echo "[INFO] Installing DeepSpeed ..."
    ${PIP} install deepspeed

    # vllm 0.10.2: fixes Qwen3MoE FusedMoE _load_w2 bug from 0.8.5,
    # satisfies trl 0.28.0's requirement (vllm>=0.10.2,<0.13.0),
    # and supports swift rollout for server-mode GRPO.
    echo "[INFO] Installing vLLM 0.10.2 (torch 2.8.0 compatible, Qwen3MoE fix) ..."
    ${PIP} install "vllm==0.10.2"

    # Pin AFTER vllm to ensure vllm's deps don't override these versions.
    echo "[INFO] Pinning transformers + trl (verified working combo) ..."
    ${PIP} install "transformers==4.57.6" "trl==0.28.0"

    echo "[INFO] Installing bitsandbytes for QLoRA ..."
    ${PIP} install bitsandbytes

    echo "[INFO] Verifying ..."
    ${PYTHON} -c \
        "import torch, swift, deepspeed, trl, vllm; print('torch:', torch.__version__, '| swift:', swift.__version__, '| deepspeed:', deepspeed.__version__, '| trl:', trl.__version__, '| vllm:', vllm.__version__)"

    echo "[OK] ${ENV} ready."
}

# ── Main ──────────────────────────────────────────────────────────────────────
TARGET="${1:-both}"

case "${TARGET}" in
    vllm_infer)  setup_vllm_infer ;;
    swift_train) setup_swift_train ;;
    both)        setup_vllm_infer; setup_swift_train ;;
    *)
        echo "[ERROR] Unknown target: ${TARGET}. Use vllm_infer, swift_train, or both."
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "All done. Next steps:"
echo "  Inference : conda init bash && exec bash && conda activate vllm_infer  && bash eval_swift_cot.sh"
echo "  Training  : conda init bash && exec bash && conda activate swift_train && bash train_swift_cot.sh"
echo "============================================"
