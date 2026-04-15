# 1. 重装 PyTorch (兼容 CUDA 12.x 驱动)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 3. 配置 accelerate 多卡
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 8
num_machines: 1
mixed_precision: bf16
EOF
