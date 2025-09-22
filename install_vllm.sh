#!/bin/bash

# Install vLLM with FP8 support
echo "Installing vLLM with FP8 support..."

# Activate virtual environment
source .venv/bin/activate

# Set UV link mode
export UV_LINK_MODE=copy

# First, ensure we have the right PyTorch version
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install vLLM dependencies first
echo "Installing vLLM dependencies..."
uv pip install packaging ninja setuptools>=49.4.0 numpy

# Install vLLM
echo "Installing vLLM (this may take a few minutes)..."
# Use pre-built wheel if available
uv pip install vllm --no-build-isolation

# Install FP8 support dependencies
echo "Installing FP8 dependencies..."
uv pip install compressed-tensors

# Verify installation
echo "Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || echo "vLLM import failed"

echo "Installation complete!"