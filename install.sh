#!/bin/bash
# Installation script for MoE-aware BitsAndBytes Qwen3-Local

echo "================================================"
echo "Qwen3-Local MoE BitsAndBytes Installation Script"
echo "================================================"

# Set UV link mode to avoid warnings
export UV_LINK_MODE=copy

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Verify uv installation
    if ! command -v uv &> /dev/null; then
        echo "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
fi

# Create virtual environment using uv
echo "Creating virtual environment with uv..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers from main branch
echo "Installing transformers from main branch..."
uv pip install git+https://github.com/huggingface/transformers.git@main

# Install BitsAndBytes
echo "Installing BitsAndBytes..."
uv pip install "bitsandbytes>=0.41.0"

# Install other dependencies
echo "Installing remaining dependencies..."
uv pip install "accelerate>=0.20.0"
uv pip install "sentencepiece>=0.1.99"
uv pip install "psutil>=5.9.0"
uv pip install "numpy>=1.24.0"
uv pip install "fastapi>=0.100.0"
uv pip install "uvicorn>=0.23.0"
uv pip install "pydantic>=2.0.0"
uv pip install "websockets>=10.0"
uv pip install "huggingface-hub>=0.17.0"

# Install development dependencies
echo "Installing development dependencies..."
uv pip install pytest pytest-cov pytest-asyncio black isort

# Check if setup.py or pyproject.toml exists for package installation
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing package in development mode..."
    uv pip install -e .
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import sys
import torch
import transformers
import bitsandbytes
import accelerate

print('✓ Python:', sys.version.split()[0])
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
    print('✓ VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
print('✓ Transformers version:', transformers.__version__)
print('✓ BitsAndBytes version:', bitsandbytes.__version__)
print('✓ Accelerate version:', accelerate.__version__)
print('')
print('Installation complete!')
"

echo ""
echo "================================================"
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download the model: ./run.sh download-model"
echo "2. Run quick test: ./run.sh quick-test"
echo "3. Start the server: ./run.sh serve"
echo "================================================"