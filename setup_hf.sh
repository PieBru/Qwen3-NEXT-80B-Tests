#!/bin/bash
# Setup script for HuggingFace authentication and model download

echo "================================================"
echo "HuggingFace Setup for Qwen3-80B BnB Model"
echo "================================================"

# Activate virtual environment
source venv/bin/activate

# Install huggingface-cli if not already installed
echo "Installing HuggingFace CLI..."
export UV_LINK_MODE=copy
uv pip install huggingface-hub

echo ""
echo "To authenticate with HuggingFace, run:"
echo "  huggingface-cli login"
echo ""
echo "After authentication, download the model with:"
echo ""
echo "Method 1: Using Python (recommended for automatic caching):"
echo "python -c \""
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "print('Downloading Qwen3-Next-80B-A3B-Instruct-bnb-4bit model...')"
echo "model_name = 'unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit'"
echo "print('Downloading tokenizer...')"
echo "tokenizer = AutoTokenizer.from_pretrained(model_name)"
echo "print('Downloading model weights (this may take a while, ~40GB)...')"
echo "model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')"
echo "print('Model downloaded successfully!')"
echo "\""
echo ""
echo "Method 2: Using huggingface-cli (for manual download):"
echo "  huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit --local-dir models/qwen3-80b-bnb"
echo ""
echo "The model will be cached in ~/.cache/huggingface/ by default."