#!/bin/bash
# Run script for Qwen3-Local MoE BitsAndBytes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "Qwen3-Local MoE BitsAndBytes Runner"
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running install script...${NC}"
    ./install.sh
fi

# Activate virtual environment
source .venv/bin/activate

# Parse command
COMMAND=$1
shift

case "$COMMAND" in
    "serve")
        echo -e "${GREEN}Starting API server...${NC}"
        python main.py serve "$@"
        ;;

    "generate")
        echo -e "${GREEN}Generating text...${NC}"
        python main.py generate "$@"
        ;;

    "benchmark")
        echo -e "${GREEN}Running performance benchmark...${NC}"
        python main.py benchmark "$@"
        ;;

    "profile")
        echo -e "${GREEN}Profiling expert usage...${NC}"
        python main.py profile "$@"
        ;;

    "test")
        echo -e "${GREEN}Running tests...${NC}"
        pytest tests/ -v "$@"
        ;;

    "quick-test")
        echo -e "${GREEN}Running quick inference test...${NC}"
        python -c "
import sys
sys.path.insert(0, 'src')
from config import default_config
print('Configuration loaded successfully!')
print(f'Model: {default_config.model.model_name}')
print(f'VRAM allocation: {default_config.memory.gpu_memory_gb}GB')
print(f'RAM allocation: {default_config.memory.cpu_memory_gb}GB')
print('Ready to load model.')
"
        ;;

    "memory-check")
        echo -e "${GREEN}Checking memory availability...${NC}"
        python -c "
import torch
import psutil

# Check CUDA
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu_name}')
    print(f'VRAM: {vram_gb:.1f}GB')
else:
    print('No CUDA GPU detected')

# Check RAM
mem = psutil.virtual_memory()
print(f'System RAM: {mem.total / 1024**3:.1f}GB total')
print(f'Available RAM: {mem.available / 1024**3:.1f}GB')

# Check requirements
print('')
if torch.cuda.is_available() and vram_gb >= 15:
    print('✓ GPU memory sufficient')
else:
    print('✗ Insufficient GPU memory (need 16GB)')

if mem.available / 1024**3 >= 55:
    print('✓ System RAM sufficient')
else:
    print('✗ Insufficient RAM (need 55GB available)')
"
        ;;

    "download-model")
        echo -e "${GREEN}Downloading model weights...${NC}"
        python src/download_model.py "$@"
        ;;

    *)
        echo -e "${YELLOW}Usage: ./run.sh [command] [options]${NC}"
        echo ""
        echo "Commands:"
        echo "  serve          - Start the API server"
        echo "  generate       - Generate text from a prompt"
        echo "  benchmark      - Run performance benchmarks"
        echo "  profile        - Profile expert usage"
        echo "  test           - Run the test suite"
        echo "  quick-test     - Quick configuration test"
        echo "  memory-check   - Check memory availability"
        echo "  download-model - Download model weights"
        echo ""
        echo "Examples:"
        echo "  ./run.sh serve --port 8000"
        echo "  ./run.sh generate \"Hello world\" --max-tokens 50"
        echo "  ./run.sh benchmark --output results/bench"
        echo "  ./run.sh test"
        ;;
esac