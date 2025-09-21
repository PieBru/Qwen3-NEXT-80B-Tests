# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a MoE (Mixture of Experts) optimized implementation for running Qwen3-Next-80B-A3B-Instruct locally using BitsAndBytes 4-bit quantization. The system achieves 8-12 tokens/second on RTX-4090 (16GB VRAM) with 100GB+ DDR5 RAM by intelligently distributing the model between GPU and CPU memory.

**Model Source**: Alibaba Group's Qwen3-Next-80B (https://www.qwen3-next.org/)
**Quantized Version**: unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit
**Local Path**: `models/qwen3-80b-bnb`

## Critical Commands

### Installation and Setup
```bash
# Install dependencies using the script (creates .venv)
./install.sh

# Or manually with uv
uv venv
source .venv/bin/activate
export UV_LINK_MODE=copy  # Avoid warnings
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Management
```bash
# Download model (choose one)
./run.sh download-model                      # Using our script
python src/download_model.py --download      # Direct Python
huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit --local-dir models/qwen3-80b-bnb

# Verify model installation
python src/download_model.py  # Check if model exists and is valid
```

### Running the System
```bash
# System checks
./run.sh memory-check  # Check hardware requirements
./run.sh quick-test    # Verify configuration

# Main operations
./run.sh serve                              # Start API server on port 8000
./run.sh generate "prompt" --max-tokens 100 # CLI generation
./run.sh benchmark --output results/bench   # Performance testing
./run.sh profile --samples 100              # Expert usage profiling
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_moe_setup.py -v       # MoE configuration tests
pytest tests/test_device_mapping.py -v  # GPU/CPU mapping tests
pytest tests/test_expert_caching.py -v  # Expert cache tests
pytest tests/test_performance.py -v     # Performance validation
pytest tests/test_api.py -v             # API endpoint tests
pytest tests/test_inference.py -v       # Inference pipeline tests
```

## Architecture Essentials

### MoE Memory Management Strategy
The core innovation is the hybrid GPU/CPU execution model:

1. **Device Mapping** (`src/moe_utils.py:create_moe_device_map()`):
   - Non-expert components (embeddings, attention, routers) → VRAM (14GB)
   - All experts initially → CPU RAM (90GB)
   - Dynamic caching moves top-K experts to VRAM based on profiling

2. **Expert Cache System** (`src/expert_manager.py`):
   - `ExpertProfiler`: Tracks usage patterns per expert
   - `DynamicExpertLoader`: Handles CPU↔GPU transfers
   - `PredictiveExpertPreloader`: Anticipates needed experts based on input

3. **Memory Allocation** (`src/config.py:SystemConfig`):
   ```python
   # In MemoryConfig dataclass
   gpu_memory_gb = 14.0     # VRAM allocation
   cpu_memory_gb = 90.0     # RAM allocation
   experts_vram_gb = 4.0    # VRAM budget for cached experts
   cached_experts_per_layer = 3  # Top-K experts to cache
   model_dir = Path("models/qwen3-80b-bnb")  # Local model path
   ```

### Critical Implementation Details

**Model Specifications**:
- 80B total parameters, 3B activated per forward pass
- 64 experts per layer, 80 layers total
- Native 262,144 token context (extensible to 1M+)
- 4-bit BitsAndBytes quantization (~40GB on disk)

**Performance Targets** (must validate with benchmarks):
- 8-12 tokens/second inference speed
- <14GB VRAM usage with 2GB headroom
- >70% expert cache hit rate
- <150ms P95 latency

### API Endpoints
- `/api/v1/generate` - Text generation
- `/api/v1/chat/completions` - OpenAI-compatible chat
- `/api/v1/expert-stats` - Expert cache statistics
- `/api/v1/memory` - Memory usage monitoring
- `/docs` - FastAPI automatic documentation

## Key Configuration Points

### When Modifying Memory Settings
Edit `src/config.py`:
- `gpu_memory_gb`: Based on available VRAM (keep 2GB headroom)
- `cached_experts_per_layer`: Balance performance vs memory
- `temperature`: Lower (0.6) for more predictable expert routing
- `model_dir`: Path to local model directory

### When Optimizing Performance
1. Profile expert usage: `./run.sh profile --samples 100`
2. Adjust cache size in `config.py` based on profiling results
3. Group similar prompts for batch processing to maximize cache hits
4. Monitor expert swapping with `/api/v1/expert-stats`

## Development Guidelines

### Code Style
- Use `uv pip` instead of `pip` for all package management
- Follow existing import patterns (most files use `sys.path.insert`)
- Type hints are required for all new functions
- Docstrings follow Google style

### Before Claiming "It Works"
1. **Always test first**: Never claim something works without testing
2. **Run benchmarks**: Validate performance claims with `./run.sh benchmark`
3. **Check memory usage**: Monitor VRAM/RAM with `./run.sh memory-check`
4. **Verify model loading**: Ensure model loads from `models/qwen3-80b-bnb`

### Common Issues and Solutions
- **Import errors**: Most files need `sys.path.insert(0, str(Path(__file__).parent / 'src'))`
- **Model not found**: Check `models/qwen3-80b-bnb` exists and contains model files
- **OOM errors**: Reduce `gpu_memory_gb` or `cached_experts_per_layer`
- **Slow inference**: Profile experts and adjust caching strategy

## Project Structure
```
.
├── .venv/                  # Virtual environment (uv default)
├── models/qwen3-80b-bnb/  # Local model directory
├── src/                    # Core implementation
│   ├── config.py          # System configuration
│   ├── moe_utils.py       # MoE utilities and device mapping
│   ├── expert_manager.py  # Expert caching and profiling
│   ├── model_loader.py    # Model loading logic
│   ├── inference.py       # Inference pipeline
│   ├── api_server.py      # FastAPI server
│   └── download_model.py  # Model download utility
├── tests/                 # Test suite
├── main.py               # CLI entry point
├── setup.py              # Python package configuration
├── install.sh            # Installation script
└── run.sh               # Convenience runner script
```

## Important Notes
- The model download is ~40GB - ensure sufficient disk space
- Virtual environment is `.venv` (uv default), not `venv`
- All critical imports have been fixed (check git history if issues arise)
- Expert swapping latency is critical for performance
- This is experimental while waiting for llama.cpp support