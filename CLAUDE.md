# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a MoE (Mixture of Experts) optimized implementation for running Qwen3-Next-80B-A3B-Instruct locally using BitsAndBytes 4-bit quantization. The system achieves 8-12 tokens/second on RTX-4090 (16GB VRAM) with 100GB+ DDR5 RAM by intelligently distributing the model between GPU and CPU memory.

## Critical Commands

### Installation and Setup
```bash
# Use uv pip instead of pip for all installations
export UV_LINK_MODE=copy  # Avoid warnings with uv pip

# Install dependencies
./install.sh  # Note: Update this script to use 'uv pip' instead of 'pip'

# Activate environment
source venv/bin/activate
```

### Running the System
```bash
# Quick system check
./run.sh memory-check  # Check if hardware meets requirements
./run.sh quick-test    # Verify configuration

# Main operations
./run.sh serve         # Start API server on port 8000
./run.sh generate "prompt" --max-tokens 100  # CLI generation
./run.sh benchmark --output results/bench    # Performance testing
./run.sh profile --samples 100              # Expert usage profiling
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_moe_setup.py -v       # MoE configuration tests
pytest tests/test_device_mapping.py -v   # GPU/CPU mapping tests
pytest tests/test_expert_caching.py -v   # Expert cache tests
pytest tests/test_performance.py -v      # Performance validation
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

3. **Memory Allocation** (`src/config.py`):
   ```python
   gpu_memory_gb = 14.0     # VRAM allocation
   cpu_memory_gb = 90.0     # RAM allocation
   experts_vram_gb = 4.0    # VRAM budget for cached experts
   cached_experts_per_layer = 3  # Top-K experts to cache
   ```

### Critical Implementation Details

**Model**: `unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit`
- 80B total parameters, 3B activated per forward pass
- 64 experts per layer, 80 layers total
- Native 262,144 token context (extensible to 1M+)

**Performance Targets**:
- 8-12 tokens/second (must validate with benchmarks)
- <14GB VRAM usage
- >70% expert cache hit rate

### API Endpoints
- `/api/v1/generate` - Text generation
- `/api/v1/chat/completions` - OpenAI-compatible chat
- `/api/v1/expert-stats` - Expert cache statistics
- `/api/v1/memory` - Memory usage monitoring

## Key Configuration Points

### When Modifying Memory Settings
Edit `src/config.py`:
- Adjust `gpu_memory_gb` based on available VRAM (keep 2GB headroom)
- Modify `cached_experts_per_layer` to balance performance vs memory
- Lower `temperature` (0.6 instead of 0.7) for more predictable expert routing

### When Optimizing Performance
1. Profile expert usage first: `./run.sh profile --samples 100`
2. Adjust cache size in `config.py` based on profiling results
3. Group similar prompts together for batch processing to maximize cache hits

## Important Notes
- Always use `uv pip` instead of `pip` for package management
- The model download is ~40GB - use `./run.sh download-model` to fetch
- Never claim performance targets are met without running `./run.sh benchmark`
- Expert swapping latency is critical - monitor with `/api/v1/expert-stats`