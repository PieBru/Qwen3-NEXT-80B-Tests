# Performance Issue Analysis

## Problem
The Qwen3-80B MoE model is running at ~0.12 tokens/second instead of the expected 10+ tokens/second.

## Root Cause
The model is running **entirely on CPU** due to meta tensor issues with BitsAndBytes quantized models. This causes:
- 100% CPU usage on all 32 cores
- ~248GB RAM usage
- Extremely slow inference (~100x slower than expected)

## Why It's Happening

1. **BitsAndBytes Quantization Incompatibility**:
   - The pre-quantized model from Unsloth has meta tensors
   - These meta tensors cause errors when using `device_map` for hybrid placement
   - To avoid crashes, we force CPU-only loading

2. **Current Loading Strategy** (model_loader.py:153-176):
   ```python
   # Forces CPU-only to avoid meta tensor errors
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   device_map={'': 'cpu'}
   ```

3. **MoE Architecture Not Utilized**:
   - MoE models should have non-experts on GPU (fast)
   - Only experts on CPU (for memory)
   - Current: Everything on CPU (extremely slow)

## Expected Architecture

### Proper MoE Placement
- **GPU (10GB)**: Embeddings, attention layers, routers, layer norms
- **CPU (90GB)**: 512 experts per layer × 48 layers
- **Performance**: 10+ tokens/second

### Current Placement
- **GPU (0GB)**: Nothing
- **CPU (248GB)**: Everything
- **Performance**: 0.12 tokens/second

## Solutions to Try

### Option 1: Load with device_map="auto"
Let transformers automatically handle placement, but this often fails with BitsAndBytes models.

### Option 2: Post-load GPU Migration
Load to CPU first, then manually move non-expert components to GPU.

### Option 3: Use Different Quantization
Switch from BitsAndBytes to GPTQ or AWQ quantization that supports device mapping.

### Option 4: Wait for llama.cpp
The model will eventually be supported by llama.cpp which handles MoE efficiently.

## Immediate Fix Needed
The model needs hybrid GPU/CPU execution to achieve reasonable performance. The current CPU-only execution makes it unusable for practical inference.