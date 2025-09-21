# Spec Requirements Document

> Spec: vLLM FP8 Integration
> Created: 2025-09-21

## Overview

Implement vLLM with FP8 quantization to achieve 10+ tokens/second inference speed for the Qwen3-80B MoE model. This replaces the current slow BitsAndBytes implementation with a proven high-performance solution that properly utilizes hybrid GPU/CPU execution.

## User Stories

### High-Performance Inference

As a developer, I want to run Qwen3-80B at 10+ tokens/second, so that I can use the model for production workloads.

The current BitsAndBytes implementation runs at 0.12 tok/s due to CPU-only execution. By switching to vLLM with FP8 quantization and proper MoE CPU offloading, we can achieve 10-20 tok/s as confirmed by Reddit users with identical hardware.

### Efficient Resource Utilization

As a system administrator, I want the model to properly utilize both GPU and CPU resources, so that inference is fast while memory usage is optimized.

vLLM will place non-expert components (embeddings, attention, routers) on GPU while offloading the 24,576 experts to CPU RAM, achieving optimal performance with available hardware.

## Spec Scope

1. **Model Download** - Download and setup neuralmagic/Qwen3-NEXT-80B-A3B-Instruct-FP8 model
2. **vLLM Installation** - Install vLLM >=0.6.0 with FP8 and CPU offloading support
3. **Server Implementation** - Configure vLLM server with optimal settings for MoE
4. **Performance Validation** - Verify 10+ tok/s inference speed
5. **Migration Guide** - Document transition from BitsAndBytes to vLLM

## Out of Scope

- Modifying the existing BitsAndBytes implementation
- Supporting other quantization formats (AWQ, GPTQ)
- Multi-GPU setup
- Custom model training or fine-tuning

## Expected Deliverable

1. Working vLLM server achieving 10-20 tokens/second on RTX 4090 + 128GB RAM
2. Simple curl/API interface for testing inference speed
3. Documented configuration matching successful Reddit implementation