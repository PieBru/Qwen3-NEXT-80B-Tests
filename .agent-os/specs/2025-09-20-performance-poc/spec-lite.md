# Qwen3-Next-80B Performance PoC - Lite Summary

Implement a Proof of Concept to demonstrate Qwen3-Next-80B-A3B-Instruct can achieve >5 tokens/second inference speed locally using GPTQ Int4 quantization on 16GB VRAM + 120GB RAM hardware.

## Key Points
- Deploy 80B model with GPTQ Int4 quantization for memory efficiency
- Achieve and benchmark >5 tok/sec performance as primary success criteria
- Create basic inference API for testing and demonstration
- Optimize memory usage across VRAM and system RAM
- Provide concrete performance metrics for stakeholder validation