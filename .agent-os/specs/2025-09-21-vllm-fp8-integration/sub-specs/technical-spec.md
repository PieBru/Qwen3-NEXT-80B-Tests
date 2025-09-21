# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-21-vllm-fp8-integration/spec.md

## Technical Requirements

### Model Requirements
- Download neuralmagic/Qwen3-NEXT-80B-A3B-Instruct-FP8 (~80GB)
- Store in models/qwen3-80b-fp8/ directory
- Verify model integrity after download

### vLLM Configuration
- Version: vLLM >= 0.6.0 with FP8 support
- CPU offloading: 100GB for expert layers
- GPU memory utilization: 0.9 (use 90% of VRAM)
- Enforce eager mode for MoE compatibility
- Trust remote code for custom model architecture

### Server Parameters
- Port: 8000 (configurable)
- Max model length: 8192 tokens
- Tensor parallel size: 1 (single GPU)
- dtype: float8 for FP8 quantization

### Performance Targets
- Inference speed: 10-20 tokens/second
- First token latency: <2 seconds
- Model load time: 2-3 minutes
- Memory usage: ~10GB VRAM, ~100GB RAM

## External Dependencies

- **vllm >= 0.6.0** - Main inference engine with MoE support
- **Justification:** Only inference framework with proven MoE CPU offloading

- **flashinfer** - Optional performance enhancement
- **Justification:** Improves MoE routing performance (optional)

- **huggingface-hub** - For model downloading
- **Justification:** Required to download FP8 model from HuggingFace