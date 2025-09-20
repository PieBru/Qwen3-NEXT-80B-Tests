# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-20-awq-model-migration/spec.md

> Created: 2025-09-20
> Version: 1.0.0

## Technical Requirements

### vLLM Installation with AWQ Support
- Install vLLM from main branch with precompiled AWQ support using: `VLLM_USE_PRECOMPILED=1 pip install git+https://github.com/vllm-project/vllm.git@main`
- Ensure transformers library is installed from main branch for compatibility
- Verify CUDA and PyTorch compatibility for AWQ acceleration

### Model Configuration
- **Model ID**: cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
- **Quantization**: AWQ 4-bit quantization
- **Model Parameters**: 80B total parameters, 3B activated per forward pass
- **Context Length**: Native 262,144 tokens (extensible to 1M+)
- **Data Type**: float16 for optimal memory usage

### Tensor Parallel Configuration
- **Tensor Parallel Size**: 4 (distribute across 4 GPUs)
- **Max Model Length**: 8192 tokens for inference
- **GPU Memory**: Optimize for 16GB VRAM per GPU
- **System Memory**: Utilize 120GB RAM for model weights and KV cache

### Performance Optimization
- Target inference speed: >5 tokens/second
- Memory efficiency: Minimize GPU memory fragmentation
- KV cache optimization for longer sequences
- Batch processing optimization for throughput

### Server Configuration
- Host: localhost (0.0.0.0 for external access)
- Port: 8000 (configurable)
- API compatibility: OpenAI-compatible endpoints
- Request timeout: 300 seconds for long generations

### Command Line Configuration
```bash
vllm serve cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8000
```

## Approach

### Migration Strategy
1. **Environment Preparation**: Install vLLM with AWQ support and verify dependencies
2. **Model Download**: Pre-download AWQ model weights to local cache
3. **Configuration Testing**: Test tensor parallel configuration with reduced parameters
4. **Performance Validation**: Benchmark against current GPTQ setup
5. **Production Deployment**: Replace GPTQ with AWQ in production environment

### Memory Management
- Pre-allocate GPU memory to prevent fragmentation
- Configure KV cache size based on available VRAM
- Monitor memory usage during inference to prevent OOM errors
- Implement memory cleanup between requests if needed

### Performance Monitoring
- Track tokens per second during inference
- Monitor GPU utilization across all tensor parallel workers
- Measure memory usage patterns for optimization
- Compare latency and throughput with GPTQ baseline

## External Dependencies

- **vLLM**: Main branch with AWQ support (VLLM_USE_PRECOMPILED=1)
- **transformers**: Latest main branch for Qwen3 compatibility
- **torch**: Compatible version with CUDA support
- **autoawq**: AWQ quantization library (may be bundled with vLLM)
- **accelerate**: For distributed model loading