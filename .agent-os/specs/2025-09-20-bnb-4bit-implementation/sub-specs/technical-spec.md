# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-20-bnb-4bit-implementation/spec.md

> Created: 2025-09-20
> Version: 1.0.0

## Technical Requirements

### MoE-Aware BitsAndBytes 4-bit Quantization Setup
- Install transformers from main branch for latest BnB support: `pip install git+https://github.com/huggingface/transformers`
- Install BitsAndBytes: `pip install bitsandbytes>=0.41.0`
- Install accelerate for device mapping: `pip install accelerate>=0.20.0`
- Configure BitsAndBytesConfig with 4-bit quantization, nf4 compute dtype, and nested quantization
- Set bnb_4bit_quant_type="nf4" for optimal quality-size tradeoff
- Configure max_memory={0: "14GB", "cpu": "90GB"} for hybrid execution

### MoE Model Loading Configuration
- Load model from "unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit" repository
- Implement custom device mapping strategy:
  * Non-expert layers (embeddings, attention, layer norms) → GPU (VRAM)
  * Router/gating networks → GPU (VRAM)
  * Expert networks → CPU (RAM) with dynamic caching
  * Top-K frequently used experts → GPU (VRAM) after profiling
- Enable torch_dtype=torch.bfloat16 for memory efficiency
- Configure trust_remote_code=True for custom model components
- Set low_cpu_mem_usage=True with offload_folder="offload" for overflow

### Hybrid GPU/CPU Memory Optimization
- **VRAM Allocation (14GB target)**:
  * Token embeddings and output projections: ~2GB
  * Attention layers (Q,K,V) for all layers: ~4GB
  * Layer normalization and routers: ~2GB
  * Top 3-5 frequently used experts per layer: ~4GB
  * KV cache and activations: ~2GB
- **System RAM Allocation (90GB)**:
  * Remaining experts (majority of model parameters): ~80GB
  * Buffer for expert swapping: ~10GB
- Implement expert profiling to identify frequently activated experts
- Create ExpertCacheManager for dynamic expert placement
- Monitor and optimize expert activation patterns

### Inference Pipeline Configuration
- Tokenizer setup with proper padding and truncation for 262K context
- Generation parameters: temperature=0.7, top_p=0.8, top_k=20, do_sample=True
- Implement streaming generation for long outputs
- Configure repetition penalty and length penalty for quality control
- Set up proper stopping criteria and EOS token handling

### Context Window Management
- Support context lengths up to 262,144 tokens
- Implement sliding window attention for ultra-long contexts
- Memory-efficient batching for variable-length inputs
- Context truncation strategies for inputs exceeding limits
- Efficient padding and attention mask generation

### MoE Performance Optimizations
- **Expert Placement Strategy**:
  * Profile expert usage on representative data
  * Cache top 3-5 experts per layer in VRAM
  * Implement predictive expert loading based on input patterns
  * Batch similar inputs to reuse cached experts
- **Runtime Optimizations**:
  * Enable model.eval() mode for inference
  * Use torch.no_grad() context manager
  * Implement asynchronous expert loading from CPU to GPU
  * Optimize expert routing decisions with lower temperature
  * Monitor expert swap overhead and adjust caching strategy
- **Target Performance**: 8-12 tokens/second with optimal expert placement

## MoE Implementation Details

### Custom Device Mapping Implementation
```python
def create_moe_device_map(model_name):
    device_map = {
        # Core components in VRAM
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
    }

    # For each layer
    for i in range(num_layers):
        # Non-expert components to GPU
        device_map[f"model.layers.{i}.self_attn"] = 0
        device_map[f"model.layers.{i}.input_layernorm"] = 0
        device_map[f"model.layers.{i}.post_attention_layernorm"] = 0
        device_map[f"model.layers.{i}.block_sparse_moe.gate"] = 0

        # Experts to CPU initially
        for j in range(num_experts):
            device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = "cpu"

    return device_map
```

### Expert Profiling and Caching
```python
class ExpertCacheManager:
    def __init__(self, vram_budget_gb=6, num_cached_experts=3):
        self.vram_budget = vram_budget_gb * 1024**3
        self.num_cached_experts = num_cached_experts
        self.expert_usage_stats = {}
        self.cached_experts = set()

    def profile_expert_usage(self, sample_inputs):
        # Track expert activation patterns
        # Return top-K most frequently used experts
        pass

    def optimize_placement(self, top_experts):
        # Move frequently used experts to VRAM
        # Maintain memory budget constraints
        pass
```

### Runtime Expert Management
- Implement predictive expert loading based on input token patterns
- Asynchronous expert transfer between CPU and GPU during inference
- Monitor expert swap latency and adjust caching threshold dynamically
- Group similar prompts in batches to maximize expert cache hits

## External Dependencies

- **transformers** (main branch) - Latest BitsAndBytes integration and Qwen3 MoE support
- **bitsandbytes>=0.41.0** - 4-bit quantization backend with latest optimizations
- **torch>=2.0.0** - PyTorch with native compilation and memory optimizations
- **accelerate>=0.20.0** - Custom device mapping and hybrid GPU/CPU execution
- **sentencepiece>=0.1.99** - Tokenizer backend for Qwen3 models
- **psutil>=5.9.0** - Memory monitoring for expert placement optimization
- **numpy>=1.24.0** - Expert usage statistics and profiling