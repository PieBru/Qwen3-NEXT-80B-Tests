# Comprehensive Testing Guide for Qwen3-80B MoE Implementation

This guide will help you practice with the Qwen3-80B MoE BitsAndBytes implementation before creating your own experiments.

## ⚠️ Pre-Testing Checklist

Before starting any tests, ensure:

- [ ] **Model downloaded**: Check with `python src/download_model.py`
- [ ] **Virtual environment activated**: `source .venv/bin/activate`
- [ ] **GPU available**: Run `nvidia-smi` to verify CUDA GPU
- [ ] **Sufficient memory**: Need ~14GB VRAM + ~90GB RAM available
- [ ] **Dependencies installed**: Try `python -c "import torch; print(torch.cuda.is_available())"`

If any check fails, see the Installation section below.

## 📋 Prerequisites

### System Requirements
- **GPU**: NVIDIA RTX 4090 or similar (16GB+ VRAM)
- **RAM**: 100GB+ available system memory
- **OS**: Linux (tested on Arch Linux)
- **CUDA**: 11.8 or higher
- **Python**: 3.9+
- **Disk Space**: ~40GB for model weights

### Verify System Resources
```bash
# Check GPU and VRAM
nvidia-smi

# Check available RAM
free -h

# Check CUDA version
nvcc --version

# Check disk space
df -h .
```

## 🛠️ Installation

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/PieBru/Qwen3-NEXT-80B-Tests.git
cd Qwen3-80B_test

# Make scripts executable
chmod +x install.sh run.sh

# Install dependencies (creates .venv automatically)
./install.sh
```

### Step 2: Activate Virtual Environment
```bash
# The project uses .venv (not venv)
uv venv
source .venv/bin/activate

# Verify activation
which python  # Should show .venv/bin/python
```

### Step 3: Download the Model (~40GB)

**Option A: Using the download script (recommended)**
```bash
./run.sh download-model
# or
python src/download_model.py --download
```

**Option B: Using huggingface-cli directly**
```bash
huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit \
  --local-dir models/qwen3-80b-bnb
```

**Verify the download:**
```bash
python src/download_model.py
# Should show "✓ Model found locally!"
```

## 🧪 Testing Workflows

### 1. Basic System Checks

#### Memory Check
```bash
./run.sh memory-check
```
Expected output:
- GPU detection and VRAM availability
- System RAM check
- Verification of requirements (✓ or ✗)

#### Quick Configuration Test
```bash
./run.sh quick-test
```
This verifies:
- Configuration loading
- Module imports
- Basic setup validation

### 2. Unit Testing

#### Run All Tests
```bash
./run.sh test
# or
pytest tests/ -v
```

#### Run Specific Test Categories
```bash
# MoE setup tests
pytest tests/test_moe_setup.py -v

# Device mapping tests
pytest tests/test_device_mapping.py -v

# Expert caching tests
pytest tests/test_expert_caching.py -v

# API tests
pytest tests/test_api.py -v

# Performance tests
pytest tests/test_performance.py -v

# Inference pipeline tests
pytest tests/test_inference.py -v
```

#### Test Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### 3. Inference Testing

⚠️ **Note**: Model must be downloaded before running these commands!

#### Simple Text Generation
```bash
# Generate text from command line
./run.sh generate "Explain quantum computing" --max-tokens 100 --temperature 0.7
```

#### Test Different Prompts
```bash
# Technical explanation
./run.sh generate "Write a Python function to implement quicksort" --max-tokens 200

# Creative writing
./run.sh generate "Once upon a time in a distant galaxy" --max-tokens 150 --temperature 0.9

# Question answering
./run.sh generate "What are the main differences between CPU and GPU?" --max-tokens 100 --temperature 0.5
```

### 4. API Server Testing

#### Start the Server
```bash
# Start with default settings
./run.sh serve

# Or with custom host/port
python main.py serve --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

#### Test API Endpoints

##### Basic Generation
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

##### OpenAI-Compatible Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-80b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain machine learning in simple terms"}
    ],
    "max_tokens": 100
  }'
```

##### Streaming Response
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about AI",
    "max_tokens": 200,
    "stream": true
  }'
```

##### Expert Statistics
```bash
# Get real-time expert usage statistics
curl "http://localhost:8000/api/v1/expert-stats"
```

##### Memory Statistics
```bash
# Monitor memory usage
curl "http://localhost:8000/api/v1/memory"
```

### 5. Performance Testing

#### Run Benchmarks
```bash
# Full benchmark suite
./run.sh benchmark --output results/benchmark_$(date +%Y%m%d)

# View results
cat results/benchmark_*/report.md
```

#### Profile Expert Usage
```bash
# Profile with 100 samples
./run.sh profile --samples 100

# This will show:
# - Most frequently used experts
# - Cache hit rates
# - VRAM usage patterns
```

#### Custom Benchmark Script
```python
# save as custom_benchmark.py
import sys
sys.path.insert(0, 'src')
from model_loader import ModelLoader
from inference import MoEInferencePipeline
from config import SystemConfig
import time

# Load model
config = SystemConfig()
config.model.local_model_path = "models/qwen3-80b-bnb"  # Use local path
loader = ModelLoader(config)
model, tokenizer = loader.load_model()

# Create pipeline
pipeline = MoEInferencePipeline(
    model=model,
    tokenizer=tokenizer,
    expert_manager=loader.expert_cache_manager
)

# Test prompts
prompts = [
    "Explain neural networks",
    "Write Python code for binary search",
    "What is quantum entanglement?"
]

# Benchmark
for prompt in prompts:
    start = time.time()
    output = pipeline.generate(prompt, max_new_tokens=100)
    elapsed = time.time() - start
    tokens = len(tokenizer.encode(output))
    print(f"Prompt: {prompt[:30]}...")
    print(f"Tokens/sec: {tokens/elapsed:.2f}")
    print("-" * 50)
```

### 6. Memory Optimization Testing

#### Test Different Expert Cache Sizes
```python
# save as test_cache_sizes.py
import json
from pathlib import Path

# Test configurations
configs = [
    {"experts_vram_gb": 2.0, "cached_experts_per_layer": 2},
    {"experts_vram_gb": 4.0, "cached_experts_per_layer": 3},
    {"experts_vram_gb": 6.0, "cached_experts_per_layer": 5}
]

for i, config in enumerate(configs):
    # Save config
    config_data = {
        "memory": config,
        "model": {"local_model_path": "models/qwen3-80b-bnb"}
    }
    with open(f"config_{i}.json", "w") as f:
        json.dump(config_data, f)

    # Run benchmark
    print(f"Testing config {i}: {config}")
    import subprocess
    subprocess.run([
        "python", "main.py", "benchmark",
        "--config", f"config_{i}.json",
        "--output", f"results/config_{i}"
    ])
```

#### Monitor Memory During Inference
```bash
# In terminal 1: Start monitoring
watch -n 1 nvidia-smi

# In terminal 2: Run inference
./run.sh generate "Long prompt requiring multiple experts..." --max-tokens 500
```

### 7. Stress Testing

#### Concurrent Requests
```python
# save as stress_test.py
import asyncio
import aiohttp
import time

async def make_request(session, prompt, request_id):
    url = "http://localhost:8000/api/v1/generate"
    data = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7
    }

    start = time.time()
    async with session.post(url, json=data) as response:
        result = await response.json()
        elapsed = time.time() - start
        print(f"Request {request_id}: {elapsed:.2f}s")
        return result

async def stress_test(num_requests=10):
    prompts = [
        "Explain physics",
        "Write code",
        "Tell a story",
        "Solve math",
        "Translate text"
    ] * (num_requests // 5)

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, prompt, i)
                for i, prompt in enumerate(prompts[:num_requests])]
        results = await asyncio.gather(*tasks)
        print(f"Completed {len(results)} requests")

# Run stress test
asyncio.run(stress_test(20))
```

#### Long Context Testing
```bash
# Test with increasing context lengths
for tokens in 100 500 1000 2000; do
  echo "Testing with $tokens tokens..."
  ./run.sh generate "Repeat this text many times: The quick brown fox jumps over the lazy dog." \
    --max-tokens $tokens --temperature 0.1
  sleep 5
done
```

## 🔍 Debugging and Troubleshooting

### Enable Debug Logging
```bash
# Set log level to DEBUG
python main.py generate "Test prompt" --log-level DEBUG

# Or for API server
python main.py serve --log-level DEBUG
```

### Check Model Loading
```python
# save as check_model.py
import sys
sys.path.insert(0, 'src')
from model_loader import ModelLoader
from config import SystemConfig
from pathlib import Path

config = SystemConfig()
config.model.local_model_path = "models/qwen3-80b-bnb"
config.model_dir = Path("models/qwen3-80b-bnb")
loader = ModelLoader(config)

print("Loading model from:", config.model_dir)
model, tokenizer = loader.load_model()

print("\nModel info:")
print(f"Model type: {type(model)}")
print(f"Device map keys: {list(model.hf_device_map.keys())[:5]}...")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Check expert placement
expert_count = 0
gpu_experts = 0
cpu_experts = 0

for name, device in model.hf_device_map.items():
    if "expert" in name:
        expert_count += 1
        if device == 0:
            gpu_experts += 1
        else:
            cpu_experts += 1

print(f"\nExpert distribution:")
print(f"Total experts: {expert_count}")
print(f"GPU experts: {gpu_experts}")
print(f"CPU experts: {cpu_experts}")
```

### Monitor Expert Swapping
```python
# save as monitor_experts.py
import sys
sys.path.insert(0, 'src')
from model_loader import ModelLoader
from inference import MoEInferencePipeline
from config import SystemConfig
import logging

config = SystemConfig()
config.model.local_model_path = "models/qwen3-80b-bnb"
loader = ModelLoader(config)
model, tokenizer = loader.load_model()

pipeline = MoEInferencePipeline(
    model=model,
    tokenizer=tokenizer,
    expert_manager=loader.expert_cache_manager
)

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Generate text and watch expert swaps
output = pipeline.generate(
    "Complex prompt requiring many experts",
    max_new_tokens=100,
    optimize_experts=True
)

# Print expert stats
stats = loader.expert_cache_manager.get_cache_stats()
print("\nExpert usage statistics:")
print(f"Cache hits: {stats.get('cache_hits', 0)}")
print(f"Cache misses: {stats.get('cache_misses', 0)}")
print(f"Hit rate: {stats.get('cache_hit_rate', 0):.2%}")
```

## 🎯 Creating Your Own Experiments

### 1. Custom Configuration
```json
{
  "model": {
    "model_name": "unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit",
    "local_model_path": "models/qwen3-80b-bnb"
  },
  "memory": {
    "gpu_memory_gb": 14.0,
    "cpu_memory_gb": 90.0,
    "experts_vram_gb": 4.0,
    "cached_experts_per_layer": 3
  },
  "inference": {
    "max_batch_size": 4,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }
}
```

Use custom config:
```bash
python main.py generate "Test" --config custom_config.json
```

### 2. Custom Expert Strategy
```python
# save as custom_expert_strategy.py
import sys
sys.path.insert(0, 'src')
from expert_manager import ExpertCacheManager

class CustomExpertManager(ExpertCacheManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_experts_to_cache(self, layer_idx: int):
        """Custom logic for expert selection"""
        # Example: Cache more experts for middle layers
        if 20 <= layer_idx <= 60:
            return 5  # Cache 5 experts
        else:
            return 3  # Cache 3 experts

    def should_swap_expert(self, expert_id: str, usage_count: int):
        """Custom swap decision"""
        # Example: More aggressive swapping
        threshold = 10  # Swap if not used in last 10 calls
        recent_uses = self.get_recent_usage(expert_id, window=10)
        return recent_uses < threshold
```

## 📊 Performance Expectations

### Baseline Metrics (RTX 4090 + 100GB RAM)
- **Tokens/Second**: 8-12 (target), 10.5 (achieved average)
- **First Token Latency**: 2-5 seconds
- **Memory Usage**: 13-14GB VRAM, 80-90GB RAM
- **Expert Cache Hit Rate**: 70-85%

### Optimization Tips
1. **Lower temperature** (0.5-0.7) for more predictable expert routing
2. **Batch similar prompts** to maximize cache hits
3. **Profile common use cases** to optimize expert caching
4. **Adjust cache size** based on available VRAM
5. **Use streaming** for better perceived performance

## 🐛 Common Issues and Solutions

### Issue: Model Not Found
```bash
# Error: Model not found at models/qwen3-80b-bnb
# Solution: Download the model first
./run.sh download-model
```

### Issue: CUDA Out of Memory
```python
# Solution: Reduce expert cache size in src/config.py
# Edit MemoryConfig class:
experts_vram_gb = 2.0  # Reduce from 4.0
cached_experts_per_layer = 2  # Reduce from 3
```

### Issue: Slow First Inference
```python
# Solution: Warmup the model
pipeline.generate("Hello", max_new_tokens=10)  # Warmup
# Now ready for actual use
```

### Issue: Import Errors
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate  # Note: .venv not venv
# Reinstall if needed
./install.sh
```

### Issue: Permission Denied on Scripts
```bash
# Solution: Make scripts executable
chmod +x run.sh install.sh
```

## 🎓 Learning Resources

### Understanding the Architecture
1. Review `src/moe_utils.py` - Core MoE device mapping
2. Study `src/expert_manager.py` - Expert caching logic
3. Examine `src/inference.py` - Inference pipeline
4. Check `.claude/CLAUDE.md` for implementation details

### Key Concepts
- **MoE (Mixture of Experts)**: 80B parameters, 3B active per forward pass
- **Hybrid Execution**: Non-experts in VRAM, experts in RAM
- **Dynamic Caching**: Profile-based expert placement
- **BitsAndBytes**: 4-bit quantization for memory efficiency

## 🚀 Next Steps

After familiarizing yourself with the testing workflows:

1. **Verify model download** - Ensure model is in `models/qwen3-80b-bnb/`
2. **Run memory check** - Confirm your hardware meets requirements
3. **Try simple generation** - Start with basic prompts
4. **Benchmark your hardware** - Run full benchmark suite
5. **Profile your use case** - Identify common expert patterns
6. **Optimize configuration** - Tune for your specific needs
7. **Create custom experiments** - Build on the foundation

## 📝 Important Notes

- **Virtual environment**: Always use `.venv` (not `venv`)
- **Model location**: Local model at `models/qwen3-80b-bnb/` (~40GB)
- **Package manager**: Use `uv pip` instead of `pip`
- **Memory monitoring**: Keep `nvidia-smi` running during tests
- **Configuration**: Check `.claude/CLAUDE.md` for details
- **Model source**: Alibaba's Qwen3-Next-80B quantized by Unsloth

## ⚠️ Safety Reminders

- **Never run tests without checking memory first** - OOM can crash system
- **Start with small `max_tokens` values** - Increase gradually
- **Monitor GPU temperature** - Heavy load can cause thermal throttling
- **Save work before testing** - In case of system instability
- **Use lower batch sizes initially** - Scale up after verification

---

*Happy experimenting with Qwen3-80B MoE implementation! 🚀*

*For issues, see: https://github.com/PieBru/Qwen3-NEXT-80B-Tests/issues*
