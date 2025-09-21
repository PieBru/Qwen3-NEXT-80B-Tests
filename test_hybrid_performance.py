#!/usr/bin/env python3
"""Test the hybrid model loader performance"""

import sys
import torch
import time
from pathlib import Path
sys.path.insert(0, 'src')

from model_loader_hybrid import HybridModelLoader
from config import SystemConfig
from inference import MoEInferencePipeline

print("=" * 50)
print("Testing Hybrid GPU/CPU Model Loading")
print("=" * 50)

# Configure
config = SystemConfig()
config.model.local_model_path = "models/qwen3-80b-bnb"

# Load model
print("\n1. Loading model with hybrid strategy...")
loader = HybridModelLoader(config)
start = time.time()
model, tokenizer = loader.load_model()
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")

# Check device placement
print("\n2. Checking device placement...")
gpu_params = 0
cpu_params = 0
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        gpu_params += 1
    else:
        cpu_params += 1

print(f"Parameters on GPU: {gpu_params}")
print(f"Parameters on CPU: {cpu_params}")

# Create pipeline
print("\n3. Creating inference pipeline...")
pipeline = MoEInferencePipeline(
    model=model,
    tokenizer=tokenizer,
    expert_manager=loader.expert_cache_manager
)

# Test inference speed
print("\n4. Testing inference speed...")
test_prompt = "2+2="

print(f"Prompt: '{test_prompt}'")
start = time.time()
output = pipeline.generate(test_prompt, max_new_tokens=5, temperature=0.1)
inference_time = time.time() - start

print(f"Output: '{output}'")
print(f"Inference time: {inference_time:.1f}s")

tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(test_prompt))
if inference_time > 0:
    tok_per_sec = tokens / inference_time
    print(f"Speed: {tok_per_sec:.2f} tokens/second")

    if tok_per_sec < 1:
        print("\n❌ Still too slow! Need further optimization.")
    elif tok_per_sec < 10:
        print("\n⚠️ Better but not at target speed yet.")
    else:
        print("\n✅ Achieved target speed of 10+ tokens/second!")

print("\n" + "=" * 50)