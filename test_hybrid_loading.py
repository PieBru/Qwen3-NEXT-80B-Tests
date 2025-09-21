#!/usr/bin/env python3
"""Test hybrid GPU/CPU loading for the MoE model"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, 'src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_utils import create_moe_device_map

print("Testing hybrid GPU/CPU loading...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

model_path = "models/qwen3-80b-bnb"

# Create MoE device map
device_map = create_moe_device_map()
print(f"\nDevice map summary:")
gpu_count = sum(1 for v in device_map.values() if v == 0)
cpu_count = sum(1 for v in device_map.values() if v == "cpu")
print(f"- GPU components: {gpu_count}")
print(f"- CPU components: {cpu_count}")

# Try loading with device map
print("\nAttempting to load model with hybrid device map...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "10GB", "cpu": "100GB"}
    )
    print("✓ Model loaded successfully with hybrid placement!")

    # Check actual device placement
    if hasattr(model, 'hf_device_map'):
        print("\nActual device placement:")
        for key in ["model.embed_tokens", "model.norm", "lm_head"]:
            if key in model.hf_device_map:
                print(f"  {key}: {model.hf_device_map[key]}")

        # Check a few expert placements
        expert_on_gpu = sum(1 for k, v in model.hf_device_map.items() if "expert" in k and v == 0)
        expert_on_cpu = sum(1 for k, v in model.hf_device_map.items() if "expert" in k and v == "cpu")
        print(f"  Experts on GPU: {expert_on_gpu}")
        print(f"  Experts on CPU: {expert_on_cpu}")

except Exception as e:
    print(f"✗ Failed to load with hybrid placement: {e}")
    print(f"Error type: {type(e).__name__}")

    # Try to understand the error
    if "meta" in str(e).lower():
        print("\n⚠️ Meta tensor issue detected - BitsAndBytes models may not support device_map")
    elif "cuda" in str(e).lower() and "oom" in str(e).lower():
        print("\n⚠️ CUDA OOM - reduce GPU memory allocation")
    else:
        print("\n⚠️ Unknown error - may need different loading strategy")