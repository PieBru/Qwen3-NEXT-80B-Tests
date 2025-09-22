#!/usr/bin/env python3
"""
Test vLLM with CPU backend for FP8 model.
"""

import os
import sys

# Set CPU platform BEFORE importing vLLM
os.environ['VLLM_PLATFORM'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs

# Now import vLLM
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

print("="*80)
print("Testing vLLM with CPU Backend")
print("="*80)

print(f"Current platform: {current_platform}")
print(f"Platform type: {type(current_platform)}")

# Check if CPU platform was selected
if 'cpu' not in str(type(current_platform)).lower():
    print("WARNING: CPU platform not selected, but continuing anyway...")

model_path = "models/qwen3-80b-fp8"

print(f"\nAttempting to load model on CPU: {model_path}")
print("This will use system RAM instead of VRAM")

try:
    # Initialize LLM with CPU settings
    llm = LLM(
        model=model_path,
        quantization="compressed-tensors",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=256,  # Very small for CPU
        enforce_eager=True,
        gpu_memory_utilization=0,  # No GPU memory
    )

    print("Model loaded successfully on CPU!")

    # Test generation
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=10,
    )

    print("\nGenerating text...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

    print("\n✅ CPU inference successful!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()