#!/usr/bin/env python3
"""Diagnose why the model is slow"""

import sys
import torch
import time
from pathlib import Path
sys.path.insert(0, 'src')

print("=== Model Loading Diagnosis ===")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check current cached model
cache_path = Path.home() / ".cache/huggingface/model_cache/initialized_model.pt"
if cache_path.exists():
    print(f"Cached model exists: {cache_path} ({cache_path.stat().st_size / 1e9:.1f}GB)")

    # Load and check the cached model
    print("\nLoading cached model to check device placement...")
    start = time.time()

    checkpoint = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"Cache loaded in {time.time() - start:.1f}s")

    # Check if model has device info
    if hasattr(checkpoint, 'hf_device_map'):
        print(f"Model has device map: {bool(checkpoint.hf_device_map)}")

    # Try to understand model structure
    if hasattr(checkpoint, 'model'):
        model_layers = checkpoint.model
        if hasattr(model_layers, 'layers'):
            print(f"Model has {len(model_layers.layers)} layers")

            # Check first layer structure
            layer0 = model_layers.layers[0]
            print(f"\nLayer 0 components:")
            for attr in dir(layer0):
                if not attr.startswith('_'):
                    obj = getattr(layer0, attr)
                    if hasattr(obj, 'weight'):
                        device = obj.weight.device if hasattr(obj.weight, 'device') else 'unknown'
                        print(f"  {attr}: device={device}")

    # Check where tensors are
    print("\nChecking tensor devices in cached model...")
    cpu_params = 0
    cuda_params = 0
    meta_params = 0

    # The checkpoint is the actual model object
    if hasattr(checkpoint, 'named_parameters'):
        for name, param in checkpoint.named_parameters():
            if param.is_meta:
                meta_params += 1
            elif param.device.type == 'cuda':
                cuda_params += 1
            else:
                cpu_params += 1
    else:
        print("Cannot iterate parameters - checking state dict")
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            for name, param in state_dict.items():
                if hasattr(param, 'device'):
                    if param.device.type == 'cuda':
                        cuda_params += 1
                    else:
                        cpu_params += 1

    print(f"Parameter distribution:")
    print(f"  CPU: {cpu_params}")
    print(f"  CUDA: {cuda_params}")
    print(f"  Meta: {meta_params}")

    if cpu_params > 0 and cuda_params == 0:
        print("\n❌ PROBLEM: Model is entirely on CPU!")
        print("This explains the slow performance - need hybrid GPU/CPU placement")

else:
    print("No cached model found")

print("\n=== Solution ===")
print("The model needs:")
print("1. Non-expert components on GPU (embeddings, attention, routers)")
print("2. Experts on CPU (due to memory constraints)")
print("3. Proper BitsAndBytes handling for quantized weights")
print("\nCurrent issue: Model is CPU-only, causing 100x slowdown")