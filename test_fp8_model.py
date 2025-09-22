#!/usr/bin/env python3
"""Test script for FP8 quantized Qwen3-Next-80B model loading and inference."""

import sys
import torch
import gc
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    # GPU memory
    gpus = GPUtil.getGPUs()
    gpu_used = gpus[0].memoryUsed / 1024 if gpus else 0  # Convert to GB
    gpu_total = gpus[0].memoryTotal / 1024 if gpus else 0

    # CPU memory
    cpu_stats = psutil.virtual_memory()
    cpu_used = (cpu_stats.total - cpu_stats.available) / (1024**3)  # Convert to GB
    cpu_total = cpu_stats.total / (1024**3)

    return {
        "gpu_used_gb": gpu_used,
        "gpu_total_gb": gpu_total,
        "gpu_percent": (gpu_used / gpu_total * 100) if gpu_total > 0 else 0,
        "cpu_used_gb": cpu_used,
        "cpu_total_gb": cpu_total,
        "cpu_percent": cpu_stats.percent
    }


def print_memory_stats(label: str = ""):
    """Print formatted memory statistics."""
    stats = get_memory_stats()
    print(f"\n{'='*60}")
    if label:
        print(f"Memory Usage - {label}")
    else:
        print("Memory Usage")
    print(f"{'='*60}")
    print(f"GPU: {stats['gpu_used_gb']:.2f}/{stats['gpu_total_gb']:.2f} GB ({stats['gpu_percent']:.1f}%)")
    print(f"CPU: {stats['cpu_used_gb']:.2f}/{stats['cpu_total_gb']:.2f} GB ({stats['cpu_percent']:.1f}%)")
    print(f"{'='*60}\n")


def test_fp8_model_loading():
    """Test loading the FP8 quantized model."""

    model_path = Path("models/qwen3-80b-fp8")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False

    print(f"✅ Found FP8 model at {model_path}")
    print_memory_stats("Before Loading")

    try:
        # Load configuration
        print("\n1. Loading model configuration...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"   - Model type: {config.model_type}")
        print(f"   - Quantization: FP8 (8-bit float)")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Num layers: {config.num_hidden_layers}")
        print(f"   - Num experts: {config.num_experts}")
        print(f"   - Experts per token: {config.num_experts_per_tok}")

        # Load tokenizer
        print("\n2. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        print(f"   - Vocab size: {len(tokenizer)}")
        print(f"   - Model max length: {tokenizer.model_max_length}")

        # Set up device mapping for hybrid loading
        print("\n3. Setting up device mapping...")
        max_memory = {
            0: "14GiB",  # GPU memory limit
            "cpu": "90GiB"  # CPU memory limit
        }

        # Load model with automatic device mapping
        print("\n4. Loading FP8 model with device mapping...")
        print("   This may take 2-3 minutes...")

        start_time = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"  # Use eager attention (flash_attention_2 not installed)
        )

        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.1f} seconds")

        print_memory_stats("After Loading")

        # Test inference
        print("\n5. Testing inference...")
        test_prompt = "The capital of France is"

        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move inputs to appropriate device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        print(f"   Input: '{test_prompt}'")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=False
            )

        inference_time = time.time() - start_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Output: '{response}'")
        print(f"   Inference time: {inference_time:.2f} seconds")

        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / inference_time
        print(f"   Tokens/second: {tokens_per_second:.2f}")

        print_memory_stats("After Inference")

        print("\n✅ FP8 model test completed successfully!")

        # Analyze device placement
        print("\n6. Analyzing model device placement...")
        device_map = getattr(model, 'hf_device_map', {})
        if device_map:
            gpu_layers = sum(1 for d in device_map.values() if d == 0 or d == "cuda:0")
            cpu_layers = sum(1 for d in device_map.values() if d == "cpu")
            print(f"   - Layers on GPU: {gpu_layers}")
            print(f"   - Layers on CPU: {cpu_layers}")

        return True

    except Exception as e:
        print(f"\n❌ Error loading FP8 model: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'outputs' in locals():
            del outputs
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_stats("After Cleanup")


def main():
    """Main test execution."""
    print("="*80)
    print("FP8 Model Loading Test for Qwen3-Next-80B")
    print("="*80)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("\n⚠️  CUDA not available - will use CPU only")

    # Run the test
    success = test_fp8_model_loading()

    if success:
        print("\n" + "="*80)
        print("✅ FP8 MODEL TEST PASSED")
        print("="*80)
        print("\nNext steps:")
        print("1. Update src/config.py to use 'models/qwen3-80b-fp8' as model_dir")
        print("2. Run: ./run.sh quick-test")
        print("3. Run: ./run.sh benchmark")
        print("4. Start API server: ./run.sh serve")
    else:
        print("\n" + "="*80)
        print("❌ FP8 MODEL TEST FAILED")
        print("="*80)
        print("\nTroubleshooting:")
        print("1. Verify model files in models/qwen3-80b-fp8/")
        print("2. Check available GPU memory (need ~14GB)")
        print("3. Check available CPU memory (need ~90GB)")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())