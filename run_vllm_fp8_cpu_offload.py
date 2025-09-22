#!/usr/bin/env python3
"""
Run vLLM with FP8 model using CPU offloading for layers.
This script configures vLLM to offload some layers to CPU to fit in available VRAM.
"""

import subprocess
import sys
import os

def run_vllm_server():
    """Start vLLM server with CPU offloading."""

    model_path = "models/qwen3-80b-fp8"

    # vLLM command with CPU offloading options
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", "8001",  # Different port to avoid conflicts
        "--host", "0.0.0.0",

        # Quantization settings
        "--quantization", "compressed-tensors",

        # Memory management - key settings for CPU offloading
        "--gpu-memory-utilization", "0.9",  # Use most of GPU
        "--cpu-offload-gb", "100",  # Offload up to 100GB to CPU

        # CPU offloading for KV cache
        "--num-cpu-blocks-override", "8192",  # Force CPU blocks for KV cache
        "--max-cpu-loras", "4",

        # Model parallelism settings
        "--tensor-parallel-size", "1",
        "--pipeline-parallel-size", "1",

        # Context and batch settings
        "--max-model-len", "2048",
        "--max-num-seqs", "1",  # Single sequence to minimize memory
        "--max-num-batched-tokens", "2048",

        # Performance settings
        "--enforce-eager",  # Disable CUDA graphs for memory efficiency
        "--disable-log-stats",

        # Trust remote code for Qwen3-Next
        "--trust-remote-code",

        # Additional memory optimization
        "--enable-prefix-caching",
        "--disable-sliding-window",

        # Force CPU execution for some operations
        "--device", "cuda",
        "--dtype", "bfloat16",  # Use bfloat16 for better compatibility
    ]

    # Set environment variables for better memory management
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_CPU_OFFLOAD_GB"] = "100"  # Ensure CPU offloading is enabled
    env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Try FlashInfer backend

    print("Starting vLLM server with CPU offloading...")
    print(f"Command: {' '.join(cmd)}")
    print("\nEnvironment variables set:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  VLLM_CPU_OFFLOAD_GB={env['VLLM_CPU_OFFLOAD_GB']}")
    print(f"  VLLM_ATTENTION_BACKEND={env['VLLM_ATTENTION_BACKEND']}")
    print("\n" + "="*80)

    try:
        # Run the server
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def test_with_cli():
    """Alternative: Test using vLLM CLI directly."""

    model_path = "models/qwen3-80b-fp8"

    # Simple generation test with vLLM CLI
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_client",
        "--model", model_path,
        "--quantization", "compressed-tensors",
        "--gpu-memory-utilization", "0.9",
        "--cpu-offload-gb", "100",
        "--max-model-len", "1024",
        "--trust-remote-code",
        "--prompt", "The capital of France is",
        "--max-tokens", "20",
    ]

    print("Testing with vLLM CLI...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nOutput:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM with CPU offloading")
    parser.add_argument("--test", action="store_true", help="Run quick test instead of server")
    args = parser.parse_args()

    if args.test:
        test_with_cli()
    else:
        run_vllm_server()