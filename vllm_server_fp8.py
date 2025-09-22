#!/usr/bin/env python3
"""
vLLM server for FP8 quantized Qwen3-Next-80B model.
Optimized for RTX 4090 + 128GB RAM with CPU offloading.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Configuration for optimal performance
MODEL_PATH = "models/qwen3-80b-fp8"  # Local FP8 model path
# Alternative: HuggingFace model
# MODEL_PATH = "neuralmagic/Qwen3-NEXT-80B-A3B-Instruct-FP8"

# Memory configuration
GPU_MEMORY_GB = 14  # RTX 4090 has 16GB, leave 2GB headroom
CPU_OFFLOAD_GB = 90  # Use 90GB of RAM for expert offloading

# Server configuration
PORT = 8000
HOST = "0.0.0.0"


def check_prerequisites():
    """Check if system meets requirements."""
    print("Checking system prerequisites...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Performance will be degraded.")
    else:
        gpu = torch.cuda.get_device_properties(0)
        print(f"✅ GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")

    # Check model exists
    model_path = Path(MODEL_PATH)
    if model_path.exists() and model_path.is_dir():
        print(f"✅ Found local model at {MODEL_PATH}")
    else:
        print(f"ℹ️  Will download model from HuggingFace: {MODEL_PATH}")

    # Check vLLM
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed. Run: pip install vllm")
        sys.exit(1)


def create_vllm_server_command():
    """Create the vLLM server command with optimal settings."""

    cmd_parts = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(PORT),
        "--host", HOST,

        # Memory settings
        "--gpu-memory-utilization", "0.85",  # Use 85% of VRAM
        "--cpu-offload-gb", str(CPU_OFFLOAD_GB),  # CPU offloading for experts

        # MoE optimizations
        "--enforce-eager",  # Disable CUDA graphs for MoE models
        "--enable-prefix-caching",  # Cache KV for better performance

        # Quantization
        "--quantization", "compressed-tensors",  # For FP8 support

        # Performance settings
        "--max-model-len", "4096",  # Start with smaller context
        "--max-num-seqs", "4",  # Batch size

        # Logging
        "--disable-log-stats",  # Reduce overhead
    ]

    return " ".join(cmd_parts)


def create_test_script():
    """Create a test script for the vLLM server."""

    test_script = '''#!/bin/bash
# Test script for vLLM server

echo "Testing vLLM server..."

# Wait for server to start
sleep 5

# Test generation endpoint
curl -X POST http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "MODEL_PATH",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

echo ""
echo "Test complete!"
'''.replace("MODEL_PATH", MODEL_PATH)

    with open("test_vllm_server.sh", "w") as f:
        f.write(test_script)
    os.chmod("test_vllm_server.sh", 0o755)
    print("✅ Created test_vllm_server.sh")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="vLLM FP8 Server for Qwen3-80B")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check prerequisites without starting server")
    parser.add_argument("--print-command", action="store_true",
                       help="Print the vLLM command without running")
    args = parser.parse_args()

    print("="*60)
    print("vLLM FP8 Server Configuration")
    print("="*60)

    check_prerequisites()

    if args.check_only:
        print("\nPrerequisite check complete.")
        return

    # Create test script
    create_test_script()

    # Get the command
    cmd = create_vllm_server_command()

    if args.print_command:
        print("\nvLLM Server Command:")
        print("-"*60)
        print(cmd)
        print("-"*60)
        print("\nRun this command to start the server.")
    else:
        print(f"\nStarting vLLM server on http://{HOST}:{PORT}")
        print("Press Ctrl+C to stop the server.")
        print("-"*60)

        # Execute the command
        os.system(cmd)


if __name__ == "__main__":
    main()