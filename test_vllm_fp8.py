#!/usr/bin/env python3
"""
Test vLLM with FP8 Qwen3-Next model.
This script tests the model with the new vLLM version that supports Qwen3-Next.
"""

import os
import sys
from pathlib import Path

def test_vllm_qwen3_next():
    """Test vLLM with Qwen3-Next FP8 model."""
    print("=" * 80)
    print("Testing vLLM with Qwen3-Next FP8 Model")
    print("=" * 80)

    try:
        from vllm import LLM, SamplingParams
        import vllm

        print(f"vLLM version: {vllm.__version__}")

        model_path = "models/qwen3-80b-fp8"

        # Check if model exists
        if not Path(model_path).exists():
            print(f"Error: Model not found at {model_path}")
            return False

        print(f"\nLoading model from {model_path}...")
        print("This may take a few minutes...")

        # Initialize LLM with FP8 support
        llm = LLM(
            model=model_path,
            quantization="compressed-tensors",  # For FP8
            gpu_memory_utilization=0.3,  # Use minimal GPU memory
            trust_remote_code=True,
            enforce_eager=True,  # Disable compilation for first test
            max_model_len=512,  # Very small context for testing
            tensor_parallel_size=1,
        )

        print("\nModel loaded successfully!")

        # Test generation
        prompts = [
            "The capital of France is",
            "Machine learning is",
        ]

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.95,
        )

        print("\nGenerating text...")
        outputs = llm.generate(prompts, sampling_params)

        print("\n" + "=" * 80)
        print("Generation Results:")
        print("=" * 80)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")

        print("\n" + "=" * 80)
        print("✅ vLLM FP8 test successful!")
        print("=" * 80)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure vLLM is installed with: uv pip install vllm")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vllm_support():
    """Check which architectures vLLM supports."""
    try:
        import vllm
        from vllm.model_executor.models import ModelRegistry

        print(f"\nvLLM version: {vllm.__version__}")
        print("\nSupported Qwen architectures:")

        # List all registered models that contain "Qwen"
        for name in dir(ModelRegistry):
            if "qwen" in name.lower():
                print(f"  - {name}")

        # Try to check if Qwen3Next is supported
        try:
            from vllm.model_executor.models.qwen import QwenForCausalLM
            print("  - QwenForCausalLM (found)")
        except:
            pass

        try:
            from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
            print("  - Qwen2ForCausalLM (found)")
        except:
            pass

        try:
            from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
            print("  - Qwen3NextForCausalLM (found)")
        except:
            print("  - Qwen3NextForCausalLM (NOT found)")

    except Exception as e:
        print(f"Error checking vLLM support: {e}")


def main():
    """Main test function."""
    print("\nChecking vLLM architecture support...")
    check_vllm_support()

    print("\n" + "=" * 80)
    print("Starting FP8 model test...")
    print("=" * 80)

    success = test_vllm_qwen3_next()

    if not success:
        print("\n⚠️ Test failed. Please check the error messages above.")
        sys.exit(1)

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()