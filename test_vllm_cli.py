#!/usr/bin/env python3
"""
Test vLLM with FP8 model using Python API directly.
This bypasses the server and tests the model loading directly.
"""

import sys
from pathlib import Path

# Test different approaches for Qwen3-Next model

def test_direct_vllm_api():
    """Test using vLLM's Python API directly."""
    print("=" * 60)
    print("Testing vLLM Direct API")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams

        model_path = "models/qwen3-80b-fp8"

        # Try different configurations
        configs_to_try = [
            {
                "name": "Standard FP8",
                "kwargs": {
                    "model": model_path,
                    "quantization": "compressed-tensors",
                    "gpu_memory_utilization": 0.85,
                    "enforce_eager": True,
                    "max_model_len": 2048,
                }
            },
            {
                "name": "With trust_remote_code",
                "kwargs": {
                    "model": model_path,
                    "quantization": "compressed-tensors",
                    "trust_remote_code": True,
                    "gpu_memory_utilization": 0.85,
                    "enforce_eager": True,
                    "max_model_len": 2048,
                }
            },
            {
                "name": "Force Qwen2 architecture",
                "kwargs": {
                    "model": model_path,
                    "quantization": "compressed-tensors",
                    "gpu_memory_utilization": 0.85,
                    "enforce_eager": True,
                    "max_model_len": 2048,
                    "hf_overrides": {"architectures": ["Qwen2ForCausalLM"]},
                }
            }
        ]

        for config in configs_to_try:
            print(f"\n📌 Trying: {config['name']}")
            print("-" * 40)

            try:
                # Create LLM instance
                print(f"Loading model from {model_path}...")
                llm = LLM(**config['kwargs'])

                # Test generation
                prompts = ["The capital of France is"]
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=50
                )

                print("Generating text...")
                outputs = llm.generate(prompts, sampling_params)

                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    print(f"Prompt: {prompt!r}")
                    print(f"Generated: {generated_text!r}")

                print(f"✅ {config['name']} worked!")
                return True

            except Exception as e:
                print(f"❌ {config['name']} failed: {e}")
                continue

        return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_vllm_offline():
    """Test offline generation without server."""
    print("\n" + "=" * 60)
    print("Testing vLLM Offline Generation")
    print("=" * 60)

    try:
        from vllm.entrypoints.offline_inference import offline_inference

        # This is a hypothetical test - the actual module might differ
        print("Note: Offline inference module structure may vary")

    except ImportError:
        print("Offline inference module not found in expected location")

    # Alternative approach using the LLM class
    test_direct_vllm_api()


def check_model_architecture():
    """Check the actual architecture in the model config."""
    print("\n" + "=" * 60)
    print("Checking Model Architecture")
    print("=" * 60)

    import json
    config_path = Path("models/qwen3-80b-fp8/config.json")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        print(f"Model architectures: {config.get('architectures', 'Not found')}")
        print(f"Model type: {config.get('model_type', 'Not found')}")

        # Check if we can modify the architecture
        print("\n💡 Suggestion: You could try modifying the config.json")
        print("   Change 'Qwen3NextForCausalLM' to 'Qwen2ForCausalLM'")
        print("   (This is risky and may not work correctly)")
    else:
        print(f"Config not found at {config_path}")


def test_with_modified_config():
    """Test with a temporarily modified config."""
    print("\n" + "=" * 60)
    print("Testing with Modified Config (Temporary)")
    print("=" * 60)

    import json
    import shutil
    from pathlib import Path

    config_path = Path("models/qwen3-80b-fp8/config.json")
    backup_path = Path("models/qwen3-80b-fp8/config.json.backup")

    try:
        # Backup original config
        if config_path.exists():
            shutil.copy(config_path, backup_path)

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Modify architecture
            original_arch = config.get('architectures', [])
            config['architectures'] = ['Qwen2ForCausalLM']

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Modified architecture from {original_arch} to ['Qwen2ForCausalLM']")

            # Try loading with modified config
            from vllm import LLM, SamplingParams

            try:
                llm = LLM(
                    model="models/qwen3-80b-fp8",
                    quantization="compressed-tensors",
                    gpu_memory_utilization=0.85,
                    enforce_eager=True,
                    max_model_len=512,  # Start small
                    trust_remote_code=True
                )

                # Test generation
                prompts = ["Hello, world!"]
                sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
                outputs = llm.generate(prompts, sampling_params)

                print("✅ Modified config approach worked!")
                for output in outputs:
                    print(f"Generated: {output.outputs[0].text}")

            except Exception as e:
                print(f"❌ Modified config approach failed: {e}")

            # Restore original config
            shutil.move(backup_path, config_path)
            print("Restored original config")

    except Exception as e:
        print(f"Error in config modification test: {e}")
        # Ensure we restore the original config
        if backup_path.exists():
            shutil.move(backup_path, config_path)


def main():
    print("vLLM CLI Testing for FP8 Model")
    print("=" * 80)

    # Check model architecture
    check_model_architecture()

    # Test direct API
    success = test_direct_vllm_api()

    if not success:
        print("\n⚠️  Standard approaches failed.")
        print("Would you like to try the modified config approach? (risky)")
        # Uncomment to test:
        # test_with_modified_config()

    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()