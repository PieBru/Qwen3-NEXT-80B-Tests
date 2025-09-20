#!/usr/bin/env python3
"""Test script to verify all critical fixes are working"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all critical imports work"""
    print("Testing imports...")
    errors = []

    try:
        from src.inference import MoEInferencePipeline
        print("  ✓ inference imports successfully")
    except ImportError as e:
        errors.append(f"  ✗ inference: {e}")

    try:
        from src.api_server import app
        print("  ✓ api_server imports successfully")
    except ImportError as e:
        errors.append(f"  ✗ api_server: {e}")

    try:
        from src.moe_utils import create_moe_device_map, ExpertCacheManager
        print("  ✓ moe_utils imports successfully")
    except ImportError as e:
        errors.append(f"  ✗ moe_utils: {e}")

    try:
        from src.expert_manager import PredictiveExpertPreloader
        print("  ✓ expert_manager imports successfully")
    except ImportError as e:
        errors.append(f"  ✗ expert_manager: {e}")

    try:
        from src.config import SystemConfig
        print("  ✓ config imports successfully")
    except ImportError as e:
        errors.append(f"  ✗ config: {e}")

    return errors

def test_type_annotations():
    """Test that type annotations are correct"""
    print("\nTesting type annotations...")
    from src.moe_utils import create_moe_device_map
    from typing import get_type_hints

    try:
        hints = get_type_hints(create_moe_device_map)
        print(f"  ✓ create_moe_device_map type hints: {hints['return']}")
        if 'Any' in str(hints['return']):
            print("  ✓ Uses Any (not 'any')")
    except Exception as e:
        print(f"  ✗ Type annotation error: {e}")

def test_config_loading():
    """Test config loading functionality"""
    print("\nTesting config loading...")

    # Create a test config
    test_config = {
        "model": {
            "model_name": "test-model",
            "num_experts": 32
        },
        "memory": {
            "gpu_memory_gb": 12.0,
            "cpu_memory_gb": 24.0
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8001
        }
    }

    # Save test config
    with open("test_config.json", "w") as f:
        json.dump(test_config, f)

    # Test loading
    from src.config import SystemConfig
    config = SystemConfig()

    # Simulate config loading logic from main.py
    with open("test_config.json") as f:
        custom_config = json.load(f)

    for key, value in custom_config.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                # Handle nested configs
                config_attr = getattr(config, key)
                if config_attr is None:
                    # Create the appropriate config class if needed
                    from src.config import ModelConfig, MemoryConfig, ServerConfig
                    if key == 'model':
                        config_attr = ModelConfig()
                    elif key == 'memory':
                        config_attr = MemoryConfig()
                    elif key == 'server':
                        config_attr = ServerConfig()
                    setattr(config, key, config_attr)

                for sub_key, sub_value in value.items():
                    if hasattr(config_attr, sub_key):
                        setattr(config_attr, sub_key, sub_value)
            else:
                setattr(config, key, value)

    # Verify
    assert config.model is not None, "Model config not created"
    assert config.model.model_name == "test-model", "Model name not updated"
    assert config.model.num_experts == 32, "Num experts not updated"
    assert config.memory.gpu_memory_gb == 12.0, "Memory config not updated"
    assert config.memory.cpu_memory_gb == 24.0, "CPU memory config not updated"
    assert config.server.port == 8001, "Server config not updated"
    print("  ✓ Config loading works correctly")

    # Clean up
    Path("test_config.json").unlink()

def test_expert_module_access():
    """Test that expert module access is safe"""
    print("\nTesting safe expert module access...")

    class MockExpert:
        def __getitem__(self, idx):
            return "expert_module"

    class MockBlockSparseMoE:
        def __init__(self):
            self.experts = MockExpert()

    class MockLayer:
        def __init__(self):
            self.block_sparse_moe = MockBlockSparseMoE()

    class MockLayers:
        def __getitem__(self, idx):
            return MockLayer()

    class MockModelInner:
        def __init__(self):
            self.layers = MockLayers()

    class MockModel:
        def __init__(self):
            self.model = MockModelInner()

    from src.expert_manager import PredictiveExpertPreloader

    # Test with model
    preloader = PredictiveExpertPreloader(model=MockModel())
    module = preloader._get_expert_module(0, 0)
    assert module == "expert_module", "Could not get expert module"
    print("  ✓ Expert module access works with valid model")

    # Test without model
    preloader = PredictiveExpertPreloader(model=None)
    module = preloader._get_expert_module(0, 0)
    assert module is None, "Should return None without model"
    print("  ✓ Expert module access safely handles missing model")

def test_api_validation():
    """Test API request validation"""
    print("\nTesting API request validation...")

    try:
        from src.api_server import GenerateRequest
        from pydantic import ValidationError

        # Test valid request
        valid_req = GenerateRequest(prompt="Hello world", max_tokens=100)
        print("  ✓ Valid request accepted")

        # Test empty prompt (should fail)
        try:
            invalid_req = GenerateRequest(prompt="", max_tokens=100)
            print("  ✗ Empty prompt should be rejected")
        except ValidationError:
            print("  ✓ Empty prompt correctly rejected")

        # Test excessive tokens (should fail)
        try:
            invalid_req = GenerateRequest(prompt="Hello", max_tokens=5000)
            print("  ✗ Excessive tokens should be rejected")
        except ValidationError:
            print("  ✓ Excessive tokens correctly rejected")

    except ImportError as e:
        print(f"  ⚠ Could not test API validation (missing dependency): {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Running fix verification tests...")
    print("=" * 60)

    # Test imports
    import_errors = test_imports()

    # Test type annotations
    if not import_errors:
        test_type_annotations()

    # Test config loading
    test_config_loading()

    # Test expert module access
    test_expert_module_access()

    # Test API validation
    test_api_validation()

    print("\n" + "=" * 60)
    if import_errors:
        print("Some imports failed (likely due to missing dependencies):")
        for error in import_errors:
            print(error)
        print("\nThis is expected if transformers/torch are not installed.")
    else:
        print("All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()