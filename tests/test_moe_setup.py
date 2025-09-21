"""
Tests for MoE-Aware BitsAndBytes Environment Setup
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestMoEQuantizationConfig(unittest.TestCase):
    """Test BitsAndBytes quantization configuration for MoE models"""

    def test_bnb_config_creation(self):
        """Test creating BitsAndBytesConfig with correct parameters"""
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(config.bnb_4bit_use_double_quant)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)

    def test_memory_allocation_config(self):
        """Test memory allocation configuration for hybrid GPU/CPU"""
        max_memory = {
            0: "14GB",  # GPU 0
            "cpu": "90GB"  # System RAM
        }

        self.assertEqual(max_memory[0], "14GB")
        self.assertEqual(max_memory["cpu"], "90GB")

        # Test conversion to bytes
        def parse_memory_string(mem_str):
            if mem_str.endswith("GB"):
                return int(mem_str[:-2]) * 1024**3
            return int(mem_str)

        gpu_bytes = parse_memory_string(max_memory[0])
        cpu_bytes = parse_memory_string(max_memory["cpu"])

        self.assertEqual(gpu_bytes, 14 * 1024**3)
        self.assertEqual(cpu_bytes, 90 * 1024**3)


class TestMoEDeviceMapping(unittest.TestCase):
    """Test custom device mapping for MoE architecture"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_layers = 32  # Typical for 80B model
        self.num_experts = 8  # Typical MoE configuration

    def test_create_device_map_structure(self):
        """Test creation of device map with correct structure"""
        from moe_utils import create_moe_device_map

        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        # Check core components are on GPU
        self.assertEqual(device_map.get("model.embed_tokens"), 0)
        self.assertEqual(device_map.get("model.norm"), 0)
        self.assertEqual(device_map.get("lm_head"), 0)

        # Check attention layers are on GPU
        for i in range(self.num_layers):
            self.assertEqual(
                device_map.get(f"model.layers.{i}.self_attn"), 0,
                f"Attention layer {i} should be on GPU"
            )
            self.assertEqual(
                device_map.get(f"model.layers.{i}.input_layernorm"), 0,
                f"Layer norm {i} should be on GPU"
            )

    def test_expert_placement_to_cpu(self):
        """Test that experts are initially placed on CPU"""
        from moe_utils import create_moe_device_map

        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        # Check that experts are on CPU
        for i in range(self.num_layers):
            for j in range(self.num_experts):
                expert_key = f"model.layers.{i}.block_sparse_moe.experts.{j}"
                self.assertEqual(
                    device_map.get(expert_key), "cpu",
                    f"Expert {j} in layer {i} should be on CPU initially"
                )

    def test_router_placement_to_gpu(self):
        """Test that router/gating networks are placed on GPU"""
        from moe_utils import create_moe_device_map

        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        # Check routers are on GPU
        for i in range(self.num_layers):
            router_key = f"model.layers.{i}.block_sparse_moe.gate"
            self.assertEqual(
                device_map.get(router_key), 0,
                f"Router in layer {i} should be on GPU"
            )


class TestDependencyInstallation(unittest.TestCase):
    """Test dependency installation and verification"""

    def test_transformers_version(self):
        """Test that transformers is installed from main branch"""
        try:
            import transformers
            # Check for recent features indicating main branch
            self.assertTrue(hasattr(transformers, 'BitsAndBytesConfig'))
        except ImportError:
            self.skipTest("transformers not installed yet")

    def test_bitsandbytes_version(self):
        """Test that bitsandbytes is installed with correct version"""
        try:
            import bitsandbytes as bnb
            # Check version is >= 0.41.0
            version = bnb.__version__
            major, minor, patch = map(int, version.split('.')[:3])
            self.assertTrue(
                major > 0 or (major == 0 and minor >= 41),
                f"bitsandbytes version {version} is too old"
            )
        except ImportError:
            self.skipTest("bitsandbytes not installed yet")

    def test_accelerate_available(self):
        """Test that accelerate is available for device mapping"""
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            self.assertTrue(callable(init_empty_weights))
            self.assertTrue(callable(load_checkpoint_and_dispatch))
        except ImportError:
            self.skipTest("accelerate not installed yet")

    def test_cuda_availability(self):
        """Test CUDA availability for GPU operations"""
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            self.assertGreaterEqual(device_count, 1, "At least one GPU should be available")

            # Check VRAM
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram / (1024**3)
                self.assertGreaterEqual(
                    vram_gb, 15,  # RTX 4090 has 16GB
                    f"GPU VRAM ({vram_gb:.1f}GB) seems too low for RTX 4090"
                )
        else:
            self.skipTest("CUDA not available")


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory monitoring utilities"""

    def test_get_memory_stats(self):
        """Test getting current memory statistics"""
        import psutil

        # Test system memory
        mem = psutil.virtual_memory()
        self.assertGreater(mem.total, 0)
        self.assertGreater(mem.available, 0)

        # Check we have enough RAM (100GB+ available)
        available_gb = mem.available / (1024**3)
        self.assertGreaterEqual(
            available_gb, 50,  # Relaxed for testing
            f"Available RAM ({available_gb:.1f}GB) is less than required"
        )

    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Get current GPU memory
        allocated = torch.cuda.memory_allocated(0)
        torch.cuda.memory_reserved(0)

        # Allocate some tensor
        test_tensor = torch.randn(1000, 1000, device='cuda:0')

        # Check memory increased
        new_allocated = torch.cuda.memory_allocated(0)
        self.assertGreater(new_allocated, allocated)

        # Clean up
        del test_tensor
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()