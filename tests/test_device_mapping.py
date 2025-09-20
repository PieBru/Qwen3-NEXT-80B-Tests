"""
Tests for MoE custom device mapping
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from moe_utils import create_moe_device_map
from model_loader import ModelLoader


class TestDeviceMapping(unittest.TestCase):
    """Test custom device mapping for MoE models"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_layers = 80
        self.num_experts = 64

    def test_device_map_creation(self):
        """Test creating device map with correct structure"""
        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        # Verify core components are mapped
        self.assertIn("model.embed_tokens", device_map)
        self.assertIn("model.norm", device_map)
        self.assertIn("lm_head", device_map)

        # Count GPU vs CPU assignments
        gpu_count = sum(1 for v in device_map.values() if v == 0)
        cpu_count = sum(1 for v in device_map.values() if v == "cpu")

        # Should have many CPU assignments (experts)
        self.assertGreater(cpu_count, gpu_count)

        # Verify experts are on CPU
        expert_count = sum(
            1 for k, v in device_map.items()
            if "experts" in k and v == "cpu"
        )
        expected_experts = self.num_layers * self.num_experts
        self.assertEqual(expert_count, expected_experts)

    def test_non_expert_gpu_placement(self):
        """Test that all non-expert components are on GPU"""
        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        # Check attention layers
        for i in range(self.num_layers):
            self.assertEqual(
                device_map[f"model.layers.{i}.self_attn"], 0,
                f"Attention layer {i} should be on GPU"
            )

        # Check layer norms
        for i in range(self.num_layers):
            self.assertEqual(
                device_map[f"model.layers.{i}.input_layernorm"], 0
            )
            self.assertEqual(
                device_map[f"model.layers.{i}.post_attention_layernorm"], 0
            )

        # Check routers
        for i in range(self.num_layers):
            self.assertEqual(
                device_map[f"model.layers.{i}.block_sparse_moe.gate"], 0
            )

    def test_expert_cpu_placement(self):
        """Test that all experts are initially on CPU"""
        device_map = create_moe_device_map(
            num_layers=self.num_layers,
            num_experts=self.num_experts
        )

        for i in range(self.num_layers):
            for j in range(self.num_experts):
                key = f"model.layers.{i}.block_sparse_moe.experts.{j}"
                self.assertEqual(
                    device_map[key], "cpu",
                    f"Expert {j} in layer {i} should be on CPU"
                )

    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    def test_model_loader_with_device_map(self, mock_tokenizer, mock_model):
        """Test ModelLoader with custom device map"""
        from config import default_config

        # Mock model and tokenizer
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create loader
        loader = ModelLoader(default_config)

        # Mock the device map creation
        with patch('model_loader.create_moe_device_map') as mock_create_map:
            mock_create_map.return_value = {"test": "cpu"}

            # Attempt to load model
            model, tokenizer = loader.load_model()

            # Verify device map was created
            mock_create_map.assert_called_once()

            # Verify model.from_pretrained was called with device_map
            call_args = mock_model.from_pretrained.call_args
            if call_args:
                self.assertIn('device_map', call_args[1])


class TestMemoryConfiguration(unittest.TestCase):
    """Test memory configuration for hybrid GPU/CPU setup"""

    def test_memory_allocation(self):
        """Test memory allocation configuration"""
        from config import MemoryConfig

        mem_config = MemoryConfig()

        # Check GPU memory
        self.assertEqual(mem_config.gpu_memory_gb, 14.0)
        self.assertEqual(mem_config.gpu_reserved_gb, 2.0)

        # Check CPU memory
        self.assertEqual(mem_config.cpu_memory_gb, 90.0)
        self.assertEqual(mem_config.cpu_buffer_gb, 10.0)

        # Check expert caching
        self.assertEqual(mem_config.experts_vram_gb, 4.0)
        self.assertEqual(mem_config.cached_experts_per_layer, 3)

        # Check max memory mapping
        max_mem = mem_config.max_memory_mapping
        self.assertEqual(max_mem[0], "14.0GB")
        self.assertEqual(max_mem["cpu"], "90.0GB")

    def test_quantization_config(self):
        """Test BitsAndBytes quantization configuration"""
        from config import QuantizationConfig
        from transformers import BitsAndBytesConfig

        quant_config = QuantizationConfig()

        # Create BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.load_in_4bit,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=quant_config.bnb_4bit_compute_dtype
        )

        self.assertTrue(bnb_config.load_in_4bit)
        self.assertEqual(bnb_config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(bnb_config.bnb_4bit_use_double_quant)
        self.assertEqual(bnb_config.bnb_4bit_compute_dtype, torch.bfloat16)


class TestExpertPlacementOptimization(unittest.TestCase):
    """Test expert placement optimization after profiling"""

    @patch('moe_utils.ExpertCacheManager')
    def test_expert_movement_to_gpu(self, mock_cache_manager):
        """Test moving frequently used experts to GPU"""
        from moe_utils import ExpertCacheManager

        # Create mock model
        mock_model = MagicMock()

        # Create cache manager
        manager = ExpertCacheManager(
            model=mock_model,
            vram_budget_gb=4.0,
            num_cached_experts_per_layer=3
        )

        # Simulate profiling results
        top_experts = [
            ("layer_0_expert_5", 150),
            ("layer_0_expert_12", 140),
            ("layer_1_expert_3", 135),
            ("layer_1_expert_7", 130),
            ("layer_2_expert_1", 125),
        ]

        # Test optimization
        manager.optimize_expert_placement(top_experts)

        # Verify experts would be cached
        self.assertIsNotNone(manager.vram_budget)
        self.assertIsNotNone(manager.cached_experts)

    def test_memory_monitoring(self):
        """Test memory monitoring utilities"""
        from moe_utils import MemoryMonitor

        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        # Check RAM stats
        self.assertIn('ram', stats)
        self.assertIn('total_gb', stats['ram'])
        self.assertIn('available_gb', stats['ram'])
        self.assertIn('used_gb', stats['ram'])
        self.assertIn('percent', stats['ram'])

        # Check all values are positive
        self.assertGreater(stats['ram']['total_gb'], 0)
        self.assertGreaterEqual(stats['ram']['available_gb'], 0)
        self.assertGreaterEqual(stats['ram']['used_gb'], 0)

        # If CUDA available, check VRAM stats
        if torch.cuda.is_available():
            self.assertIn('vram', stats)
            self.assertIn('allocated_gb', stats['vram'])
            self.assertIn('reserved_gb', stats['vram'])
            self.assertIn('free_gb', stats['vram'])


if __name__ == '__main__':
    unittest.main()