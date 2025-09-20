"""
Tests for expert profiling and caching system
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from moe_utils import ExpertCacheManager
from expert_manager import (
    ExpertProfiler,
    DynamicExpertLoader,
    ExpertSwapScheduler,
    PredictiveExpertPreloader
)


class TestExpertProfiler(unittest.TestCase):
    """Test expert profiling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.profiler = ExpertProfiler(self.mock_model)

    def test_usage_tracking(self):
        """Test tracking expert usage patterns"""
        # Simulate expert activations
        layer_idx = 0
        expert_indices = [5, 12, 5, 7, 5, 12]  # Expert 5 used most

        for expert_idx in expert_indices:
            self.profiler.record_expert_usage(layer_idx, expert_idx, strength=1.0)

        # Get usage stats
        stats = self.profiler.get_usage_stats()

        # Expert 5 should have highest count
        expert_5_stats = stats.get(f"layer_{layer_idx}_expert_5")
        self.assertIsNotNone(expert_5_stats)
        self.assertEqual(expert_5_stats['count'], 3)

    def test_top_k_experts(self):
        """Test getting top-k most used experts"""
        # Record usage for multiple experts
        for layer in range(3):
            for _ in range(10):
                self.profiler.record_expert_usage(layer, 1, strength=1.0)
            for _ in range(5):
                self.profiler.record_expert_usage(layer, 2, strength=0.8)
            for _ in range(2):
                self.profiler.record_expert_usage(layer, 3, strength=0.5)

        # Get top 2 experts per layer
        top_experts = self.profiler.get_top_experts_per_layer(k=2)

        # Each layer should have 2 experts
        for layer in range(3):
            layer_experts = top_experts.get(layer, [])
            self.assertEqual(len(layer_experts), 2)
            # Expert 1 should be first
            self.assertEqual(layer_experts[0][0], 1)
            # Expert 2 should be second
            self.assertEqual(layer_experts[1][0], 2)

    def test_usage_pattern_analysis(self):
        """Test analyzing usage patterns over time"""
        # Simulate usage over time
        for step in range(100):
            # Early steps use experts 1-3
            if step < 30:
                expert = step % 3
            # Middle steps use experts 4-6
            elif step < 70:
                expert = 3 + (step % 3)
            # Late steps use experts 7-9
            else:
                expert = 6 + (step % 3)

            self.profiler.record_expert_usage(0, expert, strength=1.0)

        # Analyze patterns
        patterns = self.profiler.analyze_patterns()

        self.assertIn('temporal_shifts', patterns)
        self.assertIn('frequency_distribution', patterns)


class TestDynamicExpertLoader(unittest.TestCase):
    """Test dynamic expert loading between CPU and GPU"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.loader = DynamicExpertLoader(
            model=self.mock_model,
            vram_budget_gb=4.0
        )

    @patch('torch.cuda.memory_allocated')
    def test_load_expert_to_gpu(self, mock_mem_allocated):
        """Test loading an expert to GPU"""
        mock_mem_allocated.return_value = 1 * 1024**3  # 1GB allocated

        # Create mock expert
        mock_expert = MagicMock()
        mock_expert.to = MagicMock()

        # Load expert
        success = self.loader.load_expert_to_gpu(
            layer_idx=0,
            expert_idx=5,
            expert_module=mock_expert
        )

        # Verify expert was moved to GPU
        mock_expert.to.assert_called_with('cuda:0')
        self.assertTrue(success)

    @patch('torch.cuda.memory_allocated')
    def test_memory_budget_enforcement(self, mock_mem_allocated):
        """Test that memory budget is enforced"""
        # Simulate near-full VRAM
        mock_mem_allocated.return_value = 3.8 * 1024**3  # 3.8GB allocated

        MagicMock()

        # Estimate expert size
        self.loader.expert_size_bytes = 0.5 * 1024**3  # 500MB per expert

        # Check if can load
        can_load = self.loader.can_load_expert()

        # Should return False due to budget
        self.assertFalse(can_load)

    def test_expert_swapping(self):
        """Test swapping experts between CPU and GPU"""
        # Create mock experts
        gpu_expert = MagicMock()
        cpu_expert = MagicMock()

        # Add to loader's cache
        self.loader.gpu_experts = {'layer_0_expert_1': gpu_expert}

        # Perform swap
        self.loader.swap_experts(
            gpu_to_cpu=[('layer_0_expert_1', gpu_expert)],
            cpu_to_gpu=[('layer_0_expert_2', cpu_expert)]
        )

        # Verify moves
        gpu_expert.to.assert_called_with('cpu')
        cpu_expert.to.assert_called_with('cuda:0')


class TestExpertSwapScheduler(unittest.TestCase):
    """Test expert swap scheduling"""

    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = ExpertSwapScheduler(
            swap_threshold=0.7,
            min_usage_for_gpu=10
        )

    def test_swap_decision_making(self):
        """Test making swap decisions based on usage"""
        # Create usage stats
        usage_stats = {
            'layer_0_expert_1': {'count': 100, 'last_used': 10},  # High use
            'layer_0_expert_2': {'count': 5, 'last_used': 50},    # Low use
            'layer_0_expert_3': {'count': 50, 'last_used': 15},   # Medium use
        }

        gpu_experts = {'layer_0_expert_2'}  # Low use expert on GPU
        cpu_experts = {'layer_0_expert_1', 'layer_0_expert_3'}

        # Get swap decisions
        to_gpu, to_cpu = self.scheduler.get_swap_decisions(
            usage_stats, gpu_experts, cpu_experts, max_gpu_experts=2
        )

        # High use expert should move to GPU
        self.assertIn('layer_0_expert_1', to_gpu)
        # Low use expert should move to CPU
        self.assertIn('layer_0_expert_2', to_cpu)

    def test_swap_throttling(self):
        """Test that swaps are throttled to avoid thrashing"""
        # Record multiple swaps
        for i in range(10):
            self.scheduler.record_swap('layer_0_expert_1', direction='to_gpu')

        # Check if should throttle
        should_swap = self.scheduler.should_swap('layer_0_expert_1')

        # Should throttle after many swaps
        self.assertFalse(should_swap)


class TestPredictiveExpertPreloader(unittest.TestCase):
    """Test predictive expert preloading"""

    def setUp(self):
        """Set up test fixtures"""
        self.preloader = PredictiveExpertPreloader()

    def test_pattern_learning(self):
        """Test learning expert activation patterns"""
        # Train with sequence patterns
        sequences = [
            [1, 2, 3, 4],  # Pattern 1
            [1, 2, 3, 4],  # Pattern 1 repeated
            [5, 6, 7, 8],  # Pattern 2
            [1, 2, 3, 4],  # Pattern 1 again
        ]

        for seq in sequences:
            self.preloader.record_sequence(seq)

        # Predict next experts after seeing [1, 2]
        predicted = self.preloader.predict_next_experts([1, 2])

        # Should predict [3, 4] as likely next
        self.assertIn(3, predicted)
        self.assertIn(4, predicted)

    def test_input_based_prediction(self):
        """Test predicting experts based on input characteristics"""
        # Record patterns for different input types
        technical_inputs = ["code", "algorithm", "function"]
        technical_experts = [10, 11, 12]

        conversational_inputs = ["hello", "thanks", "bye"]
        conversational_experts = [20, 21, 22]

        for text in technical_inputs:
            self.preloader.record_input_pattern(text, technical_experts)

        for text in conversational_inputs:
            self.preloader.record_input_pattern(text, conversational_experts)

        # Predict for technical input
        predicted = self.preloader.predict_from_input("algorithm optimization")

        # Should predict technical experts
        self.assertTrue(any(e in predicted for e in technical_experts))

    def test_cache_warming(self):
        """Test cache warming with predicted experts"""
        # Create mock loader
        mock_loader = MagicMock()

        # Predict experts
        predicted_experts = [
            ('layer_0_expert_5', 0.9),
            ('layer_0_expert_12', 0.85),
            ('layer_1_expert_3', 0.8),
        ]

        # Warm cache
        self.preloader.warm_cache(predicted_experts, mock_loader)

        # Verify loader was called to load high-confidence experts
        self.assertEqual(mock_loader.load_expert_to_gpu.call_count, 3)


class TestExpertCacheManager(unittest.TestCase):
    """Test the main expert cache manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()

    def test_initialization(self):
        """Test cache manager initialization"""
        manager = ExpertCacheManager(
            model=self.mock_model,
            vram_budget_gb=4.0,
            num_cached_experts_per_layer=3
        )

        self.assertEqual(manager.vram_budget, 4.0 * 1024**3)
        self.assertEqual(manager.num_cached_experts_per_layer, 3)
        self.assertEqual(len(manager.cached_experts), 0)

    @patch('torch.cuda.is_available')
    def test_cache_statistics(self, mock_cuda):
        """Test getting cache statistics"""
        mock_cuda.return_value = True

        manager = ExpertCacheManager(
            model=self.mock_model,
            vram_budget_gb=4.0
        )

        # Add some cached experts
        manager.cached_experts = {
            'layer_0_expert_5',
            'layer_0_expert_12',
            'layer_1_expert_3'
        }

        stats = manager.get_cache_stats()

        self.assertEqual(stats['cached_experts'], 3)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('vram_usage_gb', stats)
        self.assertIn('top_experts', stats)

    def test_cache_clearing(self):
        """Test clearing the expert cache"""
        manager = ExpertCacheManager(
            model=self.mock_model,
            vram_budget_gb=4.0
        )

        # Add cached experts
        manager.cached_experts = {
            'layer_0_expert_5',
            'layer_0_expert_12'
        }

        # Clear cache
        manager.clear_cache()

        # Cache should be empty
        self.assertEqual(len(manager.cached_experts), 0)


if __name__ == '__main__':
    unittest.main()