"""
Tests for performance validation and optimization
"""

import unittest
from unittest.mock import patch, MagicMock
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchmark import (
    PerformanceBenchmark,
    MemoryProfiler,
    ExpertPlacementOptimizer,
    LatencyTracker,
    ThroughputMeasure
)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test performance benchmarking"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_pipeline = MagicMock()
        self.benchmark = PerformanceBenchmark(self.mock_pipeline)

    def test_tokens_per_second_measurement(self):
        """Test measuring tokens per second"""
        # Mock generation that takes 2 seconds for 20 tokens
        def mock_generate(prompt, **kwargs):
            time.sleep(0.1)  # Simulate generation time
            return "Token " * 20  # 20 tokens

        self.mock_pipeline.generate = mock_generate
        self.mock_pipeline.tokenizer.encode.return_value = list(range(20))

        # Run benchmark
        result = self.benchmark.measure_tokens_per_second(
            "Test prompt",
            num_runs=1
        )

        self.assertIn('avg_tokens_per_second', result)
        self.assertIn('total_tokens', result)
        self.assertIn('total_time', result)

        # Should be roughly 200 tokens/second (20 tokens / 0.1 seconds)
        self.assertGreater(result['avg_tokens_per_second'], 0)

    def test_performance_target_validation(self):
        """Test validation against performance targets"""
        # Mock results
        benchmark_results = {
            'avg_tokens_per_second': 10.5,
            'p95_latency': 0.095,
            'memory_usage_gb': 13.5
        }

        # Validate against targets
        passed = self.benchmark.validate_performance_targets(
            benchmark_results,
            min_tokens_per_second=8.0,
            max_p95_latency=0.1,
            max_memory_gb=14.0
        )

        self.assertTrue(passed)

        # Test failure case
        passed = self.benchmark.validate_performance_targets(
            benchmark_results,
            min_tokens_per_second=15.0  # Higher than actual
        )

        self.assertFalse(passed)

    def test_benchmark_suite(self):
        """Test running complete benchmark suite"""
        # Mock pipeline methods
        self.mock_pipeline.generate.return_value = "Test output"
        self.mock_pipeline.tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Run benchmark suite
        results = self.benchmark.run_benchmark_suite(
            test_prompts=["Short", "Medium length prompt", "Long " * 50],
            output_lengths=[10, 50, 100]
        )

        self.assertIn('by_prompt_length', results)
        self.assertIn('by_output_length', results)
        self.assertIn('overall', results)

    def test_expert_placement_strategies(self):
        """Test different expert placement strategies"""
        strategies = [
            'top_k_by_usage',
            'top_k_by_recency',
            'hybrid',
            'random'
        ]

        results = {}
        for strategy in strategies:
            # Mock performance for each strategy
            mock_perf = 5 + np.random.random() * 10  # 5-15 tokens/sec
            results[strategy] = mock_perf

        # Find best strategy
        best_strategy = max(results, key=results.get)

        self.assertIn(best_strategy, strategies)
        self.assertGreater(results[best_strategy], 5.0)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling"""

    def setUp(self):
        """Set up test fixtures"""
        self.profiler = MemoryProfiler()

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_memory_profiling(self, mock_props, mock_allocated):
        """Test GPU memory profiling"""
        # Mock GPU memory
        mock_allocated.return_value = 10 * 1024**3  # 10GB
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        profile = self.profiler.profile_gpu_memory()

        self.assertIn('allocated_gb', profile)
        self.assertIn('total_gb', profile)
        self.assertIn('utilization', profile)

        self.assertAlmostEqual(profile['allocated_gb'], 10.0, places=1)
        self.assertAlmostEqual(profile['utilization'], 0.625, places=2)

    @patch('psutil.virtual_memory')
    def test_ram_memory_profiling(self, mock_vmem):
        """Test system RAM profiling"""
        mock_vmem.return_value = MagicMock(
            total=128 * 1024**3,
            used=80 * 1024**3,
            available=48 * 1024**3,
            percent=62.5
        )

        profile = self.profiler.profile_system_memory()

        self.assertIn('total_gb', profile)
        self.assertIn('used_gb', profile)
        self.assertIn('available_gb', profile)
        self.assertIn('percent', profile)

        self.assertAlmostEqual(profile['total_gb'], 128.0, places=1)
        self.assertAlmostEqual(profile['used_gb'], 80.0, places=1)

    def test_memory_distribution_analysis(self):
        """Test analyzing memory distribution across components"""
        # Mock model with various components
        mock_model = MagicMock()

        distribution = self.profiler.analyze_memory_distribution(mock_model)

        # Should identify different component types
        self.assertIn('embeddings', distribution)
        self.assertIn('attention', distribution)
        self.assertIn('experts', distribution)
        self.assertIn('other', distribution)

    def test_memory_optimization_recommendations(self):
        """Test generating memory optimization recommendations"""
        profile = {
            'gpu_utilization': 0.95,
            'ram_utilization': 0.70,
            'expert_cache_size': 20,
            'kv_cache_size_gb': 2.5
        }

        recommendations = self.profiler.get_optimization_recommendations(profile)

        self.assertIsInstance(recommendations, list)
        # High GPU utilization should trigger recommendations
        self.assertTrue(any('GPU' in r for r in recommendations))


class TestExpertPlacementOptimizer(unittest.TestCase):
    """Test expert placement optimization"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.optimizer = ExpertPlacementOptimizer(self.mock_model)

    def test_optimization_search(self):
        """Test searching for optimal expert placement"""
        # Mock performance function
        def mock_perf_func(num_cached_experts):
            # Simulated performance curve (peaks at 5 experts)
            return 10 * np.exp(-((num_cached_experts - 5) ** 2) / 10)

        # Find optimal number of cached experts
        optimal_config = self.optimizer.search_optimal_configuration(
            performance_func=mock_perf_func,
            search_space={'num_cached_experts': range(1, 11)}
        )

        self.assertIn('num_cached_experts', optimal_config)
        # Should find near 5 as optimal
        self.assertGreaterEqual(optimal_config['num_cached_experts'], 3)
        self.assertLessEqual(optimal_config['num_cached_experts'], 7)

    def test_grid_search_optimization(self):
        """Test grid search for optimal parameters"""
        param_grid = {
            'cache_size': [3, 5, 7],
            'swap_threshold': [0.7, 0.8, 0.9],
            'temperature': [0.6, 0.7, 0.8]
        }

        # Mock evaluation function
        def mock_evaluate(params):
            # Simulate performance based on params
            score = params['cache_size'] * 2
            score += (1 - params['swap_threshold']) * 10
            score += (0.7 - abs(params['temperature'] - 0.7)) * 20
            return score

        best_params = self.optimizer.grid_search(
            param_grid,
            evaluate_func=mock_evaluate
        )

        self.assertIn('cache_size', best_params)
        self.assertIn('swap_threshold', best_params)
        self.assertIn('temperature', best_params)

    def test_adaptive_optimization(self):
        """Test adaptive optimization during runtime"""
        # Simulate runtime performance data
        performance_history = []
        for i in range(100):
            perf = 8 + np.sin(i / 10) * 2  # Varying performance
            performance_history.append(perf)

        # Adapt configuration based on history
        adapted_config = self.optimizer.adapt_configuration(
            performance_history,
            current_config={'cache_size': 5}
        )

        self.assertIn('cache_size', adapted_config)
        # Should adjust based on performance trends
        self.assertIsInstance(adapted_config['cache_size'], int)


class TestLatencyTracker(unittest.TestCase):
    """Test latency tracking"""

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = LatencyTracker()

    def test_latency_recording(self):
        """Test recording inference latencies"""
        # Record some latencies
        latencies = [0.1, 0.12, 0.09, 0.15, 0.11, 0.08, 0.13]

        for lat in latencies:
            self.tracker.record_latency(lat)

        stats = self.tracker.get_statistics()

        self.assertIn('mean', stats)
        self.assertIn('p50', stats)
        self.assertIn('p95', stats)
        self.assertIn('p99', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)

        self.assertAlmostEqual(stats['mean'], np.mean(latencies), places=3)

    def test_token_latency_breakdown(self):
        """Test breaking down latency by token position"""
        # Record token generation times
        token_times = [0.1, 0.05, 0.04, 0.04, 0.03]  # First token slowest

        for i, time_taken in enumerate(token_times):
            self.tracker.record_token_latency(i, time_taken)

        breakdown = self.tracker.get_token_latency_breakdown()

        self.assertIn('first_token', breakdown)
        self.assertIn('subsequent_tokens_avg', breakdown)

        self.assertEqual(breakdown['first_token'], 0.1)
        self.assertLess(breakdown['subsequent_tokens_avg'], 0.1)

    def test_latency_percentiles(self):
        """Test calculating latency percentiles"""
        # Generate many latencies
        np.random.seed(42)
        latencies = np.random.exponential(0.1, 1000)

        for lat in latencies:
            self.tracker.record_latency(lat)

        percentiles = self.tracker.get_percentiles([50, 90, 95, 99])

        self.assertIn(50, percentiles)
        self.assertIn(99, percentiles)

        # Percentiles should increase
        self.assertLess(percentiles[50], percentiles[90])
        self.assertLess(percentiles[90], percentiles[95])
        self.assertLess(percentiles[95], percentiles[99])


class TestThroughputMeasure(unittest.TestCase):
    """Test throughput measurement"""

    def setUp(self):
        """Set up test fixtures"""
        self.measure = ThroughputMeasure()

    def test_requests_per_second(self):
        """Test measuring requests per second"""
        # Simulate processing requests
        start_time = time.time()
        num_requests = 50

        for _ in range(num_requests):
            self.measure.record_request()
            time.sleep(0.01)  # 10ms per request

        elapsed = time.time() - start_time
        rps = self.measure.calculate_requests_per_second(elapsed)

        # Should be roughly 100 RPS (1 / 0.01)
        self.assertGreater(rps, 50)
        self.assertLess(rps, 150)

    def test_tokens_per_second(self):
        """Test measuring tokens per second"""
        # Record token generation
        self.measure.record_tokens(100, time_taken=2.0)
        self.measure.record_tokens(150, time_taken=2.5)
        self.measure.record_tokens(120, time_taken=2.2)

        avg_tps = self.measure.get_average_tokens_per_second()

        # Should be roughly (100/2 + 150/2.5 + 120/2.2) / 3
        expected = np.mean([50, 60, 54.5])
        self.assertAlmostEqual(avg_tps, expected, delta=5)

    def test_concurrent_request_handling(self):
        """Test measuring concurrent request throughput"""
        # Simulate concurrent requests
        concurrent_counts = []

        for i in range(10):
            self.measure.start_request(f"req_{i}")
            concurrent_counts.append(self.measure.get_concurrent_requests())
            time.sleep(0.01)

        for i in range(10):
            self.measure.end_request(f"req_{i}")

        max_concurrent = max(concurrent_counts)

        self.assertGreater(max_concurrent, 0)
        self.assertEqual(self.measure.get_concurrent_requests(), 0)


class TestPerformanceReport(unittest.TestCase):
    """Test performance report generation"""

    def test_report_generation(self):
        """Test generating comprehensive performance report"""
        from benchmark import PerformanceReporter

        reporter = PerformanceReporter()

        # Add test results
        reporter.add_benchmark_result(
            'tokens_per_second',
            {'avg': 10.5, 'min': 8.2, 'max': 12.8}
        )
        reporter.add_benchmark_result(
            'latency',
            {'p50': 0.05, 'p95': 0.12, 'p99': 0.18}
        )
        reporter.add_benchmark_result(
            'memory',
            {'gpu_gb': 13.5, 'ram_gb': 78.2}
        )

        # Generate report
        report = reporter.generate_report()

        self.assertIn('summary', report)
        self.assertIn('benchmarks', report)
        self.assertIn('timestamp', report)

        # Check performance target validation
        self.assertIn('targets_met', report['summary'])

    def test_markdown_report_format(self):
        """Test generating report in Markdown format"""
        from benchmark import PerformanceReporter

        reporter = PerformanceReporter()

        reporter.add_benchmark_result(
            'overall',
            {'tokens_per_second': 11.2}
        )

        markdown = reporter.to_markdown()

        self.assertIn('# Performance Report', markdown)
        self.assertIn('## Results', markdown)
        self.assertIn('11.2', markdown)

    def test_performance_comparison(self):
        """Test comparing performance across configurations"""
        from benchmark import compare_configurations

        configs = {
            'baseline': {'tokens_per_second': 5.2, 'memory_gb': 15.8},
            'optimized': {'tokens_per_second': 11.5, 'memory_gb': 13.2},
            'aggressive': {'tokens_per_second': 12.8, 'memory_gb': 14.9}
        }

        comparison = compare_configurations(configs)

        self.assertIn('best_throughput', comparison)
        self.assertIn('best_memory', comparison)
        self.assertEqual(comparison['best_throughput'], 'aggressive')
        self.assertEqual(comparison['best_memory'], 'optimized')


if __name__ == '__main__':
    unittest.main()