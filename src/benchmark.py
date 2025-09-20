"""
Performance benchmarking and optimization tools
"""

import torch
import time
import numpy as np
import psutil
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""

    def __init__(self, pipeline, target_tokens_per_second: float = 8.0):
        """
        Initialize performance benchmark.

        Args:
            pipeline: The inference pipeline to benchmark
            target_tokens_per_second: Target performance in tokens/second
        """
        self.pipeline = pipeline
        self.target_tps = target_tokens_per_second
        self.results = []

    def measure_tokens_per_second(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Dict:
        """
        Measure tokens per second.

        Args:
            prompt: Input prompt
            max_new_tokens: Tokens to generate
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Performance metrics
        """
        logger.info(f"Measuring tokens/second with {num_runs} runs")

        # Warmup
        for _ in range(warmup_runs):
            self.pipeline.generate(prompt, max_new_tokens=max_new_tokens)

        # Benchmark runs
        timings = []
        token_counts = []

        for i in range(num_runs):
            start_time = time.time()
            output = self.pipeline.generate(prompt, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start_time

            # Count tokens
            tokens = self.pipeline.tokenizer.encode(output)
            token_count = len(tokens)

            timings.append(elapsed)
            token_counts.append(token_count)

            logger.debug(f"Run {i+1}: {token_count} tokens in {elapsed:.2f}s")

        # Calculate statistics
        total_tokens = sum(token_counts)
        total_time = sum(timings)
        avg_tps = total_tokens / total_time

        return {
            'avg_tokens_per_second': avg_tps,
            'min_tokens_per_second': min(t/time for t, time in zip(token_counts, timings)),
            'max_tokens_per_second': max(t/time for t, time in zip(token_counts, timings)),
            'total_tokens': total_tokens,
            'total_time': total_time,
            'runs': num_runs
        }

    def validate_performance_targets(
        self,
        results: Dict,
        min_tokens_per_second: float = 8.0,
        max_p95_latency: float = 0.15,
        max_memory_gb: float = 14.0
    ) -> bool:
        """
        Validate performance against targets.

        Args:
            results: Benchmark results
            min_tokens_per_second: Minimum required tokens/second
            max_p95_latency: Maximum acceptable P95 latency
            max_memory_gb: Maximum memory usage

        Returns:
            True if all targets met
        """
        passed = True

        # Check tokens per second
        if results.get('avg_tokens_per_second', 0) < min_tokens_per_second:
            logger.warning(
                f"Tokens/second {results['avg_tokens_per_second']:.2f} "
                f"below target {min_tokens_per_second}"
            )
            passed = False

        # Check latency
        if results.get('p95_latency', float('inf')) > max_p95_latency:
            logger.warning(
                f"P95 latency {results['p95_latency']:.3f}s "
                f"above target {max_p95_latency}s"
            )
            passed = False

        # Check memory
        if results.get('memory_usage_gb', float('inf')) > max_memory_gb:
            logger.warning(
                f"Memory usage {results['memory_usage_gb']:.1f}GB "
                f"above target {max_memory_gb}GB"
            )
            passed = False

        return passed

    def run_benchmark_suite(
        self,
        test_prompts: List[str],
        output_lengths: List[int] = [10, 50, 100, 500],
        expert_configs: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Run comprehensive benchmark suite.

        Args:
            test_prompts: List of test prompts
            output_lengths: List of output lengths to test
            expert_configs: Optional expert cache configurations to test

        Returns:
            Comprehensive benchmark results
        """
        results = {
            'by_prompt_length': {},
            'by_output_length': {},
            'by_expert_config': {},
            'overall': {}
        }

        # Benchmark by prompt length
        for prompt in test_prompts:
            prompt_len = len(prompt.split())
            bucket = f"{(prompt_len // 10) * 10}-{((prompt_len // 10) + 1) * 10}_words"

            if bucket not in results['by_prompt_length']:
                results['by_prompt_length'][bucket] = []

            perf = self.measure_tokens_per_second(prompt, max_new_tokens=50)
            results['by_prompt_length'][bucket].append(perf)

        # Benchmark by output length
        for length in output_lengths:
            perf = self.measure_tokens_per_second(
                test_prompts[0],
                max_new_tokens=length,
                num_runs=3
            )
            results['by_output_length'][f"{length}_tokens"] = perf

        # Benchmark expert configurations if provided
        if expert_configs:
            for config in expert_configs:
                # Apply configuration
                self.pipeline.expert_manager.num_cached_experts_per_layer = config.get('cache_size', 3)

                perf = self.measure_tokens_per_second(
                    test_prompts[0],
                    max_new_tokens=100,
                    num_runs=3
                )
                results['by_expert_config'][str(config)] = perf

        # Overall performance
        all_tps = []
        for prompt in test_prompts[:3]:  # Sample
            perf = self.measure_tokens_per_second(prompt)
            all_tps.append(perf['avg_tokens_per_second'])

        results['overall'] = {
            'avg_tokens_per_second': np.mean(all_tps),
            'std_tokens_per_second': np.std(all_tps),
            'target_met': np.mean(all_tps) >= self.target_tps
        }

        return results


class MemoryProfiler:
    """Profile memory usage patterns"""

    def __init__(self):
        """Initialize memory profiler"""
        self.memory_snapshots = []

    def profile_gpu_memory(self) -> Dict:
        """Profile GPU memory usage"""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'utilization': torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
        }

    def profile_system_memory(self) -> Dict:
        """Profile system RAM usage"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024**3,
            'used_gb': mem.used / 1024**3,
            'available_gb': mem.available / 1024**3,
            'percent': mem.percent
        }

    def analyze_memory_distribution(self, model) -> Dict:
        """Analyze memory distribution across model components"""
        distribution = {
            'embeddings': 0,
            'attention': 0,
            'experts': 0,
            'other': 0
        }

        for name, param in model.named_parameters():
            param_bytes = param.numel() * param.element_size()

            if 'embed' in name:
                distribution['embeddings'] += param_bytes
            elif 'attn' in name or 'attention' in name:
                distribution['attention'] += param_bytes
            elif 'expert' in name or 'moe' in name:
                distribution['experts'] += param_bytes
            else:
                distribution['other'] += param_bytes

        # Convert to GB
        for key in distribution:
            distribution[key] = distribution[key] / 1024**3

        return distribution

    def get_optimization_recommendations(self, profile: Dict) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []

        # Check GPU utilization
        if profile.get('gpu_utilization', 0) > 0.9:
            recommendations.append(
                "GPU memory nearly full. Consider: "
                "1) Reducing expert cache size, "
                "2) Lowering batch size, "
                "3) Enabling gradient checkpointing"
            )

        # Check RAM utilization
        if profile.get('ram_utilization', 0) > 0.85:
            recommendations.append(
                "System RAM heavily utilized. Consider: "
                "1) Offloading fewer experts to CPU, "
                "2) Reducing KV cache size, "
                "3) Clearing unused caches"
            )

        # Check expert cache efficiency
        if profile.get('expert_cache_size', 0) > 30:
            recommendations.append(
                "Large expert cache may cause thrashing. "
                "Consider reducing to 3-5 experts per layer"
            )

        return recommendations


class ExpertPlacementOptimizer:
    """Optimize expert placement strategies"""

    def __init__(self, model):
        """Initialize optimizer"""
        self.model = model
        self.performance_history = []

    def search_optimal_configuration(
        self,
        performance_func: Callable,
        search_space: Dict
    ) -> Dict:
        """
        Search for optimal expert placement configuration.

        Args:
            performance_func: Function to evaluate performance
            search_space: Parameter search space

        Returns:
            Optimal configuration
        """
        best_config = None
        best_performance = 0

        # Grid search (simple approach)
        for num_experts in search_space.get('num_cached_experts', range(1, 11)):
            perf = performance_func(num_experts)

            if perf > best_performance:
                best_performance = perf
                best_config = {'num_cached_experts': num_experts}

            logger.info(f"Config {num_experts} experts: {perf:.2f} tokens/sec")

        return best_config

    def grid_search(
        self,
        param_grid: Dict,
        evaluate_func: Callable
    ) -> Dict:
        """
        Grid search over parameter space.

        Args:
            param_grid: Parameter grid
            evaluate_func: Evaluation function

        Returns:
            Best parameters
        """
        from itertools import product

        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))

        best_params = None
        best_score = -float('inf')

        for combo in combinations:
            params = dict(zip(keys, combo))
            score = evaluate_func(params)

            if score > best_score:
                best_score = score
                best_params = params

        logger.info(f"Best parameters: {best_params} with score {best_score:.2f}")
        return best_params

    def adapt_configuration(
        self,
        performance_history: List[float],
        current_config: Dict
    ) -> Dict:
        """
        Adapt configuration based on performance history.

        Args:
            performance_history: Recent performance measurements
            current_config: Current configuration

        Returns:
            Adapted configuration
        """
        if len(performance_history) < 10:
            return current_config

        # Calculate trend
        recent_avg = np.mean(performance_history[-5:])
        older_avg = np.mean(performance_history[-10:-5])

        new_config = current_config.copy()

        # If performance degrading, adjust
        if recent_avg < older_avg * 0.95:
            # Reduce cache size if performance dropping
            new_config['cache_size'] = max(
                1,
                current_config.get('cache_size', 5) - 1
            )
            logger.info(f"Reducing cache size to {new_config['cache_size']}")

        elif recent_avg > older_avg * 1.05:
            # Increase cache size if performance improving
            new_config['cache_size'] = min(
                10,
                current_config.get('cache_size', 5) + 1
            )
            logger.info(f"Increasing cache size to {new_config['cache_size']}")

        return new_config


class LatencyTracker:
    """Track and analyze inference latencies"""

    def __init__(self, window_size: int = 1000):
        """Initialize latency tracker"""
        self.latencies = deque(maxlen=window_size)
        self.token_latencies = defaultdict(list)

    def record_latency(self, latency: float):
        """Record a latency measurement"""
        self.latencies.append(latency)

    def record_token_latency(self, token_position: int, latency: float):
        """Record latency for specific token position"""
        self.token_latencies[token_position].append(latency)

    def get_statistics(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}

        latencies_array = np.array(self.latencies)

        return {
            'mean': np.mean(latencies_array),
            'std': np.std(latencies_array),
            'min': np.min(latencies_array),
            'max': np.max(latencies_array),
            'p50': np.percentile(latencies_array, 50),
            'p90': np.percentile(latencies_array, 90),
            'p95': np.percentile(latencies_array, 95),
            'p99': np.percentile(latencies_array, 99)
        }

    def get_token_latency_breakdown(self) -> Dict:
        """Get latency breakdown by token position"""
        breakdown = {}

        if 0 in self.token_latencies:
            breakdown['first_token'] = np.mean(self.token_latencies[0])

        subsequent = []
        for pos in range(1, max(self.token_latencies.keys()) + 1):
            if pos in self.token_latencies:
                subsequent.extend(self.token_latencies[pos])

        if subsequent:
            breakdown['subsequent_tokens_avg'] = np.mean(subsequent)

        return breakdown

    def get_percentiles(self, percentiles: List[int]) -> Dict:
        """Get specific percentiles"""
        if not self.latencies:
            return {}

        latencies_array = np.array(self.latencies)
        return {
            p: np.percentile(latencies_array, p)
            for p in percentiles
        }


class ThroughputMeasure:
    """Measure throughput metrics"""

    def __init__(self):
        """Initialize throughput measure"""
        self.request_count = 0
        self.token_measurements = []
        self.concurrent_requests = {}

    def record_request(self):
        """Record a completed request"""
        self.request_count += 1

    def record_tokens(self, token_count: int, time_taken: float):
        """Record token generation"""
        self.token_measurements.append({
            'tokens': token_count,
            'time': time_taken,
            'tps': token_count / time_taken
        })

    def calculate_requests_per_second(self, elapsed_time: float) -> float:
        """Calculate requests per second"""
        if elapsed_time == 0:
            return 0
        return self.request_count / elapsed_time

    def get_average_tokens_per_second(self) -> float:
        """Get average tokens per second"""
        if not self.token_measurements:
            return 0

        return np.mean([m['tps'] for m in self.token_measurements])

    def start_request(self, request_id: str):
        """Start tracking a request"""
        self.concurrent_requests[request_id] = time.time()

    def end_request(self, request_id: str):
        """End tracking a request"""
        if request_id in self.concurrent_requests:
            del self.concurrent_requests[request_id]

    def get_concurrent_requests(self) -> int:
        """Get current number of concurrent requests"""
        return len(self.concurrent_requests)


class PerformanceReporter:
    """Generate performance reports"""

    def __init__(self):
        """Initialize reporter"""
        self.results = {}
        self.timestamp = datetime.now()

    def add_benchmark_result(self, name: str, result: Dict):
        """Add a benchmark result"""
        self.results[name] = result

    def generate_report(self) -> Dict:
        """Generate comprehensive report"""
        report = {
            'timestamp': self.timestamp.isoformat(),
            'benchmarks': self.results,
            'summary': self._generate_summary()
        }

        return report

    def _generate_summary(self) -> Dict:
        """Generate summary of results"""
        summary = {
            'targets_met': True,
            'highlights': [],
            'warnings': []
        }

        # Check tokens per second
        if 'tokens_per_second' in self.results:
            tps = self.results['tokens_per_second'].get('avg', 0)
            if tps >= 8.0:
                summary['highlights'].append(
                    f"✓ Achieved {tps:.1f} tokens/second (target: 8.0)"
                )
            else:
                summary['warnings'].append(
                    f"✗ Only {tps:.1f} tokens/second (target: 8.0)"
                )
                summary['targets_met'] = False

        # Check memory usage
        if 'memory' in self.results:
            gpu_gb = self.results['memory'].get('gpu_gb', 0)
            if gpu_gb <= 14.0:
                summary['highlights'].append(
                    f"✓ GPU memory {gpu_gb:.1f}GB within limit"
                )
            else:
                summary['warnings'].append(
                    f"✗ GPU memory {gpu_gb:.1f}GB exceeds 14GB limit"
                )

        return summary

    def to_markdown(self) -> str:
        """Convert report to Markdown format"""
        md = ["# Performance Report"]
        md.append(f"\n*Generated: {self.timestamp}*\n")

        # Summary
        summary = self._generate_summary()
        md.append("## Summary")
        md.append(f"**Targets Met:** {'✅ Yes' if summary['targets_met'] else '❌ No'}\n")

        if summary['highlights']:
            md.append("### Highlights")
            for highlight in summary['highlights']:
                md.append(f"- {highlight}")

        if summary['warnings']:
            md.append("\n### Warnings")
            for warning in summary['warnings']:
                md.append(f"- {warning}")

        # Detailed results
        md.append("\n## Results")
        for name, result in self.results.items():
            md.append(f"\n### {name.replace('_', ' ').title()}")
            for key, value in result.items():
                if isinstance(value, float):
                    md.append(f"- **{key}**: {value:.2f}")
                else:
                    md.append(f"- **{key}**: {value}")

        return "\n".join(md)

    def save_report(self, filepath: Path):
        """Save report to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.generate_report(), f, indent=2, default=str)

        # Save Markdown
        with open(filepath.with_suffix('.md'), 'w') as f:
            f.write(self.to_markdown())

        logger.info(f"Report saved to {filepath}")


def compare_configurations(configs: Dict[str, Dict]) -> Dict:
    """
    Compare performance across different configurations.

    Args:
        configs: Dictionary of configuration names to results

    Returns:
        Comparison analysis
    """
    comparison = {
        'best_throughput': None,
        'best_memory': None,
        'best_overall': None,
        'rankings': {}
    }

    # Find best for each metric
    max_tps = 0
    min_memory = float('inf')

    for name, results in configs.items():
        tps = results.get('tokens_per_second', 0)
        memory = results.get('memory_gb', float('inf'))

        if tps > max_tps:
            max_tps = tps
            comparison['best_throughput'] = name

        if memory < min_memory:
            min_memory = memory
            comparison['best_memory'] = name

    # Overall ranking (simple weighted score)
    for name, results in configs.items():
        tps = results.get('tokens_per_second', 0)
        memory = results.get('memory_gb', 16)

        # Normalize and weight
        tps_score = (tps / max_tps) if max_tps > 0 else 0
        memory_score = (min_memory / memory) if memory > 0 else 0

        overall_score = 0.7 * tps_score + 0.3 * memory_score
        comparison['rankings'][name] = overall_score

    # Best overall
    comparison['best_overall'] = max(
        comparison['rankings'],
        key=comparison['rankings'].get
    )

    return comparison