"""
Advanced expert management system for MoE models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ExpertProfiler:
    """Profile and analyze expert usage patterns"""

    def __init__(self, model, window_size: int = 1000):
        """
        Initialize expert profiler.

        Args:
            model: The MoE model
            window_size: Size of sliding window for pattern analysis
        """
        self.model = model
        self.window_size = window_size
        self.usage_history = defaultdict(lambda: deque(maxlen=window_size))
        self.usage_counts = defaultdict(int)
        self.usage_strengths = defaultdict(float)
        self.temporal_patterns = defaultdict(list)
        self.step_count = 0

    def record_expert_usage(
        self,
        layer_idx: int,
        expert_idx: int,
        strength: float = 1.0,
        timestamp: Optional[float] = None
    ):
        """Record expert usage with activation strength"""
        if timestamp is None:
            timestamp = time.time()

        expert_id = f"layer_{layer_idx}_expert_{expert_idx}"

        self.usage_counts[expert_id] += 1
        self.usage_strengths[expert_id] += strength
        self.usage_history[expert_id].append({
            'step': self.step_count,
            'timestamp': timestamp,
            'strength': strength
        })
        self.temporal_patterns[expert_id].append(self.step_count)
        self.step_count += 1

    def get_usage_stats(self) -> Dict:
        """Get comprehensive usage statistics"""
        stats = {}
        for expert_id, count in self.usage_counts.items():
            avg_strength = self.usage_strengths[expert_id] / count if count > 0 else 0
            stats[expert_id] = {
                'count': count,
                'avg_strength': avg_strength,
                'total_strength': self.usage_strengths[expert_id],
                'recent_usage': len(self.usage_history[expert_id])
            }
        return stats

    def get_top_experts_per_layer(self, k: int = 3) -> Dict[int, List[Tuple[int, float]]]:
        """Get top-k experts for each layer"""
        layer_experts = defaultdict(list)

        for expert_id, count in self.usage_counts.items():
            parts = expert_id.split('_')
            layer_idx = int(parts[1])
            expert_idx = int(parts[3])
            score = count * (self.usage_strengths[expert_id] / count if count > 0 else 0)
            layer_experts[layer_idx].append((expert_idx, score))

        # Sort and take top-k for each layer
        top_experts = {}
        for layer_idx, experts in layer_experts.items():
            sorted_experts = sorted(experts, key=lambda x: x[1], reverse=True)
            top_experts[layer_idx] = sorted_experts[:k]

        return top_experts

    def analyze_patterns(self) -> Dict:
        """Analyze temporal patterns in expert usage"""
        patterns = {
            'temporal_shifts': {},
            'frequency_distribution': {},
            'burst_patterns': {},
            'correlation_matrix': None
        }

        # Analyze temporal shifts
        for expert_id, history in self.usage_history.items():
            if len(history) > 10:
                recent_steps = [h['step'] for h in list(history)[-10:]]
                avg_gap = np.mean(np.diff(recent_steps)) if len(recent_steps) > 1 else 0
                patterns['temporal_shifts'][expert_id] = avg_gap

        # Frequency distribution
        total_usage = sum(self.usage_counts.values())
        for expert_id, count in self.usage_counts.items():
            patterns['frequency_distribution'][expert_id] = count / total_usage if total_usage > 0 else 0

        return patterns


class DynamicExpertLoader:
    """Dynamically load and unload experts between CPU and GPU"""

    def __init__(
        self,
        model,
        vram_budget_gb: float = 4.0,
        device: str = "cuda:0"
    ):
        """
        Initialize dynamic expert loader.

        Args:
            model: The MoE model
            vram_budget_gb: VRAM budget for experts
            device: Target GPU device
        """
        self.model = model
        self.vram_budget_bytes = vram_budget_gb * 1024**3
        self.device = device
        self.gpu_experts: Dict[str, any] = {}
        self.cpu_experts: Dict[str, any] = {}
        self.expert_size_bytes = None
        self.loading_times = []

        # Thread pool for asynchronous loading
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"Initialized DynamicExpertLoader with {vram_budget_gb}GB budget")

    def estimate_expert_size(self, expert_module) -> int:
        """Estimate memory size of an expert module"""
        total_params = sum(p.numel() for p in expert_module.parameters())
        # 4-bit quantization = 0.5 bytes per parameter
        size_bytes = total_params * 0.5
        return size_bytes

    def can_load_expert(self) -> bool:
        """Check if there's room to load another expert"""
        if not torch.cuda.is_available():
            return False

        current_usage = torch.cuda.memory_allocated(self.device)
        available = self.vram_budget_bytes - current_usage

        if self.expert_size_bytes is None:
            # Estimate from a sample expert
            self.expert_size_bytes = 100 * 1024**2  # Default 100MB

        return available > self.expert_size_bytes * 1.2  # 20% safety margin

    def load_expert_to_gpu(
        self,
        layer_idx: int,
        expert_idx: int,
        expert_module
    ) -> bool:
        """Load a specific expert to GPU"""
        expert_id = f"layer_{layer_idx}_expert_{expert_idx}"

        if expert_id in self.gpu_experts:
            logger.debug(f"Expert {expert_id} already on GPU")
            return True

        if not self.can_load_expert():
            logger.warning(f"Cannot load expert {expert_id}: insufficient VRAM")
            return False

        try:
            start_time = time.time()
            expert_module.to(self.device)
            load_time = time.time() - start_time

            self.gpu_experts[expert_id] = expert_module
            if expert_id in self.cpu_experts:
                del self.cpu_experts[expert_id]

            self.loading_times.append(load_time)
            logger.info(f"Loaded expert {expert_id} to GPU in {load_time:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load expert {expert_id}: {e}")
            return False

    def offload_expert_to_cpu(
        self,
        layer_idx: int,
        expert_idx: int,
        expert_module
    ) -> bool:
        """Offload a specific expert to CPU"""
        expert_id = f"layer_{layer_idx}_expert_{expert_idx}"

        if expert_id in self.cpu_experts:
            logger.debug(f"Expert {expert_id} already on CPU")
            return True

        try:
            expert_module.to("cpu")
            self.cpu_experts[expert_id] = expert_module
            if expert_id in self.gpu_experts:
                del self.gpu_experts[expert_id]

            logger.info(f"Offloaded expert {expert_id} to CPU")
            return True

        except Exception as e:
            logger.error(f"Failed to offload expert {expert_id}: {e}")
            return False

    def swap_experts(
        self,
        gpu_to_cpu: List[Tuple[str, any]],
        cpu_to_gpu: List[Tuple[str, any]]
    ):
        """Swap experts between GPU and CPU"""
        # First, move experts from GPU to CPU to free space
        for expert_id, expert_module in gpu_to_cpu:
            parts = expert_id.split('_')
            layer_idx = int(parts[1])
            expert_idx = int(parts[3])
            self.offload_expert_to_cpu(layer_idx, expert_idx, expert_module)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Then, move experts from CPU to GPU
        for expert_id, expert_module in cpu_to_gpu:
            parts = expert_id.split('_')
            layer_idx = int(parts[1])
            expert_idx = int(parts[3])
            self.load_expert_to_gpu(layer_idx, expert_idx, expert_module)

    async def async_load_expert(self, layer_idx: int, expert_idx: int, expert_module):
        """Asynchronously load expert to GPU"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.load_expert_to_gpu,
            layer_idx, expert_idx, expert_module
        )
        return result

    def get_loading_stats(self) -> Dict:
        """Get statistics about expert loading"""
        if not self.loading_times:
            return {}

        return {
            'avg_load_time': np.mean(self.loading_times),
            'max_load_time': np.max(self.loading_times),
            'total_loads': len(self.loading_times),
            'gpu_experts': len(self.gpu_experts),
            'cpu_experts': len(self.cpu_experts)
        }


class ExpertSwapScheduler:
    """Schedule expert swaps based on usage patterns"""

    def __init__(
        self,
        swap_threshold: float = 0.7,
        min_usage_for_gpu: int = 10,
        throttle_window: int = 100
    ):
        """
        Initialize swap scheduler.

        Args:
            swap_threshold: Threshold for swap decisions
            min_usage_for_gpu: Minimum usage count to consider for GPU
            throttle_window: Window for swap throttling
        """
        self.swap_threshold = swap_threshold
        self.min_usage_for_gpu = min_usage_for_gpu
        self.throttle_window = throttle_window
        self.swap_history = defaultdict(lambda: deque(maxlen=throttle_window))
        self.last_swap_time = defaultdict(float)

    def get_swap_decisions(
        self,
        usage_stats: Dict,
        gpu_experts: Set[str],
        cpu_experts: Set[str],
        max_gpu_experts: int
    ) -> Tuple[List[str], List[str]]:
        """Decide which experts to swap"""
        to_gpu = []
        to_cpu = []

        # Score all experts
        expert_scores = {}
        for expert_id, stats in usage_stats.items():
            if stats['count'] >= self.min_usage_for_gpu:
                # Score based on usage count and recency
                time_since_use = max(0.001, time.time() - stats.get('last_used', time.time()))
                recency_weight = 1.0 / (1 + time_since_use)
                score = stats['count'] * recency_weight
                expert_scores[expert_id] = score

        # Sort by score
        sorted_experts = sorted(
            expert_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top experts for GPU
        target_gpu_experts = set()
        for expert_id, score in sorted_experts[:max_gpu_experts]:
            target_gpu_experts.add(expert_id)

        # Determine swaps needed
        to_gpu = list(target_gpu_experts - gpu_experts)
        to_cpu = list(gpu_experts - target_gpu_experts)

        # Apply throttling
        to_gpu = [e for e in to_gpu if self.should_swap(e)]
        to_cpu = [e for e in to_cpu if self.should_swap(e)]

        return to_gpu, to_cpu

    def should_swap(self, expert_id: str) -> bool:
        """Check if an expert should be swapped (throttling)"""
        current_time = time.time()
        last_swap = self.last_swap_time.get(expert_id, 0)

        # Don't swap if recently swapped
        if current_time - last_swap < 10:  # 10 second cooldown
            return False

        # Check swap frequency
        recent_swaps = len(self.swap_history[expert_id])
        if recent_swaps > 5:  # Too many recent swaps
            return False

        return True

    def record_swap(self, expert_id: str, direction: str):
        """Record a swap event"""
        current_time = time.time()
        self.swap_history[expert_id].append({
            'time': current_time,
            'direction': direction
        })
        self.last_swap_time[expert_id] = current_time


class PredictiveExpertPreloader:
    """Predict and preload experts based on patterns"""

    def __init__(self, sequence_length: int = 10, model=None):
        """
        Initialize predictive preloader.

        Args:
            sequence_length: Length of sequences to analyze
            model: The loaded model for accessing expert modules
        """
        self.sequence_length = sequence_length
        self.sequence_patterns = defaultdict(list)
        self.input_patterns = defaultdict(list)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.model = model

    def _get_expert_module(self, layer_idx: int, expert_idx: int):
        """
        Safely get expert module from model.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            Expert module or None if not accessible
        """
        if self.model is None:
            return None

        try:
            # Try the standard Qwen3 MoE structure
            return self.model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
        except (AttributeError, IndexError):
            # Try alternative structures
            try:
                # Alternative 1: direct experts attribute
                return self.model.model.layers[layer_idx].experts[expert_idx]
            except (AttributeError, IndexError):
                try:
                    # Alternative 2: mlp_experts
                    return self.model.model.layers[layer_idx].mlp_experts[expert_idx]
                except (AttributeError, IndexError):
                    return None

    def record_sequence(self, expert_sequence: List[int]):
        """Record an expert activation sequence"""
        for i in range(len(expert_sequence) - 1):
            current = expert_sequence[i]
            next_expert = expert_sequence[i + 1]
            self.transition_matrix[current][next_expert] += 1

        # Store sequence patterns
        for i in range(len(expert_sequence) - self.sequence_length):
            pattern = tuple(expert_sequence[i:i + self.sequence_length])
            next_experts = expert_sequence[i + self.sequence_length:i + self.sequence_length + 2]
            self.sequence_patterns[pattern].extend(next_experts)

    def record_input_pattern(self, input_text: str, activated_experts: List[int]):
        """Record which experts are activated for certain input patterns"""
        # Simple keyword extraction
        keywords = set(input_text.lower().split())

        for keyword in keywords:
            self.input_patterns[keyword].extend(activated_experts)

    def predict_next_experts(self, recent_experts: List[int], k: int = 5) -> List[int]:
        """Predict next likely experts based on recent activations"""
        predictions = defaultdict(float)

        # Use transition matrix
        if recent_experts:
            last_expert = recent_experts[-1]
            for next_expert, count in self.transition_matrix[last_expert].items():
                predictions[next_expert] += count

        # Use sequence patterns
        if len(recent_experts) >= self.sequence_length:
            pattern = tuple(recent_experts[-self.sequence_length:])
            if pattern in self.sequence_patterns:
                for expert in self.sequence_patterns[pattern]:
                    predictions[expert] += 1

        # Sort and return top-k
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [expert for expert, _ in sorted_predictions[:k]]

    def predict_from_input(self, input_text: str, k: int = 5) -> List[int]:
        """Predict likely experts based on input characteristics"""
        predictions = defaultdict(float)
        keywords = set(input_text.lower().split())

        for keyword in keywords:
            if keyword in self.input_patterns:
                for expert in self.input_patterns[keyword]:
                    predictions[expert] += 1

        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [expert for expert, _ in sorted_predictions[:k]]

    def warm_cache(self, predicted_experts: List[Tuple[str, float]], loader: DynamicExpertLoader):
        """Warm the cache with predicted experts"""
        if self.model is None:
            logger.warning("Model not provided to PredictiveExpertPreloader, skipping cache warming")
            return

        for expert_id, confidence in predicted_experts:
            if confidence > 0.7:  # High confidence threshold
                parts = expert_id.split('_')
                layer_idx = int(parts[1])
                expert_idx = int(parts[3])

                # Get expert module from model
                expert_module = self._get_expert_module(layer_idx, expert_idx)
                if expert_module is not None:
                    logger.info(f"Preloading expert {expert_id} with confidence {confidence:.2f}")
                    loader.load_expert_to_gpu(layer_idx, expert_idx, expert_module)
                else:
                    logger.warning(f"Failed to access expert module {expert_id}")
                    continue