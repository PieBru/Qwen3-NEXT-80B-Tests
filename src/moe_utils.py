"""
MoE utility functions for device mapping and expert management
"""

import torch
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
import logging
import psutil

logger = logging.getLogger(__name__)


def create_moe_device_map(
    num_layers: int = 80,
    num_experts: int = 64,
    model_type: str = "qwen3"
) -> Dict[str, Any]:
    """
    Create custom device map for MoE model with non-experts on GPU
    and experts on CPU.

    Args:
        num_layers: Number of transformer layers
        num_experts: Number of experts per layer
        model_type: Type of model (affects layer naming)

    Returns:
        Device map dictionary
    """
    device_map = {}

    # Core components always on GPU
    gpu_components = [
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    ]

    for component in gpu_components:
        device_map[component] = 0

    # For each transformer layer
    for i in range(num_layers):
        # Non-expert components go to GPU
        device_map[f"model.layers.{i}.self_attn"] = 0
        device_map[f"model.layers.{i}.input_layernorm"] = 0
        device_map[f"model.layers.{i}.post_attention_layernorm"] = 0

        # Router/gate must be on GPU for efficient routing
        device_map[f"model.layers.{i}.block_sparse_moe.gate"] = 0

        # Initially place all experts on CPU
        for j in range(num_experts):
            device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = "cpu"

    return device_map


@dataclass
class ExpertUsageStats:
    """Statistics for expert usage tracking"""
    expert_id: str
    layer_idx: int
    expert_idx: int
    usage_count: int = 0
    total_tokens: int = 0
    avg_activation_strength: float = 0.0
    last_used_step: int = -1


class ExpertCacheManager:
    """
    Manages dynamic loading of frequently used experts into VRAM
    """

    def __init__(
        self,
        model,
        vram_budget_gb: float = 6.0,
        num_cached_experts_per_layer: int = 3,
        device: str = "cuda:0"
    ):
        """
        Initialize expert cache manager.

        Args:
            model: The MoE model
            vram_budget_gb: VRAM budget for expert caching in GB
            num_cached_experts_per_layer: Number of experts to cache per layer
            device: Target GPU device
        """
        self.model = model
        self.vram_budget = vram_budget_gb * 1024**3  # Convert to bytes
        self.num_cached_experts_per_layer = num_cached_experts_per_layer
        self.device = device

        # Track expert usage
        self.expert_stats: Dict[str, ExpertUsageStats] = {}
        self.cached_experts: Set[str] = set()
        self.step_count = 0

        # Memory tracking
        self.expert_memory_size = None  # Will be calculated on first expert

        logger.info(f"Initialized ExpertCacheManager with {vram_budget_gb}GB VRAM budget")

    def _get_expert_module(self, layer_idx: int, expert_idx: int):
        """
        Safely get expert module from model.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            Expert module or None if not accessible
        """
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
                    logger.warning(f"Could not access expert module at layer {layer_idx}, expert {expert_idx}")
                    return None

    def profile_expert_usage(
        self,
        input_ids: torch.Tensor,
        num_samples: int = 100
    ) -> List[Tuple[str, int]]:
        """
        Profile which experts are used most frequently.

        Args:
            input_ids: Sample input IDs for profiling
            num_samples: Number of samples to profile

        Returns:
            List of (expert_id, usage_count) sorted by usage
        """
        logger.info(f"Profiling expert usage with {num_samples} samples")

        with torch.no_grad():
            for i in range(min(num_samples, len(input_ids))):
                self.step_count += 1

                # Run forward pass with router logit output
                outputs = self.model(
                    input_ids[i:i+1],
                    output_router_logits=True,
                    use_cache=False
                )

                # Analyze router decisions if available
                if hasattr(outputs, 'router_logits') and outputs.router_logits:
                    for layer_idx, logits in enumerate(outputs.router_logits):
                        # Get top-k experts (usually k=2 for MoE)
                        topk_values, topk_indices = torch.topk(logits, k=2, dim=-1)

                        for expert_idx in topk_indices.flatten().tolist():
                            expert_id = f"layer_{layer_idx}_expert_{expert_idx}"

                            if expert_id not in self.expert_stats:
                                self.expert_stats[expert_id] = ExpertUsageStats(
                                    expert_id=expert_id,
                                    layer_idx=layer_idx,
                                    expert_idx=expert_idx
                                )

                            stats = self.expert_stats[expert_id]
                            stats.usage_count += 1
                            stats.last_used_step = self.step_count
                            stats.total_tokens += logits.size(1)  # Batch size * seq len

        # Sort experts by usage count
        sorted_experts = sorted(
            self.expert_stats.items(),
            key=lambda x: x[1].usage_count,
            reverse=True
        )

        logger.info(f"Profiling complete. Top 5 experts: {sorted_experts[:5]}")

        return [(k, v.usage_count) for k, v in sorted_experts]

    def optimize_expert_placement(self, top_experts: List[Tuple[str, int]]) -> None:
        """
        Move frequently used experts to VRAM based on profiling results.

        Args:
            top_experts: List of (expert_id, usage_count) to cache
        """
        logger.info("Optimizing expert placement based on usage patterns")

        # Calculate how many experts we can fit in VRAM
        if self.expert_memory_size is None:
            self._estimate_expert_memory_size()

        max_experts = int(self.vram_budget / self.expert_memory_size) if self.expert_memory_size else 10
        logger.info(f"Can cache up to {max_experts} experts in {self.vram_budget / 1024**3:.1f}GB VRAM")

        # Select top experts to cache
        experts_to_cache = []
        for expert_id, usage_count in top_experts[:max_experts]:
            if expert_id not in self.cached_experts:
                experts_to_cache.append((expert_id, usage_count))

        # Move experts to GPU
        for expert_id, usage_count in experts_to_cache:
            stats = self.expert_stats[expert_id]
            layer_idx = stats.layer_idx
            expert_idx = stats.expert_idx

            # Get the expert module
            expert_module = self._get_expert_module(layer_idx, expert_idx)
            if expert_module is not None:
                # Move to GPU
                expert_module.to(self.device)
                self.cached_experts.add(expert_id)
                logger.info(f"Cached {expert_id} to VRAM (usage: {usage_count})")
            else:
                logger.error(f"Failed to cache expert {expert_id}: Could not access expert module")

    def _estimate_expert_memory_size(self) -> None:
        """Estimate memory size of a single expert"""
        # Get a sample expert
        sample_expert = self._get_expert_module(0, 0)
        if sample_expert is not None:
            # Calculate parameter size
            total_params = sum(p.numel() for p in sample_expert.parameters())

            # Assume 4-bit quantization (0.5 bytes per parameter)
            self.expert_memory_size = total_params * 0.5

            logger.info(f"Estimated expert size: {self.expert_memory_size / 1024**2:.1f}MB")
        else:
            logger.error("Failed to estimate expert size: Could not access expert module")
            # Default estimate: 100MB per expert
            self.expert_memory_size = 100 * 1024**2

    def clear_cache(self) -> None:
        """Clear all cached experts and move them back to CPU"""
        logger.info("Clearing expert cache")

        for expert_id in list(self.cached_experts):
            stats = self.expert_stats[expert_id]
            layer_idx = stats.layer_idx
            expert_idx = stats.expert_idx

            expert_module = self._get_expert_module(layer_idx, expert_idx)
            if expert_module is not None:
                expert_module.to("cpu")
                self.cached_experts.remove(expert_id)
            else:
                logger.error(f"Failed to move expert {expert_id} to CPU: Could not access expert module")

    def get_cache_stats(self) -> Dict:
        """Get current cache statistics"""
        return {
            "cached_experts": len(self.cached_experts),
            "total_experts_tracked": len(self.expert_stats),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "vram_usage_gb": (len(self.cached_experts) * self.expert_memory_size) / 1024**3 if self.expert_memory_size else 0,
            "top_experts": list(self.cached_experts)[:10]
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate based on recent usage"""
        if not self.expert_stats:
            return 0.0

        recent_threshold = self.step_count - 100
        recent_uses = sum(
            1 for stats in self.expert_stats.values()
            if stats.last_used_step > recent_threshold and stats.expert_id in self.cached_experts
        )
        total_recent = sum(
            1 for stats in self.expert_stats.values()
            if stats.last_used_step > recent_threshold
        )

        return recent_uses / total_recent if total_recent > 0 else 0.0


class MemoryMonitor:
    """Monitor system and GPU memory usage"""

    @staticmethod
    def get_memory_stats() -> Dict:
        """Get current memory statistics"""
        stats = {}

        # System RAM
        mem = psutil.virtual_memory()
        stats['ram'] = {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent
        }

        # GPU VRAM
        if torch.cuda.is_available():
            stats['vram'] = {
                'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3,
                'free_gb': (torch.cuda.get_device_properties(0).total_memory -
                           torch.cuda.memory_allocated(0)) / 1024**3
            }

        return stats

    @staticmethod
    def check_memory_available(required_ram_gb: float, required_vram_gb: float) -> bool:
        """Check if enough memory is available"""
        stats = MemoryMonitor.get_memory_stats()

        ram_ok = stats['ram']['available_gb'] >= required_ram_gb
        vram_ok = True

        if torch.cuda.is_available():
            vram_ok = stats['vram']['free_gb'] >= required_vram_gb

        return ram_ok and vram_ok