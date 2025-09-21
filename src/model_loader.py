"""
Model loader with MoE-aware device mapping
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Dict
import logging

from moe_utils import create_moe_device_map, ExpertCacheManager, MemoryMonitor
from config import SystemConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads MoE model with custom device mapping and quantization"""

    def __init__(self, config: SystemConfig):
        """
        Initialize model loader.

        Args:
            config: System configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.expert_cache_manager = None
        self.device_map = None

        logger.info(f"Initialized ModelLoader for {config.model.model_name}")

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model with MoE-aware device mapping.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info("Starting model loading process...")

        # Check memory availability
        if not self._check_memory_requirements():
            raise RuntimeError("Insufficient memory for model loading")

        # Use balanced device mapping to properly materialize tensors
        # Sequential can cause meta tensor issues
        self.device_map = "balanced"
        logger.info("Using balanced device mapping")

        # No BitsAndBytes config needed - model is already quantized
        # bnb_config = self._create_bnb_config()

        # Determine model path (local or remote)
        from pathlib import Path
        model_path = Path(self.config.model.local_model_path)
        if model_path.exists():
            model_name_or_path = str(model_path)
            logger.info(f"Using local model from: {model_name_or_path}")
        else:
            model_name_or_path = self.config.model.model_name
            logger.info(f"Using remote model: {model_name_or_path}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.config.quantization.trust_remote_code,
            cache_dir=self.config.cache_dir
        )

        # Load model with custom device map
        logger.info("Loading model with MoE device mapping...")

        # For pre-quantized models, we need to modify the config directly
        if model_path.exists():
            config_path = model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    model_config = json.load(f)

                # Update quantization config to allow CPU offloading
                if 'quantization_config' in model_config:
                    model_config['quantization_config']['llm_int8_enable_fp32_cpu_offload'] = True
                    # Temporarily save modified config
                    with open(config_path, 'w') as f:
                        json.dump(model_config, f, indent=2)
                    logger.info("Updated model config to enable CPU offloading")

        try:
            # Use very conservative memory limits to avoid CUDA OOM
            # Reserve most memory for activations and KV cache
            conservative_max_memory = {
                0: "8GB",  # Very conservative, only for essential components
                "cpu": "100GB"
            }

            # Set environment variables for better performance
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Note: Safetensors loading is inherently single-threaded
            # Multi-threading doesn't help with checkpoint loading

            # The model config has been updated, now load without extra quantization params
            # Note: Don't use dtype parameter with pre-quantized models, it can cause meta tensor issues
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=self.device_map,
                max_memory=conservative_max_memory,
                trust_remote_code=self.config.quantization.trust_remote_code,
                low_cpu_mem_usage=self.config.quantization.low_cpu_mem_usage,
                offload_folder=str(self.config.offload_dir),
                offload_state_dict=True,  # Ensure tensors are materialized
                cache_dir=self.config.cache_dir
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Initialize expert cache manager
        self.expert_cache_manager = ExpertCacheManager(
            model=self.model,
            vram_budget_gb=self.config.memory.experts_vram_gb,
            num_cached_experts_per_layer=self.config.memory.cached_experts_per_layer,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Set model to evaluation mode
        self.model.eval()

        return self.model, self.tokenizer

    def _create_device_map(self) -> Dict[str, any]:
        """Create custom device map for MoE model"""
        return create_moe_device_map(
            num_layers=self.config.model.num_layers,
            num_experts=self.config.model.num_experts,
            model_type="qwen3"
        )

    def _create_conservative_device_map(self) -> Dict[str, any]:
        """Create a conservative device map that minimizes GPU usage to avoid OOM"""
        device_map = {}

        # Only put the absolute essentials on GPU
        # Embeddings and final layers are small enough
        device_map["model.embed_tokens"] = 0
        device_map["model.norm"] = 0
        device_map["lm_head"] = 0

        # Put ALL layers on CPU to avoid CUDA OOM
        # We'll rely on CPU offloading for now
        for i in range(self.config.model.num_layers):
            device_map[f"model.layers.{i}"] = "cpu"

        logger.info(f"Conservative device map: {len([v for v in device_map.values() if v == 0])} components on GPU, "
                   f"{len([v for v in device_map.values() if v == 'cpu'])} on CPU")

        return device_map

    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes quantization configuration with CPU offloading support"""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.config.quantization.bnb_4bit_compute_dtype,
            llm_int8_enable_fp32_cpu_offload=self.config.quantization.llm_int8_enable_fp32_cpu_offload
        )

    def _check_memory_requirements(self) -> bool:
        """Check if system has enough memory"""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        # Check RAM
        ram_available = stats['ram']['available_gb']
        ram_required = self.config.memory.cpu_memory_gb

        if ram_available < ram_required:
            logger.error(
                f"Insufficient RAM: {ram_available:.1f}GB available, "
                f"{ram_required:.1f}GB required"
            )
            return False

        # Check VRAM if CUDA available
        if torch.cuda.is_available():
            vram_available = stats['vram']['free_gb']
            vram_required = self.config.memory.gpu_memory_gb

            if vram_available < vram_required * 0.8:  # Allow 20% tolerance
                logger.warning(
                    f"Low VRAM: {vram_available:.1f}GB available, "
                    f"{vram_required:.1f}GB requested"
                )

        logger.info(
            f"Memory check passed: {ram_available:.1f}GB RAM, "
            f"{stats.get('vram', {}).get('free_gb', 0):.1f}GB VRAM available"
        )
        return True

    def profile_and_optimize_experts(
        self,
        sample_texts: list,
        num_samples: int = 100
    ) -> None:
        """
        Profile expert usage and optimize placement.

        Args:
            sample_texts: Sample texts for profiling
            num_samples: Number of samples to profile
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        logger.info("Starting expert profiling...")

        # Tokenize sample texts
        inputs = self.tokenizer(
            sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Profile expert usage
        top_experts = self.expert_cache_manager.profile_expert_usage(
            inputs.input_ids,
            num_samples=num_samples
        )

        # Optimize expert placement
        self.expert_cache_manager.optimize_expert_placement(top_experts)

        # Log cache statistics
        cache_stats = self.expert_cache_manager.get_cache_stats()
        logger.info(f"Expert cache stats: {cache_stats}")

    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        if self.expert_cache_manager:
            stats['expert_cache'] = self.expert_cache_manager.get_cache_stats()

        return stats

    def clear_expert_cache(self) -> None:
        """Clear expert cache and move experts back to CPU"""
        if self.expert_cache_manager:
            self.expert_cache_manager.clear_cache()
            logger.info("Expert cache cleared")

    def update_expert_cache(self, top_k: int = 5) -> None:
        """
        Update expert cache with new top-k configuration.

        Args:
            top_k: Number of experts to cache per layer
        """
        if self.expert_cache_manager:
            self.expert_cache_manager.num_cached_experts_per_layer = top_k
            logger.info(f"Updated expert cache to top-{top_k} per layer")