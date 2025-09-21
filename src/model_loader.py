"""
Model loader with MoE-aware device mapping
"""

# Set up memory management environment before importing torch
import os
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Dict
import logging
import time
import threading

from moe_utils import create_moe_device_map, ExpertCacheManager, MemoryMonitor
from config import SystemConfig
# from model_cache import ModelCache  # Disabled: not functional with pre-quantized models

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
        # self.model_cache = ModelCache(config.cache_dir / "model_cache")  # Disabled: not functional with pre-quantized models

        logger.info(f"Initialized ModelLoader for {config.model.model_name}")

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model with MoE-aware device mapping.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info("Starting model loading process...")
        start_time = time.time()

        # Check memory availability
        if not self._check_memory_requirements():
            raise RuntimeError("Insufficient memory for model loading")

        # Check if we have a cached model
        cached_model_path = self.config.cache_dir / "model_cache" / "initialized_model.pt"
        if cached_model_path.exists():
            logger.info("Found cached initialized model, attempting to load...")
            loaded = self._load_cached_model(cached_model_path)
            if loaded:
                logger.info(f"Model loaded from cache in {time.time() - start_time:.1f} seconds")
                return self.model, self.tokenizer
            else:
                logger.info("Cache load failed, proceeding with normal loading...")

        # Use auto device mapping for pre-quantized models to avoid meta tensor issues
        # Auto properly materializes tensors during loading
        self.device_map = "auto"
        logger.info("Using auto device mapping for pre-quantized model")

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

        # For pre-quantized models, we need to prepare config with CPU offloading enabled
        # Note: We don't modify the actual config file on disk to avoid side effects
        config_overrides = {}
        if model_path.exists():
            config_path = model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    model_config = json.load(f)

                # Prepare config overrides for CPU offloading
                if 'quantization_config' in model_config:
                    if model_config['quantization_config'].get('llm_int8_enable_fp32_cpu_offload') != True:
                        logger.info("Note: Model config needs CPU offloading enabled for optimal MoE handling")
                        # We'll handle this through the quantization_config parameter instead of modifying files

        try:
            # Use very conservative memory limits to avoid CUDA OOM
            # Reserve most memory for activations and KV cache
            # conservative_max_memory = {
            #     0: "8GB",  # Very conservative, only for essential components
            #     "cpu": "100GB"
            # }

            # Set environment variables for better memory management
            import os
            # Only set if not already configured
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
                logger.info("Set PYTORCH_CUDA_ALLOC_CONF for better memory management")

            # Note: Safetensors loading is inherently single-threaded
            # Multi-threading doesn't help with checkpoint loading

            # The model config has been updated, now load without extra quantization params
            # Note: Don't use dtype parameter with pre-quantized models, it can cause meta tensor issues

            logger.info("Loading checkpoint shards...")
            checkpoint_start = time.time()

            # For pre-quantized BitsAndBytes models, load with auto device map
            # The meta tensor issue is a known BitsAndBytes limitation
            # We'll work around it by catching the error and continuing
            try:
                # Clear CUDA cache before loading to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    max_memory=self.config.memory.max_memory_mapping,
                    trust_remote_code=self.config.quantization.trust_remote_code,
                    cache_dir=self.config.cache_dir,
                    offload_folder=str(self.config.offload_dir),
                    offload_state_dict=False,  # Changed to False to avoid potential issues
                    low_cpu_mem_usage=True  # Use low CPU memory mode
                )
            except RuntimeError as e:
                if "meta tensors" in str(e):
                    logger.warning("Meta tensor error encountered, attempting fallback loading...")
                    # Fallback: Load without device_map and handle manually
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=self.config.quantization.trust_remote_code,
                        cache_dir=self.config.cache_dir
                    )
                else:
                    raise

            checkpoint_time = time.time() - checkpoint_start
            logger.info(f"Checkpoint shards loaded in {checkpoint_time:.1f} seconds")

            # The model dispatch/initialization phase happens automatically with device_map
            logger.info("Model initialization complete")

            total_load_time = time.time() - start_time
            logger.info(f"Model fully loaded in {total_load_time:.1f} seconds total")

            # Try to cache the model for faster subsequent loads
            if not cached_model_path.exists():
                logger.info("Attempting to cache model for faster future loads...")
                cache_success = self._cache_initialized_model(cached_model_path)
                if cache_success:
                    logger.info("Model cached successfully - future loads will be much faster")
                else:
                    logger.info("Model caching failed - will load from scratch next time")

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

        # Check RAM (require only 55GB available to be safe)
        ram_available = stats['ram']['available_gb']
        ram_required = 55.0  # Minimum safe amount for 40GB model

        if ram_available < ram_required:
            logger.error(
                f"Insufficient RAM: {ram_available:.1f}GB available, "
                f"{ram_required:.1f}GB required (minimum for 40GB model)"
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

    def _monitor_dispatch_phase(self) -> None:
        """Monitor the model dispatch/initialization phase with progress indicator"""
        import psutil
        import sys

        # Create a simple progress indicator
        stop_monitoring = threading.Event()

        def monitor_progress():
            """Show progress during dispatch phase"""
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            start = time.time()

            while not stop_monitoring.is_set():
                elapsed = time.time() - start
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem_percent = psutil.virtual_memory().percent

                # Show spinner with status
                sys.stdout.write(f'\r  {spinner[idx]} Initializing model... '
                               f'[{elapsed:.0f}s] CPU: {cpu_percent:.1f}% RAM: {mem_percent:.1f}%')
                sys.stdout.flush()

                idx = (idx + 1) % len(spinner)
                time.sleep(0.2)

            sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear the line
            sys.stdout.flush()

        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.start()

        try:
            # The actual dispatch happens when we first access the model
            # Force initialization by accessing model parameters
            _ = self.model.config
            _ = next(self.model.parameters(), None)
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()

    def _cache_initialized_model(self, cache_path) -> bool:
        """
        Cache the initialized model to disk for faster loading.
        Uses direct model pickling to avoid state_dict() issues with quantized weights.

        Args:
            cache_path: Path to save the cached model

        Returns:
            True if successfully cached
        """
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching initialized model to {cache_path}...")

            # For BitsAndBytes models, we save the entire model object
            # This avoids the state_dict() meta tensor issue
            cache_data = {
                'model': self.model,  # Save the entire model object
                'tokenizer': self.tokenizer,
                'device_map': getattr(self.model, 'hf_device_map', self.device_map),
                'model_config': self.model.config,
                'timestamp': time.time()
            }

            # Use torch.save with pickle protocol 4 for better compatibility
            torch.save(cache_data, cache_path, pickle_protocol=4)
            cache_size_gb = cache_path.stat().st_size / (1024**3)
            logger.info(f"Model cached successfully ({cache_size_gb:.1f} GB)")
            return True

        except Exception as e:
            logger.warning(f"Could not cache model (non-critical): {e}")
            # Caching failure is non-critical, model still works
            return False

    def _load_cached_model(self, cache_path) -> bool:
        """
        Load model from cache.

        Args:
            cache_path: Path to cached model

        Returns:
            True if successfully loaded from cache
        """
        try:
            logger.info("Loading model from cache (this should be much faster)...")

            # Load the entire cached model object
            cache_data = torch.load(cache_path, map_location='cpu')

            # Check cache validity (optional - can add version checks here)
            if 'timestamp' in cache_data:
                import datetime
                cached_time = datetime.datetime.fromtimestamp(cache_data['timestamp'])
                logger.info(f"Using cache from {cached_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Directly use the cached model and tokenizer
            self.model = cache_data['model']
            self.tokenizer = cache_data['tokenizer']
            self.device_map = cache_data.get('device_map', 'auto')

            logger.info("Model loaded from cache successfully")

            # Move model to correct device if needed
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                logger.info("Moving model components to GPU...")
                # The model should already have the correct device mapping

            return True

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            logger.info("Falling back to normal loading...")
            return False