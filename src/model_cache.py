"""
Model caching utilities for faster loading
"""

import torch
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import hashlib
import json

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache loaded models to speed up subsequent loads"""

    def __init__(self, cache_dir: Path = None):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "qwen3-local" / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")

    def _get_cache_key(self, model_path: str, device_map: Any, max_memory: dict) -> str:
        """
        Generate a unique cache key based on model configuration.

        Args:
            model_path: Path to model
            device_map: Device mapping configuration
            max_memory: Memory limits

        Returns:
            Cache key string
        """
        # Create a unique key based on configuration
        config_str = f"{model_path}_{str(device_map)}_{json.dumps(max_memory, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_model(self, model: Any, tokenizer: Any, model_path: str,
                   device_map: Any, max_memory: dict) -> bool:
        """
        Save model to cache.

        Args:
            model: The loaded model
            tokenizer: The tokenizer
            model_path: Original model path
            device_map: Device mapping used
            max_memory: Memory limits used

        Returns:
            True if saved successfully
        """
        try:
            cache_key = self._get_cache_key(model_path, device_map, max_memory)
            cache_file = self.cache_dir / f"model_{cache_key}.pt"
            tokenizer_file = self.cache_dir / f"tokenizer_{cache_key}.pkl"
            metadata_file = self.cache_dir / f"metadata_{cache_key}.json"

            logger.info(f"Saving model to cache: {cache_file}")

            # Save model state dict (more reliable than pickling the whole model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            }, cache_file)

            # Save tokenizer
            with open(tokenizer_file, 'wb') as f:
                pickle.dump(tokenizer, f)

            # Save metadata
            metadata = {
                'model_path': model_path,
                'device_map': str(device_map),
                'max_memory': max_memory,
                'cache_key': cache_key
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            logger.info("Model cached successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            return False

    def load_model(self, model_path: str, device_map: Any,
                   max_memory: dict) -> Optional[Tuple[Any, Any]]:
        """
        Load model from cache if available.

        Args:
            model_path: Original model path
            device_map: Device mapping configuration
            max_memory: Memory limits

        Returns:
            Tuple of (model, tokenizer) if cached, None otherwise
        """
        try:
            cache_key = self._get_cache_key(model_path, device_map, max_memory)
            cache_file = self.cache_dir / f"model_{cache_key}.pt"
            tokenizer_file = self.cache_dir / f"tokenizer_{cache_key}.pkl"
            metadata_file = self.cache_dir / f"metadata_{cache_key}.json"

            # Check if all cache files exist
            if not all(f.exists() for f in [cache_file, tokenizer_file, metadata_file]):
                logger.info("No cached model found")
                return None

            logger.info(f"Loading model from cache: {cache_file}")

            # Load metadata and verify it matches
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if metadata['cache_key'] != cache_key:
                logger.warning("Cache key mismatch, skipping cache")
                return None

            # Load tokenizer
            with open(tokenizer_file, 'rb') as f:
                tokenizer = pickle.load(f)

            # For now, return None - full model reconstruction is complex
            # This would need the original model class and proper initialization
            logger.info("Note: Full model cache loading not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached models"""
        try:
            for file in self.cache_dir.glob("*"):
                file.unlink()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


class CheckpointCache:
    """Cache for faster checkpoint loading using memory mapping"""

    @staticmethod
    def create_memory_mapped_checkpoint(model_path: Path, cache_dir: Path = None) -> Optional[Path]:
        """
        Create memory-mapped version of model checkpoints for faster loading.

        Args:
            model_path: Path to model directory
            cache_dir: Directory for memory-mapped files

        Returns:
            Path to memory-mapped checkpoint if created
        """
        try:
            cache_dir = cache_dir or Path.home() / ".cache" / "qwen3-local" / "mmap_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # This would require custom implementation for safetensors format
            # For now, return None - placeholder for future optimization
            logger.info("Memory-mapped checkpoint caching not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Failed to create memory-mapped checkpoint: {e}")
            return None