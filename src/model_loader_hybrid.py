"""
Enhanced model loader with proper hybrid GPU/CPU placement for MoE models
"""

import torch
import time
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import SystemConfig
from moe_utils import ExpertCacheManager

logger = logging.getLogger(__name__)

class HybridModelLoader:
    """Load MoE model with proper GPU/CPU placement for performance"""

    def __init__(self, config: SystemConfig):
        self.config = config
        logger.info(f"Initialized HybridModelLoader for {config.model.model_name}")

    def load_model(self):
        """Load model with hybrid GPU/CPU placement"""
        model_path = self.config.model.local_model_path or self.config.model.model_name

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check for cached model
        cache_path = Path.home() / ".cache/huggingface/model_cache/initialized_model.pt"

        if cache_path.exists():
            logger.info("Loading model from cache...")
            start = time.time()
            cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)

            # The cached data might be a dict or the model itself
            if isinstance(cached_data, dict):
                # If it's a dict, it might have the model under a key
                if 'model' in cached_data:
                    model = cached_data['model']
                else:
                    # It might be the state dict, need to load model structure first
                    logger.info("Cache contains state dict, loading model structure...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map={'': 'cpu'},
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            else:
                model = cached_data

            logger.info(f"Model loaded from cache in {time.time() - start:.1f}s")
        else:
            logger.info("Loading model from checkpoint files...")
            # Load to CPU first to avoid meta tensor issues
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={'': 'cpu'},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        # Now move non-expert components to GPU
        logger.info("Optimizing model placement for MoE architecture...")
        model = self._optimize_moe_placement(model)

        # Initialize expert manager
        self.expert_cache_manager = ExpertCacheManager(
            model=model,
            vram_budget_gb=self.config.memory.experts_vram_gb,
            num_cached_experts_per_layer=self.config.memory.cached_experts_per_layer,
            device="cuda:0"
        )

        model.eval()
        return model, tokenizer

    def _optimize_moe_placement(self, model):
        """Move non-expert components to GPU for performance"""
        if not torch.cuda.is_available():
            logger.warning("No GPU available, model will run slowly on CPU")
            return model

        device = torch.device("cuda:0")
        moved_components = []

        try:
            # Move embeddings
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                logger.info("Moving embeddings to GPU...")
                model.model.embed_tokens = model.model.embed_tokens.to(device)
                moved_components.append("embeddings")

            # Move output layers
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                logger.info("Moving final norm to GPU...")
                model.model.norm = model.model.norm.to(device)
                moved_components.append("final_norm")

            if hasattr(model, 'lm_head'):
                logger.info("Moving lm_head to GPU...")
                model.lm_head = model.lm_head.to(device)
                moved_components.append("lm_head")

            # Move critical components from each layer
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                logger.info(f"Optimizing {len(model.model.layers)} layers...")

                for i, layer in enumerate(model.model.layers):
                    # Move layer norms
                    if hasattr(layer, 'input_layernorm'):
                        layer.input_layernorm = layer.input_layernorm.to(device)
                    if hasattr(layer, 'post_attention_layernorm'):
                        layer.post_attention_layernorm = layer.post_attention_layernorm.to(device)

                    # Move attention components
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn = layer.self_attn.to(device)
                    if hasattr(layer, 'linear_attn'):
                        layer.linear_attn = layer.linear_attn.to(device)

                    # Move MoE routers/gates (critical for performance)
                    if hasattr(layer, 'block_sparse_moe'):
                        moe = layer.block_sparse_moe
                        if hasattr(moe, 'gate'):
                            moe.gate = moe.gate.to(device)
                        # Shared expert if exists
                        if hasattr(moe, 'shared_expert'):
                            moe.shared_expert = moe.shared_expert.to(device)

                        # Keep individual experts on CPU for memory efficiency
                        # They will be dynamically loaded as needed

                logger.info(f"Moved attention, routers and norms to GPU for all layers")
                moved_components.append(f"{len(model.model.layers)} layers (non-expert components)")

            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

            logger.info(f"✓ Model optimized for MoE: {', '.join(moved_components)} on GPU")
            logger.info("Experts remain on CPU and will be loaded dynamically")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during optimization: {e}")
            logger.warning("Model will run slower with components on CPU")
            torch.cuda.empty_cache()

        return model

# Export the loader
ModelLoader = HybridModelLoader