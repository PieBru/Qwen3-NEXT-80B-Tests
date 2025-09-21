#!/usr/bin/env python3
"""Test conservative model loading to avoid CUDA OOM"""

import os
import sys
import torch
import logging

# Set memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_load():
    """Test loading with conservative settings"""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA available. Free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB")

    from src.model_loader import ModelLoader
    from src.config import SystemConfig

    config = SystemConfig()
    logger.info("Creating model loader with conservative settings...")
    logger.info("- Sequential device mapping")
    logger.info("- Max VRAM: 8GB")
    logger.info("- Max RAM: 100GB")

    loader = ModelLoader(config)

    try:
        logger.info("Starting model load (this will take 10-15 minutes)...")
        model, tokenizer = loader.load_model()
        logger.info("✓ Model loaded successfully!")

        # Check memory usage
        if torch.cuda.is_available():
            used_vram = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"VRAM used: {used_vram:.2f} GB")

            if used_vram > 8:
                logger.warning(f"VRAM usage exceeds target of 8GB!")

        # Quick inference test
        logger.info("Testing inference...")
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Inference test successful: '{response}'")

        return True

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM: {e}")
        logger.error("Need to reduce VRAM allocation further")
        return False

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load()
    sys.exit(0 if success else 1)