#!/usr/bin/env python3
"""Final test of CUDA OOM fixes"""

import os
import sys

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting model load test with conservative settings...")

    # Import after env vars are set
    from src.model_loader import ModelLoader
    from src.config import SystemConfig

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_info = torch.cuda.mem_get_info(0)
        logger.info(f"GPU Memory: {mem_info[0]/1024**3:.2f} GB free of {mem_info[1]/1024**3:.2f} GB total")

    config = SystemConfig()
    loader = ModelLoader(config)

    logger.info("Configuration:")
    logger.info("- Device mapping: sequential")
    logger.info("- Max GPU memory: 8GB")
    logger.info("- Max CPU memory: 100GB")
    logger.info("- Model path: models/qwen3-80b-bnb")
    logger.info("")
    logger.info("Loading model (expect 10-15 minutes for 40GB model)...")
    logger.info("Note: Loading is CPU-bound (single-threaded safetensors limitation)")

    try:
        model, tokenizer = loader.load_model()
        logger.info("✅ SUCCESS: Model loaded without CUDA OOM!")

        # Check actual memory usage
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory Used: {used:.2f} GB allocated, {reserved:.2f} GB reserved")

        # Quick test
        logger.info("Testing inference...")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Response: {response}")

        logger.info("")
        logger.info("✅ All tests passed! Model loads and runs without CUDA OOM.")
        return True

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA OOM Error: {e}")
        logger.error("The 8GB limit is still too high. May need further reduction.")
        return False

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)