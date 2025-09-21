#!/usr/bin/env python3
"""Download or verify the Qwen3-Next-80B-A3B-Instruct-bnb-4bit model."""

import os
import sys
import argparse
from pathlib import Path
import logging
import subprocess
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SystemConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_local_model(model_path: Path) -> bool:
    """Check if the local model directory exists and contains model files."""
    if not model_path.exists():
        return False

    # Check for essential model files
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    for file in essential_files:
        if not (model_path / file).exists():
            return False

    # Check for model weight files (safetensors or bin)
    has_weights = any(
        model_path.glob("*.safetensors")
    ) or any(
        model_path.glob("*.bin")
    )

    return has_weights

def download_with_hf_cli(repo_id: str, local_dir: Path) -> bool:
    """Download model using huggingface-cli."""
    try:
        logger.info(f"Downloading {repo_id} to {local_dir}")
        logger.info("Using huggingface-cli for download...")

        cmd = [
            "huggingface-cli", "download",
            repo_id,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False"
        ]

        # Add token if available
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            cmd.extend(["--token", hf_token])

        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0

    except FileNotFoundError:
        logger.error("huggingface-cli not found. Please install with: pip install huggingface-hub[cli]")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def verify_model_config(model_path: Path) -> dict:
    """Verify and display model configuration."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info("Model configuration verified:")
        logger.info(f"  • Model type: {config.get('model_type', 'Unknown')}")
        logger.info(f"  • Architecture: {config.get('architectures', ['Unknown'])[0] if 'architectures' in config else 'Unknown'}")
        logger.info(f"  • Hidden size: {config.get('hidden_size', 'Unknown')}")
        logger.info(f"  • Num layers: {config.get('num_hidden_layers', 'Unknown')}")

        # Check for MoE configuration
        if 'num_experts' in config:
            logger.info(f"  • MoE experts: {config.get('num_experts')}")
            logger.info(f"  • Active experts: {config.get('num_experts_per_tok', 'Unknown')}")

        return config
    except Exception as e:
        logger.error(f"Failed to read model config: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Download or verify Qwen3-Next-80B-A3B-Instruct BnB 4-bit model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if model exists locally
  python src/download_model.py

  # Download model if not present
  python src/download_model.py --download

  # Force re-download even if exists
  python src/download_model.py --force

  # Use custom model directory
  python src/download_model.py --model-dir /path/to/model

Default model location: models/qwen3-80b-bnb
HuggingFace repo: unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit
        """
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model if not present locally"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/qwen3-80b-bnb",
        help="Local directory for the model (default: models/qwen3-80b-bnb)"
    )

    args = parser.parse_args()

    # Load config
    config = SystemConfig()
    model_path = Path(args.model_dir).resolve()

    logger.info("=" * 60)
    logger.info("Qwen3-Next-80B Model Manager")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Model directory: {model_path}")
    logger.info("")

    # Check if model exists locally
    if check_local_model(model_path) and not args.force:
        logger.info("✓ Model found locally!")
        logger.info("")

        # Verify model configuration
        verify_model_config(model_path)

        # Check model size
        import shutil
        if model_path.exists():
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            logger.info("")
            logger.info(f"Total model size: {total_size / (1024**3):.1f}GB")

        logger.info("")
        logger.info("Model is ready to use!")
        logger.info("Run './run.sh serve' to start the API server")

    elif args.download or args.force:
        if args.force and check_local_model(model_path):
            logger.warning("Model exists. Force re-downloading...")

        logger.info("Model not found locally or force download requested")
        logger.info("")
        logger.info("Model details:")
        logger.info("  • 80B parameters (3B active per forward pass)")
        logger.info("  • 4-bit quantized with BitsAndBytes")
        logger.info("  • Optimized for MoE inference")
        logger.info("  • Download size: ~40GB")
        logger.info("")

        # Check disk space
        import shutil
        parent_dir = model_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        free_space = shutil.disk_usage(parent_dir).free / (1024**3)  # GB
        if free_space < 50:
            logger.warning(f"Low disk space: {free_space:.1f}GB free")
            logger.warning("Model download requires ~40GB of space")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Download cancelled")
                sys.exit(0)
        else:
            logger.info(f"✓ Sufficient disk space available: {free_space:.1f}GB")

        logger.info("")
        logger.info("Starting download with huggingface-cli...")
        logger.info("This may take a while depending on your connection speed")
        logger.info("")

        # Create parent directory if needed
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        if download_with_hf_cli(config.model.model_name, model_path):
            logger.info("")
            logger.info("✓ Model downloaded successfully!")

            # Verify the downloaded model
            if check_local_model(model_path):
                verify_model_config(model_path)
                logger.info("")
                logger.info("Next steps:")
                logger.info("  1. Run './run.sh quick-test' to verify installation")
                logger.info("  2. Run './run.sh serve' to start the API server")
                logger.info("  3. Run './run.sh generate <prompt>' for CLI generation")
            else:
                logger.error("Model download may be incomplete. Please check and retry.")
                sys.exit(1)
        else:
            logger.error("Download failed. Please check the error messages above.")
            sys.exit(1)
    else:
        logger.warning("✗ Model not found locally")
        logger.info("")
        logger.info("To download the model, run one of:")
        logger.info("  • python src/download_model.py --download")
        logger.info("  • huggingface-cli download unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit --local-dir models/qwen3-80b-bnb")
        logger.info("")
        logger.info("The model is approximately 40GB in size.")
        sys.exit(1)

if __name__ == "__main__":
    main()