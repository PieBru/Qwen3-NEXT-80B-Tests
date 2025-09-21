"""
Setup script for Qwen3-Local MoE BitsAndBytes implementation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="qwen3-local-moe",
    version="0.1.0",
    description="MoE-aware BitsAndBytes implementation for Qwen3-Next-80B local deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PieBru",
    url="https://github.com/PieBru/Qwen3-NEXT-80B-Tests",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core ML dependencies
        "torch>=2.0.0",  # PyTorch - install with CUDA via install.sh
        "transformers>=4.35.0",  # HuggingFace Transformers
        "bitsandbytes>=0.41.0",  # 4-bit quantization
        "accelerate>=0.20.0",  # HuggingFace Accelerate

        # Model dependencies
        "sentencepiece>=0.1.99",  # Tokenizer
        "huggingface-hub>=0.17.0",  # Model downloading

        # System monitoring
        "psutil>=5.9.0",  # Process and system utilities
        "numpy>=1.24.0",  # Numerical operations

        # API server
        "fastapi>=0.100.0",  # REST API framework
        "uvicorn>=0.23.0",  # ASGI server
        "pydantic>=2.0.0",  # Data validation
        "websockets>=10.0",  # WebSocket support
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.21.0",  # Async test support
            "black>=22.0.0",  # Code formatter
            "isort>=5.0.0",  # Import sorter
        ]
    },
    entry_points={
        "console_scripts": [
            "qwen3-local=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",  # Update if different
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords="qwen3, moe, mixture-of-experts, llm, transformers, bitsandbytes, quantization",
)