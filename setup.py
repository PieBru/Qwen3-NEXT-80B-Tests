"""
Setup script for Qwen3-Local MoE BitsAndBytes implementation
"""

from setuptools import setup, find_packages

setup(
    name="qwen3-local-moe",
    version="0.1.0",
    description="MoE-aware BitsAndBytes implementation for Qwen3-Next-80B",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ]
    },
)