"""
Configuration for MoE-aware BitsAndBytes implementation
"""

import torch
from dataclasses import dataclass
from typing import Dict
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_name: str = "unsloth/Qwen3-Next-80B-A3B-Instruct-bnb-4bit"
    local_model_path: str = "models/qwen3-80b-bnb"  # Local model directory
    num_layers: int = 80
    num_experts: int = 64
    num_activated_experts: int = 6  # A3B means ~3B active params
    context_length: int = 262144
    vocab_size: int = 151936


@dataclass
class MemoryConfig:
    """Memory allocation configuration"""
    # GPU VRAM allocation
    gpu_memory_gb: float = 14.0  # Leave 2GB headroom on RTX 4090 16GB
    gpu_reserved_gb: float = 2.0  # Reserved for KV cache and activations

    # CPU RAM allocation
    cpu_memory_gb: float = 90.0  # Use 90GB of 100+ available
    cpu_buffer_gb: float = 10.0  # Buffer for expert swapping

    # Expert caching
    experts_vram_gb: float = 4.0  # VRAM budget for expert caching
    cached_experts_per_layer: int = 3  # Top-K experts to cache per layer

    @property
    def max_memory_mapping(self) -> Dict:
        """Get max_memory dict for model loading"""
        return {
            0: f"{self.gpu_memory_gb}GB",
            "cpu": f"{self.cpu_memory_gb}GB"
        }


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization configuration"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    llm_int8_enable_fp32_cpu_offload: bool = True  # Enable CPU offloading for MoE
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1

    # Optimization settings
    use_cache: bool = True
    torch_compile: bool = False  # Set True if torch>=2.0 with compile support
    flash_attention: bool = True
    streaming: bool = True

    # Batch settings
    batch_size: int = 1
    max_batch_size: int = 4


@dataclass
class ServerConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    api_prefix: str = "/api/v1"
    cors_origins: list = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class SystemConfig:
    """Complete system configuration"""
    model: ModelConfig = None
    memory: MemoryConfig = None
    quantization: QuantizationConfig = None
    inference: InferenceConfig = None
    server: ServerConfig = None

    # Paths
    cache_dir: Path = Path.home() / ".cache" / "huggingface"
    offload_dir: Path = Path.home() / ".cache" / "qwen3-local" / "offload"
    logs_dir: Path = Path.home() / ".cache" / "qwen3-local" / "logs"
    model_dir: Path = Path("models/qwen3-80b-bnb")  # Local model directory

    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.model is None:
            self.model = ModelConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.quantization is None:
            self.quantization = QuantizationConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.server is None:
            self.server = ServerConfig()

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "model": {
                "name": self.model.model_name,
                "num_layers": self.model.num_layers,
                "num_experts": self.model.num_experts,
                "context_length": self.model.context_length
            },
            "memory": {
                "gpu_memory_gb": self.memory.gpu_memory_gb,
                "cpu_memory_gb": self.memory.cpu_memory_gb,
                "experts_vram_gb": self.memory.experts_vram_gb,
                "cached_experts_per_layer": self.memory.cached_experts_per_layer
            },
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "quant_type": self.quantization.bnb_4bit_quant_type,
                "use_double_quant": self.quantization.bnb_4bit_use_double_quant
            },
            "inference": {
                "temperature": self.inference.temperature,
                "top_p": self.inference.top_p,
                "top_k": self.inference.top_k,
                "max_new_tokens": self.inference.max_new_tokens
            }
        }


# Default configuration instance
default_config = SystemConfig()