"""
MoE-optimized inference pipeline
"""

import torch
import torch.nn.functional as F
import asyncio
from typing import List, Optional, Dict, Union, Generator
from dataclasses import dataclass
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor

from expert_manager import (
    ExpertProfiler,
    DynamicExpertLoader,
    ExpertSwapScheduler,
    PredictiveExpertPreloader
)
from moe_utils import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True
    max_new_tokens: int = 2048
    min_new_tokens: int = 1
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True

    def __post_init__(self):
        """Validate configuration parameters"""
        assert 0 < self.temperature <= 2.0, "Temperature must be between 0 and 2.0"
        assert 0 < self.top_p <= 1.0, "top_p must be between 0 and 1.0"
        assert self.top_k > 0, "top_k must be positive"
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"

    def to_dict(self) -> Dict:
        """Convert to dictionary for HuggingFace"""
        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'do_sample': self.do_sample,
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size,
            'early_stopping': self.early_stopping,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'use_cache': self.use_cache
        }


class BatchManager:
    """Manage batching for efficient inference"""

    def __init__(self, max_batch_size: int = 4):
        """
        Initialize batch manager.

        Args:
            max_batch_size: Maximum batch size
        """
        self.max_batch_size = max_batch_size

    def create_batches(
        self,
        inputs: List[str],
        max_batch_size: Optional[int] = None,
        group_by_length: bool = True
    ) -> List[List[str]]:
        """
        Create batches from inputs.

        Args:
            inputs: List of input strings
            max_batch_size: Override default max batch size
            group_by_length: Group similar length inputs

        Returns:
            List of batches
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        if not group_by_length:
            # Simple batching
            batches = []
            for i in range(0, len(inputs), max_batch_size):
                batches.append(inputs[i:i + max_batch_size])
            return batches

        # Group by length for efficiency
        length_groups = {}
        for inp in inputs:
            length_bucket = len(inp) // 100 * 100  # Bucket by 100 chars
            if length_bucket not in length_groups:
                length_groups[length_bucket] = []
            length_groups[length_bucket].append(inp)

        # Create batches from groups
        batches = []
        for group in length_groups.values():
            for i in range(0, len(group), max_batch_size):
                batches.append(group[i:i + max_batch_size])

        return batches

    def create_expert_aware_batches(
        self,
        inputs: List[str],
        expert_predictor: callable,
        max_batch_size: Optional[int] = None
    ) -> List[List[str]]:
        """
        Create batches based on predicted expert usage.

        Args:
            inputs: List of input strings
            expert_predictor: Function to predict expert usage
            max_batch_size: Override default max batch size

        Returns:
            List of batches grouped by expert usage
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        # Group by predicted experts
        expert_groups = {}
        for inp in inputs:
            expert_key = expert_predictor(inp)
            if expert_key not in expert_groups:
                expert_groups[expert_key] = []
            expert_groups[expert_key].append(inp)

        # Create batches from groups
        batches = []
        for group in expert_groups.values():
            for i in range(0, len(group), max_batch_size):
                batches.append(group[i:i + max_batch_size])

        return batches


class MemoryManager:
    """Manage memory during inference"""

    def __init__(
        self,
        max_memory_gb: float = 16.0,
        oom_threshold: float = 0.9
    ):
        """
        Initialize memory manager.

        Args:
            max_memory_gb: Maximum GPU memory in GB
            oom_threshold: Threshold for OOM prevention
        """
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.oom_threshold = oom_threshold
        self.memory_monitor = MemoryMonitor()

    def should_clear_cache(self) -> bool:
        """Check if GPU cache should be cleared"""
        if not torch.cuda.is_available():
            return False

        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory

        usage_ratio = allocated / total
        return usage_ratio > self.oom_threshold

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared GPU cache")

    def check_system_memory(self, required_gb: float) -> bool:
        """Check if enough system memory is available"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1024**3
        return available_gb >= required_gb

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        return self.memory_monitor.get_memory_stats()


class StreamingGenerator:
    """Generate text in streaming fashion"""

    def __init__(
        self,
        model,
        tokenizer,
        stop_sequences: Optional[List[str]] = None
    ):
        """
        Initialize streaming generator.

        Args:
            model: The language model
            tokenizer: The tokenizer
            stop_sequences: Sequences to stop generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences or []

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        config: Optional[GenerationConfig] = None
    ) -> Generator[str, None, None]:
        """
        Generate text in streaming fashion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            config: Generation configuration

        Yields:
            Generated tokens one by one
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            # Keep on CPU to match model device
            # input_ids = input_ids.cuda()
            pass

        # Generate token by token
        generated = []
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Sample next token
                if config.do_sample:
                    # Apply temperature
                    logits = logits / config.temperature

                    # Apply top-p and top-k
                    filtered_logits = self._top_k_top_p_filtering(
                        logits, config.top_k, config.top_p
                    )

                    # Sample
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Decode token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)

                # Check stop sequences
                generated.append(token_text)
                generated_text = "".join(generated)

                if any(stop in generated_text for stop in self.stop_sequences):
                    break

                # Yield token
                yield token_text

                # Update input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering"""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits


class MoEInferencePipeline:
    """Main inference pipeline with MoE optimizations"""

    def __init__(
        self,
        model,
        tokenizer,
        expert_manager,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize inference pipeline.

        Args:
            model: The MoE model
            tokenizer: The tokenizer
            expert_manager: Expert cache manager
            config: Default generation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.expert_manager = expert_manager
        self.config = config or GenerationConfig()

        # Initialize components
        self.batch_manager = BatchManager()
        self.memory_manager = MemoryManager()
        self.streaming_generator = StreamingGenerator(model, tokenizer)

        # Expert optimization components
        self.expert_profiler = ExpertProfiler(model)
        self.expert_loader = DynamicExpertLoader(model)
        self.swap_scheduler = ExpertSwapScheduler()
        self.expert_preloader = PredictiveExpertPreloader(model=model)

        logger.info("Initialized MoEInferencePipeline")

    def _get_expert_module(self, layer_idx: int, expert_idx: int):
        """
        Safely get expert module from model.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            Expert module or None if not accessible
        """
        try:
            # Try the standard Qwen3 MoE structure
            return self.model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
        except (AttributeError, IndexError):
            # Try alternative structures
            try:
                # Alternative 1: direct experts attribute
                return self.model.model.layers[layer_idx].experts[expert_idx]
            except (AttributeError, IndexError):
                try:
                    # Alternative 2: mlp_experts
                    return self.model.model.layers[layer_idx].mlp_experts[expert_idx]
                except (AttributeError, IndexError):
                    logger.warning(f"Could not access expert module at layer {layer_idx}, expert {expert_idx}")
                    return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        optimize_experts: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            optimize_experts: Whether to optimize expert placement
            stream: Whether to stream output
            **kwargs: Additional generation parameters

        Returns:
            Generated text or stream
        """
        # Update config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            **kwargs
        )

        # Optimize experts if requested
        if optimize_experts:
            self._optimize_experts_for_input(prompt)

        # Check memory
        if self.memory_manager.should_clear_cache():
            self.memory_manager.clear_gpu_cache()

        # Generate
        if stream:
            return self.streaming_generator.stream_generate(
                prompt, gen_config.max_new_tokens, gen_config
            )
        else:
            return self._generate_text(prompt, gen_config)

    def _generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Internal text generation method"""
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings
        )

        # Keep inputs on CPU since model is mostly on CPU
        # Moving to CUDA causes device mismatch errors
        # if torch.cuda.is_available():
        #     inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **config.to_dict()
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        optimize_experts: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens to generate
            optimize_experts: Whether to optimize expert placement
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Create batches
        batches = self.batch_manager.create_batches(prompts, group_by_length=True)

        results = []
        for batch in batches:
            # Optimize experts for batch
            if optimize_experts:
                self._optimize_experts_for_batch(batch)

            # Check memory
            if self.memory_manager.should_clear_cache():
                self.memory_manager.clear_gpu_cache()

            # Generate for batch
            batch_results = self._generate_batch(batch, max_new_tokens, **kwargs)
            results.extend(batch_results)

        return results

    def _generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Internal batch generation method"""
        config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            **kwargs
        )

        # Encode prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings
        )

        # Keep inputs on CPU since model is mostly on CPU
        # Moving to CUDA causes device mismatch errors
        # if torch.cuda.is_available():
        #     inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **config.to_dict()
            )

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        # Remove prompts from outputs
        results = []
        for prompt, generated in zip(prompts, generated_texts):
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            results.append(generated)

        return results

    def _optimize_experts_for_input(self, prompt: str):
        """Optimize expert placement for specific input"""
        # Predict likely experts
        predicted_experts = self.expert_preloader.predict_from_input(prompt)

        # Load predicted experts
        for expert_idx in predicted_experts[:5]:
            layer_idx = expert_idx // self.model.config.num_experts
            expert_module_idx = expert_idx % self.model.config.num_experts

            expert_module = self._get_expert_module(layer_idx, expert_module_idx)
            if expert_module is not None:
                self.expert_loader.load_expert_to_gpu(layer_idx, expert_module_idx, expert_module)
            else:
                logger.debug(f"Could not preload expert at layer {layer_idx}, expert {expert_module_idx}")

    def _optimize_experts_for_batch(self, prompts: List[str]):
        """Optimize expert placement for batch of inputs"""
        # Predict experts for all prompts
        all_predictions = []
        for prompt in prompts:
            predicted = self.expert_preloader.predict_from_input(prompt)
            all_predictions.extend(predicted)

        # Count frequencies
        expert_counts = {}
        for expert in all_predictions:
            expert_counts[expert] = expert_counts.get(expert, 0) + 1

        # Load most common experts
        sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
        for expert_idx, count in sorted_experts[:10]:
            if count >= len(prompts) / 2:  # Used by at least half the batch
                layer_idx = expert_idx // self.model.config.num_experts
                expert_module_idx = expert_idx % self.model.config.num_experts

                try:
                    expert_module = self.model.model.layers[layer_idx].block_sparse_moe.experts[expert_module_idx]
                    self.expert_loader.load_expert_to_gpu(layer_idx, expert_module_idx, expert_module)
                except Exception as e:
                    logger.debug(f"Could not preload expert: {e}")

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate text in streaming fashion"""
        config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            **kwargs
        )

        return self.streaming_generator.stream_generate(prompt, config.max_new_tokens, config)


class AsyncMoEInferencePipeline(MoEInferencePipeline):
    """Asynchronous version of MoE inference pipeline"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_async(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            kwargs.get('max_new_tokens'),
            kwargs.get('temperature'),
            kwargs.get('top_p'),
            kwargs.get('top_k'),
            kwargs.get('optimize_experts', True),
            False  # Don't stream in async
        )
        return result

    async def generate_batch_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate batch asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.generate_batch,
            prompts,
            kwargs.get('max_new_tokens'),
            kwargs.get('optimize_experts', True)
        )
        return result