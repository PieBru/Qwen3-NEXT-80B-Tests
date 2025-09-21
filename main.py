#!/usr/bin/env python3
"""
Main entry point for Qwen3-Local MoE BitsAndBytes implementation
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_loader import ModelLoader
from inference import MoEInferencePipeline
from benchmark import PerformanceBenchmark, PerformanceReporter
from config import SystemConfig


def setup_logging(level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('qwen3_local.log')
        ]
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Qwen3-Local MoE BitsAndBytes Implementation"
    )

    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Server command
    server_parser = subparsers.add_parser('serve', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--output', help='Output file for results')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('prompt', help='Input prompt')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')

    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile expert usage')
    profile_parser.add_argument('--samples', type=int, default=100, help='Number of samples')

    # Common arguments
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--config', help='Config file path')

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logging.getLogger(__name__)

    # Load configuration
    config = SystemConfig()
    if args.config:
        # Load custom config if provided
        import json
        with open(args.config) as f:
            custom_config = json.load(f)
        # Update config with custom values
        from config import ModelConfig, MemoryConfig, QuantizationConfig, InferenceConfig, ServerConfig

        for key, value in custom_config.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    # Handle nested configs
                    config_attr = getattr(config, key)
                    if config_attr is None:
                        # Create the appropriate config class if needed
                        if key == 'model':
                            config_attr = ModelConfig()
                        elif key == 'memory':
                            config_attr = MemoryConfig()
                        elif key == 'quantization':
                            config_attr = QuantizationConfig()
                        elif key == 'inference':
                            config_attr = InferenceConfig()
                        elif key == 'server':
                            config_attr = ServerConfig()
                        setattr(config, key, config_attr)

                    for sub_key, sub_value in value.items():
                        if hasattr(config_attr, sub_key):
                            setattr(config_attr, sub_key, sub_value)
                else:
                    setattr(config, key, value)

    # Execute command
    if args.command == 'serve':
        run_server(config, args.host, args.port)

    elif args.command == 'benchmark':
        run_benchmark(config, args.output)

    elif args.command == 'generate':
        run_generation(
            config,
            args.prompt,
            args.max_tokens,
            args.temperature
        )

    elif args.command == 'profile':
        run_profiling(config, args.samples)

    else:
        parser.print_help()


def run_server(config: SystemConfig, host: str, port: int):
    """Run the API server"""
    import uvicorn
    from api_server import app

    logging.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_benchmark(config: SystemConfig, output_file: Optional[str]):
    """Run performance benchmark"""
    logger = logging.getLogger(__name__)
    logger.info("Starting performance benchmark...")

    # Load model
    loader = ModelLoader(config)
    model, tokenizer = loader.load_model()

    # Create pipeline
    pipeline = MoEInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        expert_manager=loader.expert_cache_manager
    )

    # Profile and optimize experts
    logger.info("Profiling expert usage...")
    sample_texts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What is the capital of France?",
        "How do neural networks work?",
        "Translate 'Hello world' to Spanish."
    ]
    loader.profile_and_optimize_experts(sample_texts, num_samples=50)

    # Run benchmark
    benchmark = PerformanceBenchmark(pipeline, target_tokens_per_second=8.0)

    # Test prompts
    test_prompts = [
        "Hello",
        "Write a short story about",
        "Explain the concept of machine learning in detail",
        "What are the key differences between supervised and unsupervised learning approaches in machine learning?"
    ]

    logger.info("Running benchmark suite...")
    results = benchmark.run_benchmark_suite(
        test_prompts=test_prompts,
        output_lengths=[10, 50, 100],
        expert_configs=[
            {'cache_size': 3},
            {'cache_size': 5},
            {'cache_size': 7}
        ]
    )

    # Generate report
    reporter = PerformanceReporter()
    reporter.add_benchmark_result('overall', results['overall'])
    reporter.add_benchmark_result('by_output_length', results['by_output_length'])

    # Memory stats
    memory_stats = loader.get_memory_usage()
    reporter.add_benchmark_result('memory', memory_stats)

    # Expert cache stats
    cache_stats = loader.expert_cache_manager.get_cache_stats()
    reporter.add_benchmark_result('expert_cache', cache_stats)

    # Save report
    if output_file:
        output_path = Path(output_file)
        reporter.save_report(output_path)
        logger.info(f"Report saved to {output_path}")
    else:
        print("\n" + reporter.to_markdown())

    # Validate targets
    targets_met = benchmark.validate_performance_targets(results['overall'])
    if targets_met:
        logger.info("✅ All performance targets met!")
    else:
        logger.warning("❌ Some performance targets not met")

    return 0 if targets_met else 1


def run_generation(
    config: SystemConfig,
    prompt: str,
    max_tokens: int,
    temperature: float
):
    """Run text generation"""
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")

    # Load model
    loader = ModelLoader(config)
    model, tokenizer = loader.load_model()

    # Create pipeline
    pipeline = MoEInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        expert_manager=loader.expert_cache_manager
    )

    # Generate
    logger.info("Generating text...")
    start_time = time.time()

    output = pipeline.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        optimize_experts=True
    )

    elapsed = time.time() - start_time

    # Display results
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"{'='*50}")
    print(f"Response: {output}")
    print(f"{'='*50}")
    print(f"Generated in {elapsed:.2f} seconds")

    # Show expert stats
    cache_stats = loader.expert_cache_manager.get_cache_stats()
    print(f"Cached experts: {cache_stats['cached_experts']}")
    print(f"Cache hit rate: {cache_stats.get('cache_hit_rate', 0):.2%}")


def run_profiling(config: SystemConfig, num_samples: int):
    """Run expert profiling"""
    logger = logging.getLogger(__name__)
    logger.info(f"Profiling experts with {num_samples} samples...")

    # Load model
    loader = ModelLoader(config)
    model, tokenizer = loader.load_model()

    # Sample prompts for profiling
    sample_prompts = [
        "Explain artificial intelligence",
        "Write code to implement quicksort",
        "What is photosynthesis?",
        "Translate this to French: Good morning",
        "Solve this math problem: 15 * 27",
        "Tell me about the solar system",
        "How does a computer work?",
        "What are the benefits of exercise?",
        "Explain blockchain technology",
        "Write a haiku about nature"
    ] * (num_samples // 10)

    # Profile
    loader.profile_and_optimize_experts(
        sample_prompts[:num_samples],
        num_samples=num_samples
    )

    # Display results
    cache_stats = loader.expert_cache_manager.get_cache_stats()
    print(f"\n{'='*50}")
    print("Expert Profiling Results")
    print(f"{'='*50}")
    print(f"Total experts tracked: {cache_stats['total_experts_tracked']}")
    print(f"Cached experts: {cache_stats['cached_experts']}")
    print(f"VRAM usage: {cache_stats.get('vram_usage_gb', 0):.2f} GB")
    print("\nTop cached experts:")
    for expert in cache_stats.get('top_experts', [])[:10]:
        print(f"  - {expert}")


if __name__ == "__main__":
    import time
    sys.exit(main())