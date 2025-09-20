"""
Tests for MoE-optimized inference pipeline
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference import (
    MoEInferencePipeline,
    BatchManager,
    GenerationConfig,
    StreamingGenerator,
    MemoryManager
)


class TestMoEInferencePipeline(unittest.TestCase):
    """Test MoE-optimized inference pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_expert_manager = MagicMock()

        self.pipeline = MoEInferencePipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            expert_manager=self.mock_expert_manager
        )

    def test_single_generation(self):
        """Test single text generation"""
        # Mock tokenizer
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        self.mock_tokenizer.decode.return_value = "Generated text"
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 2

        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 5, 50000)
        self.mock_model.return_value = mock_output

        # Generate text
        result = self.pipeline.generate(
            "Test prompt",
            max_new_tokens=10,
            temperature=0.7
        )

        self.assertIsInstance(result, str)
        self.mock_tokenizer.encode.assert_called()
        self.mock_model.assert_called()

    def test_batch_generation(self):
        """Test batch text generation"""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        # Mock tokenizer for batch
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [1, 3], [1, 4]]),
            'attention_mask': torch.ones(3, 2)
        }
        self.mock_tokenizer.batch_decode.return_value = [
            "Result 1", "Result 2", "Result 3"
        ]
        self.mock_tokenizer.pad_token_id = 0

        # Mock model for batch
        mock_output = MagicMock()
        mock_output.logits = torch.randn(3, 2, 50000)
        self.mock_model.return_value = mock_output

        # Generate batch
        results = self.pipeline.generate_batch(
            prompts,
            max_new_tokens=10
        )

        self.assertEqual(len(results), 3)
        self.mock_tokenizer.batch_decode.assert_called()

    @patch('inference.MoEInferencePipeline._optimize_experts_for_input')
    def test_expert_optimization_during_inference(self, mock_optimize):
        """Test that experts are optimized before inference"""
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "Result"

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 50000)
        self.mock_model.return_value = mock_output

        # Generate with expert optimization
        self.pipeline.generate(
            "Test prompt",
            optimize_experts=True
        )

        # Verify expert optimization was called
        mock_optimize.assert_called_once()

    def test_streaming_generation(self):
        """Test streaming text generation"""
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "Word"

        # Mock model to return one token at a time
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 50000)
        self.mock_model.return_value = mock_output

        # Test streaming
        stream = self.pipeline.generate_stream(
            "Test prompt",
            max_new_tokens=5
        )

        tokens = list(stream)
        self.assertGreater(len(tokens), 0)


class TestBatchManager(unittest.TestCase):
    """Test batch management for efficient inference"""

    def setUp(self):
        """Set up test fixtures"""
        self.batch_manager = BatchManager(max_batch_size=4)

    def test_batch_grouping_by_length(self):
        """Test grouping inputs by similar length"""
        inputs = [
            "Short",
            "This is a medium length prompt",
            "Short too",
            "This is another medium length prompt that is similar",
            "Very short"
        ]

        batches = self.batch_manager.create_batches(
            inputs,
            group_by_length=True
        )

        # Should group similar length prompts
        self.assertGreater(len(batches), 0)
        self.assertLessEqual(len(batches[0]), 4)

    def test_batch_size_limit(self):
        """Test that batch size limit is enforced"""
        inputs = ["Input"] * 10

        batches = self.batch_manager.create_batches(
            inputs,
            max_batch_size=3
        )

        for batch in batches:
            self.assertLessEqual(len(batch), 3)

        # Total should equal original
        total = sum(len(batch) for batch in batches)
        self.assertEqual(total, 10)

    def test_expert_aware_batching(self):
        """Test batching inputs that likely use similar experts"""
        # Technical inputs
        technical = [
            "Write a Python function",
            "Implement an algorithm",
            "Debug this code"
        ]

        # Conversational inputs
        conversational = [
            "Hello, how are you?",
            "Thanks for your help",
            "Have a great day"
        ]

        all_inputs = technical + conversational

        batches = self.batch_manager.create_expert_aware_batches(
            all_inputs,
            expert_predictor=lambda x: 0 if "code" in x or "algorithm" in x else 1
        )

        # Should separate technical and conversational
        self.assertEqual(len(batches), 2)


class TestMemoryManager(unittest.TestCase):
    """Test memory management during inference"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory_manager = MemoryManager(
            max_memory_gb=16.0,
            oom_threshold=0.9
        )

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_oom_prevention(self, mock_props, mock_allocated):
        """Test OOM prevention mechanism"""
        # Mock near-full memory
        mock_allocated.return_value = 14 * 1024**3  # 14GB used
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        # Check if should prevent OOM
        should_clear = self.memory_manager.should_clear_cache()

        self.assertTrue(should_clear)

    @patch('torch.cuda.empty_cache')
    def test_cache_clearing(self, mock_clear_cache):
        """Test GPU cache clearing"""
        self.memory_manager.clear_gpu_cache()

        mock_clear_cache.assert_called_once()

    @patch('psutil.virtual_memory')
    def test_system_memory_check(self, mock_vmem):
        """Test system memory checking"""
        mock_vmem.return_value = MagicMock(
            total=128 * 1024**3,
            available=50 * 1024**3,
            percent=60.0
        )

        has_memory = self.memory_manager.check_system_memory(required_gb=40)

        self.assertTrue(has_memory)

        has_memory = self.memory_manager.check_system_memory(required_gb=60)

        self.assertFalse(has_memory)


class TestGenerationConfig(unittest.TestCase):
    """Test generation configuration"""

    def test_config_validation(self):
        """Test that generation config validates parameters"""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_new_tokens=100
        )

        # Validate temperature
        self.assertGreater(config.temperature, 0)
        self.assertLessEqual(config.temperature, 2.0)

        # Validate top_p
        self.assertGreater(config.top_p, 0)
        self.assertLessEqual(config.top_p, 1.0)

        # Validate top_k
        self.assertGreater(config.top_k, 0)

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = GenerationConfig(
            temperature=0.8,
            top_p=0.95,
            do_sample=True
        )

        config_dict = config.to_dict()

        self.assertEqual(config_dict['temperature'], 0.8)
        self.assertEqual(config_dict['top_p'], 0.95)
        self.assertTrue(config_dict['do_sample'])


class TestStreamingGenerator(unittest.TestCase):
    """Test streaming text generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

    def test_token_streaming(self):
        """Test streaming tokens one by one"""
        generator = StreamingGenerator(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer
        )

        # Mock tokenizer
        self.mock_tokenizer.encode.return_value = torch.tensor([1, 2, 3])
        self.mock_tokenizer.decode.side_effect = lambda x: f"Token{x[0]}"

        # Mock model to generate tokens
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 50000)
        self.mock_model.return_value = mock_output

        # Stream tokens
        tokens = []
        for token in generator.stream_generate("Test", max_tokens=3):
            tokens.append(token)

        self.assertEqual(len(tokens), 3)

    def test_streaming_with_stop_sequence(self):
        """Test streaming stops at stop sequence"""
        generator = StreamingGenerator(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            stop_sequences=["STOP"]
        )

        self.mock_tokenizer.encode.return_value = torch.tensor([1, 2])
        self.mock_tokenizer.decode.side_effect = [
            "Hello", "World", "STOP", "More"
        ]

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 1, 50000)
        self.mock_model.return_value = mock_output

        tokens = list(generator.stream_generate("Test", max_tokens=10))

        # Should stop before "More"
        self.assertNotIn("More", tokens)


class TestAsyncInference(unittest.TestCase):
    """Test asynchronous inference capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_expert_manager = MagicMock()

    async def test_async_generation(self):
        """Test async text generation"""
        from inference import AsyncMoEInferencePipeline

        pipeline = AsyncMoEInferencePipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            expert_manager=self.mock_expert_manager
        )

        # Mock async generation
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "Generated text"

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 3, 50000)
        self.mock_model.return_value = mock_output

        # Generate asynchronously
        result = await pipeline.generate_async("Test prompt")

        self.assertIsInstance(result, str)

    async def test_concurrent_generations(self):
        """Test multiple concurrent generations"""
        from inference import AsyncMoEInferencePipeline

        pipeline = AsyncMoEInferencePipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            expert_manager=self.mock_expert_manager
        )

        # Mock for concurrent calls
        self.mock_tokenizer.encode.return_value = [1, 2]
        self.mock_tokenizer.decode.side_effect = [
            "Result 1", "Result 2", "Result 3"
        ]

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 2, 50000)
        self.mock_model.return_value = mock_output

        # Generate multiple prompts concurrently
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        tasks = [pipeline.generate_async(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 3)


# Run async tests
def run_async_tests():
    """Run async test cases"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    suite = unittest.TestSuite()
    suite.addTest(TestAsyncInference('test_async_generation'))
    suite.addTest(TestAsyncInference('test_concurrent_generations'))

    unittest.TextTestRunner()

    async def run_tests():
        for test in suite:
            if asyncio.iscoroutinefunction(test._testMethodName):
                await getattr(test, test._testMethodName)()

    loop.run_until_complete(run_tests())
    loop.close()


if __name__ == '__main__':
    # Run regular tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run async tests
    print("\nRunning async tests...")
    run_async_tests()