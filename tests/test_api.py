"""
Tests for API endpoints with MoE optimization
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import after adding to path
from api_server import app, ModelService


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)

        # Mock the model service
        self.mock_service = MagicMock()
        app.state.model_service = self.mock_service

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_generate_endpoint(self):
        """Test text generation endpoint"""
        self.mock_service.generate.return_value = "Generated text response"

        payload = {
            "prompt": "Hello, how are you?",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"], "Generated text response")

    def test_chat_completions_endpoint(self):
        """Test OpenAI-compatible chat completions endpoint"""
        self.mock_service.chat_completion.return_value = {
            "id": "chat-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3-80b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well."
                },
                "finish_reason": "stop"
            }]
        }

        payload = {
            "model": "qwen3-80b",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7
        }

        response = self.client.post("/api/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("choices", data)
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")

    def test_expert_stats_endpoint(self):
        """Test expert statistics endpoint"""
        self.mock_service.get_expert_stats.return_value = {
            "cached_experts": 15,
            "cache_hit_rate": 0.85,
            "top_experts": ["layer_0_expert_5", "layer_1_expert_12"],
            "memory_usage": {
                "vram_gb": 3.5,
                "ram_gb": 75.2
            }
        }

        response = self.client.get("/api/v1/expert-stats")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("cached_experts", data)
        self.assertIn("cache_hit_rate", data)
        self.assertIn("top_experts", data)
        self.assertEqual(data["cached_experts"], 15)

    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        self.mock_service.get_model_info.return_value = {
            "model_name": "qwen3-80b-moe",
            "num_parameters": "80B",
            "num_experts": 64,
            "context_length": 262144,
            "quantization": "4-bit BnB"
        }

        response = self.client.get("/api/v1/model")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("model_name", data)
        self.assertIn("num_parameters", data)
        self.assertEqual(data["model_name"], "qwen3-80b-moe")

    def test_streaming_generation(self):
        """Test streaming text generation"""
        # Mock streaming response
        def mock_stream():
            yield "Hello"
            yield " world"
            yield "!"

        self.mock_service.generate_stream.return_value = mock_stream()

        payload = {
            "prompt": "Say hello",
            "stream": True,
            "max_tokens": 10
        }

        # Streaming endpoints are handled differently
        # This is a simplified test
        response = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_batch_generation(self):
        """Test batch generation endpoint"""
        self.mock_service.generate_batch.return_value = [
            "Response 1",
            "Response 2",
            "Response 3"
        ]

        payload = {
            "prompts": [
                "Prompt 1",
                "Prompt 2",
                "Prompt 3"
            ],
            "max_tokens": 50,
            "optimize_experts": True
        }

        response = self.client.post("/api/v1/generate/batch", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 3)

    def test_expert_optimization_endpoint(self):
        """Test expert optimization control endpoint"""
        payload = {
            "enable": True,
            "cache_size": 5,
            "profiling_samples": 100
        }

        response = self.client.post("/api/v1/expert-optimization", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)

    def test_memory_stats_endpoint(self):
        """Test memory statistics endpoint"""
        self.mock_service.get_memory_stats.return_value = {
            "ram": {
                "total_gb": 128,
                "available_gb": 50,
                "used_gb": 78
            },
            "vram": {
                "total_gb": 16,
                "allocated_gb": 14,
                "free_gb": 2
            }
        }

        response = self.client.get("/api/v1/memory")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("ram", data)
        self.assertIn("vram", data)
        self.assertEqual(data["vram"]["total_gb"], 16)

    def test_error_handling(self):
        """Test API error handling"""
        # Test missing required field
        payload = {
            # Missing 'prompt' field
            "max_tokens": 100
        }

        response = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

        # Test server error
        self.mock_service.generate.side_effect = Exception("Model error")

        payload = {
            "prompt": "Test prompt",
            "max_tokens": 100
        }

        response = self.client.post("/api/v1/generate", json=payload)
        self.assertEqual(response.status_code, 500)

        data = response.json()
        self.assertIn("detail", data)


class TestModelService(unittest.TestCase):
    """Test model service"""

    def setUp(self):
        """Set up model service"""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_expert_manager = MagicMock()

        self.service = ModelService(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            expert_manager=self.mock_expert_manager
        )

    def test_initialization(self):
        """Test service initialization"""
        self.assertIsNotNone(self.service.pipeline)
        self.assertEqual(self.service.model, self.mock_model)
        self.assertEqual(self.service.tokenizer, self.mock_tokenizer)

    @patch('api_server.MoEInferencePipeline')
    def test_generate_method(self, mock_pipeline_class):
        """Test generate method"""
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "Generated text"
        mock_pipeline_class.return_value = mock_pipeline

        service = ModelService(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            expert_manager=self.mock_expert_manager
        )

        result = service.generate("Test prompt", max_tokens=50)
        self.assertEqual(result, "Generated text")

    def test_chat_completion_format(self):
        """Test chat completion response format"""
        self.service.pipeline.generate.return_value = "Assistant response"

        result = self.service.chat_completion(
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            model="qwen3"
        )

        self.assertIn("id", result)
        self.assertIn("object", result)
        self.assertIn("created", result)
        self.assertIn("model", result)
        self.assertIn("choices", result)
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(len(result["choices"]), 1)

    def test_expert_stats_retrieval(self):
        """Test getting expert statistics"""
        self.mock_expert_manager.get_cache_stats.return_value = {
            "cached_experts": 10,
            "cache_hit_rate": 0.75
        }

        stats = self.service.get_expert_stats()

        self.assertIn("cached_experts", stats)
        self.assertIn("cache_hit_rate", stats)
        self.assertEqual(stats["cached_experts"], 10)

    def test_model_info_retrieval(self):
        """Test getting model information"""
        self.mock_model.config = MagicMock()
        self.mock_model.config.num_hidden_layers = 80
        self.mock_model.config.num_experts = 64
        self.mock_model.config.max_position_embeddings = 262144

        info = self.service.get_model_info()

        self.assertIn("num_layers", info)
        self.assertIn("num_experts", info)
        self.assertIn("context_length", info)
        self.assertEqual(info["num_layers"], 80)


class TestWebSocket(unittest.TestCase):
    """Test WebSocket endpoints for streaming"""

    def setUp(self):
        """Set up WebSocket test client"""
        self.client = TestClient(app)
        self.mock_service = MagicMock()
        app.state.model_service = self.mock_service

    def test_websocket_connection(self):
        """Test WebSocket connection"""
        with self.client.websocket_connect("/api/v1/ws") as websocket:
            # Send generation request
            websocket.send_json({
                "type": "generate",
                "prompt": "Hello",
                "max_tokens": 10,
                "stream": True
            })

            # Mock streaming response
            self.mock_service.generate_stream.return_value = ["Hello", " world"]

            # Note: Actual WebSocket testing would require more setup
            # This is a simplified test
            websocket.close()

    def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        with self.client.websocket_connect("/api/v1/ws") as websocket:
            # Send invalid request
            websocket.send_json({
                "type": "invalid"
            })

            # Should receive error response
            # Note: Implementation depends on actual WebSocket handling
            websocket.close()


if __name__ == '__main__':
    unittest.main()