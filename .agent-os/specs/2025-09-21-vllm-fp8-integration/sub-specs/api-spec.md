# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-21-vllm-fp8-integration/spec.md

## Endpoints

### POST /v1/completions

**Purpose:** Generate text completions (OpenAI-compatible)
**Parameters:**
- model: string (model name)
- prompt: string (input text)
- max_tokens: integer (maximum tokens to generate)
- temperature: float (0.0-2.0, sampling temperature)
- top_p: float (nucleus sampling)
- stream: boolean (streaming response)

**Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "qwen3-80b",
  "choices": [{
    "text": "generated text",
    "index": 0,
    "logprobs": null,
    "finish_reason": "stop"
  }]
}
```

**Errors:**
- 400: Invalid parameters
- 503: Model not loaded
- 500: Inference error

### POST /v1/chat/completions

**Purpose:** Chat completion endpoint (OpenAI-compatible)
**Parameters:**
- model: string
- messages: array of message objects
- max_tokens: integer
- temperature: float
- stream: boolean

**Response:** OpenAI chat completion format

### GET /health

**Purpose:** Check server and model status
**Parameters:** None
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_memory_used": "10.2GB",
  "cpu_memory_used": "98.5GB"
}
```