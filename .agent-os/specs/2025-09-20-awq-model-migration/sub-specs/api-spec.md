# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-20-awq-model-migration/spec.md

> Created: 2025-09-20
> Version: 1.0.0

## Endpoints

### POST /v1/chat/completions

**Purpose:** Generate chat completions using the AWQ-quantized Qwen3 model
**Parameters:**
- `model`: "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
- `messages`: Array of chat messages with role and content
- `max_tokens`: Maximum tokens to generate (default: 8192)
- `temperature`: Sampling temperature (0.0 to 2.0)
- `top_p`: Nucleus sampling parameter
- `stream`: Boolean for streaming responses

**Response:** OpenAI-compatible chat completion format
**Errors:** 400 (Bad Request), 422 (Validation Error), 500 (Server Error)

### POST /v1/completions

**Purpose:** Generate text completions using the AWQ model
**Parameters:**
- `model`: "cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit"
- `prompt`: Input text prompt
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling parameter
- `stream`: Boolean for streaming responses

**Response:** OpenAI-compatible completion format
**Errors:** 400 (Bad Request), 422 (Validation Error), 500 (Server Error)

### GET /v1/models

**Purpose:** List available models including the AWQ model
**Parameters:** None
**Response:** Array of model objects with AWQ model information
**Errors:** 500 (Server Error)

### GET /health

**Purpose:** Health check endpoint for the vLLM server
**Parameters:** None
**Response:** JSON with server status and model readiness
**Errors:** 503 (Service Unavailable)

### GET /metrics

**Purpose:** Prometheus-style metrics for monitoring (if enabled)
**Parameters:** None
**Response:** Text format metrics including performance data
**Errors:** 404 (Not Found if metrics disabled)

## Controllers

### ChatCompletionController
- **Actions**: create_chat_completion, stream_chat_completion
- **Business Logic**: Process chat messages, apply AWQ model optimizations
- **Error Handling**: Validate input length against 8192 token limit, handle OOM gracefully

### CompletionController
- **Actions**: create_completion, stream_completion
- **Business Logic**: Process text prompts, optimize for AWQ performance characteristics
- **Error Handling**: Input validation, token limit enforcement, timeout management

### ModelController
- **Actions**: list_models, get_model_info
- **Business Logic**: Return AWQ model metadata, capabilities, and status
- **Error Handling**: Handle model loading states, availability checks

### HealthController
- **Actions**: health_check, readiness_check
- **Business Logic**: Verify AWQ model is loaded and ready for inference
- **Error Handling**: Report model loading issues, memory constraints

## Performance Considerations

### Request Processing
- Batch similar requests for improved throughput
- Implement request queuing for high load scenarios
- Monitor token generation speed (target >5 tok/sec)
- Handle long context requests (up to 8192 tokens) efficiently

### Response Formatting
- Maintain OpenAI API compatibility for existing clients
- Include AWQ-specific metadata in response headers
- Provide performance metrics in debug mode
- Support both streaming and non-streaming responses

### Error Responses
- Standardized error format with clear messages
- Include performance hints for optimization
- Rate limiting information in headers
- Memory usage warnings when approaching limits