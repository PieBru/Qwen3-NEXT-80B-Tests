# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-20-performance-poc/spec.md

> Created: 2025-09-20
> Version: 1.0.0

## Endpoints

### POST /inference

**Purpose:** Generate text completion using Qwen3-Next-80B model
**Parameters:**
- `prompt` (string, required): Input text prompt
- `max_tokens` (integer, optional, default: 100): Maximum tokens to generate
- `temperature` (float, optional, default: 0.7): Sampling temperature
- `stream` (boolean, optional, default: false): Enable streaming response

**Response:**
```json
{
  "text": "Generated completion text",
  "tokens_generated": 125,
  "generation_time": 2.3,
  "tokens_per_second": 5.43,
  "memory_used_gb": 14.2
}
```

**Errors:** 400 (invalid parameters), 500 (inference error), 503 (model not ready)

### GET /metrics

**Purpose:** Retrieve current performance and system metrics
**Parameters:** None

**Response:**
```json
{
  "model_loaded": true,
  "gpu_memory_used_gb": 15.8,
  "gpu_memory_total_gb": 16.0,
  "system_memory_used_gb": 45.2,
  "avg_tokens_per_second": 5.67,
  "total_inferences": 1247,
  "uptime_seconds": 3600
}
```

**Errors:** 500 (metrics collection error)

### POST /benchmark

**Purpose:** Run standardized performance benchmark
**Parameters:**
- `test_type` (string, optional, default: "standard"): Benchmark type (standard, stress, memory)
- `iterations` (integer, optional, default: 10): Number of test iterations

**Response:**
```json
{
  "benchmark_type": "standard",
  "iterations": 10,
  "avg_tokens_per_second": 5.42,
  "min_tokens_per_second": 4.98,
  "max_tokens_per_second": 5.89,
  "p95_tokens_per_second": 5.67,
  "avg_memory_usage_gb": 14.1,
  "test_prompts": [
    {
      "prompt_length": 50,
      "output_length": 100,
      "tokens_per_second": 5.34
    }
  ]
}
```

**Errors:** 400 (invalid test type), 500 (benchmark error)

### GET /health

**Purpose:** Check model and system health status
**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "model_ready": true,
  "gpu_available": true,
  "memory_sufficient": true,
  "last_inference": "2025-09-20T10:30:00Z"
}
```

**Errors:** 503 (unhealthy state)

## Controllers

### InferenceController
- **generate_completion()** - Handle text generation requests with performance tracking
- **validate_parameters()** - Ensure prompt and generation parameters are valid
- **track_performance()** - Record tokens/second and memory usage metrics

### MetricsController
- **get_current_metrics()** - Collect real-time system and model performance data
- **calculate_averages()** - Compute rolling averages for key performance indicators
- **format_response()** - Structure metrics data for API response

### BenchmarkController
- **run_standard_benchmark()** - Execute predefined test scenarios for performance validation
- **run_stress_test()** - Test sustained performance under continuous load
- **run_memory_test()** - Validate memory usage patterns across different prompt sizes
- **generate_report()** - Compile comprehensive benchmark results

### HealthController
- **check_model_status()** - Verify model is loaded and ready for inference
- **check_system_resources()** - Monitor GPU and system memory availability
- **validate_performance()** - Ensure current performance meets >5 tok/sec requirement