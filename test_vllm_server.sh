#!/bin/bash
# Test script for vLLM server

echo "Testing vLLM server..."

# Wait for server to start
sleep 5

# Test generation endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/qwen3-80b-fp8",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

echo ""
echo "Test complete!"
