# AWQ Model Migration - Lite Summary

Migrate from GPTQ to AWQ quantization using cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit for improved inference performance targeting >5 tok/sec.

## Key Points
- Replace GPTQ with AWQ 4-bit quantization for better performance
- Configure vLLM with tensor parallel size 4 for distributed inference
- Optimize memory usage for 16GB VRAM + 120GB RAM configuration