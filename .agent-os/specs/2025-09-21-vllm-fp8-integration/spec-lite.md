# Spec Summary (Lite)

Replace the slow BitsAndBytes implementation (0.12 tok/s) with vLLM + FP8 quantization to achieve 10-20 tokens/second for Qwen3-80B inference. This leverages vLLM's native MoE support with CPU offloading for experts, matching the successful configuration reported by Reddit users with identical hardware.