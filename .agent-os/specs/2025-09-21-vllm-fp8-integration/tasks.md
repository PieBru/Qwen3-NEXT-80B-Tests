# Spec Tasks

## Tasks

- [ ] 1. Set up vLLM with FP8 support
  - [ ] 1.1 Write tests for vLLM installation verification
  - [ ] 1.2 Install vLLM >=0.6.0 with CUDA 11.8 support
  - [ ] 1.3 Install required dependencies (vllm-flash-attn, compressed-tensors)
  - [ ] 1.4 Verify vLLM can import and detect GPU
  - [ ] 1.5 Verify all tests pass

- [ ] 2. Configure vLLM for FP8 model loading
  - [ ] 2.1 Write tests for FP8 model configuration
  - [ ] 2.2 Create vLLM configuration script for neuralmagic/Qwen3-NEXT-80B-A3B-Instruct-FP8
  - [ ] 2.3 Set up CPU offloading parameters (--cpu-offload-gb 90)
  - [ ] 2.4 Configure tensor parallel settings
  - [ ] 2.5 Verify all tests pass

- [ ] 3. Implement vLLM server with optimal MoE settings
  - [ ] 3.1 Write tests for vLLM server endpoints
  - [ ] 3.2 Create vLLM server launcher script with proper parameters
  - [ ] 3.3 Implement health check and inference endpoints
  - [ ] 3.4 Add memory monitoring and logging
  - [ ] 3.5 Verify all tests pass

- [ ] 4. Performance validation and benchmarking
  - [ ] 4.1 Write performance benchmark tests
  - [ ] 4.2 Create benchmark script to measure tokens/second
  - [ ] 4.3 Validate 10+ tok/s target is achieved
  - [ ] 4.4 Document actual performance metrics
  - [ ] 4.5 Verify all tests pass

- [ ] 5. Create migration documentation
  - [ ] 5.1 Document vLLM installation steps
  - [ ] 5.2 Create configuration reference guide
  - [ ] 5.3 Write troubleshooting section for common issues
  - [ ] 5.4 Add performance comparison table (BNB vs vLLM)
  - [ ] 5.5 Verify documentation completeness