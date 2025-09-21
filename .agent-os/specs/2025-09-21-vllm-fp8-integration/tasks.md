# Spec Tasks

## Tasks

- [ ] 1. Download and Setup FP8 Model
  - [ ] 1.1 Install huggingface-hub if not present
  - [ ] 1.2 Download neuralmagic/Qwen3-NEXT-80B-A3B-Instruct-FP8 model (~80GB)
  - [ ] 1.3 Store in models/qwen3-80b-fp8/ directory
  - [ ] 1.4 Verify model files integrity and size

- [ ] 2. Install and Configure vLLM
  - [ ] 2.1 Remove any existing vLLM installation
  - [ ] 2.2 Install vLLM >=0.6.0 with pip
  - [ ] 2.3 Install optional flashinfer for better performance
  - [ ] 2.4 Verify vLLM installation and FP8 support

- [ ] 3. Create vLLM Server Script
  - [ ] 3.1 Write vllm_server.sh with proven Reddit configuration
  - [ ] 3.2 Configure CPU offloading (100GB)
  - [ ] 3.3 Set GPU memory utilization (0.9)
  - [ ] 3.4 Enable eager mode for MoE

- [ ] 4. Test and Validate Performance
  - [ ] 4.1 Start vLLM server on port 8000
  - [ ] 4.2 Test with simple "2+2" prompt
  - [ ] 4.3 Measure tokens/second performance
  - [ ] 4.4 Verify 10+ tok/s achievement

- [ ] 5. Create Migration Documentation
  - [ ] 5.1 Document vLLM setup process
  - [ ] 5.2 Create comparison with BitsAndBytes performance
  - [ ] 5.3 Write troubleshooting guide
  - [ ] 5.4 Update README with new vLLM instructions