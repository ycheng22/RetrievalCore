# Architecture & Technical Decisions

This document records key technical decisions, environment configurations, and dependency trade-offs.

## Local Hardware & Deep Learning Runtime Environment

- **GPU Model**: NVIDIA GeForce RTX 2060 SUPER
- **VRAM**: 8192 MiB (8 GB)
- **Driver Version**: 595.95
- **CUDA Version**: CUDA 12.4 (Driver supports up to CUDA 13.2)
- **PyTorch Version**: 2.6.0+cu124
- **vLLM Version**: N/A (Official wheels do not support native Windows; recommended to use WSL2 or local serving via Ollama/llama.cpp)
