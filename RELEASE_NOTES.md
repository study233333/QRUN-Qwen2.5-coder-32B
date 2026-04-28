# v0.1-beta Release Notes

## Overview
This is the initial open-source release of TheWakeSystems-QRUN-Qwen2.5-coder-32B. Based on the Qwen2.5-Coder-32B foundation, this project replaces 8 of the 64 original Transformer blocks with Hybrid Q-RUN quantum-informed layers to significantly compress the parameter count and explore the feasibility of quantum-classical hybrid inference.

## Highlights
- **Significant Parameter Compression**: The originally replaced layers contained approximately 3398M parameters, while our drop-in quantum modules contain only 43.7M parameters (a compression to approximately 1.3%).
- **Multi-GPU Parallel Inference**: Supports layer-wise shard loading for massive models (e.g., distributed across 16 GPUs) and resolves specific memory fragmentation issues introduced by layer substitution.
- **Mathematical and Logical Reasoning**: Retains standard mathematical calculation and logical reasoning capabilities.

## Known Issues / Limitations
- **Code Generation Degradation**: Some semantic breaks and token repetition phenomena are observed in multilingual code generation, long-context dependencies, and commonsense tasks. Not recommended for core generation in production environments.
- **High Memory Footprint**: Due to current underlying architectural constraints, a high overall memory budget is still required (~58.8GB GPU memory in BF16 mode).

## Next Steps / Mitigation Plan
1. [Priority High] **Fix Generation Repetition**: Plan to optimize the Q-RUN layer's attention propagation stability and training annealing strategies.
2. [Priority Medium] Provide single-card quantization options (e.g., AWQ/GPTQ) to further reduce memory requirements and improve edge deployment viability.
3. [Priority Medium] Improve `examples` to include fine-tuning guidelines and more comprehensive mathematical performance benchmark comparisons.
