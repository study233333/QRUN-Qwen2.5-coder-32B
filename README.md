# TheWakeSystems — QRUN Hybrid Compression for Qwen2.5-Coder-32B

This repository contains TheWakeSystems' internally developed QRUN Hybrid compression project for the Qwen2.5-Coder-32B model. TheWakeSystems presents a quantum-classical hybrid architecture (Hybrid Q-RUN) that replaces selected model layers with quantum-informed modules to achieve extreme parameter compression and enterprise-grade multi-GPU distributed inference.

## Project Overview

TheWakeSystems developed this Hybrid Q-RUN solution to address enterprise requirements for low-parameter, high-throughput large-model inference. Leveraging a quantum-classical hybrid layer replacement strategy and engineering optimizations for distributed execution, the project demonstrates how to reduce model parameter count dramatically while retaining utility for reasoning and code-related tasks in constrained hardware environments.

Key facts retained from the original implementation:

- **Base model:** Qwen2.5-Coder-32B
- **Proprietary architecture:** TheWakeSystems' Hybrid Q-RUN quantum-classical hybrid layers
- **Parameter compression:** **3398M → 43.7M (≈ 1.3%)**
- **Recommended hardware:** 16× CUDA GPUs (BF16), total effective GPU memory ≈ **58.8 GB**
- **Primary entrypoint:** `scripts/benchmark_hybrid.py`

## Core Technical Capabilities (TheWakeSystems R&D Highlights)

- **Quantum-classical hybrid layer replacement:** Selective replacement of classical transformer components with quantum-informed modules to reduce parameter count while preserving representational capacity where most needed.
- **Extreme parameter compression:** Engineering and algorithmic optimizations enabling compression from billions of parameters to tens of millions without wholesale architecture redesign.
- **Enterprise multi-GPU deployment:** Native support and deployment patterns for multi-card distributed inference with BF16 precision to maximize throughput and resource utilization.
- **Inference optimizations:** Memory and compute optimizations targeted at minimizing runtime memory footprint and maximizing batch throughput across synchronized GPUs.

## Performance Summary

- **Compression ratio:** **3398M → 43.7M** (~**1.3%**) of original parameter count.
- **Memory profile:** Designed for deployment on a **16× GPU** BF16 cluster with an aggregate usable memory of **~58.8 GB**.
- **Intended workloads:** Mathematical and logical reasoning, code generation (note: see Known Limitations below).

These metrics represent TheWakeSystems' internal optimization targets and expected deployment envelope for enterprise environments.

## Enterprise Deployment — Quick Start

The following steps describe a concise enterprise-oriented deployment flow for evaluation and integration.

1. Prepare environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\\Scripts\\activate    # Windows PowerShell
pip install -r requirements.txt
```

2. Handle model weights

- Place official weights under `checkpoints/` following `checkpoints/README.md` guidance.
- **Large files (>100 MB):** use Git LFS or host weights in an enterprise artifact store (recommended). TheWakeSystems recommends hosting large model artifacts in dedicated storage (e.g., internal S3, private Hugging Face repository, or artifact registry) and referencing them in this repository.

3. Run single-node inference benchmark

```bash
python scripts/benchmark_hybrid.py --model-path checkpoints/checkpoints_hybrid_v2
```

4. Distributed multi-GPU execution

Follow your cluster's orchestration for multi-node or multi-GPU runs. Typical pattern for PyTorch-based distributed launch (example):

```bash
# Example (adjust to your cluster/orchestrator):
python -m torch.distributed.run --nproc_per_node=16 scripts/benchmark_hybrid.py --model-path checkpoints/checkpoints_hybrid_v2
```

For enterprise deployments, integrate this repository with your cluster scheduler (Slurm, Kubernetes, or proprietary orchestration), configure BF16 support and NCCL networking, and monitor GPU memory/temperature and interconnect usage.

## Project Structure

The repository follows an enterprise project layout to separate concerns and accelerate integration:

```
THeWakeSystems-QRUN-Qwen2.5-coder-32B/
├── README.md                     # This document (TheWakeSystems project overview)
├── LICENSE                       # Project license and usage terms
├── requirements.txt              # Python dependencies
├── MODEL_CARD.md                 # Model card and evaluation notes
├── model/                        # Model definitions and loader utilities
├── scripts/                      # Benchmarks, inference and training scripts
│   └── benchmark_hybrid.py       # Primary evaluation / demo entrypoint
├── examples/                     # Minimal examples and usage snippets
└── checkpoints/                  # Weight placement and storage guidance (NOT included in repo)
```

> Note: The `checkpoints/` directory is a placeholder for large artifacts; TheWakeSystems does not include weight files in the repository. See `checkpoints/README.md` for storage and retrieval guidance.

## Known Limitations and Responsible Disclosure

TheWakeSystems maintains a rigorous and transparent disclosure of known issues observed during internal evaluation:

- **Generation quality regressions:** Code generation, general commonsense reasoning, and some multilingual tasks may exhibit token repetition and semantic discontinuities. This behavior is under active investigation by TheWakeSystems R&D and is documented to ensure responsible use in production.
- **Model fidelity trade-offs:** Extreme compression strategies can introduce task-specific degradations. Evaluate the model against target enterprise tasks and apply domain-specific fine-tuning as required.

If you discover critical vulnerabilities or safety issues, please follow TheWakeSystems' security disclosure process (see `SECURITY.md` if available) or contact the maintainers directly.

## License & Open Source Statement

This project is published by TheWakeSystems under the terms provided in the `LICENSE` file. By using or distributing artifacts derived from this repository, you agree to comply with the license terms and any applicable export-control rules. TheWakeSystems retains intellectual property attribution for proprietary components and contributions in this repository where noted.

## About TheWakeSystems

TheWakeSystems is a research-driven engineering organization specializing in large-model compression, quantum-classical hybrid architectures, and enterprise-scale inference systems. Our team combines expertise in model compression, quantum-informed algorithms, distributed systems, and production-grade ML engineering to deliver solutions that reduce deployment cost while enabling high-throughput model inference for enterprise applications.

For more information, partnership inquiries, or enterprise licensing, contact: contact@thewakesystems.example (replace with official contact)

---

For detailed model specifications and experimental provenance, refer to `MODEL_CARD.md` and `checkpoints/README.md`.
