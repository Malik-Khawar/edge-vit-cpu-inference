# ⚡ EdgeViT: High-Throughput CPU Inference for Vision Transformers

This repository contains a standalone, high-performance systems optimization pipeline for deploying **Vision Transformers (ViT)** on memory-constrained Edge/CPU environments using PyTorch and Hugging Face.

Deploying state-of-the-art vision models like ViT on edge CPUs is challenging due to their large model footprint, high computation latency, and quadratic sequence length complexity. This project addresses these constraints by combining three powerful systems-level optimizations: **Token Merging (ToMe)**, **Entropy-Based Early Exits**, and **INT8 Dynamic Quantization**.

---

## 🚀 Key Features

1. **Token Merging (ToMe)**: Reduces sequence length on the fly by identifying and merging redundant token embeddings (such as background patches) using bipartite soft matching.
2. **Entropy-Based Early Exits**: Short-circuits the model forward pass at layer 6 (out of 12) if prediction entropy falls below a strict confidence threshold.
3. **INT8 Dynamic Quantization**: Compresses model parameters by 4x, mapping 32-bit floating-point weights to 8-bit integers for high-throughput CPU vector execution.
4. **Systems CPU Profiler**: Measures exact inference latency, CPU memory (RAM) usage, and GFLOPs complexity.
5. **Trade-off Visualizations**: Generates diagnostic Pareto plots mapping speed-to-similarity frontiers and latency-size comparisons.

---

## 🔬 Mathematical Formulation

### 1. Token Merging via Bipartite Soft Matching
Attention patches are divided into two disjoint sets, $A$ and $B$. We compute the cosine similarity between the key vectors $K_A$ and $K_B$:
$$\text{Sim}(A_i, B_j) = \frac{K_{A,i} \cdot K_{B,j}}{\|K_{A,i}\| \|K_{B,j}\|}$$
The top $r$ matches are merged by taking their weighted averages, decreasing the sequence length from $N$ to $N - r$ at each layer.

### 2. Entropy-Based Early Exits
An auxiliary classification head is trained at layer $L$. During inference, we evaluate the prediction probability distribution $P$ and calculate Shannon Entropy $H(P)$:
$$H(P) = -\sum_{c=1}^{C} P_c \log_2(P_c)$$
If $H(P) < \theta$ (confidence threshold), the model raises an early exit event and bypasses the remaining layers, saving significant compute.

### 3. INT8 Dynamic Quantization
Weight values $w$ are quantized dynamically:
$$q = \text{clip}\left( \text{round}\left( \frac{w}{S} \right), -128, 127 \right)$$
Where the scale factor $S$ is calculated per layer:
$$S = \frac{\max(|w|)}{127}$$

---

## 📊 Benchmarking & Results

Evaluated on a proxy CIFAR-10 dataset (resized to 224x224) using a standard Hugging Face `google/vit-base-patch16-224` model:

| Model Configuration | CPU Latency | Model Size | Compute Complexity | Feature similarity |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (FP32)** | `114.40 ms` | `330.30 MB` | `17.50 GFLOPs` | `100.00%` (Baseline) |
| **ToMe Only ($r=4$)** | `105.24 ms` | `330.30 MB` | `13.50 GFLOPs` | `96.27%` |
| **Early Exit Only ($\theta=1.5$)** | `82.50 ms` | `330.30 MB` | `11.50 GFLOPs` | `98.10%` |
| **Quantized Only (INT8)** | `71.20 ms` | `85.89 MB` | `17.50 GFLOPs` | `99.55%` |
| **EdgeViT (Combined)** | **`58.64 ms`** | **`85.89 MB`** | **`7.96 GFLOPs`** | **`99.39%`** |

*EdgeViT delivers a **2.0x latency reduction**, a **3.8x memory footprint savings**, and **2.2x fewer FLOPs** while retaining **99.39% cosine similarity** to the FP32 embedding representations.*

### Saved Outputs
- **Latency & Size Impact (`results/latency_size_comparison.png`)**: A dual-axis bar chart comparing speedups and model footprint reductions.
- **Pareto Frontier (`results/pareto_frontier.png`)**: A scatter plot charting the optimization Pareto frontier (Latency vs. Embedding Similarity).

---

## 📁 Repository Structure

```text
├── src/
│   ├── data.py             # Benchmarking dataset loader (subsampled CIFAR-10)
│   ├── vit_baseline.py     # Hugging Face Vision Transformer model wrapper
│   ├── token_merging.py    # Custom bipartite token matching & merging hooks
│   ├── early_exit.py       # Auxiliary exit head, train loops, and exit exception handler
│   ├── quantization.py     # INT8 dynamic quantization pipeline
│   ├── profile.py          # Latency, RAM, and FLOPs hardware metrics
│   ├── evaluate.py         # Embedding cosine similarity calculation
│   └── plot.py             # Matplotlib code for benchmark charts
├── results/                # Comparative bar charts and Pareto plots
├── main.py                 # Benchmarking pipeline orchestrator
├── pyproject.toml          # Workspace configurations and dependency lists
└── README.md               # Documentation
```

---

## ⚙️ Installation & Usage

### Setup Environment
Configure the project virtual environment and synchronize dependencies:
```bash
uv sync
```

### Run Benchmark Pipeline
Run the full optimization and plotting pipeline:
```bash
uv run python main.py
```
This runs the baseline, applies each optimization individually, trains the early exit head on the CPU, and saves final comparison charts to the `results/` folder.
