# Numerical Stability of Backpropagation in Neural Network Training

> A Systematic Analysis of Gradient Dynamics, Floating-Point Error Propagation, and Mitigation Strategies

**Author:** Raya Aldawoud  
**Course:** COM-701 – Numerical Mathematics and Computing  
**Institution:** Prince Sattam bin Abdulaziz University, AY 2025/2026

---

## Overview

This repository contains the complete implementation for a systematic numerical investigation of **gradient propagation stability** in deep neural networks. All experiments are implemented from scratch using **pure NumPy** — no PyTorch or TensorFlow — to give full transparency over every floating-point operation.

The study examines five controlled experiments:

| Experiment | Research Question |
|------------|-------------------|
| E1 | How does network depth affect gradient stability? |
| E2 | How does weight initialization affect gradient flow? |
| E3 | Does ZNorm reduce inter-layer gradient variance? |
| E4 | What do Jacobian spectral radii reveal at initialization? |
| E5 | How do MLP and vanilla RNN gradient profiles differ? |

---

## Key Results

- Gradient norm ratios decay from **0.9515** at depth L=2 to **0.6933** at L=8 under He initialization
- Gaussian initialization produces **near-zero gradients** from layer 3 onward at depth L=8
- Jacobian spectral radii (**0.73–0.82** in early layers, **0.96–1.06** in later layers) accurately predict gradient behavior *before* training
- ZNorm reduces inter-layer gradient variance consistently across all 100 iterations at **no convergence cost**
- Vanilla RNN first/last gradient ratio: **0.1909** vs. MLP: **1.0947** — a 5.24× gap explained by the shared recurrent weight matrix

---

## Repository Structure

```
├── experiments.py          # Main script — runs all 5 experiments
├── README.md               # This file
└── results/                # Auto-generated figures (created on first run)
    ├── E1_depth_gradients.png
    ├── E2_initialization.png
    ├── E3_znorm.png
    ├── E4_spectral_radius.png
    └── E5_mlp_vs_rnn.png
```

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Usage

```bash
python experiments.py
```

All figures are saved automatically to the `results/` directory. The script also prints numerical tables for E1 (gradient norm ratios) and E4 (spectral radii) to the console.

**Reproducibility:** All experiments use `seed=42` for both data generation and weight initialization. Results are fully deterministic.

---

## Implementation Highlights

- **Manual backpropagation** via the explicit chain rule — no automatic differentiation
- **Explicit Jacobian construction** at each layer for spectral radius analysis via `numpy.linalg.eigvals`
- Three initialization schemes: **Gaussian** (σ=0.01), **Xavier** (Glorot & Bengio, 2010), **He** (He et al., 2015)
- **ZNorm** gradient normalization (Yun, 2024): z-score standardization of layer-wise gradients
- **Vanilla RNN** with BPTT implemented from scratch for architecture comparison

---

## References

1. Zucchet & Orvieto (NeurIPS 2024) — Vanishing/exploding gradients are not the end of the story  
2. Rehmer & Kroll (IFAC 2020) — Vanishing/exploding gradients in gated recurrent units  
3. Engelken (NeurIPS 2023) — Gradient flossing  
4. Dadoun et al. (arXiv 2025) — Stability of the Jacobian matrix in deep networks  
5. Glorot & Bengio (AISTATS 2010) — Xavier initialization  
6. He et al. (ICCV 2015) — He initialization for ReLU networks  
7. Yun (arXiv 2024) — ZNorm gradient normalization  
