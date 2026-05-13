"""
Numerical Stability of Backpropagation in Neural Network Training
=================================================================
A Systematic Analysis of Gradient Dynamics, Floating-Point Error
Propagation, and Mitigation Strategies

Author  : Raya Aldawoud
Course  : COM-701 – Numerical Mathematics and Computing
Affil.  : Prince Sattam bin Abdulaziz University, AY 2025/2026

Experiments
-----------
E1 – Effect of network depth on gradient stability (MLP)
E2 – Effect of weight initialization on gradient stability
E3 – Effect of ZNorm gradient normalization
E4 – Jacobian spectral radius analysis
E5 – MLP vs. vanilla RNN gradient comparison

Usage
-----
    python experiments.py

All figures are saved to the `results/` directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED        = 42
RESULTS_DIR = "results"
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"    : 150,
    "font.size"     : 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(n_samples=128, input_dim=64, n_classes=10, seed=42):
    """Generate synthetic Gaussian input data with random integer labels."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_samples, input_dim))
    y   = rng.integers(0, n_classes, size=n_samples)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def tanh_act(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    """Numerically stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_weights(n_in, n_out, scheme="he", seed=None):
    """
    Initialize weight matrix W and bias vector b.

    Schemes
    -------
    gaussian : W ~ N(0, 0.01)  — naive small-variance init
    xavier   : W ~ U(-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out)))
    he       : W ~ N(0, sqrt(2/n_in))  — recommended for ReLU
    """
    rng = np.random.default_rng(seed)
    if scheme == "gaussian":
        W = rng.standard_normal((n_in, n_out)) * 0.01
    elif scheme == "xavier":
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, (n_in, n_out))
    elif scheme == "he":
        std = np.sqrt(2.0 / n_in)
        W = rng.standard_normal((n_in, n_out)) * std
    else:
        raise ValueError(f"Unknown initialization scheme: '{scheme}'")
    b = np.zeros((1, n_out))
    return W, b


# ══════════════════════════════════════════════════════════════════════════════
# MLP FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def mlp_forward(X, weights, biases, activation="relu"):
    """
    Forward pass through a deep MLP.

    Returns
    -------
    probs  : softmax output probabilities  (batch, n_classes)
    caches : list of (pre-activation z, input h) per hidden layer
    h_last : final hidden representation before output layer
    """
    act_fn = relu if activation == "relu" else tanh_act
    caches = []
    h = X

    # Hidden layers
    for W, b in zip(weights[:-1], biases[:-1]):
        z     = h @ W + b
        h_new = act_fn(z)
        caches.append((z, h))
        h = h_new

    # Output layer (linear → softmax)
    z_out = h @ weights[-1] + biases[-1]
    probs = softmax(z_out)
    return probs, caches, h


def cross_entropy_loss(probs, y):
    """Mean cross-entropy loss with a small epsilon for numerical safety."""
    n     = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_p.mean()


# ══════════════════════════════════════════════════════════════════════════════
# MLP BACKWARD PASS (manual chain rule)
# ══════════════════════════════════════════════════════════════════════════════

def mlp_backward(probs, y, weights, biases, caches, activation="relu"):
    """
    Backpropagation via explicit chain-rule computation.

    Returns
    -------
    grad_norms   : list[float]   — ||dL/dz^(l)||_2 per hidden layer (earliest first)
    jacobians    : list[ndarray] — batch-averaged Jacobian per hidden layer
    weight_grads : list[ndarray] — dL/dW per hidden layer
    """
    act_grad = relu_grad if activation == "relu" else tanh_grad
    n = len(y)

    # Output layer delta: dL/d(z_out) = (p - one_hot(y)) / n
    delta = probs.copy()
    delta[np.arange(n), y] -= 1
    delta /= n

    grad_norms   = []
    jacobians    = []
    weight_grads = []

    for i in reversed(range(len(caches))):
        z, h_prev = caches[i]

        # Gradient w.r.t. pre-activation of layer i
        dz = delta @ weights[i + 1].T * act_grad(z)

        # Batch-averaged Jacobian: diag(sigma'(z)) @ W  [shape: (n_out, n_in)]
        act_d = act_grad(z)                         # (batch, hidden)
        J_avg = act_d.mean(axis=0)[:, None] * weights[i]

        jacobians.insert(0, J_avg)
        weight_grads.insert(0, h_prev.T @ dz)
        grad_norms.insert(0, np.linalg.norm(dz))
        delta = dz

    return grad_norms, jacobians, weight_grads


# ══════════════════════════════════════════════════════════════════════════════
# ZNORM GRADIENT NORMALIZATION  (Yun, 2024)
# ══════════════════════════════════════════════════════════════════════════════

def znorm(grads, eps=1e-8):
    """
    Z-score normalise each gradient array independently.

    g_hat = (g - mean(g)) / (std(g) + eps)

    Unlike gradient clipping, ZNorm rescales the full distribution at
    each layer, preventing any single layer from dominating updates.
    """
    return [(g - g.mean()) / (g.std() + eps) for g in grads]


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK BUILDER & TRAINING STEP
# ══════════════════════════════════════════════════════════════════════════════

def build_mlp(depth, input_dim=64, hidden_dim=64, output_dim=10,
              scheme="he", seed=0):
    """Construct weight/bias lists for a depth-layer MLP."""
    dims = [input_dim] + [hidden_dim] * depth + [output_dim]
    weights, biases = [], []
    for i in range(len(dims) - 1):
        W, b = init_weights(dims[i], dims[i + 1], scheme=scheme, seed=seed + i)
        weights.append(W)
        biases.append(b)
    return weights, biases


def train_step(X, y, weights, biases, lr=0.01,
               activation="relu", use_znorm=False):
    """One gradient-descent step; returns loss, gradient norms, and Jacobians."""
    probs, caches, _ = mlp_forward(X, weights, biases, activation)
    loss             = cross_entropy_loss(probs, y)
    grad_norms, jacobians, weight_grads = mlp_backward(
        probs, y, weights, biases, caches, activation)

    if use_znorm:
        weight_grads = znorm(weight_grads)

    # Update hidden-layer weights only (output layer excluded)
    for i in range(len(weights) - 1):
        weights[i] -= lr * weight_grads[i]

    return loss, grad_norms, jacobians


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 – Effect of Network Depth
# ══════════════════════════════════════════════════════════════════════════════

def experiment1():
    """
    Train MLP networks of depths L in {2,4,6,8,10} for 100 iterations
    under He initialization and measure layer-wise gradient norms.

    Research question: How does depth affect gradient stability?
    """
    print("\nExperiment 1: Effect of Network Depth")
    X, y    = generate_data()
    depths  = [2, 4, 6, 8, 10]
    n_iters = 100
    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(depths)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary   = {}

    for depth, color in zip(depths, colors):
        weights, biases = build_mlp(depth, scheme="he", seed=0)
        all_norms       = []

        for _ in range(n_iters):
            loss, grad_norms, _ = train_step(X, y, weights, biases)
            all_norms.append(grad_norms)

        all_norms   = np.array(all_norms)   # (iters, layers)
        final_norms = all_norms[-1]
        summary[depth] = final_norms

        layer_idx = np.arange(1, depth + 1)
        axes[0].plot(layer_idx, np.log10(final_norms + 1e-12),
                     marker="o", color=color, label=f"L={depth}")
        axes[1].plot(all_norms[:, 0], color=color, label=f"L={depth}")

    axes[0].set_xlabel("Layer index (1 = earliest)")
    axes[0].set_ylabel(r"$\log_{10}(\|\mathbf{g}^{(l)}\|_2)$")
    axes[0].set_title("Layer-wise Gradient Norms vs. Depth\n(final iteration, He init)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel(r"First-layer gradient norm $\|\mathbf{g}^{(1)}\|_2$")
    axes[1].set_title("First-Layer Gradient Norm During Training")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/E1_depth_gradients.png", bbox_inches="tight")
    plt.close()
    print("  Saved: E1_depth_gradients.png")

    # Gradient norm ratio table
    print(f"\n  {'Depth':>6} | {'First-layer norm':>18} | {'Last-layer norm':>16} | {'Ratio':>12}")
    print("  " + "-" * 60)
    for depth in depths:
        norms = summary[depth]
        ratio = norms[0] / (norms[-1] + 1e-12)
        print(f"  {depth:>6} | {norms[0]:>18.6e} | {norms[-1]:>16.6e} | {ratio:>12.6e}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 – Effect of Weight Initialization
# ══════════════════════════════════════════════════════════════════════════════

def experiment2():
    """
    Compare Gaussian, Xavier, and He initialization at depth L=8.

    Research question: How does initialization affect gradient stability?
    """
    print("\nExperiment 2: Effect of Weight Initialization")
    X, y    = generate_data()
    depth   = 8
    n_iters = 100
    schemes = ["gaussian", "xavier", "he"]
    colors  = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for scheme, color in zip(schemes, colors):
        weights, biases = build_mlp(depth, scheme=scheme, seed=0)
        all_norms, losses = [], []

        for _ in range(n_iters):
            loss, grad_norms, _ = train_step(X, y, weights, biases)
            all_norms.append(grad_norms)
            losses.append(loss)

        final_norms = np.array(all_norms)[-1]
        layer_idx   = np.arange(1, depth + 1)

        axes[0].plot(layer_idx, np.log10(final_norms + 1e-12),
                     marker="s", color=color, label=scheme.capitalize())
        axes[1].plot(losses, color=color, label=scheme.capitalize())

    axes[0].set_xlabel("Layer index (1 = earliest)")
    axes[0].set_ylabel(r"$\log_{10}(\|\mathbf{g}^{(l)}\|_2)$")
    axes[0].set_title(f"Layer-wise Gradient Norms by Initialization\n(depth={depth}, final iteration)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training Loss by Initialization Scheme")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/E2_initialization.png", bbox_inches="tight")
    plt.close()
    print("  Saved: E2_initialization.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 – ZNorm Gradient Normalization
# ══════════════════════════════════════════════════════════════════════════════

def experiment3():
    """
    Compare training with and without ZNorm at depth L=8 (He init).

    Research question: Does ZNorm reduce inter-layer gradient variance?
    """
    print("\nExperiment 3: ZNorm Gradient Normalization")
    X, y    = generate_data()
    depth   = 8
    n_iters = 100
    configs = [
        ("Baseline (no ZNorm)", False, "#e74c3c"),
        ("ZNorm",               True,  "#2ecc71"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, use_znorm, color in configs:
        weights, biases = build_mlp(depth, scheme="he", seed=0)
        grad_vars, losses = [], []

        for _ in range(n_iters):
            loss, grad_norms, _ = train_step(
                X, y, weights, biases, use_znorm=use_znorm)
            grad_vars.append(np.var(grad_norms))
            losses.append(loss)

        axes[0].plot(grad_vars, color=color, label=label)
        axes[1].plot(losses,    color=color, label=label)

    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Gradient norm variance across layers")
    axes[0].set_title("Inter-layer Gradient Variance\n(depth=8, He init)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training Loss: Baseline vs. ZNorm")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/E3_znorm.png", bbox_inches="tight")
    plt.close()
    print("  Saved: E3_znorm.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 – Jacobian Spectral Radius
# ══════════════════════════════════════════════════════════════════════════════

def experiment4():
    """
    Compute the spectral radius rho(J^(l)) at each layer before training.

    Research question: What is the per-layer spectral radius, and does it
    predict the gradient behaviour observed in E1?
    """
    print("\nExperiment 4: Jacobian Spectral Radius Analysis")
    X, y   = generate_data()
    depths = [2, 4, 6, 8, 10]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths)))

    fig, ax = plt.subplots(figsize=(9, 5))

    print(f"\n  {'Depth':>6} | Layer spectral radii")
    print("  " + "-" * 60)

    for depth, color in zip(depths, colors):
        weights, biases  = build_mlp(depth, scheme="he", seed=0)
        probs, caches, _ = mlp_forward(X, weights, biases)
        _, jacobians, _  = mlp_backward(probs, y, weights, biases, caches)

        spectral_radii = [
            np.max(np.abs(np.linalg.eigvals(J))) for J in jacobians
        ]

        layer_idx = np.arange(1, depth + 1)
        ax.plot(layer_idx, spectral_radii,
                marker="D", color=color, label=f"L={depth}")

        radii_str = "  ".join(f"{r:.4f}" for r in spectral_radii)
        print(f"  {depth:>6} | {radii_str}")

    ax.axhline(y=1.0, color="black", linestyle="--",
               linewidth=1.5, label=r"$\rho=1$ (stability boundary)")
    ax.set_xlabel("Layer index (1 = earliest)")
    ax.set_ylabel(r"Spectral radius $\rho(J^{(l)})$")
    ax.set_title("Jacobian Spectral Radius per Layer\n(at initialization, He init)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/E4_spectral_radius.png", bbox_inches="tight")
    plt.close()
    print("  Saved: E4_spectral_radius.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 – MLP vs. Vanilla RNN Gradient Comparison
# ══════════════════════════════════════════════════════════════════════════════

class VanillaRNN:
    """
    Minimal single-layer RNN implemented in pure NumPy.

    The recurrent weight matrix W_h is shared at every time step,
    which causes its spectral properties to compound uniformly across T
    steps — producing stronger gradient instability than the layer-wise
    weight variation in MLPs.
    """

    def __init__(self, input_dim=64, hidden_dim=64,
                 output_dim=10, T=10, scheme="he", seed=0):
        self.T          = T
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        std = {"he": np.sqrt(2.0 / input_dim),
               "xavier": np.sqrt(2.0 / (input_dim + hidden_dim))}.get(scheme, 0.01)

        self.Wx = rng.standard_normal((input_dim, hidden_dim)) * std
        self.Wh = rng.standard_normal((hidden_dim, hidden_dim)) * std
        self.bh = np.zeros((1, hidden_dim))
        self.Wy = rng.standard_normal((hidden_dim, output_dim)) * std
        self.by = np.zeros((1, output_dim))

    def forward(self, X):
        """X: (batch, T, input_dim)"""
        batch = X.shape[0]
        h     = np.zeros((batch, self.hidden_dim))
        hs, zs = [h], []

        for t in range(self.T):
            z = X[:, t, :] @ self.Wx + h @ self.Wh + self.bh
            h = tanh_act(z)
            hs.append(h)
            zs.append(z)

        probs = softmax(h @ self.Wy + self.by)
        return probs, hs, zs

    def backward(self, probs, y, hs, zs):
        """BPTT — returns per-time-step gradient norms (earliest first)."""
        n     = len(y)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        dh_next    = delta @ self.Wy.T
        grad_norms = []

        for t in reversed(range(self.T)):
            dz = dh_next * tanh_grad(zs[t])
            grad_norms.insert(0, np.linalg.norm(dz))
            dh_next = dz @ self.Wh.T

        return grad_norms


def experiment5():
    """
    Single forward+backward pass comparing a 10-layer MLP and
    vanilla RNN (T=10), both under He initialization.

    Research question: How do MLP and RNN gradient profiles differ?
    """
    print("\nExperiment 5: MLP vs. RNN Gradient Comparison")
    X_flat, y = generate_data()
    depth = T = 10

    # MLP gradients
    weights, biases  = build_mlp(depth, scheme="he", seed=0)
    probs, caches, _ = mlp_forward(X_flat, weights, biases)
    mlp_grad_norms, _, _ = mlp_backward(probs, y, weights, biases, caches)

    # RNN gradients
    X_seq = np.stack([X_flat] * T, axis=1)            # (128, 10, 64)
    rnn   = VanillaRNN(input_dim=64, hidden_dim=64,
                       output_dim=10, T=T, scheme="he", seed=0)
    probs_rnn, hs, zs = rnn.forward(X_seq)
    rnn_grad_norms    = rnn.backward(probs_rnn, y, hs, zs)

    # ── Plotting ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(1, depth + 1)

    axes[0].plot(idx, np.log10(np.array(mlp_grad_norms) + 1e-12),
                 marker="o", color="#3498db", label="MLP (L=10)")
    axes[0].plot(idx, np.log10(np.array(rnn_grad_norms) + 1e-12),
                 marker="s", color="#e74c3c", label="RNN (T=10)")
    axes[0].set_xlabel("Layer / Time-step index (1 = earliest)")
    axes[0].set_ylabel(r"$\log_{10}(\|\mathbf{g}\|_2)$")
    axes[0].set_title("Gradient Norms: MLP vs. RNN\n(He init, depth/T = 10)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    mlp_ratio = np.array(mlp_grad_norms) / (mlp_grad_norms[-1] + 1e-12)
    rnn_ratio = np.array(rnn_grad_norms) / (rnn_grad_norms[-1] + 1e-12)
    axes[1].plot(idx, mlp_ratio, marker="o", color="#3498db", label="MLP")
    axes[1].plot(idx, rnn_ratio, marker="s", color="#e74c3c", label="RNN")
    axes[1].set_xlabel("Layer / Time-step index (1 = earliest)")
    axes[1].set_ylabel("Gradient norm (normalized to last layer/step)")
    axes[1].set_title("Normalized Gradient Decay: MLP vs. RNN")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/E5_mlp_vs_rnn.png", bbox_inches="tight")
    plt.close()
    print("  Saved: E5_mlp_vs_rnn.png")

    # ── Numerical summary ─────────────────────────────────────────────────
    mlp_r = mlp_grad_norms[0] / (mlp_grad_norms[-1] + 1e-12)
    rnn_r = rnn_grad_norms[0] / (rnn_grad_norms[-1] + 1e-12)
    print(f"\n  MLP grad norms (layer 1→{depth}): "
          f"{mlp_grad_norms[0]:.4e} → {mlp_grad_norms[-1]:.4e}  |  ratio = {mlp_r:.4f}")
    print(f"  RNN grad norms (step  1→{T}):    "
          f"{rnn_grad_norms[0]:.4e} → {rnn_grad_norms[-1]:.4e}  |  ratio = {rnn_r:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    experiment1()
    experiment2()
    experiment3()
    experiment4()
    experiment5()
    print(f"\n  All experiments complete. Figures saved to '{RESULTS_DIR}/'")
