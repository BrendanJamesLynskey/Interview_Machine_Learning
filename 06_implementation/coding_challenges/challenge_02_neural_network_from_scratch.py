"""
Challenge 02: Two-Layer Neural Network from Scratch (NumPy only)
================================================================

Task:
    1. Implement a two-layer neural network: Input -> Hidden (ReLU) -> Output (Softmax).
    2. Implement forward pass, softmax cross-entropy loss, and backpropagation by hand.
    3. Train on a synthetic multi-class classification dataset.
    4. Verify gradients using numerical differentiation.

Learning objectives:
    - Understand the chain rule applied to a multi-layer computation graph.
    - Implement numerically stable softmax and cross-entropy.
    - Understand backpropagation: how gradients flow through ReLU and linear layers.
    - Build intuition for weight initialisation (Xavier) and its effect on training.

Architecture:
    Input (n_features) -> Linear -> ReLU -> Linear -> Softmax -> Loss (CE)
    Notation: W1, b1 (first layer), W2, b2 (second layer)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# Activations and loss functions
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, z) element-wise."""
    return np.maximum(0.0, z)


def relu_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Backward pass through ReLU.

    d(ReLU(z))/dz = 1 if z > 0 else 0.
    dL/dz = dL/dA * (z > 0)  (element-wise mask).
    """
    return dA * (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last axis.

    Stability trick: subtract max(z) per row before exponentiating.
    Without this, exp(z) overflows for large z values.

    For a batch of shape (N, K):
        softmax[i, k] = exp(z[i,k] - max_k(z[i])) / sum_k(exp(z[i,k] - max_k(z[i])))
    """
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_one_hot: np.ndarray) -> float:
    """
    Average cross-entropy loss over a batch.

    L = -(1/N) * sum_i sum_k y[i,k] * log(probs[i,k])

    Clipping probs avoids log(0) = -inf.
    """
    N = probs.shape[0]
    eps = 1e-12
    return -np.sum(y_one_hot * np.log(probs + eps)) / N


def softmax_cross_entropy_backward(probs: np.ndarray, y_one_hot: np.ndarray) -> np.ndarray:
    """
    Combined backward pass for softmax + cross-entropy loss.

    When softmax and cross-entropy are combined, the gradient with respect to
    the pre-softmax logits z simplifies beautifully:
        dL/dz = (probs - y_one_hot) / N

    Derivation:
        L = -sum_k y_k * log(p_k)
        p_k = softmax(z_k)
        dL/dz_j = p_j - y_j  (for one sample; proof uses Jacobian of softmax)
        For a batch: dL/dZ = (P - Y) / N
    """
    N = probs.shape[0]
    return (probs - y_one_hot) / N


# ---------------------------------------------------------------------------
# Two-layer neural network
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """
    Two-layer neural network for multi-class classification.

    Architecture: Linear(n_in -> n_hidden) -> ReLU -> Linear(n_hidden -> n_out) -> Softmax

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_output : int
        Number of output classes.
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # --- Xavier (Glorot) uniform initialisation ---
        # Scale: sqrt(6 / (fan_in + fan_out))
        # Ensures variance of activations is approximately preserved across layers.
        # Avoids vanishing or exploding activations in early training.
        limit1 = np.sqrt(6.0 / (n_input + n_hidden))
        limit2 = np.sqrt(6.0 / (n_hidden + n_output))

        self.W1 = rng.uniform(-limit1, limit1, size=(n_input, n_hidden))   # (D, H)
        self.b1 = np.zeros(n_hidden)                                        # (H,)
        self.W2 = rng.uniform(-limit2, limit2, size=(n_hidden, n_output))  # (H, K)
        self.b2 = np.zeros(n_output)                                        # (K,)

        # Cached values from forward pass (needed for backward pass)
        self._cache: dict = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : ndarray of shape (N, n_input)

        Returns
        -------
        probs : ndarray of shape (N, n_output)
            Softmax probabilities.
        """
        # Layer 1: Linear
        Z1 = X @ self.W1 + self.b1          # (N, H)

        # Activation: ReLU
        A1 = relu(Z1)                        # (N, H)

        # Layer 2: Linear
        Z2 = A1 @ self.W2 + self.b2         # (N, K)

        # Output: Softmax
        probs = softmax(Z2)                  # (N, K)

        # Cache for backward pass
        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}
        return probs

    def backward(self, y_one_hot: np.ndarray) -> dict:
        """
        Backward pass via backpropagation.

        Parameters
        ----------
        y_one_hot : ndarray of shape (N, K)
            One-hot encoded true labels.

        Returns
        -------
        grads : dict with keys 'dW1', 'db1', 'dW2', 'db2'
        """
        X  = self._cache["X"]
        Z1 = self._cache["Z1"]
        A1 = self._cache["A1"]
        probs = self._cache["probs"]

        # --- Backward through softmax + cross-entropy (combined) ---
        # dL/dZ2 shape: (N, K)
        dZ2 = softmax_cross_entropy_backward(probs, y_one_hot)

        # --- Backward through Layer 2 (Linear: Z2 = A1 @ W2 + b2) ---
        # dL/dW2 = A1^T @ dZ2   shape: (H, K)
        dW2 = A1.T @ dZ2

        # dL/db2 = sum over batch dim   shape: (K,)
        db2 = dZ2.sum(axis=0)

        # dL/dA1 = dZ2 @ W2^T   shape: (N, H)
        dA1 = dZ2 @ self.W2.T

        # --- Backward through ReLU ---
        # dL/dZ1 = dL/dA1 * (Z1 > 0)   shape: (N, H)
        dZ1 = relu_backward(dA1, Z1)

        # --- Backward through Layer 1 (Linear: Z1 = X @ W1 + b1) ---
        # dL/dW1 = X^T @ dZ1   shape: (D, H)
        dW1 = X.T @ dZ1

        # dL/db1 = sum over batch dim   shape: (H,)
        db1 = dZ1.sum(axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update(self, grads: dict, lr: float) -> None:
        """Vanilla gradient descent parameter update."""
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (argmax of softmax probabilities)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# ---------------------------------------------------------------------------
# Gradient checking (numerical differentiation)
# ---------------------------------------------------------------------------

def numerical_gradient(
    net: TwoLayerNet,
    X: np.ndarray,
    y_one_hot: np.ndarray,
    param_name: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Estimate the gradient of the loss w.r.t. a parameter using finite differences.

    Centred finite difference: (f(p+eps) - f(p-eps)) / (2*eps)

    Used to verify the analytical backpropagation gradients.
    Only practical for small networks (O(n_params) forward passes required).
    """
    param = getattr(net, param_name)
    grad_numerical = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index

        original = param[idx]

        # f(p + eps)
        param[idx] = original + eps
        probs_plus = net.forward(X)
        loss_plus = cross_entropy_loss(probs_plus, y_one_hot)

        # f(p - eps)
        param[idx] = original - eps
        probs_minus = net.forward(X)
        loss_minus = cross_entropy_loss(probs_minus, y_one_hot)

        grad_numerical[idx] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore original value
        param[idx] = original
        it.iternext()

    return grad_numerical


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    net: TwoLayerNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.01,
    n_epochs: int = 200,
    batch_size: int = 64,
) -> list:
    """
    Mini-batch gradient descent training loop.

    Returns list of per-epoch average loss values.
    """
    N = X_train.shape[0]
    rng = np.random.default_rng(seed=0)
    loss_history = []

    def one_hot(y: np.ndarray, k: int) -> np.ndarray:
        oh = np.zeros((len(y), k))
        oh[np.arange(len(y)), y] = 1.0
        return oh

    for epoch in range(n_epochs):
        # Shuffle training data each epoch
        indices = rng.permutation(N)
        X_shuf = X_train[indices]
        y_shuf = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            y_oh    = one_hot(y_batch, n_classes)

            probs = net.forward(X_batch)
            loss  = cross_entropy_loss(probs, y_oh)
            grads = net.backward(y_oh)
            net.update(grads, lr)

            epoch_loss += loss
            n_batches  += 1

        loss_history.append(epoch_loss / n_batches)

    return loss_history


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_spiral_dataset(
    n_samples_per_class: int = 200,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple:
    """
    Generate a 2D spiral dataset for multi-class classification.

    Classic benchmark for neural networks -- not linearly separable.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples_per_class * n_classes, 2))
    y = np.zeros(n_samples_per_class * n_classes, dtype=int)

    for c in range(n_classes):
        ix = range(n_samples_per_class * c, n_samples_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_samples_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_samples_per_class)
        t += rng.normal(0, 0.2, n_samples_per_class)
        X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        y[ix] = c

    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- Generate spiral dataset ---
    N_PER_CLASS = 200
    N_CLASSES   = 3
    X, y = make_spiral_dataset(n_samples_per_class=N_PER_CLASS, n_classes=N_CLASSES)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {N_CLASSES} classes\n")

    # --- Train/test split ---
    split = int(0.8 * len(y))
    perm = np.random.permutation(len(y))
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_test,  y_test  = X[perm[split:]], y[perm[split:]]

    # --- Instantiate network ---
    net = TwoLayerNet(n_input=2, n_hidden=64, n_output=N_CLASSES, seed=42)

    # --- Gradient check (on a small batch before training) ---
    print("Running gradient check on W1 (small random subset)...")
    n_check = 8
    X_check = X_train[:n_check]
    y_check = y_train[:n_check]
    oh_check = np.zeros((n_check, N_CLASSES))
    oh_check[np.arange(n_check), y_check] = 1.0

    # Analytical gradient
    net.forward(X_check)
    grads_analytical = net.backward(oh_check)

    # Numerical gradient for W1 (first 2x2 block to keep it fast)
    net_copy = TwoLayerNet(n_input=2, n_hidden=64, n_output=N_CLASSES, seed=42)
    net_copy.W1 = net.W1.copy()
    net_copy.b1 = net.b1.copy()
    net_copy.W2 = net.W2.copy()
    net_copy.b2 = net.b2.copy()

    # Only check a 2x4 slice of W1 to keep runtime short
    dW1_numerical = numerical_gradient(net_copy, X_check, oh_check, "W1")
    dW1_slice_analytical = grads_analytical["dW1"][:2, :4]
    dW1_slice_numerical  = dW1_numerical[:2, :4]

    max_rel_error = np.max(
        np.abs(dW1_slice_analytical - dW1_slice_numerical)
        / (np.abs(dW1_slice_analytical) + np.abs(dW1_slice_numerical) + 1e-10)
    )
    print(f"Max relative gradient error (W1 slice): {max_rel_error:.2e}")
    print(f"  (Expect < 1e-5 for correct implementation)\n")

    # --- Train ---
    print("Training...")
    loss_history = train(net, X_train, y_train, N_CLASSES, lr=0.5, n_epochs=300, batch_size=64)

    # --- Evaluate ---
    train_acc = np.mean(net.predict(X_train) == y_train)
    test_acc  = np.mean(net.predict(X_test)  == y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print(f"Final loss:     {loss_history[-1]:.4f}\n")

    # --- Decision boundary plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Two-Layer Neural Network from Scratch (NumPy)", fontsize=13)

    # Loss curve
    axes[0].plot(loss_history, color="#3498db", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss Curve")
    axes[0].grid(True, alpha=0.3)

    # Decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = net.predict(grid).reshape(xx.shape)

    colors_map = ["#fadbd8", "#d5f5e3", "#d6eaf8"]
    scatter_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for c in range(N_CLASSES):
        mask = (Z == c)
        axes[1].contourf(xx, yy, mask.astype(float), alpha=0.3, colors=[colors_map[c]])

    for c in range(N_CLASSES):
        idx = y == c
        axes[1].scatter(X[idx, 0], X[idx, 1], s=15, alpha=0.7,
                        color=scatter_colors[c], label=f"Class {c}")

    axes[1].set_title(f"Decision Boundary (Test acc = {test_acc:.3f})")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("neural_network_scratch_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Expected results:")
    print("  Gradient check: max relative error < 1e-4 (ideally < 1e-6).")
    print("  Train accuracy: > 0.95 on spiral data with hidden_size=64.")
    print("  Test  accuracy: > 0.90 (spiral is not linearly separable; logistic")
    print("                   regression would score ~33 %).")
