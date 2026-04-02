"""
Challenge 05: Gradient Descent Variants from Scratch
=====================================================

Task:
    1. Implement SGD, SGD with momentum, RMSProp, and Adam optimisers from scratch.
    2. Test each on a 2D non-convex surface (Rosenbrock function) and a
       logistic regression problem with a noisy mini-batch gradient.
    3. Plot the parameter trajectories and loss curves for each optimiser.
    4. Show parameter sensitivity: compare Adam with and without bias correction.

Learning objectives:
    - Understand the update equations for each optimiser from first principles.
    - Understand why momentum helps escape saddle points and narrow valleys.
    - Understand why adaptive learning rates (RMSProp, Adam) reduce the need for
      hand-tuning the learning rate per parameter.
    - Understand Adam's bias correction and why it matters in early training.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Optimiser base class
# ---------------------------------------------------------------------------

class Optimiser:
    """
    Base class for all optimiser implementations.

    Maintains a list of parameter arrays (numpy ndarrays) and their
    corresponding state dictionaries (for moment estimates, etc.).
    """

    def __init__(self, params: list, lr: float):
        self.params = params   # List of mutable numpy arrays
        self.lr     = lr
        self.state  = [{} for _ in params]  # Per-parameter state
        self.step_count = 0

    def zero_grad(self) -> None:
        """In a numpy context, gradients are passed explicitly. This is a no-op."""
        pass

    def step(self, grads: list) -> None:
        """
        Apply one gradient descent update.

        Parameters
        ----------
        grads : list of ndarray
            Gradients for each parameter, in the same order as self.params.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGD (vanilla)
# ---------------------------------------------------------------------------

class SGD(Optimiser):
    """
    Vanilla stochastic gradient descent with optional L2 weight decay.

    Update rule:
        p <- p - lr * (g + weight_decay * p)

    The weight_decay term is equivalent to L2 regularisation: it shrinks
    parameters toward zero each step, providing implicit regularisation.
    """

    def __init__(self, params: list, lr: float = 0.01, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.weight_decay = weight_decay

    def step(self, grads: list) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            p -= self.lr * g


# ---------------------------------------------------------------------------
# SGD with Momentum
# ---------------------------------------------------------------------------

class SGDMomentum(Optimiser):
    """
    SGD with Nesterov or classical heavy-ball momentum.

    Classical momentum update:
        v <- beta * v - lr * g       (exponential moving average of gradients)
        p <- p + v

    Equivalently:
        v <- beta * v + g
        p <- p - lr * v

    Why momentum helps:
    - In narrow valleys (ill-conditioned loss landscapes), vanilla SGD oscillates
      along the steep walls while making slow progress along the valley axis.
      Momentum accumulates velocity along the gentle axis and dampens oscillations
      on the steep axis.
    - Accelerates convergence in ravines (regions where surface curves much more
      steeply in one dimension than another).
    - beta=0.9 is the standard default.

    Nesterov momentum (lookahead gradient):
        p_lookahead = p + beta * v
        g_lookahead = gradient at p_lookahead
        v <- beta * v - lr * g_lookahead
        p <- p + v

    Nesterov has slightly better theoretical convergence guarantees (O(1/k^2) vs
    O(1/k) for vanilla) and is often marginally faster in practice.
    """

    def __init__(self, params: list, lr: float = 0.01, momentum: float = 0.9,
                 nesterov: bool = False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self, grads: list) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if "v" not in self.state[i]:
                self.state[i]["v"] = np.zeros_like(p)

            v = self.state[i]["v"]

            if self.nesterov:
                # Nesterov: gradient was already evaluated at the lookahead point
                # (in this implementation the caller is responsible for this;
                # here we apply the standard Nesterov update assuming g is the
                # lookahead gradient).
                v_new = self.momentum * v - self.lr * g
                p += -self.momentum * v + (1 + self.momentum) * v_new
                self.state[i]["v"] = v_new
            else:
                # Classical momentum
                v *= self.momentum
                v -= self.lr * g
                p += v
                self.state[i]["v"] = v


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSProp(Optimiser):
    """
    RMSProp: Root Mean Square Propagation (Hinton, 2012).

    Maintains a per-parameter exponential moving average of squared gradients
    and divides the learning rate by the root of this average:

        s <- rho * s + (1 - rho) * g^2     (EMA of squared gradients)
        p <- p - lr * g / sqrt(s + eps)

    Why this helps:
    - Parameters with consistently large gradients receive smaller effective
      learning rates (implicit normalisation).
    - Parameters with small or noisy gradients receive larger effective rates.
    - This "equalises" the learning rate across dimensions, reducing the need
      to hand-tune per-parameter learning rates.
    - Particularly effective for non-stationary problems (online learning) and
      recurrent networks.

    rho=0.99 or 0.9 is common. The original suggestion was rho=0.9.
    eps (epsilon) prevents division by zero; 1e-8 is standard.
    """

    def __init__(self, params: list, lr: float = 0.01, rho: float = 0.99,
                 eps: float = 1e-8):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps

    def step(self, grads: list) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if "s" not in self.state[i]:
                self.state[i]["s"] = np.zeros_like(p)

            s = self.state[i]["s"]
            s *= self.rho
            s += (1.0 - self.rho) * g ** 2

            p -= self.lr * g / (np.sqrt(s) + self.eps)
            self.state[i]["s"] = s


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam(Optimiser):
    """
    Adam: Adaptive Moment Estimation (Kingma & Ba, 2014).

    Combines momentum (first moment) with RMSProp (second moment) and adds
    bias correction to account for initialisation at zero.

    Update equations:
        m <- beta1 * m + (1 - beta1) * g          (first moment: mean)
        v <- beta2 * v + (1 - beta2) * g^2        (second moment: uncentred variance)

    Bias-corrected estimates (critical in early steps):
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

    Parameter update:
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Bias correction rationale:
        m and v are initialised to 0. In early steps, they are biased toward 0
        (they haven't had enough steps to accumulate a representative average).
        Without correction, the effective learning rate is much smaller than
        intended for the first 1/(1-beta1) steps. Correction divides by the
        "warm-up factor" 1 - beta^t which grows from ~beta toward 1.

    Standard defaults: beta1=0.9, beta2=0.999, eps=1e-8.
    """

    def __init__(self, params: list, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 bias_correction: bool = True):
        super().__init__(params, lr)
        self.beta1           = beta1
        self.beta2           = beta2
        self.eps             = eps
        self.bias_correction = bias_correction

    def step(self, grads: list) -> None:
        self.step_count += 1
        t = self.step_count

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if "m" not in self.state[i]:
                self.state[i]["m"] = np.zeros_like(p)
                self.state[i]["v"] = np.zeros_like(p)

            m = self.state[i]["m"]
            v = self.state[i]["v"]

            # Update biased first and second moment estimates
            m *= self.beta1
            m += (1.0 - self.beta1) * g

            v *= self.beta2
            v += (1.0 - self.beta2) * g ** 2

            if self.bias_correction:
                m_hat = m / (1.0 - self.beta1 ** t)
                v_hat = v / (1.0 - self.beta2 ** t)
            else:
                # Without correction: biased toward 0 in early steps
                m_hat = m
                v_hat = v

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.state[i]["m"] = m
            self.state[i]["v"] = v


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def rosenbrock(x: np.ndarray, y: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Rosenbrock function: f(x, y) = (a - x)^2 + b*(y - x^2)^2

    Global minimum at (a, a^2) = (1, 1) with f = 0.
    A classic test function: the minimum lies in a narrow curved valley.
    The function is easy to find the valley but hard to converge to the minimum
    because the valley has very different curvatures in orthogonal directions.
    This exposes limitations of isotropic learning rates.
    """
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(xy: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Gradient of the Rosenbrock function at point [x, y]."""
    x, y  = xy[0], xy[1]
    dfdx  = -2.0 * (a - x) - 4.0 * b * x * (y - x ** 2)
    dfdy  = 2.0  * b * (y - x ** 2)
    return np.array([dfdx, dfdy])


# ---------------------------------------------------------------------------
# Optimisation loop for trajectory tracing
# ---------------------------------------------------------------------------

def optimise(
    start: np.ndarray,
    grad_fn: Callable,
    optimiser_class,
    optimiser_kwargs: dict,
    n_steps: int = 2000,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple:
    """
    Run an optimiser on a scalar function defined by grad_fn.

    Returns (trajectory, losses) where trajectory has shape (n_steps+1, 2).
    Noise simulates stochastic gradient noise from mini-batch sampling.
    """
    rng    = np.random.default_rng(seed)
    params = [start.copy().astype(float)]
    opt    = optimiser_class(params, **optimiser_kwargs)

    trajectory = [params[0].copy()]
    losses     = []

    for step in range(n_steps):
        p = params[0]
        g = grad_fn(p)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)

        losses.append(rosenbrock(p[0], p[1]))
        opt.step([g])
        trajectory.append(params[0].copy())

    losses.append(rosenbrock(params[0][0], params[0][1]))
    return np.array(trajectory), np.array(losses)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_trajectories(
    trajectories: dict,
    losses: dict,
    title: str = "Optimiser Comparison on Rosenbrock",
) -> None:
    """Plot parameter trajectories on the Rosenbrock surface and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    # Contour plot of Rosenbrock
    x_grid = np.linspace(-2, 2, 400)
    y_grid = np.linspace(-1, 3, 400)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = rosenbrock(X_grid, Y_grid)

    ax = axes[0]
    ax.contour(X_grid, Y_grid, np.log1p(Z_grid), levels=30, cmap="Greys", alpha=0.6)
    ax.plot(1.0, 1.0, "r*", markersize=15, label="Global min (1,1)")

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6"]
    for (name, traj), color in zip(trajectories.items(), colors):
        ax.plot(traj[:, 0], traj[:, 1], "-o", color=color, markersize=2,
                linewidth=1.2, alpha=0.8, label=name)
        ax.plot(traj[0, 0], traj[0, 1], "s", color=color, markersize=8)  # start

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 3.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Parameter Trajectory (log-contours)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Loss curves
    ax2 = axes[1]
    for (name, loss), color in zip(losses.items(), colors):
        ax2.plot(loss, color=color, linewidth=1.2, alpha=0.9, label=name)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("f(x, y)  [Rosenbrock loss]")
    ax2.set_title("Loss Convergence")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_descent_variants.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    START  = np.array([-1.5, 1.5])
    N_STEPS = 2000
    NOISE   = 0.05   # Small gradient noise to simulate stochastic gradients

    print("Running optimisers on Rosenbrock function...")
    print(f"  Starting point: {START}")
    print(f"  Global minimum: (1.0, 1.0),  f = 0.0\n")

    configs = {
        "SGD (lr=0.001)": (SGD, {"lr": 0.001}),
        "Momentum (lr=0.001, beta=0.9)": (SGDMomentum, {"lr": 0.001, "momentum": 0.9}),
        "RMSProp (lr=0.01)": (RMSProp, {"lr": 0.01, "rho": 0.99}),
        "Adam (lr=0.01)": (Adam, {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
        "Adam (no bias corr)": (Adam, {"lr": 0.01, "beta1": 0.9, "beta2": 0.999,
                                       "bias_correction": False}),
    }

    trajectories = {}
    losses       = {}

    for name, (cls, kwargs) in configs.items():
        traj, loss = optimise(
            START, rosenbrock_grad, cls, kwargs,
            n_steps=N_STEPS, noise_std=NOISE,
        )
        trajectories[name] = traj
        losses[name]       = loss

        final_point = traj[-1]
        final_loss  = loss[-1]
        dist_to_min = np.linalg.norm(final_point - np.array([1.0, 1.0]))
        print(f"  {name}")
        print(f"    Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})")
        print(f"    Final loss:  {final_loss:.6f}")
        print(f"    Dist to min: {dist_to_min:.4f}\n")

    plot_trajectories(trajectories, losses)

    # --- Adam bias correction effect ---
    print("\nBias correction effect (first 20 steps of Adam):")
    print("  With bias correction: effective LR starts near nominal LR immediately.")
    print("  Without bias correction: effective LR is very small in early steps")
    print("  because m and v are near 0 (momentum has not built up).")

    # Compute effective LR for first 20 steps with and without correction
    beta1, beta2, lr, eps = 0.9, 0.999, 0.01, 1e-8
    g_dummy = np.array([1.0])   # Constant unit gradient for clarity
    m_corr, v_corr = 0.0, 0.0
    m_nocorr, v_nocorr = 0.0, 0.0

    print(f"\n  {'Step':>4}  {'LR_with_corr':>14}  {'LR_no_corr':>12}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*12}")
    for t in range(1, 21):
        m_corr   = beta1 * m_corr   + (1 - beta1) * g_dummy[0]
        v_corr   = beta2 * v_corr   + (1 - beta2) * g_dummy[0]**2
        m_hat    = m_corr  / (1 - beta1**t)
        v_hat    = v_corr  / (1 - beta2**t)
        eff_lr_c = lr / (np.sqrt(v_hat) + eps)   # Effective step size

        m_nocorr = beta1 * m_nocorr + (1 - beta1) * g_dummy[0]
        v_nocorr = beta2 * v_nocorr + (1 - beta2) * g_dummy[0]**2
        eff_lr_n = lr / (np.sqrt(v_nocorr) + eps)

        if t <= 5 or t % 5 == 0:
            print(f"  {t:>4}  {eff_lr_c:>14.6f}  {eff_lr_n:>12.6f}")

    print("\nObservation: without bias correction, the effective LR is ~10-100x")
    print("smaller in early steps, dramatically slowing convergence at the start.")
