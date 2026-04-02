"""
Challenge 01: Linear Regression from Scratch with Gradient Descent
===================================================================

Task:
    1. Implement linear regression using batch gradient descent (no sklearn).
    2. Implement the closed-form Normal Equation solution for comparison.
    3. Implement Ridge (L2) and Lasso (L1) regularisation variants.
    4. Demonstrate on a synthetic dataset with known ground-truth coefficients.
    5. Plot the loss curve and the fitted vs actual values.

Learning objectives:
    - Understand the MSE loss and its gradient derivation.
    - Understand how learning rate and iterations affect convergence.
    - Understand the effect of L1 vs L2 regularisation on weights.
    - Implement feature normalisation (StandardScaler from scratch).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class StandardScaler:
    """Zero-mean, unit-variance feature scaling."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error: (1/2N) * sum((y_pred - y_true)^2).

    The 1/2 factor simplifies the gradient by cancelling the 2 from squaring.
    """
    N = len(y_true)
    return (1.0 / (2.0 * N)) * np.sum((y_pred - y_true) ** 2)


def mse_gradient(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss with respect to weights w (including bias in column 0).

    Derivation:
        L  = (1/2N) * ||Xw - y||^2
        dL/dw = (1/N) * X^T (Xw - y)
             = (1/N) * X^T (y_pred - y_true)

    Returns gradient vector of shape (n_features + 1,).
    """
    N = len(y_true)
    residuals = y_pred - y_true          # Shape: (N,)
    grad = (1.0 / N) * X.T @ residuals  # Shape: (n_features+1,)
    return grad


# ---------------------------------------------------------------------------
# Linear Regression: Gradient Descent
# ---------------------------------------------------------------------------

class LinearRegressionGD:
    """
    Linear regression trained by batch gradient descent.

    Model:  y_hat = X_aug @ w,  where X_aug = [1 | X] (bias absorbed into w[0]).

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of full-batch gradient descent steps.
    regularisation : str or None
        'l2' for Ridge, 'l1' for Lasso, None for plain OLS.
    lam : float
        Regularisation strength (lambda). Applied to all weights except bias.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularisation: Optional[str] = None,
        lam: float = 0.01,
    ):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg = regularisation
        self.lam = lam
        self.w_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones to X for the bias term."""
        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        Train the model on (X, y) using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (N, n_features)
        y : ndarray of shape (N,)
        """
        X_aug = self._augment(X)   # Shape: (N, n_features + 1)
        N, D = X_aug.shape

        # Initialise weights to small random values (zeros also work but slow convergence)
        rng = np.random.default_rng(seed=0)
        self.w_ = rng.normal(0, 0.01, size=D)

        self.loss_history_ = []

        for iteration in range(self.n_iter):
            y_pred = X_aug @ self.w_

            # Compute and record loss
            loss = mse_loss(y_pred, y)
            self.loss_history_.append(loss)

            # Base gradient (MSE)
            grad = mse_gradient(X_aug, y_pred, y)

            # Regularisation gradient (do NOT regularise the bias term w[0])
            if self.reg == "l2":
                # Ridge: gradient += lam * w  (bias term excluded)
                reg_grad = self.lam * self.w_.copy()
                reg_grad[0] = 0.0
                grad += reg_grad

            elif self.reg == "l1":
                # Lasso: gradient += lam * sign(w)  (subgradient for non-differentiable)
                reg_grad = self.lam * np.sign(self.w_)
                reg_grad[0] = 0.0
                grad += reg_grad

            # Gradient descent update
            self.w_ -= self.lr * grad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new input X."""
        return self._augment(X) @ self.w_

    @property
    def coef_(self) -> np.ndarray:
        """Learned feature weights (excluding bias)."""
        return self.w_[1:]

    @property
    def intercept_(self) -> float:
        """Learned bias term."""
        return float(self.w_[0])


# ---------------------------------------------------------------------------
# Linear Regression: Closed-Form Normal Equation
# ---------------------------------------------------------------------------

class LinearRegressionNormalEq:
    """
    Linear regression solved via the Normal Equation (closed-form OLS).

    w* = (X_aug^T X_aug)^{-1} X_aug^T y

    For Ridge (L2) regularisation:
    w* = (X_aug^T X_aug + lam * I)^{-1} X_aug^T y

    Time complexity: O(n_features^3) due to matrix inversion.
    Use only when n_features is small (< 10,000). For large n_features, prefer
    iterative solvers.
    """

    def __init__(self, regularisation: Optional[str] = None, lam: float = 0.0):
        self.reg = regularisation
        self.lam = lam
        self.w_: Optional[np.ndarray] = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionNormalEq":
        X_aug = self._augment(X)
        D = X_aug.shape[1]

        if self.reg == "l2":
            # Tikhonov regularisation: do not regularise the bias term (index 0)
            reg_matrix = self.lam * np.eye(D)
            reg_matrix[0, 0] = 0.0
            self.w_ = np.linalg.solve(X_aug.T @ X_aug + reg_matrix, X_aug.T @ y)
        else:
            # Plain OLS: use lstsq for numerical stability (pseudo-inverse)
            self.w_, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._augment(X) @ self.w_

    @property
    def coef_(self) -> np.ndarray:
        return self.w_[1:]

    @property
    def intercept_(self) -> float:
        return float(self.w_[0])


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot
    SS_res = sum((y_true - y_pred)^2)
    SS_tot = sum((y_true - mean(y_true))^2)

    R^2 = 1.0: perfect fit.
    R^2 = 0.0: model predicts the mean of y.
    R^2 < 0.0: model is worse than predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_regression_data(
    n_samples: int = 500,
    n_features: int = 5,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Generate a synthetic linear regression dataset.

    True model: y = X @ true_coef + intercept + noise
    Returns (X, y, true_coef, intercept).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))

    # Ground-truth coefficients: mix of positive, negative, and near-zero
    true_coef = np.array([3.5, -2.1, 0.0, 1.8, -4.2])[:n_features]
    intercept = 5.0
    noise = rng.normal(0, noise_std, size=n_samples)

    y = X @ true_coef + intercept + noise
    return X, y, true_coef, intercept


# ---------------------------------------------------------------------------
# Main: demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- Generate data ---
    X, y, true_coef, true_intercept = generate_regression_data(n_samples=500, noise_std=1.0)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    print(f"True intercept:    {true_intercept}\n")

    # --- Train/test split ---
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Feature scaling (fit on train only) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- Model 1: Gradient Descent (plain OLS) ---
    gd_model = LinearRegressionGD(learning_rate=0.1, n_iterations=500, regularisation=None)
    gd_model.fit(X_train_s, y_train)
    y_pred_gd = gd_model.predict(X_test_s)

    print("=== Gradient Descent (OLS) ===")
    print(f"  Learned coef : {gd_model.coef_}")
    print(f"  Learned bias : {gd_model.intercept_:.4f}")
    print(f"  Test RMSE    : {rmse(y_test, y_pred_gd):.4f}")
    print(f"  Test R^2     : {r2_score(y_test, y_pred_gd):.4f}\n")

    # --- Model 2: Normal Equation (OLS) ---
    ne_model = LinearRegressionNormalEq(regularisation=None)
    ne_model.fit(X_train_s, y_train)
    y_pred_ne = ne_model.predict(X_test_s)

    print("=== Normal Equation (OLS) ===")
    print(f"  Learned coef : {ne_model.coef_}")
    print(f"  Learned bias : {ne_model.intercept_:.4f}")
    print(f"  Test RMSE    : {rmse(y_test, y_pred_ne):.4f}")
    print(f"  Test R^2     : {r2_score(y_test, y_pred_ne):.4f}\n")

    # --- Model 3: Ridge (L2) ---
    ridge_model = LinearRegressionGD(
        learning_rate=0.1, n_iterations=500, regularisation="l2", lam=1.0
    )
    ridge_model.fit(X_train_s, y_train)
    y_pred_ridge = ridge_model.predict(X_test_s)

    print("=== Gradient Descent (Ridge L2, lam=1.0) ===")
    print(f"  Learned coef : {ridge_model.coef_}")
    print(f"  Test RMSE    : {rmse(y_test, y_pred_ridge):.4f}")
    print(f"  Test R^2     : {r2_score(y_test, y_pred_ridge):.4f}\n")

    # --- Model 4: Lasso (L1) ---
    lasso_model = LinearRegressionGD(
        learning_rate=0.1, n_iterations=500, regularisation="l1", lam=0.5
    )
    lasso_model.fit(X_train_s, y_train)
    y_pred_lasso = lasso_model.predict(X_test_s)

    print("=== Gradient Descent (Lasso L1, lam=0.5) ===")
    print(f"  Learned coef : {lasso_model.coef_}")
    print(f"  (Note: Lasso drives near-zero weights toward 0)")
    print(f"  Test RMSE    : {rmse(y_test, y_pred_lasso):.4f}")
    print(f"  Test R^2     : {r2_score(y_test, y_pred_lasso):.4f}\n")

    # --- Verify GD converges to same solution as Normal Equation ---
    max_coef_diff = np.max(np.abs(gd_model.coef_ - ne_model.coef_))
    print(f"Max coef difference (GD vs Normal Eq): {max_coef_diff:.6f}  (expect near 0)\n")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Linear Regression from Scratch", fontsize=13)

    # Plot 1: GD loss curve
    axes[0].plot(gd_model.loss_history_, color="#3498db", linewidth=1.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("GD Training Loss Curve")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predicted vs Actual
    axes[1].scatter(y_test, y_pred_gd, alpha=0.5, s=20, color="#2ecc71", label="GD (OLS)")
    axes[1].scatter(y_test, y_pred_ne, alpha=0.3, s=20, color="#e74c3c", label="Normal Eq")
    lo, hi = y_test.min(), y_test.max()
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")
    axes[1].set_xlabel("True y")
    axes[1].set_ylabel("Predicted y")
    axes[1].set_title("Predicted vs Actual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Coefficient comparison across models
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    x_pos = np.arange(len(feature_names))
    width = 0.2
    axes[2].bar(x_pos - 1.5*width, true_coef, width, label="True",    color="#95a5a6")
    axes[2].bar(x_pos - 0.5*width, gd_model.coef_, width, label="OLS", color="#3498db")
    axes[2].bar(x_pos + 0.5*width, ridge_model.coef_, width, label="Ridge", color="#2ecc71")
    axes[2].bar(x_pos + 1.5*width, lasso_model.coef_, width, label="Lasso", color="#e74c3c")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(feature_names)
    axes[2].set_ylabel("Coefficient value")
    axes[2].set_title("True vs Learned Coefficients")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("linear_regression_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Expected outputs summary ---
    print("Expected outputs:")
    print("  GD OLS and Normal Equation coefficients should be nearly identical.")
    print("  Ridge coefficients are slightly shrunk toward zero vs OLS.")
    print("  Lasso may drive the true-zero coefficient (x2) exactly to 0 or very near 0.")
    print("  R^2 for all models should be > 0.95 on this clean synthetic dataset.")
