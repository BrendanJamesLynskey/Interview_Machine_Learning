"""
Challenge 04: Complete Training Loop with Validation, Early Stopping, and Checkpointing
========================================================================================

Task:
    1. Build a complete training harness for a PyTorch model:
       - Per-epoch train and validation evaluation.
       - Early stopping: halt training when validation loss fails to improve
         for `patience` epochs.
       - Model checkpointing: save the best model weights; load them at the end.
       - Gradient clipping: prevent exploding gradients in deep or recurrent nets.
       - Learning rate scheduling with logging.
    2. Demonstrate on a simple MLP regression task so the code runs quickly.
    3. Show how each component prevents a specific failure mode.

Learning objectives:
    - Why early stopping prevents overfitting without being a regulariser in itself.
    - Why you checkpoint the BEST model (not the last model).
    - When gradient clipping is needed and how to set the clip norm.
    - The difference between ReduceLROnPlateau (reactive) and cosine/step (schedule-based).
"""

import os
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Training configuration (dataclass for clean API)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters in one place for reproducibility and logging."""
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    batch_size:      int   = 64
    max_epochs:      int   = 200
    patience:        int   = 15          # Early stopping patience
    min_delta:       float = 1e-4        # Minimum improvement to count as progress
    grad_clip_norm:  float = 1.0         # Max gradient L2 norm; None to disable
    checkpoint_path: str   = "best_model.pt"
    scheduler:       str   = "plateau"   # 'plateau', 'cosine', or 'step'
    seed:            int   = 42


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a validation metric and signal when to stop training.

    Attributes
    ----------
    patience  : int
        How many epochs without improvement before stopping.
    min_delta : float
        Minimum decrease in monitored metric to qualify as an improvement.
        Prevents stopping on trivially small fluctuations.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score: Optional[float] = None
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update the counter based on the current validation loss.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if val_loss < self.best_score - self.min_delta:
            # Genuine improvement: reset counter
            self.best_score = val_loss
            self.counter = 0
        else:
            # No improvement: increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Model checkpointing
# ---------------------------------------------------------------------------

class Checkpointer:
    """
    Save the model state dict whenever a new best validation loss is achieved.

    Why checkpoint the best model (not the last)?
    Training loss keeps decreasing even while val loss is rising (overfitting).
    The last model epoch may have worse generalisation than epoch 50 even if we
    ran for 200 epochs. The checkpoint ensures we deploy the epoch that had the
    best held-out performance.
    """

    def __init__(self, path: str):
        self.path      = path
        self.best_loss = math.inf

    def update(self, model: nn.Module, val_loss: float) -> bool:
        """Save model if val_loss is a new best. Returns True if saved."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            return True
        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best saved weights into the model (in-place)."""
        if os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path, map_location="cpu"))
        return model


# ---------------------------------------------------------------------------
# Training step and evaluation step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """
    One epoch of mini-batch gradient descent.

    Parameters
    ----------
    grad_clip_norm : float or None
        If not None, clip gradients so their L2 norm does not exceed this value.

        Why clip? In deep networks (especially RNNs), gradients of loss with respect
        to early layers can grow exponentially ("exploding gradients"). Clipping
        prevents parameter updates from being catastrophically large. The threshold
        1.0 is a common default; monitor grad_norm metrics to choose a good value.

    Returns
    -------
    float : average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_n    = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad(set_to_none=True)
        pred = model(X_batch).squeeze(-1)
        loss = criterion(pred, y_batch)
        loss.backward()

        if grad_clip_norm is not None:
            # torch.nn.utils.clip_grad_norm_ rescales all parameter gradients so that
            # their combined L2 norm equals grad_clip_norm if it was larger.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimiser.step()

        total_loss += loss.item() * len(X_batch)
        total_n    += len(X_batch)

    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average loss on a dataset. No gradient computation."""
    model.eval()
    total_loss = 0.0
    total_n    = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch).squeeze(-1)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
        total_n    += len(X_batch)

    return total_loss / total_n


# ---------------------------------------------------------------------------
# Full training harness
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Output of the training run."""
    train_losses:   list = field(default_factory=list)
    val_losses:     list = field(default_factory=list)
    lr_history:     list = field(default_factory=list)
    best_epoch:     int  = 0
    best_val_loss:  float = math.inf
    stopped_early:  bool = False


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """
    Full training loop with:
    - Early stopping
    - Checkpointing
    - Gradient clipping
    - Learning rate scheduling
    - Logging
    """
    model = model.to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # --- Scheduler ---
    if cfg.scheduler == "plateau":
        # ReduceLROnPlateau: reduces LR by `factor` when val loss stops improving.
        # Reactive to the actual training dynamics; no fixed schedule needed.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cfg.max_epochs, eta_min=1e-6
        )
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=50, gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")

    criterion   = nn.MSELoss()
    early_stop  = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    checkpointer = Checkpointer(cfg.checkpoint_path)
    result      = TrainResult()

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, device, cfg.grad_clip_norm
        )
        vl_loss = evaluate(model, val_loader, criterion, device)

        # Record metrics
        current_lr = optimiser.param_groups[0]["lr"]
        result.train_losses.append(tr_loss)
        result.val_losses.append(vl_loss)
        result.lr_history.append(current_lr)

        # Checkpoint if improved
        saved = checkpointer.update(model, vl_loss)
        if saved:
            result.best_epoch    = epoch
            result.best_val_loss = vl_loss

        # Scheduler step
        if cfg.scheduler == "plateau":
            scheduler.step(vl_loss)  # Plateau scheduler needs the metric
        else:
            scheduler.step()         # Schedule-based schedulers advance unconditionally

        # Early stopping check
        if early_stop.step(vl_loss):
            print(f"  Early stopping triggered at epoch {epoch}. "
                  f"Best epoch was {result.best_epoch} (val_loss={result.best_val_loss:.6f})")
            result.stopped_early = True
            break

        if epoch % 10 == 0 or epoch == 1:
            marker = " [saved]" if saved else ""
            print(f"  Epoch {epoch:04d}  "
                  f"train={tr_loss:.5f}  val={vl_loss:.5f}  "
                  f"lr={current_lr:.2e}{marker}")

    # Restore best weights before returning
    checkpointer.load_best(model)
    print(f"\nRestored best model from epoch {result.best_epoch} "
          f"(val_loss={result.best_val_loss:.6f})")

    return result


# ---------------------------------------------------------------------------
# Synthetic regression dataset
# ---------------------------------------------------------------------------

def make_regression_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Synthetic regression task: y = sin(X @ w) + noise.
    Non-linear enough to benefit from an MLP.
    """
    rng = np.random.default_rng(seed)
    X   = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)
    w   = rng.normal(0, 1, n_features).astype(np.float32)
    y   = np.sin(X @ w) + rng.normal(0, noise, n_samples).astype(np.float32)
    return X, y


def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, val_frac: float = 0.2) -> tuple:
    N      = len(X)
    split  = int(N * (1 - val_frac))
    X_t    = torch.from_numpy(X[:split])
    y_t    = torch.from_numpy(y[:split])
    X_v    = torch.from_numpy(X[split:])
    y_v    = torch.from_numpy(y[split:])

    train_ds = TensorDataset(X_t, y_t)
    val_ds   = TensorDataset(X_v, y_v)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Simple MLP model
# ---------------------------------------------------------------------------

def build_mlp(n_in: int, hidden: int = 128, n_layers: int = 4, dropout: float = 0.2) -> nn.Module:
    """
    Build a simple feed-forward MLP for regression.

    Includes Dropout to create an overfitting scenario where early stopping
    and checkpointing will demonstrably help.
    """
    layers = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = TrainConfig(
        lr=1e-3, max_epochs=200, patience=20, batch_size=64,
        grad_clip_norm=1.0, scheduler="plateau", seed=42,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}\n")

    # --- Data ---
    X, y = make_regression_dataset(n_samples=3000, n_features=20, noise=0.2, seed=cfg.seed)
    train_loader, val_loader = build_loaders(X, y, cfg.batch_size)
    print(f"Train: {len(train_loader.dataset)} samples  "
          f"Val: {len(val_loader.dataset)} samples\n")

    # --- Model ---
    torch.manual_seed(cfg.seed)
    model = build_mlp(n_in=X.shape[1], hidden=256, n_layers=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- Train ---
    result = train(model, train_loader, val_loader, cfg, device)

    # --- Report ---
    print(f"\nSummary:")
    print(f"  Total epochs run:  {len(result.train_losses)}")
    print(f"  Early stopped:     {result.stopped_early}")
    print(f"  Best epoch:        {result.best_epoch}")
    print(f"  Best val MSE:      {result.best_val_loss:.6f}")
    print(f"  Final train MSE:   {result.train_losses[-1]:.6f}")
    print(f"  Final val MSE:     {result.val_losses[-1]:.6f}")

    # --- Verify checkpoint works: final model IS the best model ---
    criterion = nn.MSELoss()
    final_val_loss = evaluate(model, val_loader, criterion, device)
    print(f"  Val MSE (loaded):  {final_val_loss:.6f}  "
          f"(should equal best val loss = {result.best_val_loss:.6f})")
    assert abs(final_val_loss - result.best_val_loss) < 1e-5, \
        "Checkpoint restore failed: final val loss does not match best val loss"
    print("  Checkpoint restore: PASSED\n")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Loop: Early Stopping + Checkpointing", fontsize=13)

    epochs = range(1, len(result.train_losses) + 1)
    axes[0].plot(epochs, result.train_losses, label="Train MSE", linewidth=1.5)
    axes[0].plot(epochs, result.val_losses,   label="Val MSE",   linewidth=1.5)
    axes[0].axvline(result.best_epoch, color="green", linestyle="--",
                    label=f"Best epoch ({result.best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, result.lr_history, color="#e67e22", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule (ReduceLROnPlateau)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_loop_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Expected behaviour:")
    print("  Val loss should reach minimum well before epoch 200.")
    print("  Early stopping fires ~20 epochs after the best epoch.")
    print("  LR should drop 1-3 times as val loss plateaus.")
    print("  After checkpoint restore, val MSE matches best_val_loss exactly.")
