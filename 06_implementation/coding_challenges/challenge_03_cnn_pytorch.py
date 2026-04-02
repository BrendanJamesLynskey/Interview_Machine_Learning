"""
Challenge 03: CNN Classifier in PyTorch
========================================

Task:
    1. Build a small CNN from scratch using PyTorch nn.Module.
    2. Train it on CIFAR-10 (downloaded automatically).
    3. Implement data augmentation with torchvision transforms.
    4. Track training/validation accuracy and plot curves.
    5. Inspect what the first conv layer's filters have learned.

Learning objectives:
    - PyTorch nn.Module structure: __init__ and forward.
    - Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d.
    - DataLoader, transforms, and the train/eval mode distinction.
    - GPU/CPU device management with .to(device).
    - Weight initialisation best practices (Kaiming for ReLU networks).

Architecture:
    Input (3x32x32)
    -> ConvBlock(3->32) + ConvBlock(32->32) + MaxPool
    -> ConvBlock(32->64) + ConvBlock(64->64) + MaxPool
    -> ConvBlock(64->128) + ConvBlock(128->128)
    -> GlobalAvgPool
    -> FC(128->256) + Dropout -> FC(256->10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU. bias=False because BN has its own bias."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10: 10-class, 32x32 colour images.

    Uses Global Average Pooling instead of a large fully-connected layer to
    reduce parameter count and improve generalisation.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  32),
            nn.MaxPool2d(2, 2),         # 32x32 -> 16x16

            ConvBlock(32,  64),
            ConvBlock(64,  64),
            nn.MaxPool2d(2, 2),         # 16x16 -> 8x8

            ConvBlock(64,  128),
            ConvBlock(128, 128),        # 8x8 (no downsample)
        )

        # Reduces each 8x8 feature map to a single value per channel
        self.gap = nn.AdaptiveAvgPool2d(1)   # Output: (N, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming (He) normal initialisation.

        For ReLU activations, the correct scale is sqrt(2 / fan_in) (mode='fan_out'
        for conv). This prevents vanishing/exploding activations through deep ReLU
        stacks. Xavier initialisation assumes linear activations and underestimates
        the scale needed.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (N, 128, 8, 8)
        x = self.gap(x)               # (N, 128, 1, 1)
        x = x.flatten(start_dim=1)   # (N, 128)
        x = self.classifier(x)        # (N, 10)
        return x


# ---------------------------------------------------------------------------
# Data loading and augmentation
# ---------------------------------------------------------------------------

# Per-channel mean and std computed from the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_dataloaders(batch_size: int = 128, num_workers: int = 2) -> tuple:
    """
    Build CIFAR-10 DataLoaders with augmentation on the training split.

    Training augmentation:
    - RandomCrop(32, padding=4): randomly crops after adding 4 pixels of padding.
      Teaches translation invariance without changing image size.
    - RandomHorizontalFlip: valid for objects that appear flipped (cars, animals).
      NOT valid for tasks where orientation matters (e.g., handwriting, text).

    Validation: only normalisation -- augmentation would add randomness to
    evaluation scores, making comparisons between epochs unstable.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf
    )
    val_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training and evaluation functions
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """One epoch of mini-batch gradient descent. Returns (avg_loss, accuracy)."""
    model.train()                    # Activate Dropout and BN training mode
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate on a fixed split. Returns (avg_loss, accuracy).

    @torch.no_grad() disables gradient computation, halving memory use and
    speeding up evaluation by ~50 %.

    model.eval() is also critical:
    - BatchNorm switches from per-batch statistics to running mean/variance.
    - Dropout disables all dropout masks (full network capacity at eval time).
    Failing to call model.eval() is a common source of inconsistent val metrics.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str = "cnn_training_curves.png") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CNN Training on CIFAR-10", fontsize=13)

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to {save_path}")


def visualise_filters(model: nn.Module, n_filters: int = 32) -> None:
    """
    Display the first convolutional layer's learned RGB filters.

    First-layer filters are the only ones directly interpretable as visual
    patterns. They typically learn colour blobs, oriented edges, and
    colour-opponent patterns -- a learned analogue of Gabor wavelets.
    """
    first_conv = model.features[0].conv
    w = first_conv.weight.detach().cpu()   # (out_ch, 3, 3, 3)

    n_show = min(n_filters, w.shape[0])
    ncols  = 8
    nrows  = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.flatten()

    for i in range(n_show):
        f = w[i].permute(1, 2, 0).numpy()      # (H, W, 3)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        axes[i].imshow(f, interpolation="nearest")
        axes[i].axis("off")
        axes[i].set_title(f"F{i}", fontsize=6)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("First Conv Layer Filters", fontsize=10)
    plt.tight_layout()
    plt.savefig("cnn_filters.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Device selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Hyperparameters ---
    BATCH_SIZE   = 128
    N_EPOCHS     = 30
    LR           = 0.1
    WEIGHT_DECAY = 5e-4
    NUM_WORKERS  = 2

    # --- Data ---
    print("Downloading/loading CIFAR-10...")
    train_loader, val_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # --- Model, loss, optimiser, scheduler ---
    model = SmallCNN(num_classes=10, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Label smoothing (epsilon=0.05): prevents overconfident softmax predictions.
    # Slightly improves calibration and generalisation.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # SGD with Nesterov momentum: strong baseline for CIFAR CNN training.
    # Adam can converge faster initially but SGD often achieves better final accuracy.
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True,
    )

    # Cosine annealing: reduces LR smoothly from LR to near 0 over all epochs.
    # Avoids sharp LR drops of step decay; often finds slightly better minima.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS)

    # --- Training ---
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining for {N_EPOCHS} epochs...")
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_cnn_cifar10.pt")
            marker = " <-- best"
        else:
            marker = ""

        print(
            f"  Epoch {epoch:02d}/{N_EPOCHS}  "
            f"train={tr_acc:.3f}  val={vl_acc:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.5f}{marker}"
        )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # --- Plots ---
    plot_training_curves(history)
    visualise_filters(model)

    print("\nExpected results after 30 epochs (single T4 GPU ~5 min):")
    print("  Val accuracy ~ 82-86 %.")
    print("  Train accuracy may reach ~97 % (gap indicates some overfitting; normal).")
    print("  Adding CutMix/MixUp or AutoAugment would push val accuracy to ~90 %+.")
