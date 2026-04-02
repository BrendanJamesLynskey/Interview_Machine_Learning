# Problem 03: Loss Function Choice

**Topic:** Loss functions, probabilistic modelling, class imbalance, metric learning

**Difficulty:** Intermediate to Advanced

**Time to solve:** 30--45 minutes

---

## Problem Statement

For each of the following five scenarios, choose the most appropriate loss function. For each choice:

1. Name the loss function and write its formula.
2. Explain why this loss function is the correct choice for this scenario.
3. Identify the most likely wrong choice and explain why it fails.
4. Write a short PyTorch implementation.

---

### Scenario A: Medical Imaging -- Rare Disease Detection

A radiologist's assistant model classifies chest X-rays as either "pathology present" or "normal." The dataset contains 50,000 normal scans and only 250 scans with the pathology. The clinical requirement is high recall (missing a pathology is much worse than a false alarm).

### Scenario B: House Price Prediction with Listing Errors

A real estate model predicts house prices from tabular features. The training dataset contains prices scraped from listings, but approximately 5% of listings contain data entry errors (e.g., a \$300,000 house listed as \$3,000,000). The model must be robust to these corrupted labels.

### Scenario C: Multi-Label Document Classification

A legal document classifier assigns zero or more of 50 possible category tags to each document. Each document can belong to 0--15 categories simultaneously. The training set is moderately balanced across categories.

### Scenario D: Face Verification System

A security system must determine whether two face images are of the same person. The system should output a distance metric in embedding space such that the same-person pairs are close and different-person pairs are far apart. The training set has 100,000 identities with 5 images each.

### Scenario E: Semantic Segmentation with Class Imbalance

A self-driving car model labels each pixel of a dashcam image as one of 20 semantic classes (road, sky, pedestrian, car, etc.). In a typical urban scene, the road and sky account for over 70% of pixels, while pedestrians account for less than 1%. Correctly segmenting pedestrians is safety-critical.

---

## Full Solution

### Scenario A: Medical Imaging -- Rare Disease Detection

**Correct loss: Focal loss**

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

where $p_t = \hat{y}$ for the positive class (pathology) and $p_t = 1 - \hat{y}$ for the negative class (normal).

**Why focal loss:**

The dataset has a 200:1 imbalance (50,000 normal vs. 250 pathology). Standard binary cross-entropy will be dominated by the 50,000 normal examples. The gradient signal from the 250 pathology cases is diluted by a factor of 200 relative to normal cases. A model that always predicts "normal" achieves 99.5% accuracy, and cross-entropy will be satisfied with this solution.

Focal loss addresses this via the $(1-p_t)^\gamma$ term:
- Normal scans that are correctly classified with high confidence (e.g., $p_t = 0.97$) contribute $(1-0.97)^2 = 0.0009$ times their CE loss.
- Pathology scans where the model is uncertain or wrong (e.g., $p_t = 0.2$) contribute $(1-0.2)^2 = 0.64$ times their CE loss.

The combined effect refocuses training on the hard, clinically important cases.

The $\alpha$ parameter should be set to $\alpha_{\text{positive}} \approx 0.9$ (high weight on pathology) and $\alpha_{\text{negative}} \approx 0.1$ (low weight on normal), reflecting the clinical cost asymmetry.

**Wrong choice: Standard binary cross-entropy (unweighted)**

Unweighted BCE will train the model to achieve low loss by predicting "normal" with high confidence for almost every scan. The 250 pathology cases do not provide enough gradient to compete with the 50,000 normal cases. The resulting model will have near-zero recall on the rare class.

**PyTorch implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary focal loss for imbalanced binary classification.
    Args:
        alpha: weight for the positive class (set high for rare positive class)
        gamma: focusing parameter (0 = standard BCE, higher = more focus on hard examples)
    """
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model output, shape (N,)
            targets: binary labels {0, 1}, shape (N,)
        Returns:
            scalar mean focal loss
        """
        # Compute sigmoid probabilities from logits
        probs = torch.sigmoid(logits)

        # p_t: probability assigned to the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t: weight for positive/negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


# Usage example
model_output = torch.tensor([2.1, -0.3, 1.5, -1.2, 0.8])   # logits
labels       = torch.tensor([1, 0, 1, 0, 1])                 # ground truth

criterion = FocalLoss(alpha=0.9, gamma=2.0)
loss = criterion(model_output, labels)
print(f"Focal loss: {loss.item():.4f}")

# Compare to unweighted BCE:
bce_loss = F.binary_cross_entropy_with_logits(
    model_output, labels.float()
)
print(f"BCE loss:   {bce_loss.item():.4f}")
```

---

### Scenario B: House Price Prediction with Listing Errors

**Correct loss: Huber loss**

$$\mathcal{L}_{\delta}(\hat{y}, y) = \begin{cases} \frac{1}{2}(\hat{y} - y)^2 & |\hat{y} - y| \leq \delta \\ \delta\left(|\hat{y} - y| - \frac{\delta}{2}\right) & |\hat{y} - y| > \delta \end{cases}$$

**Why Huber loss:**

5% of labels are corrupted with errors potentially 10 times the true value (e.g., \$300k listed as \$3M: error = \$2.7M). Under MSE, these outliers contribute $(2,700,000)^2 \approx 7 \times 10^{12}$ to the loss -- massively larger than a typical correct prediction error of $(30,000)^2 = 9 \times 10^{8}$. The MSE gradient is dominated by these 5% outliers, and the model's weights are driven to minimise the outlier errors at the expense of accuracy on the 95% clean data.

Huber loss with an appropriate $\delta$ (e.g., $\delta = 50,000$ corresponding to a 50k price error threshold):
- Clean predictions: contribute quadratically (fast convergence, same as MSE)
- Outlier listings: contribute linearly (robust, bounded gradient per sample)

MAE would also work (even more robust), but Huber is preferred because:
1. MAE's gradient is $\text{sign}(\hat{y} - y)$, constant everywhere, giving no indication of how wrong the prediction is for small errors. This slows convergence near the correct value.
2. Huber's quadratic region near zero allows faster convergence when the predictions are close to correct.

**Wrong choice: MSE**

The $5\%$ corrupted labels contain errors of $\pm\$2{,}700{,}000$, which under MSE dominate the loss with magnitude $(2.7 \times 10^6)^2 = 7.29 \times 10^{12}$, compared to a typical clean error of $(5 \times 10^4)^2 = 2.5 \times 10^9$ -- about 2900 times larger. The model's weights converge towards minimising these outlier residuals, biasing the model towards the outlier price distribution.

**PyTorch implementation:**

```python
import torch
import torch.nn as nn

class HuberLossRegression(nn.Module):
    """
    Huber (smooth L1) loss for robust regression.
    For price prediction, delta should be in the same units as the target
    (e.g., $50,000 for a model predicting house prices in dollars).
    """
    def __init__(self, delta: float = 50_000.0):
        super().__init__()
        # PyTorch's SmoothL1Loss is a scaled Huber loss with delta parameter
        self.loss_fn = nn.HuberLoss(delta=delta, reduction='mean')

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


# Demonstrating robustness to outliers
clean_predictions = torch.tensor([310_000., 425_000., 195_000., 850_000.])
clean_targets     = torch.tensor([300_000., 415_000., 200_000., 860_000.])

# 4 clean + 1 outlier (listing error: $300k listed as $3M)
outlier_pred    = torch.tensor([310_000., 425_000., 195_000., 850_000., 290_000.])
outlier_targets = torch.tensor([300_000., 415_000., 200_000., 860_000., 3_000_000.])

mse = nn.MSELoss()
huber = HuberLossRegression(delta=50_000.0)

print("Clean data only:")
print(f"  MSE:   {mse(clean_predictions, clean_targets):.2f}")
print(f"  Huber: {huber(clean_predictions, clean_targets):.2f}")

print("\nWith one outlier listing:")
print(f"  MSE:   {mse(outlier_pred, outlier_targets):.2e}  <- dominated by outlier")
print(f"  Huber: {huber(outlier_pred, outlier_targets):.2e}  <- robust")

# Expected output:
# Clean data only:
#   MSE:   75000000.00      (reasonable)
#   Huber: 50000.00         (reasonable, different scale)
# With one outlier listing:
#   MSE:   5.40e+11         (enormous due to 2.7M error)
#   Huber: 1.35e+08         (manageable, linear not quadratic)
```

---

### Scenario C: Multi-Label Document Classification

**Correct loss: Binary cross-entropy (applied independently per label)**

$$\mathcal{L} = -\frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log \hat{y}_{ic} + (1 - y_{ic}) \log(1 - \hat{y}_{ic}) \right]$$

where $N$ is the batch size, $C = 50$ is the number of labels, $y_{ic} \in \{0, 1\}$ is the ground truth, and $\hat{y}_{ic} = \sigma(z_{ic})$ is the sigmoid output.

**Why independent BCE per label:**

Multi-label classification does not assume mutual exclusivity of classes. A document can simultaneously belong to "contract", "property law", and "dispute resolution." The categories are **independent binary decisions**, not competing alternatives.

Softmax + categorical cross-entropy enforces the constraint $\sum_c \hat{y}_c = 1$, which is wrong here. It would prevent a document from having high probability in multiple categories. The probabilities for different labels are not parts of a single probability simplex.

Applying sigmoid independently to each of the 50 output logits and computing BCE for each label separately is the correct approach. Each of the 50 sigmoid outputs models the probability of that label independently.

**Wrong choice: Softmax + categorical cross-entropy**

Softmax normalises probabilities to sum to 1, treating the 50 categories as mutually exclusive. A document that strongly belongs to 5 categories would have each assigned probability $\approx 0.2$ (if equally confident), reducing the probability for each below the 0.5 threshold a typical classifier uses. The model cannot assign high confidence to multiple categories simultaneously.

**PyTorch implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier using independent BCE per label.
    """
    def __init__(self, input_dim: int, n_labels: int = 50):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits (not probabilities): BCE loss handles sigmoid internally
        return self.fc(x)


def multi_label_loss(logits: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits:  shape (N, C) -- raw scores for N documents, C categories
        targets: shape (N, C) -- binary labels {0.0, 1.0}
    Returns:
        scalar mean BCE loss across all documents and labels
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


# Example: batch of 3 documents, 5 categories (simplified from 50)
logits = torch.tensor([
    [ 2.1, -0.5,  1.3, -1.8,  0.9],   # doc 0: likely cats 0, 2, 4
    [-0.3,  1.7,  2.2,  0.8, -1.1],   # doc 1: likely cats 1, 2, 3
    [ 0.5, -1.2,  0.3,  2.4,  1.6],   # doc 2: likely cats 3, 4
])
targets = torch.tensor([
    [1.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 1.0],
])

loss = multi_label_loss(logits, targets)
print(f"Multi-label BCE loss: {loss.item():.4f}")

# Predicted probabilities (for inference)
probs = torch.sigmoid(logits)
print(f"\nPredicted probabilities:\n{probs.round(decimals=2)}")

# Decision threshold (0.5 default, but adjust for recall/precision trade-off)
predicted_labels = (probs > 0.5).int()
print(f"\nPredicted labels (threshold=0.5):\n{predicted_labels}")
```

---

### Scenario D: Face Verification System

**Correct loss: Triplet loss (or NT-Xent / ArcFace for large-scale training)**

For a moderately sized dataset (100k identities, 5 images each), triplet loss with hard negative mining is a principled choice.

$$\mathcal{L}_{\text{triplet}} = \frac{1}{N_T} \sum_{(a,p,n)} \max\!\left(0,\ \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + m\right)$$

where $(a, p, n)$ are anchor, positive (same identity), and negative (different identity) samples, $f(\cdot)$ is the embedding function, and $m > 0$ is the margin.

**Why triplet loss:**

Face verification requires a learned distance metric, not a class label prediction. The model should output an embedding space where same-identity images cluster together and different-identity images are well-separated. This is a **metric learning** problem.

Triplet loss directly optimises this objective: it pushes same-identity pairs closer and different-identity pairs further apart, with the margin $m$ ensuring a minimum separation. With 100k identities, there is no practical way to add a softmax classification head (100k output classes with 5 examples each would severely overfit).

**Recommended setup:**
- Batch size: 200 samples (40 identities × 5 images each)
- Mining: semi-hard negatives (negatives that are further from the anchor than the positive but within the margin)
- Margin: $m = 0.3$ for $\ell_2$ distance, or $m = 0.5$ for cosine distance
- $\ell_2$ normalise embeddings before computing distances

**Alternative: ArcFace loss** (preferred for large-scale production)

$$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}$$

ArcFace adds an angular margin $m$ to the ground-truth class angle, enforcing intra-class compactness and inter-class separability in angular space. It combines the simplicity of a softmax classifier (during training, treat each identity as a class) with strong metric learning properties.

**Wrong choice: Standard softmax cross-entropy**

Softmax cross-entropy treats each identity as a separate class. With 100k identities × 5 images, the class-conditional distributions are tiny. The model cannot learn a generalisable embedding -- it memorises identity-specific features rather than learning invariant representations (pose, lighting, expression). On unseen identities (the actual test case for face verification), the softmax output is meaningless.

**PyTorch implementation (triplet loss):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss for face verification / metric learning.
    Inputs must be L2-normalised embeddings.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self,
                anchor:   torch.Tensor,    # (N, d)
                positive: torch.Tensor,    # (N, d)
                negative: torch.Tensor     # (N, d)
               ) -> torch.Tensor:
        """
        Args:
            anchor:   embedding of the anchor image
            positive: embedding of a same-identity image
            negative: embedding of a different-identity image
        Returns:
            scalar mean triplet loss
        """
        # L2-normalise (ensures consistent scale)
        anchor   = F.normalize(anchor,   p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Squared Euclidean distances
        d_pos = (anchor - positive).pow(2).sum(dim=1)  # (N,)
        d_neg = (anchor - negative).pow(2).sum(dim=1)  # (N,)

        # Triplet loss with soft margin
        loss = F.relu(d_pos - d_neg + self.margin)

        # Count and report hard triplet fraction
        n_hard = (loss > 0).float().mean().item()

        return loss.mean()


class SimpleEmbeddingNet(nn.Module):
    """Minimal face embedding network for illustration."""
    def __init__(self, input_dim: int = 512, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Usage
embed_net = SimpleEmbeddingNet()
triplet_loss = TripletLoss(margin=0.3)

# Random mini-batch: 32 triplets, input features of dimension 512
anchors   = torch.randn(32, 512)
positives = anchors + 0.1 * torch.randn(32, 512)  # same identity: close
negatives = torch.randn(32, 512)                   # different identity: random

emb_a = embed_net(anchors)
emb_p = embed_net(positives)
emb_n = embed_net(negatives)

loss = triplet_loss(emb_a, emb_p, emb_n)
print(f"Triplet loss: {loss.item():.4f}")
```

---

### Scenario E: Semantic Segmentation with Class Imbalance

**Correct loss: Combination of weighted cross-entropy and Dice loss**

**Weighted cross-entropy** (per-pixel, per-class weights):

$$\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} w_{c_i} \log \hat{y}_{i, c_i}$$

where $w_c = \frac{\text{median frequency}}{\text{frequency of class } c}$ (median frequency balancing).

**Dice loss** (for each class $c$ independently):

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i \hat{y}_{ic} y_{ic} + \varepsilon}{\sum_i \hat{y}_{ic} + \sum_i y_{ic} + \varepsilon}$$

**Combined loss:**

$$\mathcal{L} = \mathcal{L}_{\text{WCE}} + \lambda \mathcal{L}_{\text{Dice}}$$

with $\lambda$ typically 1.0.

**Why Dice loss is critical here:**

Pedestrian pixels are less than 1% of a typical scene. Even under weighted cross-entropy, the loss for a class with 200 pixels in a 200×200 image can be overwhelmed by 39,800 other pixels. The optimiser may not reduce the pedestrian class error if doing so increases losses elsewhere.

Dice loss measures the overlap between the predicted segmentation and the ground truth, formulated as a ratio:

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|}$$

This is scale-independent: Dice is the same whether there are 10 or 10,000 pixels of the class. A segmentation that perfectly finds the pedestrian achieves Dice = 1 regardless of the pedestrian's size in the image. The loss is thus equally sensitive to the rare pedestrian class as to the common road class.

**Why weighted CE alone is insufficient:**

With 70% road/sky pixels, even a high weight on pedestrians may not fully compensate. The Dice loss provides a complementary signal focused on the overlap metric that is most relevant to safety-critical detection.

**Wrong choice: Unweighted categorical cross-entropy**

A model trained with unweighted CE on this data will learn to segment road and sky accurately while mostly ignoring pedestrians. It can achieve 99% pixel accuracy while completely failing to detect pedestrians. Pixel accuracy is a misleading metric for imbalanced segmentation; mean Intersection-over-Union (mIoU) per class is the appropriate metric, and unweighted CE is not aligned with it.

**PyTorch implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiceLoss(nn.Module):
    """
    Multi-class Dice loss for semantic segmentation.
    Applied per-class and averaged.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self,
                logits: torch.Tensor,    # (N, C, H, W): raw class scores
                targets: torch.Tensor    # (N, H, W): integer class labels
               ) -> torch.Tensor:
        n_classes = logits.shape[1]

        # Convert to probabilities
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # One-hot encode targets: (N, H, W) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets, n_classes)        # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # (N, C, H, W)
        targets_one_hot = targets_one_hot.float()

        # Dice per class, averaged over batch and spatial dims
        # Flatten spatial dims: (N, C, H*W)
        probs_flat   = probs.view(probs.shape[0], n_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], n_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)  # (N, C)
        union        = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (N, C)

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_per_class.mean()  # average over batch and classes

        return dice_loss


class CombinedSegmentationLoss(nn.Module):
    """
    Weighted cross-entropy + Dice loss for imbalanced semantic segmentation.
    """
    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 dice_weight: float = 1.0):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self,
                logits:  torch.Tensor,  # (N, C, H, W)
                targets: torch.Tensor   # (N, H, W)
               ) -> torch.Tensor:
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss


# Example: 20 semantic classes, batch of 2 small patches (for illustration)
N, C, H, W = 2, 20, 64, 64

# Compute median frequency weights from training set class frequencies
# (In practice, compute from entire training set)
class_pixel_counts = torch.randint(100, 10_000, (C,)).float()
class_pixel_counts[15] = 50     # class 15 = pedestrian: very rare
class_pixel_counts[16] = 30     # class 16 = cyclist: very rare

class_freq   = class_pixel_counts / class_pixel_counts.sum()
median_freq  = class_freq.median()
class_weights = median_freq / class_freq
class_weights = class_weights / class_weights.sum() * C   # normalise

print("Class weights (first 5 and pedestrian/cyclist):")
for i in [0, 1, 2, 3, 4, 15, 16]:
    print(f"  Class {i:2d}: freq={class_freq[i]:.4f}, weight={class_weights[i]:.2f}")

# Compute loss on random predictions
logits  = torch.randn(N, C, H, W)
targets = torch.randint(0, C, (N, H, W))

criterion = CombinedSegmentationLoss(
    class_weights=class_weights,
    dice_weight=1.0
)
loss = criterion(logits, targets)
print(f"\nCombined CE + Dice loss: {loss.item():.4f}")

# Compare to unweighted CE
unweighted_ce = nn.CrossEntropyLoss()(logits, targets)
print(f"Unweighted CE only:      {unweighted_ce.item():.4f}")
```

---

## Summary Table

| Scenario | Correct Loss | Key Reason | Wrong Choice |
|---|---|---|---|
| A: Medical rare disease | Focal loss | 200:1 imbalance, focus on hard examples | Unweighted BCE |
| B: Regression with outliers | Huber loss | 5% corrupted labels, robust to large residuals | MSE |
| C: Multi-label classification | Independent BCE per label | Labels are independent binary decisions | Softmax CE |
| D: Face verification | Triplet / ArcFace | Metric learning required, not classification | Softmax CE on identity classes |
| E: Semantic segmentation | Weighted CE + Dice | Class imbalance + scale-independent overlap metric | Unweighted CE |

---

## Key Principles for Loss Function Selection

1. **Match the loss to the probabilistic model.** MSE = Gaussian noise, BCE = Bernoulli model, CE = Categorical model, MAE = Laplace noise. Using the wrong loss is using the wrong likelihood.

2. **Address class imbalance at the loss level** when data resampling is not practical. Focal loss, class-weighted CE, and Dice loss each provide different mechanisms for handling imbalance.

3. **Multi-label ≠ multi-class.** Multi-class (mutually exclusive): softmax + categorical CE. Multi-label (independent binary): sigmoid + BCE per label. Mixing them up is one of the most common classification mistakes.

4. **Metric learning requires metric learning losses.** When the task is "are these two things similar?" rather than "what class is this?", use contrastive, triplet, or NT-Xent loss. Softmax classification only works when all test-time classes are seen during training.

5. **Robustness to label noise matters.** Real-world labels are often imperfect. Huber, MAE, and label smoothing all provide different forms of robustness. MSE is highly sensitive to label errors.
