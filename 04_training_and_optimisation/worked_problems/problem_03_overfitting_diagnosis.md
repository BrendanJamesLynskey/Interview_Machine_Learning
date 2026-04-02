# Problem 03: Overfitting Diagnosis and Remediation

**Difficulty**: Intermediate to Advanced
**Topics covered**: Overfitting, training curves, regularisation, early stopping, dropout, weight decay, data augmentation, learning rate, batch size effects
**Time to solve**: 40--55 minutes

---

## Background

Diagnosing overfitting from training curves is a core skill that interviewers test directly. The ability to read a training curve, identify the root cause of overfitting, and prescribe the right remediation strategy -- in order of expected impact -- distinguishes candidates who have debugged real training runs from those who have only studied theory. This problem presents three progressively more complex overfitting scenarios and works through the full diagnostic and remediation process for each.

---

## Scenario Overview

You are given three training runs for a ResNet-50 trained on a custom 200-class image dataset (20,000 training images, 5,000 validation images). Each run produces training and validation loss curves and accuracy metrics logged every epoch for 100 epochs.

---

## Scenario 1: Classic Overfitting

### Training Curve Data

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 5 | 2.41 | 2.38 | 42% | 41% |
| 10 | 1.82 | 1.89 | 58% | 56% |
| 20 | 1.10 | 1.34 | 73% | 67% |
| 30 | 0.72 | 1.61 | 83% | 65% |
| 40 | 0.43 | 2.14 | 90% | 63% |
| 50 | 0.24 | 2.88 | 95% | 60% |
| 60 | 0.13 | 3.47 | 98% | 58% |

**Optimiser configuration**: SGD + momentum, $\eta = 0.01$ (fixed, no decay), $\beta = 0.9$, no weight decay, no dropout, no data augmentation.

### Diagnosis

**Signal 1: Diverging loss curves.** Train loss continues decreasing while val loss first decreases (epoch 5--20) then increases from epoch 20 onwards. The minimum validation loss is at approximately epoch 20 ($\approx 1.34$); after that, the model is memorising training data.

**Signal 2: Large and growing train-val accuracy gap.** By epoch 60, train acc = 98% vs val acc = 58% -- a 40 percentage point gap. This is a strong indicator of high variance (overfitting).

**Signal 3: Near-zero training loss trajectory.** Train loss of 0.13 at epoch 60 for a 200-class problem (where a random classifier would have loss $\approx \log(200) \approx 5.3$) indicates the model is very close to memorising every training example.

**Root cause**: No regularisation of any form (no weight decay, no dropout, no augmentation) with a model (ResNet-50, 25M parameters) that has far more capacity than the dataset size (20,000 images). The model memorises the training set rather than generalising.

### Remediation Strategy (in order of expected impact)

**Step 1: Data augmentation (highest impact).**

For a 200-class image classification task with only 20,000 images (100 images per class), the most effective intervention is expanding the effective training set via augmentation:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # aggressive crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4,
        saturation=0.4, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])
```

Expected effect: val accuracy improves from 58% to approximately 68--72%. The model is now forced to classify images under diverse transformations rather than exact pixel patterns.

**Step 2: Weight decay.**

Add L2 regularisation to prevent the model from growing extremely large weights:

```python
optimiser = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,   # standard for ResNet on small datasets
)
```

Expected incremental effect: +2--4% val accuracy over augmentation alone. Weight decay constrains the norm of the weight matrices, preventing the model from over-specialising on training patterns.

**Step 3: Learning rate schedule.**

A fixed learning rate of 0.01 throughout training is suboptimal. The model cannot converge precisely into a minimum because the constant LR keeps it wandering. Adding step decay:

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimiser, milestones=[30, 60, 80], gamma=0.1
)
# LR: 0.01 (epochs 1-29), 0.001 (30-59), 0.0001 (60-79), 0.00001 (80+)
```

Expected incremental effect: +1--2% val accuracy, and better convergence (lower val loss plateau).

**Step 4: Early stopping (optional -- reduces wasted compute, not primary fix).**

Stop training when val loss stops improving for $P = 10$ epochs:

```python
class EarlyStopping:
    """Stop training if val loss does not improve for `patience` epochs."""
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
```

Early stopping alone does not fix overfitting; it only limits how much the model can overfit. It is most valuable when compute is limited.

### Expected Results After Full Remediation

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 20 | 1.45 | 1.52 | 68% | 65% |
| 40 | 1.10 | 1.23 | 74% | 70% |
| 60 | 0.85 | 1.05 | 80% | 74% |
| 80 | 0.72 | 0.98 | 83% | 76% |
| 100 | 0.68 | 0.95 | 84% | 77% |

The train-val gap has narrowed from 40 to approximately 7 percentage points. Both curves are decreasing, with the val loss converging to a reasonable plateau.

---

## Scenario 2: Subtle Overfitting with BatchNorm Interaction

### Training Curve Data

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 10 | 1.93 | 1.98 | 55% | 53% |
| 20 | 1.45 | 1.49 | 67% | 65% |
| 30 | 1.12 | 1.15 | 74% | 73% |
| 40 | 0.89 | 0.97 | 79% | 77% |
| 50 | 0.71 | 0.84 | 84% | 79% |
| 60 | 0.56 | 0.79 | 87% | 80% |
| 70 | 0.44 | 0.77 | 90% | 80% |
| 80 | 0.36 | 0.82 | 92% | 80% |
| 90 | 0.28 | 0.91 | 94% | 79% |
| 100 | 0.22 | 1.03 | 96% | 78% |

**Configuration**: SGD + momentum, weight decay $5 \times 10^{-4}$, standard augmentation (random crop + flip), no label smoothing. Training and evaluation both done with `model.train()` mode active (this is the bug).

### Diagnosis

**Signal 1: Small early train-val gap.** The curves track closely for epochs 10--70 (gap of ~3 percentage points). This is a sign that augmentation and weight decay are working. However, starting at epoch 70, the val loss begins to increase while train loss continues decreasing.

**Signal 2: Val accuracy plateau then degradation.** Val accuracy plateaus at 80% (epoch 60--70) and then decreases. This is a classic sign of late-stage overfitting that occurs when the model starts fitting noise in the training data.

**Signal 3: Anomalously high train accuracy vs val accuracy gap.** A gap of 96% vs 78% = 18 percentage points at epoch 100 is unusually large for a model with weight decay and augmentation. Something is causing the training accuracy to be inflated.

**The hidden bug: Evaluating in training mode (`model.train()`).**

When evaluation is performed in `model.train()` mode:
- BatchNorm uses per-batch statistics instead of the running statistics.
- Dropout (if present) remains active.

Each evaluation mini-batch is normalised by its own statistics. A mini-batch of 32 validation images has very different statistics from the 50,000-image training distribution. The per-batch normalisation in eval effectively changes the feature distribution that the learned $\gamma$/$\beta$ parameters were calibrated for. This introduces stochastic noise into the evaluation metrics, making validation accuracy appear lower than the true inference performance.

Additionally, because training accuracy is computed with the same `model.train()` mode, it benefits from the fact that each training batch is normalised by its own (favourable) statistics -- effectively seeing a cleaner version of the data than validation sees.

**Verification**: If you compute a few forward passes manually with both `model.train()` and `model.eval()` on the same validation batch, the outputs will differ. The `model.eval()` outputs will typically be more confident (lower entropy) because they use the well-calibrated running statistics.

### Finding the Bug: Diagnostic Code

```python
import torch

@torch.no_grad()
def compare_eval_modes(model, val_loader, criterion, device, n_batches=5):
    """
    Compare loss and accuracy under model.train() vs model.eval() modes.
    Used to detect BatchNorm mode bugs.
    """
    results = {}
    for mode_name, set_mode in [('train_mode', model.train),
                                  ('eval_mode', model.eval)]:
        set_mode()
        total_loss, correct, total = 0.0, 0, 0
        for i, (images, labels) in enumerate(val_loader):
            if i >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        results[mode_name] = {
            'loss': total_loss / min(n_batches, len(val_loader)),
            'accuracy': correct / total
        }

    print(f"train mode: loss={results['train_mode']['loss']:.4f}, "
          f"acc={results['train_mode']['accuracy']:.3f}")
    print(f"eval  mode: loss={results['eval_mode']['loss']:.4f}, "
          f"acc={results['eval_mode']['accuracy']:.3f}")

    acc_diff = (results['eval_mode']['accuracy']
                - results['train_mode']['accuracy'])
    print(f"\nDifference (eval - train mode): {acc_diff:+.3f}")
    if abs(acc_diff) > 0.02:
        print("WARNING: Large mode difference detected. "
              "Check that evaluation uses model.eval().")
```

### Correct Evaluation Pattern

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Correct evaluation: always use model.eval()."""
    model.eval()    # <-- CRITICAL: must set before evaluation
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    model.train()   # restore train mode after evaluation
    return total_loss / len(loader), correct / total
```

### Expected Results After Fix

With the evaluation mode bug fixed, the val loss and accuracy curves will change to reflect the true model performance. The actual validation accuracy is likely higher than the reported 78%:

- **True val accuracy at epoch 70 (best checkpoint)**: approximately 82--84%, because the running statistics provide better normalisation for individual inference samples than the noisy per-batch statistics.
- **True train-val gap**: approximately 8--10 percentage points (consistent with the level of regularisation applied).

The remaining mild overfitting (train acc growing while val plateaus) is normal and expected. A minor label smoothing addition ($\epsilon = 0.1$) can further reduce it.

---

## Scenario 3: Learning Rate-Induced Overfitting

### Training Curve Data

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Notes |
|-------|-----------|----------|-----------|---------|-------|
| 1 | 4.95 | 4.98 | 5% | 4% | Initialisation |
| 5 | 3.21 | 3.18 | 21% | 22% | Normal early training |
| 10 | 2.14 | 2.19 | 48% | 46% | |
| 15 | 1.87 | 1.95 | 57% | 54% | |
| 20 | 1.79 | 1.98 | 58% | 53% | **Plateau begins** |
| 30 | 1.76 | 2.14 | 59% | 52% | **Divergence begins** |
| 40 | 1.81 | 2.41 | 57% | 49% | Train loss also rising |
| 45 | 1.63 | 2.18 | 61% | 51% | Brief recovery after LR decay |
| 50 | 1.58 | 2.05 | 62% | 53% | |
| 60 | 1.52 | 2.28 | 64% | 51% | |

**Configuration**: AdamW, $\eta = 5 \times 10^{-3}$ (fixed, no decay until epoch 45 where it drops to $5 \times 10^{-4}$), $\lambda = 0.1$, standard augmentation.

### Diagnosis

**Signal 1: Very early plateau.** Both train and val loss plateau around epoch 15--20. With a 200-class problem and only 58% accuracy, the model has not remotely converged -- it should continue improving for many more epochs. An early plateau is a classic sign that the learning rate is too large to converge precisely into a minimum.

**Signal 2: Train loss rising (epoch 30--40).** When training loss rises, the optimiser is overshooting: the learning rate is so large that each step moves past the local minimum and up the other side. The system oscillates around a minimum without converging.

**Signal 3: Brief recovery after LR decay.** At epoch 45, a manual LR decay drops the rate from $5 \times 10^{-3}$ to $5 \times 10^{-4}$. Both train and val loss briefly improve. This confirms that the learning rate was the root cause of the plateau and divergence.

**Signal 4: Val loss higher than train loss throughout.** Unlike Scenario 1 (where train loss plummeted while val stayed high), here both curves are at similar absolute values and both are too high. This is characteristic of an optimisation failure (learning rate too large) rather than a generalisation failure (overfitting).

**Root cause**: Adam's adaptive learning rate does not fully compensate for an LR that is an order of magnitude too large. $\eta = 5 \times 10^{-3}$ for AdamW is approximately 5--10 times too large for ResNet-50 on this task (typical AdamW LR is $10^{-4}$ to $10^{-3}$). The gradient noise at this LR magnitude pushes the parameters into a region of the loss landscape where the curvature is high, causing the loss to oscillate rather than decrease.

### Distinguishing LR Overfitting from Capacity Overfitting

| Symptom | LR too large | Capacity overfitting |
|---------|--------------|----------------------|
| Train loss trajectory | Plateaus early, may rise | Continuously decreases |
| Val loss trajectory | Plateaus or rises from start | Decreases then rises |
| Train-val gap | Small (both stuck at similar loss) | Large (train much lower) |
| Train accuracy | Low-to-moderate (cannot converge) | Very high |
| Response to lower LR | Immediate improvement | Improvement but still overfits |

### Remediation: Learning Rate Selection

**Step 1: LR Range Test.**

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import copy


@torch.no_grad()
def lr_range_test(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_steps: int = 200,
) -> tuple[list[float], list[float]]:
    """
    Perform the LR range test (Smith, 2018).
    Increases LR exponentially while recording the loss.
    The optimal max LR is just before the loss starts to increase.
    """
    model = copy.deepcopy(model)   # do not modify the original model
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=start_lr)
    # Exponential LR increase: lr_n = start_lr * (end_lr/start_lr)^(n/num_steps)
    gamma = (end_lr / start_lr) ** (1.0 / num_steps)
    sched = ExponentialLR(opt, gamma=gamma)

    lrs, losses = [], []
    data_iter = iter(train_loader)
    smoothed_loss = None

    for step in range(num_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        sched.step()

        loss_val = loss.item()
        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss_val

        lrs.append(opt.param_groups[0]['lr'])
        losses.append(smoothed_loss)

        # Stop if loss explodes
        if smoothed_loss > 4 * losses[0]:
            break

    return lrs, losses


# Usage:
# lrs, losses = lr_range_test(model, train_loader, criterion, device)
# Inspect the plot: pick max_lr just before the minimum of the loss curve
# Typically: max_lr = lrs[argmin(losses)] / 3  (conservative margin)
```

For the scenario, the LR range test would reveal that the loss starts rising for LR $> 10^{-3}$, confirming that $5 \times 10^{-3}$ is too large.

**Step 2: Corrected configuration.**

```python
# Correct configuration for this scenario
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,          # identified via LR range test
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=100,        # 100 epochs
    eta_min=3e-5,     # 10x reduction at end
)
```

### Expected Results After Fix

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 10 | 1.81 | 1.88 | 57% | 55% |
| 30 | 1.22 | 1.35 | 71% | 67% |
| 60 | 0.88 | 1.09 | 80% | 74% |
| 90 | 0.73 | 0.98 | 83% | 76% |
| 100 | 0.70 | 0.95 | 84% | 77% |

The model now converges steadily without plateauing early or showing oscillation. Both losses decrease throughout training. The final val accuracy of 77% compares to the "stuck" 51--53% with the over-large LR.

---

## Part: Systematic Overfitting Diagnostic Checklist

When presented with a training curve in an interview or real debugging session, work through the following checklist in order:

```
1. DESCRIBE THE CURVES
   - Is train loss decreasing? Val loss?
   - When does the divergence start?
   - What is the absolute value of the losses? (Random baseline for K classes: log(K))
   - Is there oscillation (non-monotonic train loss)?

2. IDENTIFY THE PATTERN
   Pattern A: Train loss low, val loss high and diverging
     → Capacity overfitting. Remedy: augmentation, weight decay, dropout
   Pattern B: Both losses plateau early at high values
     → LR too large or model underfit. Remedy: reduce LR / increase model
   Pattern C: Both losses decrease but val increases sharply after a point
     → Late-stage overfitting. Remedy: early stopping, lighter augmentation
   Pattern D: Val loss lower than train loss throughout
     → NOT overfitting. May be: strong augmentation on train, eval mode bug,
       test-time TTA, different distributions. Check model.eval() usage.
   Pattern E: Oscillating train loss, erratic val loss
     → LR too large or gradient explosion. Remedy: reduce LR, add grad clipping

3. CHECK FOR BUGS FIRST (before adjusting hyperparameters)
   a. Is model.eval() called before all evaluation passes?
   b. Is the validation dataloader using the same transforms as test? (No augmentation)
   c. Is the loss divided by accumulation steps if using gradient accumulation?
   d. Are BatchNorm running stats being updated during frozen fine-tuning?
   e. Is dropout active during evaluation?

4. APPLY REMEDIATIONS IN ORDER OF EXPECTED IMPACT
   For capacity overfitting:
     1. Data augmentation (largest effect for small datasets)
     2. Weight decay / L2 regularisation
     3. Reduce model capacity (if appropriate)
     4. Dropout (for dense layers)
     5. Early stopping
     6. Label smoothing
   For LR overfitting:
     1. Reduce peak LR (use LR range test)
     2. Add LR schedule (cosine or step decay)
     3. Add warmup (if using Adam)
     4. Add gradient clipping

5. VALIDATE THE FIX
   - Re-run with the change and confirm the symptom resolves
   - Avoid making multiple changes simultaneously; isolate the cause
```

---

## Quantitative Overfitting Metrics

Beyond visual inspection of curves, these quantitative measures are useful:

**Generalisation gap** (accuracy):

$$\text{Gap}_{acc} = \text{Train Acc} - \text{Val Acc}$$

Rule of thumb: $< 3\%$ is well-regularised; $3\%$--$10\%$ is moderate; $> 10\%$ warrants intervention.

**Loss ratio**:

$$\text{Ratio}_{loss} = \frac{\text{Val Loss}}{\text{Train Loss}}$$

At convergence, a ratio of $1.0$--$1.3$ is typical. Ratios $> 2.0$ indicate significant overfitting.

**Overfitting coefficient** (at epoch $t$ relative to best val epoch $t^*$):

$$\text{OC}(t) = \frac{\text{Val Loss}(t) - \text{Val Loss}(t^*)}{\text{Train Loss}(t^*) - \text{Train Loss}(t)}$$

An $\text{OC}(t) > 1$ means val loss is increasing faster than train loss is decreasing after the best epoch -- a clear sign the model is memorising noise.

---

## Key Takeaways

1. **Always check for implementation bugs before adjusting hyperparameters.** The most common "overfitting" diagnoses in practice are actually `model.eval()` bugs, augmentation applied to validation data, or gradient accumulation scaling errors.

2. **Distinguish capacity overfitting from LR overfitting.** They look different: capacity overfitting has a very low train loss and a high val loss; LR overfitting has both losses stuck at relatively high values with oscillation.

3. **Data augmentation is the highest-impact regularisation for small datasets.** For vision tasks with $< 10{,}000$ training images per class, augmentation typically improves val accuracy more than any other regularisation technique.

4. **Weight decay and augmentation are complementary, not redundant.** Augmentation increases the effective dataset size and teaches invariances; weight decay constrains the parameter magnitude. Together they address different aspects of overfitting.

5. **The batch size affects the apparent train-val gap.** Smaller batches add stochastic noise to gradient estimates, which acts as implicit regularisation. Increasing batch size without adjusting the learning rate can cause the train loss to fall much faster than val loss -- this looks like overfitting but is actually reduced gradient noise.

---

## Common Interview Questions on This Topic

**Q: You are given a training curve where the validation loss is consistently lower than the training loss. How do you interpret this?**

This is unusual and has several possible explanations:

1. **Strong augmentation on training set only.** The model sees augmented (harder) versions of images during training, making the training loss appear higher. At evaluation (with clean images), the model performs better. This is normal and expected when augmentation is applied correctly (to training only).

2. **`model.eval()` is called for evaluation but not training accuracy computation.** BatchNorm in eval mode uses well-calibrated running statistics; in train mode it uses noisy per-batch statistics. If training accuracy is computed in eval mode but validation is in eval mode, both are correct. If training accuracy is computed in train mode, it may be inflated by BatchNorm noise.

3. **Training loss includes regularisation; validation loss does not.** If the reported training loss is $\mathcal{L}_{task} + \lambda \|\theta\|^2$ (total regularised loss) and the val loss is $\mathcal{L}_{task}$ only, then training loss will always be higher by $\lambda \|\theta\|^2$.

4. **Dropout is applied at train time.** Dropout at train time applies a random mask that can make the training loss noisier and apparently higher than the clean val loss without Dropout.

5. **Dataset distribution shift.** If the validation set is genuinely easier (e.g., cleaner images, less background noise) than the training set, val loss will be lower. This is a dataset collection issue, not an overfitting issue.

To diagnose, compute train and val loss on the same 100-sample batch with `model.eval()` and no augmentation. If val loss is still lower, suspect a distribution shift.

**Q: Your team increased the batch size from 64 to 512 to improve GPU utilisation. The validation accuracy dropped from 80% to 76%. Why, and how do you fix it?**

Increasing the batch size reduces the stochastic gradient noise, which has two effects:

1. The model converges faster to the loss minimum it finds (fewer stochastic steps needed), but it converges to a sharper, less general minimum. Sharp minima are characteristic of large-batch training (Keskar et al., 2017).

2. The effective learning rate was not adjusted. With batch size 8x larger, the linear scaling rule requires the learning rate to increase by 8x to maintain the same training dynamics. Without this, the model is effectively trained with a much smaller per-token learning rate, converging to a suboptimal point.

**Fix**:
1. Apply the linear scaling rule: increase $\eta$ from (original LR) to (original LR) $\times$ (512/64) = 8x the original LR. Add a warmup phase proportional to the new batch size.
2. If accuracy is still lower, add a linear warmup at the new LR to stabilise early training.
3. If accuracy is still lower even with the correct LR, consider that large-batch training may fundamentally find sharper minima for this task. Techniques that specifically counter this: sharp-aware minimisation (SAM), Ghost Batch Normalisation (compute BN stats over virtual smaller batches), or simply accepting that the accuracy will be slightly lower in exchange for faster training.
