# Problem 01: Optimiser Comparison

**Difficulty**: Intermediate to Advanced
**Topics covered**: SGD with momentum, Adam, AdamW, weight decay, gradient scaling, generalisation gap, mixed-precision training
**Time to solve**: 35--50 minutes

---

## Background

Choosing the right optimiser for a given training scenario is one of the most impactful decisions in deep learning engineering. While Adam variants have become dominant for large-scale models, SGD with momentum can outperform them on specific tasks. This problem works through a concrete training scenario, compares optimiser behaviour analytically and numerically, and provides a framework for making principled choices.

---

## Scenario

You are given a training scenario with the following parameters:

- **Architecture**: ResNet-50 (25M parameters)
- **Task**: Image classification on a 10-class custom dataset (50,000 train, 10,000 validation images)
- **Batch size**: 128
- **Training budget**: 100 epochs
- **Hardware**: Single A100 GPU

You will compare three optimiser configurations:

1. **Config A**: SGD with momentum, $\eta = 0.1$, $\beta = 0.9$, weight decay $\lambda = 10^{-4}$, step-decay schedule (multiply by 0.1 at epochs 30, 60, 80)
2. **Config B**: Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$, cosine decay from $10^{-3}$ to $10^{-5}$
3. **Config C**: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$, $\lambda = 0.05$, cosine decay from $10^{-3}$ to $10^{-5}$, 5-epoch linear warmup

---

## Part A: Analytical Comparison of Update Rules

**Problem**: For a single parameter $\theta$ with a consistent gradient $g = 0.01$ across steps, compare the effective update magnitude after 10 steps for each optimiser. Assume all moments are initialised at zero.

### Config A: SGD with Momentum

The velocity update and parameter update:

$$v_t = \beta v_{t-1} + g, \qquad \theta_{t+1} = \theta_t - \eta v_t$$

With $\beta = 0.9$ and $g = 0.01$ (constant), expanding the recurrence:

$$v_t = g \sum_{k=0}^{t-1} \beta^k = g \cdot \frac{1 - \beta^t}{1 - \beta}$$

After 10 steps:

$$v_{10} = 0.01 \times \frac{1 - 0.9^{10}}{1 - 0.9} = 0.01 \times \frac{1 - 0.3487}{0.1} = 0.01 \times 6.513 = 0.06513$$

Total displacement $\Delta\theta = -\eta \sum_{t=1}^{10} v_t$. Since the sum telescopes:

$$\sum_{t=1}^{10} v_t = g \sum_{t=1}^{10} \frac{1-\beta^t}{1-\beta} = \frac{g}{1-\beta}\left(10 - \frac{\beta(1-\beta^{10})}{1-\beta}\right) \approx \frac{g}{1-\beta}\left(10 - \frac{0.9 \times 0.6513}{0.1}\right)$$

$$= 0.1 \left(10 - 5.862\right) = 0.1 \times 4.138 \approx 0.414$$

$$\Delta\theta_{SGD} = -0.1 \times 0.414 = -0.0414$$

Steady-state velocity (if we had run far longer): $v^* = g/(1-\beta) = 0.01/0.1 = 0.1$, giving a per-step displacement of $-0.1 \times 0.1 = -0.01$. After 10 steps, we are still building up towards the steady-state.

### Config B: Adam (no weight decay, L2 in gradient)

First moment (with bias correction):

$$m_t = 0.9 m_{t-1} + 0.1 g, \qquad \hat{m}_t = \frac{m_t}{1 - 0.9^t}$$

Second moment (with bias correction):

$$v_t = 0.999 v_{t-1} + 0.001 g^2, \qquad \hat{v}_t = \frac{v_t}{1 - 0.999^t}$$

At step $t = 10$, $g = 0.01$ constant:

$$m_{10} = 0.1 \times g \times \frac{1 - 0.9^{10}}{1 - 0.9} = 0.1 \times 0.01 \times 6.513 = 0.006513$$

$$\hat{m}_{10} = \frac{0.006513}{1 - 0.9^{10}} = \frac{0.006513}{0.6513} = 0.01 = g$$

For the second moment, since $\beta_2 = 0.999$ is very close to 1:

$$v_{10} = 0.001 \times g^2 \times \frac{1 - 0.999^{10}}{1 - 0.999} \approx 0.001 \times 10^{-4} \times \frac{0.00995}{0.001} = 10^{-4} \times 0.00995 \approx 9.95 \times 10^{-7}$$

$$\hat{v}_{10} = \frac{9.95 \times 10^{-7}}{1 - 0.999^{10}} = \frac{9.95 \times 10^{-7}}{0.00995} \approx 10^{-4} = g^2$$

Effective step:

$$\Delta\theta_{Adam,10} = -\frac{\eta \hat{m}_{10}}{\sqrt{\hat{v}_{10}} + \epsilon} = -\frac{10^{-3} \times 0.01}{\sqrt{10^{-4}} + 10^{-8}} = -\frac{10^{-5}}{0.01 + 10^{-8}} \approx -\frac{10^{-5}}{0.01} = -10^{-3}$$

Per-step displacement converges to $-\eta \times \text{sign}(g) = -10^{-3}$ (since the bias-corrected moments converge to $g$ and $g^2$). Over 10 steps, total displacement is approximately $-10^{-2}$ (exact sum depends on the step-by-step corrections, but each step contributes roughly $-10^{-3}$).

### Comparison Summary

| Optimiser | Per-step displacement (step 10) | Total displacement (10 steps) | Notes |
|---|---|---|---|
| SGD + momentum | $-0.1 \times v_{10} \approx -0.00651$ | $\approx -0.0414$ | Building up velocity; reaches $-0.01$ per step at steady state |
| Adam | $\approx -10^{-3}$ | $\approx -10^{-2}$ | Approximately sign gradient from step 1 |
| AdamW | Same as Adam (weight decay acts separately) | $\approx -10^{-2}$ | Weight decay adds $-\lambda \eta \theta$ per step |

**Key observation**: Adam makes larger progress initially (per step, for this moderate gradient scale) because it normalises the gradient, giving consistent $-\eta$ displacement per step from the start. SGD builds up velocity over $1/(1-\beta) = 10$ steps and then surpasses Adam's per-step displacement at steady state.

---

## Part B: The Weight Decay Bug in Adam

**Problem**: Show numerically that Config B (Adam with L2 regularisation in the gradient) does not apply the intended weight decay uniformly across parameters.

Consider two parameters:
- $\theta_A$: a weight matrix entry with consistently large gradients, $g_A = 1.0$ at every step
- $\theta_B$: an embedding table entry with sparse gradients, $g_B = 0.001$ for 10% of steps and $0.0$ for 90% of steps

Both parameters have current value $\theta = 1.0$ and the L2 regularisation term adds $\lambda = 0.01$ to their gradient when $g \neq 0$.

### Config B: Adam + L2 in gradient

The effective update for each parameter uses the combined gradient $g_{total} = g_{task} + \lambda \theta$. The Adam step size in the direction of the regularisation term is:

$$\text{step}_{reg} = \eta \cdot \frac{\lambda \theta}{\sqrt{\hat{v}_t} + \epsilon}$$

For $\theta_A$ (large, frequent gradients): $\hat{v}_t \approx g_A^2 = 1.0$ (converged), so:

$$\text{step}_{A,reg} \approx \frac{10^{-3} \times 0.01 \times 1.0}{\sqrt{1.0} + 10^{-8}} \approx 10^{-5}$$

Effective weight decay per step: $10^{-5}$ (much smaller than $\lambda \eta = 10^{-5}$, but this is coincidental -- the denominator $\sqrt{\hat{v}_t} \approx 1$ here).

For $\theta_B$ (small, sparse gradients): $\hat{v}_t \approx \mathbb{E}[g_B^2] = 0.1 \times 0.001^2 = 10^{-7}$ (EMA of sparse squares), so:

$$\text{step}_{B,reg} \approx \frac{10^{-3} \times 0.01 \times 1.0}{\sqrt{10^{-7}} + 10^{-8}} \approx \frac{10^{-5}}{3.16 \times 10^{-4}} \approx 0.0316$$

**The ratio is $0.0316 / 10^{-5} \approx 3160$.** The sparse-gradient parameter $\theta_B$ receives over 3000 times more regularisation than the dense-gradient parameter $\theta_A$ per weight-decay application. This is the opposite of what is usually intended: in practice, large weight matrix entries are the ones we most want to regularise.

### Config C: AdamW (decoupled weight decay)

AdamW applies weight decay directly, independent of the gradient scaling:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Both $\theta_A$ and $\theta_B$ receive the same weight decay step: $-\eta \lambda \theta = -10^{-3} \times 0.01 \times 1.0 = -10^{-5}$, regardless of their gradient history. The intended regularisation is applied uniformly.

---

## Part C: PyTorch Training Loop Implementation

The following implementation runs all three configurations and compares their training curves.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math


def get_resnet50(num_classes: int = 10) -> nn.Module:
    """Load ResNet-50 with a custom classifier head."""
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimiser(name: str, model: nn.Module):
    """Construct the optimiser and scheduler for a given config."""
    params_decay = [
        p for n, p in model.named_parameters()
        if p.ndim >= 2 and 'bn' not in n and 'norm' not in n
    ]
    params_no_decay = [
        p for n, p in model.named_parameters()
        if p.ndim < 2 or 'bn' in n or 'norm' in n
    ]

    if name == 'sgd':
        # Config A: SGD + momentum, weight decay via L2 penalty
        opt = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80], gamma=0.1
        )

    elif name == 'adam':
        # Config B: Adam + L2 regularisation in the gradient
        opt = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,   # L2 regularisation -- NOT decoupled
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100, eta_min=1e-5
        )

    elif name == 'adamw':
        # Config C: AdamW with decoupled weight decay, warmup + cosine
        opt = torch.optim.AdamW(
            [
                {'params': params_decay, 'weight_decay': 0.05},
                {'params': params_no_decay, 'weight_decay': 0.0},
            ],
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        # Linear warmup for 5 epochs, then cosine decay for 95 epochs
        warmup_epochs = 5
        total_epochs = 100
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    else:
        raise ValueError(f"Unknown optimiser: {name}")

    return opt, sched


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping -- good practice for all three configs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (average loss, top-1 accuracy) on the loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def run_experiment(config_name: str, num_epochs: int = 100):
    """Run a full training experiment for a given config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ImageNet normalisation stats (standard for pretrained-weight fine-tuning;
    # appropriate here as we mimic the ImageNet pipeline even for custom data)
    normalise = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        normalise,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalise,
    ])

    # Replace with your dataset; CIFAR-10 is used here as a proxy
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_tf
    )
    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_tf
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = get_resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser, scheduler = build_optimiser(config_name, model)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimiser, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            lr = optimiser.param_groups[0]['lr']
            print(
                f"[{config_name}] Epoch {epoch:3d}/{num_epochs} | "
                f"LR: {lr:.2e} | Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}"
            )

    return history


# Example invocation (comment out to run selectively)
# history_sgd   = run_experiment('sgd')
# history_adam  = run_experiment('adam')
# history_adamw = run_experiment('adamw')
```

---

## Part D: Expected Outcomes and Analysis

Based on the training scenario and empirical observations from the literature, here are the expected results for a custom 10-class image classification task with ResNet-50:

### Convergence Speed

```
Epoch  10: Adam   ~72% val  |  AdamW ~70% val  |  SGD ~60% val
Epoch  30: Adam   ~80% val  |  AdamW ~81% val  |  SGD ~80% val  (SGD decays LR here)
Epoch  60: Adam   ~83% val  |  AdamW ~84% val  |  SGD ~85% val
Epoch 100: Adam   ~83% val  |  AdamW ~85% val  |  SGD ~87% val
```

(Numbers are illustrative; actual values depend on dataset properties.)

**Adam converges fastest in early epochs.** The adaptive learning rate provides large, well-calibrated steps from epoch 1. **SGD is slower to start** because it must build up momentum and the initial learning rate of 0.1, while large, is uniform across all parameters and may be too large for some layers and too small for others early on.

**Late convergence advantage of SGD.** After the learning rate decays at epoch 60, SGD typically finds flatter minima and pulls ahead in validation accuracy. This is the generalisation gap discussed in Wilson et al. (2017).

**AdamW vs Adam.** AdamW's consistent weight decay (not scaled by gradient history) leads to better regularisation and typically ~1--2% higher final validation accuracy than Adam at equal learning rate and $\lambda$ values.

### Training Loss Trajectories

```
Training loss after 10 epochs (typical):
  Config A (SGD):    2.1
  Config B (Adam):   1.7    <- faster initial optimisation
  Config C (AdamW):  1.8

Training loss after 100 epochs (typical):
  Config A (SGD):    0.6
  Config B (Adam):   0.9    <- can plateau without continued LR reduction
  Config C (AdamW):  0.7
```

Adam tends to plateau earlier because its adaptive learning rate effectively decreases as the second moment grows, and without aggressive LR scheduling, it does not continue to descend.

### Gradient Norm Monitoring

```python
# Useful diagnostic: log gradient norm at each step
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        # Log per-layer gradient norm to detect exploding/vanishing gradients
```

Expected behaviour:
- **SGD with momentum**: gradient norms are relatively large and stable, reflecting the accumulated velocity.
- **Adam**: gradient norms in the update (after normalisation) are roughly $\eta / (\sqrt{\hat{v}_t} + \epsilon) \approx \eta$ per parameter, making the "normalised gradient norm" approximately the number of parameters times $\eta$.
- **AdamW**: same as Adam for the gradient component, but an additional constant offset from weight decay.

---

## Part E: Making the Optimiser Choice

### Decision Framework

```
Task type?
  ├─ Large transformer pre-training (BERT, GPT, ViT)
  │   └─ AdamW with warmup + cosine decay
  ├─ Fine-tuning pretrained transformer
  │   └─ AdamW with small LR (1e-4 to 3e-5), warmup, cosine
  ├─ Image classification from scratch (ResNet, EfficientNet)
  │   ├─ Abundant compute + time for hyperparameter tuning?
  │   │   └─ SGD + momentum with step decay or cosine (better final accuracy)
  │   └─ Limited time or rapid prototyping?
  │       └─ AdamW (converges reliably without LR sensitivity)
  └─ Sparse objectives (embeddings, GNNs, NLP fine-tuning)
      └─ AdamW (adaptive LR essential for sparse gradients)
```

### Summary of Trade-offs

| Criterion | SGD + momentum | Adam | AdamW |
|---|---|---|---|
| Final accuracy (image classification) | Best (with tuning) | Good | Good |
| Final accuracy (transformers) | Poor | Good | Best |
| Convergence speed | Slow initially | Fast | Fast |
| Hyperparameter sensitivity | High (LR, schedule critical) | Low | Moderate |
| Memory per parameter | 1 (velocity only) | 3 (param + 2 moments) | 3 |
| Correct L2 regularisation | Yes | No | Yes |
| Recommended default | Vision CNNs | Rapid prototyping | Most new projects |

---

## Key Takeaways

1. **Adam and SGD converge to different points.** Adam's adaptive scaling can converge to sharper minima than SGD's uniform scaling. For image classification, this typically means SGD achieves lower test error given equal training time and careful scheduling.

2. **L2 in Adam's gradient is not weight decay.** The weight decay strength in standard Adam varies by parameter history. AdamW fixes this with constant, decoupled decay. Never use `Adam(weight_decay=...)` for serious training; use `AdamW` with the `weight_decay` parameter.

3. **Warmup is critical for Adam/AdamW.** Without it, the second moment estimate is unreliable in the first $O(1/(1-\beta_2))$ steps, causing large, poorly calibrated updates. SGD does not have this problem.

4. **Gradient clipping is beneficial for all optimisers.** It prevents rare large-gradient batches from causing catastrophic parameter updates, particularly important at high learning rates (SGD) and during the warmup phase (Adam).

5. **The optimiser interacts with the schedule.** An SGD learning rate of 0.1 that is appropriate for epoch 0 must be decayed aggressively. An AdamW learning rate of $10^{-3}$ is much less sensitive to the exact decay schedule because the adaptive scaling already normalises per-parameter step sizes.

---

## Common Interview Questions on This Topic

**Q: Why does Adam sometimes achieve lower training loss but higher test loss than SGD?**

Adam's normalised updates find narrow, sharp minima efficiently. A sharp minimum has high curvature, meaning small perturbations of the parameters cause large changes in loss. During training, the model sits at the bottom of this narrow valley. At test time, the inputs are slightly different from training inputs, which effectively perturbs the model away from the exact minimum -- and a sharp minimum is more sensitive to such perturbations. SGD's uniform scaling naturally avoids sharp directions (it makes small steps in high-curvature directions), tending towards flatter minima that are more robust to perturbations.

**Q: If you switch from Adam to AdamW mid-training, what will you observe?**

Initially, a spike in the training loss as the weight decay suddenly starts pulling parameters towards zero. Parameters with large values (which were under-regularised by Adam) will experience the strongest decay. The model will usually recover and the training loss will decrease again, but there is typically a multi-step disruption. The correct approach is to use AdamW from the start. If you must switch, do so at the start of a new phase (e.g., after the warmup) rather than mid-training.

**Q: You see that AdamW is using a very large $\epsilon$ (e.g., $10^{-6}$ instead of $10^{-8}$). When would this be appropriate?**

A larger $\epsilon$ prevents the denominator $\sqrt{\hat{v}_t} + \epsilon$ from being very small for parameters with sparse gradients (where $\hat{v}_t \approx 0$). Without a large $\epsilon$, sparse-gradient parameters can receive enormous effective learning rates (since $\eta / \epsilon$ can be very large). This is relevant for embedding tables in NLP tasks. Some practitioners use $\epsilon = 10^{-6}$ or even $10^{-5}$ for language model training with large vocabulary embedding tables. The tradeoff is that larger $\epsilon$ reduces the per-parameter adaptivity for parameters with medium gradient magnitudes.
