# Problem 02: Learning Rate Warmup Schedule Design

**Difficulty**: Intermediate to Advanced
**Topics covered**: Learning rate warmup, cosine decay, OneCycleLR, schedule derivation, transformer training, schedule visualisation
**Time to solve**: 30--45 minutes

---

## Background

Designing a learning rate schedule is not a matter of plugging in default values. The optimal schedule depends on the model architecture, optimiser, dataset size, batch size, and training duration. This problem works through the design of a complete schedule for a specific training run, derives the learning rate at arbitrary steps, and explores the failure modes of common mistakes.

---

## Scenario

You are training a 345M-parameter language model (GPT-2 large scale) from scratch with the following specifications:

- **Optimiser**: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$
- **Total training steps**: $T = 600{,}000$
- **Batch size**: 512 sequences of 1024 tokens (effective batch = $512 \times 1024 = 524{,}288$ tokens per step)
- **Peak learning rate**: $\eta_{max} = 6 \times 10^{-4}$
- **Minimum learning rate**: $\eta_{min} = 6 \times 10^{-5}$ (i.e., $0.1 \times \eta_{max}$)
- **Warmup duration**: $W = 4{,}000$ steps

---

## Part A: Deriving the Schedule

**Problem**: Write the complete piecewise formula for $\eta_t$ and evaluate it at specific checkpoints.

### Schedule Formula

The standard warmup + cosine decay schedule:

$$\eta_t = \begin{cases}
\eta_{max} \cdot \dfrac{t}{W} & 0 \leq t \leq W \\[10pt]
\eta_{min} + \dfrac{1}{2}\!\left(\eta_{max} - \eta_{min}\right)\!\left(1 + \cos\!\left(\dfrac{\pi(t - W)}{T - W}\right)\right) & W < t \leq T
\end{cases}$$

Substituting the given values ($\eta_{max} = 6 \times 10^{-4}$, $\eta_{min} = 6 \times 10^{-5}$, $W = 4000$, $T = 600000$):

**Warmup phase** ($0 \leq t \leq 4000$):

$$\eta_t = 6 \times 10^{-4} \cdot \frac{t}{4000} = 1.5 \times 10^{-7} \cdot t$$

**Cosine decay phase** ($4000 < t \leq 600000$):

$$\eta_t = 6 \times 10^{-5} + 2.7 \times 10^{-4} \left(1 + \cos\!\left(\frac{\pi(t - 4000)}{596000}\right)\right)$$

---

## Part B: Evaluating at Key Steps

**Problem**: Compute $\eta_t$ exactly at $t \in \{1000, 4000, 100000, 302000, 600000\}$.

### Step $t = 1{,}000$ (warmup phase, 25% through warmup)

$$\eta_{1000} = 6 \times 10^{-4} \cdot \frac{1000}{4000} = 6 \times 10^{-4} \times 0.25 = 1.5 \times 10^{-4}$$

### Step $t = 4{,}000$ (end of warmup)

Using the warmup formula:

$$\eta_{4000} = 6 \times 10^{-4} \cdot \frac{4000}{4000} = 6 \times 10^{-4}$$

Verification with the cosine formula at $t = W$:

$$\eta_{4000} = 6 \times 10^{-5} + 2.7 \times 10^{-4}\left(1 + \cos(0)\right) = 6 \times 10^{-5} + 2.7 \times 10^{-4} \times 2 = 6 \times 10^{-5} + 5.4 \times 10^{-4} = 6 \times 10^{-4} \checkmark$$

Both formulas agree at the boundary. The schedule is continuous (though not smooth at the transition -- its derivative is discontinuous at $t = W$).

### Step $t = 100{,}000$ (early cosine decay)

$$\text{progress} = \frac{100000 - 4000}{600000 - 4000} = \frac{96000}{596000} \approx 0.16107$$

$$\cos(\pi \times 0.16107) = \cos(0.50609) \approx 0.87357$$

$$\eta_{100000} = 6 \times 10^{-5} + 2.7 \times 10^{-4}(1 + 0.87357) = 6 \times 10^{-5} + 2.7 \times 10^{-4} \times 1.87357$$

$$= 6 \times 10^{-5} + 5.059 \times 10^{-4} = 5.659 \times 10^{-4}$$

The learning rate has decayed only slightly from the peak at this early stage -- a consequence of the cosine curve's slow initial decay.

### Step $t = 302{,}000$ (approximate midpoint of cosine phase)

The cosine phase runs from $t = 4000$ to $t = 600000$, so the exact midpoint is at:

$$t_{mid} = 4000 + \frac{596000}{2} = 302000$$

$$\text{progress} = \frac{302000 - 4000}{596000} = \frac{298000}{596000} = 0.5$$

$$\cos(\pi \times 0.5) = \cos\!\left(\frac{\pi}{2}\right) = 0$$

$$\eta_{302000} = 6 \times 10^{-5} + 2.7 \times 10^{-4}(1 + 0) = 6 \times 10^{-5} + 2.7 \times 10^{-4} = 3.3 \times 10^{-4}$$

At the midpoint of the cosine phase, the learning rate is exactly the midpoint between $\eta_{max}$ and $\eta_{min}$:

$$\frac{\eta_{max} + \eta_{min}}{2} = \frac{6 \times 10^{-4} + 6 \times 10^{-5}}{2} = \frac{6.6 \times 10^{-4}}{2} = 3.3 \times 10^{-4} \checkmark$$

### Step $t = 600{,}000$ (end of training)

$$\text{progress} = \frac{600000 - 4000}{596000} = 1.0$$

$$\cos(\pi \times 1.0) = \cos(\pi) = -1$$

$$\eta_{600000} = 6 \times 10^{-5} + 2.7 \times 10^{-4}(1 + (-1)) = 6 \times 10^{-5} + 0 = 6 \times 10^{-5}$$

The schedule terminates exactly at $\eta_{min}$, as required.

### Summary Table

| Step | Phase | Progress (cosine) | $\eta_t$ |
|------|-------|-------------------|----------|
| 1,000 | Warmup (25%) | -- | $1.5 \times 10^{-4}$ |
| 4,000 | Warmup end | 0% | $6.0 \times 10^{-4}$ |
| 100,000 | Cosine decay | 16.1% | $\approx 5.66 \times 10^{-4}$ |
| 302,000 | Cosine midpoint | 50.0% | $3.3 \times 10^{-4}$ |
| 600,000 | End of training | 100% | $6.0 \times 10^{-5}$ |

---

## Part C: PyTorch Implementation

```python
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable


def make_warmup_cosine_scheduler(
    optimiser: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a linear warmup + cosine decay LambdaLR scheduler.

    Args:
        optimiser:      The optimiser whose LR will be scheduled.
        warmup_steps:   Number of steps to linearly ramp up to base LR.
        total_steps:    Total training steps (T).
        eta_min_ratio:  eta_min = eta_max * eta_min_ratio.

    Returns:
        A LambdaLR scheduler. Call scheduler.step() once per optimiser step.

    Notes:
        - LambdaLR multiplies the optimiser's base LR by the lambda function.
        - Set the optimiser's lr to eta_max; the lambda provides a scale in [0, 1].
        - Call scheduler.step() AFTER optimiser.step(), not before.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step <= warmup_steps:
            # Linear warmup: 0 -> 1
            return current_step / max(1, warmup_steps)
        # Cosine decay: 1 -> eta_min_ratio
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        # Clamp progress to [0, 1] in case step exceeds total_steps
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale from eta_min_ratio to 1.0
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def plot_schedule(
    eta_max: float,
    warmup_steps: int,
    total_steps: int,
    eta_min_ratio: float = 0.1,
) -> None:
    """
    Compute and print the LR at a grid of steps to verify the schedule.
    In a real environment, replace print with plt.plot for a visual.
    """
    # Minimal model to construct optimiser for the scheduler
    dummy_param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([dummy_param], lr=eta_max)
    sched = make_warmup_cosine_scheduler(
        opt, warmup_steps, total_steps, eta_min_ratio
    )

    checkpoints = [0, warmup_steps // 4, warmup_steps // 2, warmup_steps,
                   total_steps // 4, total_steps // 2, total_steps]

    print(f"{'Step':>10} | {'LR':>12} | {'Phase'}")
    print("-" * 40)
    step = 0
    for target_step in checkpoints:
        while step < target_step:
            opt.step()       # dummy step to advance
            sched.step()
            step += 1
        lr = opt.param_groups[0]['lr']
        phase = 'warmup' if target_step <= warmup_steps else 'cosine'
        print(f"{target_step:>10} | {lr:>12.3e} | {phase}")


# Example: verify the scenario from the problem statement
# plot_schedule(
#     eta_max=6e-4,
#     warmup_steps=4_000,
#     total_steps=600_000,
#     eta_min_ratio=0.1,
# )


def train_with_schedule(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    total_steps: int = 600_000,
    eta_max: float = 6e-4,
    warmup_steps: int = 4_000,
    grad_clip_norm: float = 1.0,
) -> None:
    """Skeleton training loop with warmup + cosine schedule and grad clipping."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    # Separate weight-decayed and non-decayed parameter groups
    decay_params = [
        p for n, p in model.named_parameters()
        if p.ndim >= 2 and 'norm' not in n
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.ndim < 2 or 'norm' in n
    ]
    optimiser = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=eta_max,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = make_warmup_cosine_scheduler(
        optimiser, warmup_steps, total_steps, eta_min_ratio=0.1
    )
    scaler = torch.cuda.amp.GradScaler()  # for BF16/FP16 mixed precision

    step = 0
    data_iter = iter(train_loader)

    while step < total_steps:
        try:
            tokens, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            tokens, labels = next(data_iter)

        tokens, labels = tokens.to(device), labels.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(tokens)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)

        # Clip after unscaling -- critical to do this BEFORE optimiser.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip_norm
        )

        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)

        # Schedule step happens AFTER optimiser step
        scheduler.step()
        step += 1

        if step % 1000 == 0:
            lr = optimiser.param_groups[0]['lr']
            print(
                f"Step {step:>7}/{total_steps} | "
                f"LR: {lr:.3e} | "
                f"Loss: {loss.item():.4f} | "
                f"GradNorm: {grad_norm:.3f}"
            )
```

---

## Part D: Failure Mode Analysis

### Failure Mode 1: No Warmup

**What happens**: With AdamW and $\eta = 6 \times 10^{-4}$ from step 0, the second moment $\hat{v}_t$ starts near zero for all parameters. Despite bias correction, early estimates are based on only 1--10 gradient samples. For parameters with initially small gradients (e.g., attention weights in deeper layers), the effective learning rate is:

$$\eta_{eff} = \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$$

At step 1, if a parameter has gradient $g_1 = 10^{-3}$, then $\hat{v}_1 \approx g_1^2 = 10^{-6}$ and:

$$\eta_{eff} = \frac{6 \times 10^{-4}}{\sqrt{10^{-6}} + 10^{-8}} \approx \frac{6 \times 10^{-4}}{10^{-3}} = 0.6$$

This is 1000 times the intended learning rate. The parameter may receive an update of magnitude $0.6 \times g_1 = 6 \times 10^{-4}$, which is fine here. But if another parameter has $g = 10^{-6}$ (very small), then $\eta_{eff} \approx 6 \times 10^{-4} / (10^{-6} + 10^{-8}) \approx 6 \times 10^{-4} / 10^{-6} = 600$, an astronomically large step.

**Observable symptoms**:
- Training loss spikes or diverges in the first 100--500 steps.
- Loss oscillates without decreasing.
- NaN loss values in extreme cases.
- After the initial instability, training may recover slowly or not at all.

**Fix**: Add at least $W = 0.5\%$--$1\%$ of total steps as warmup. For 600,000 total steps, $W = 3000$--$6000$.

### Failure Mode 2: Warmup Too Short

**What happens**: If $W = 100$ steps for a 600,000-step run, the second moment has only 100 samples before training operates at full learning rate. For $\beta_2 = 0.95$, the second moment half-life is approximately $\ln(2) / (1 - 0.95) \approx 14$ steps -- meaning after 100 steps, the second moment has a reasonable sample of recent gradient history. For $\beta_2 = 0.999$ (used in some recipes), the half-life is $\approx 693$ steps, requiring much longer warmup.

**Rule of thumb**: Warmup should be at least $3 \times 1/(1 - \beta_2)$ steps to allow the second moment to become representative. For $\beta_2 = 0.95$: $W \geq 60$ steps (short warmup sufficient). For $\beta_2 = 0.999$: $W \geq 3000$ steps.

### Failure Mode 3: Warmup Too Long

**What happens**: Using $W = 100{,}000$ steps for a 600,000-step run (16.7% of training). The learning rate only reaches $\eta_{max}$ at step 100,000. The first 100,000 steps operate at a fraction of the intended learning rate, effectively "wasting" early training compute.

$$\eta_{50000} = 6 \times 10^{-4} \times \frac{50000}{100000} = 3 \times 10^{-4}$$

For 50,000 steps the model is trained with $\eta = 3 \times 10^{-4}$ instead of the intended $6 \times 10^{-4}$. This is equivalent to training with a different (lower) peak learning rate, potentially reducing final performance.

**Observable symptoms**: Training loss decreases slowly in the first phase; the curve shows a clear "ramp" in the loss improvement rate that matches the learning rate ramp.

**Guideline**: Warmup fraction should be in the range of $0.1\%$--$2\%$ of total steps for most transformer pre-training recipes.

### Failure Mode 4: Ignoring the Schedule Beyond $T_{max}$

**What happens**: If training is extended beyond $T = 600,000$ steps without modifying the schedule, `CosineAnnealingLR` (or `LambdaLR` with the formula above) will hold $\eta$ at $\eta_{min}$ for all steps $t > T$. The model continues learning but at the minimum learning rate.

If the intent is to train for $T' = 900,000$ steps, the last 300,000 steps would be at $\eta_{min} = 6 \times 10^{-5}$, which is sub-optimal -- the learning rate is too small for continued large-scale learning.

**Fix**: Design the schedule for the intended total training horizon $T'$ from the start, or implement a warm restart at step $T$ with a reduced peak learning rate $\eta_{max}' = \eta_{max} / 3$.

---

## Part E: Adapting the Schedule for Fine-Tuning

When fine-tuning a pretrained model (rather than training from scratch), the schedule parameters change significantly.

### Fine-Tuning Scenario

- **Starting model**: GPT-2 large (345M parameters, pretrained)
- **Fine-tuning dataset**: 500M tokens of domain-specific text
- **Batch size**: 256 sequences of 512 tokens
- **Total fine-tuning steps**: $T_{ft} = 20{,}000$ steps

### Adapted Schedule Parameters

```python
# Fine-tuning parameters are typically:
eta_max_finetune = 1e-4   # 6x smaller than pre-training (pretrained weights
                           # are already good -- large LR would destroy them)
eta_min_finetune = 1e-5   # same ratio 10:1
warmup_steps_ft  = 200    # ~1% of 20,000; short because model is already trained
total_steps_ft   = 20_000

# The warmup is very short because the pretrained weights are near a good
# minimum and the second moment from a loaded checkpoint should be restored.
# If loading the optimiser state from a checkpoint, the second moment is
# already warm -- warmup is not strictly needed but provides a safe ramp.
```

### Why Fine-Tuning Uses a Lower Peak LR

The pretrained model's parameters represent weeks of compute-intensive training. A large learning rate would destroy this learned structure in the first few steps (catastrophic forgetting). The rule of thumb is:

$$\eta_{max}^{ft} \approx \frac{\eta_{max}^{pretrain}}{5} \text{ to } \frac{\eta_{max}^{pretrain}}{10}$$

With layer-wise learning rate decay (LLRD), earlier layers (capturing general language patterns) receive even lower LRs than later layers (capturing task-specific patterns):

```python
def build_layerwise_adamw(model: nn.Module, base_lr: float = 1e-4,
                           lr_decay: float = 0.8) -> torch.optim.Optimizer:
    """
    Assign lower LRs to earlier layers via layer-wise LR decay.
    lr_decay < 1.0 means earlier layers get lower LRs.
    """
    num_layers = len(model.transformer.h)  # GPT-2 structure
    param_groups = []

    for layer_idx in range(num_layers):
        lr_scale = lr_decay ** (num_layers - layer_idx)
        layer_lr = base_lr * lr_scale
        layer_params = list(model.transformer.h[layer_idx].parameters())
        param_groups.append({'params': layer_params, 'lr': layer_lr})

    # Embedding and head layers at base LR
    param_groups.append({
        'params': list(model.transformer.wte.parameters()) +
                  list(model.transformer.wpe.parameters()),
        'lr': base_lr * lr_decay ** num_layers   # lowest LR for embeddings
    })
    param_groups.append({
        'params': list(model.lm_head.parameters()),
        'lr': base_lr                              # highest LR for task head
    })

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
```

---

## Part F: OneCycleLR for Rapid Experimentation

For rapid experimentation on a smaller model or dataset, OneCycleLR provides a practical alternative to the warmup + cosine recipe.

**Scenario**: Training a 6-layer transformer classifier (50M parameters) on a downstream task for 5 epochs with 10,000 steps per epoch.

```python
import torch.optim as optim

model = ...   # 50M parameter classifier
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

total_steps = 50_000  # 5 epochs x 10,000 steps/epoch

scheduler = optim.lr_scheduler.OneCycleLR(
    optimiser,
    max_lr=1e-3,             # peak LR found via LR range test
    total_steps=total_steps,
    pct_start=0.1,           # 10% of steps = 5,000 steps for warmup
    anneal_strategy='cos',   # cosine annealing in both warmup and decay phases
    div_factor=10.0,         # initial LR = max_lr / 10 = 1e-4
    final_div_factor=1e3,    # final LR = max_lr / (10 * 1e3) = 1e-6
)
```

**Learning rate profile**:
- Steps 0--5,000: linear ramp from $10^{-4}$ to $10^{-3}$
- Steps 5,000--50,000: cosine decay from $10^{-3}$ to $10^{-6}$

OneCycleLR is particularly effective here because:
1. The 5-epoch training budget is small; a single cycle is appropriate.
2. The automatic LR range test (or manual experimentation) can identify the right max LR quickly.
3. The training will converge more aggressively than with a conservative warmup + long cosine schedule.

**When OneCycleLR is less suitable**:
- Very long training runs ($> 100$K steps): a single cycle may anneal too aggressively, leaving insufficient LR for continued learning.
- Distributed training with many GPUs where the per-GPU batch is very small: OneCycleLR's momentum cycling can interact poorly with noisy gradients.

---

## Key Takeaways

1. **The warmup duration should scale with $1/(1 - \beta_2)$**, the time constant of the second moment. For $\beta_2 = 0.95$, 200 warmup steps is sufficient. For $\beta_2 = 0.999$, at least 1000--3000 steps are needed.

2. **The schedule should be designed for the full intended training run.** Do not design for 300K steps and then extend to 600K; this wastes the LR budget. Use scaling laws to estimate the compute-optimal training horizon before beginning.

3. **$\eta_{min}$ acts as implicit regularisation.** A non-zero $\eta_{min}$ (e.g., $0.1 \eta_{max}$) keeps the model in a broader neighbourhood of the minimum rather than converging to the exact valley bottom, which may be sharp.

4. **The cosine curve's non-uniform decay rate is deliberate.** The slow initial decay lets the model explore the loss landscape at high LR; the rapid middle decay efficiently converges; the slow final decay gently settles into the minimum.

5. **Fine-tuning requires lower LR and shorter warmup.** Pretrained weights are already well-optimised; large steps would corrupt them. Use $\eta_{max}^{ft} \approx \eta_{max}^{pretrain} / 5$ to $/ 10$.

---

## Common Interview Questions on This Topic

**Q: You double the batch size from 256 to 512. Should you change the learning rate schedule, and if so, how?**

The linear scaling rule (Goyal et al., 2017) states: if you multiply the batch size by $k$, multiply the peak learning rate by $k$ to maintain the same training dynamics. With batch $512 = 2 \times 256$, set $\eta_{max}' = 2 \times 6 \times 10^{-4} = 1.2 \times 10^{-3}$. The warmup duration should also scale: the warmup should cover enough steps for the second moment to warm up and for data diversity to be seen, both of which are less sensitive to batch size than LR is. In practice, keep the warmup at the same number of steps (tokens seen grows proportionally with batch size).

Note: the linear scaling rule breaks down for very large batch sizes (batch sizes $> 8192$) where training becomes unstable even with proportionally larger LR. For these regimes, use warmup + gradual LR ramp to the target peak.

**Q: The training loss shows a sharp spike at step 4,000 -- exactly at the end of the warmup. What is the likely cause?**

The end-of-warmup spike is caused by the **discontinuity in the LR derivative** at $t = W$. During warmup, the LR is increasing (positive derivative). At $t = W$, the cosine phase begins, where the LR immediately starts decreasing (negative derivative). The step from increasing to decreasing LR is abrupt, and if $\eta_{max}$ is large, the optimiser was taking increasingly large steps up to $t = W$ and now suddenly takes steps that are smaller and changing direction.

A practical fix: use a cosine warmup (ramp up following the first quarter of a cosine curve) rather than linear warmup. This gives $d\eta/dt = 0$ at both $t = 0$ and $t = W$, ensuring smooth transitions.

**Q: You are given a training run that converged to 87% validation accuracy. If you re-run training with the same schedule but set $\eta_{min} = 0$ (instead of $0.1 \times \eta_{max}$), what would you expect?**

With $\eta_{min} = 0$, the schedule reaches 0 learning rate at the end of training. The last few thousand steps apply essentially no updates (the LR is near zero), so the model converges to the exact minimum of the loss in the final parameter neighbourhood. Whether this is beneficial depends on the sharpness of that minimum:

- If the model has found a **flat minimum** (low curvature), $\eta_{min} = 0$ locks it into the flat valley bottom -- this may improve test accuracy slightly by eliminating the residual "wandering" noise at $\eta_{min} > 0$.
- If the model has found a **sharp minimum** (high curvature), $\eta_{min} = 0$ locks it into a point that is very sensitive to test-time perturbations -- test accuracy may be worse.

In practice, $\eta_{min} = 0$ gives marginally lower training loss but can lead to worse generalisation than $\eta_{min} = 0.1 \eta_{max}$, particularly for models that are prone to sharp minima. The GPT-3 and Chinchilla training recipes use $\eta_{min} = 0.1 \eta_{max}$; this is a well-validated default.
