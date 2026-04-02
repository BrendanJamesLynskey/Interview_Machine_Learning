# Learning Rate Scheduling

## Prerequisites
- Gradient descent and the role of the learning rate $\eta$
- Adam and SGD update rules (see `sgd_adam_and_variants.md`)
- Basic calculus: cosine function, geometric sequences
- Familiarity with PyTorch `torch.optim.lr_scheduler`

---

## Concept Reference

### Why Learning Rate Scheduling Matters

The learning rate $\eta$ is arguably the most important hyperparameter in deep learning training. A fixed learning rate faces a fundamental tension:

- **Too large:** gradients explode or the optimiser overshoots minima, causing divergence or oscillation around a minimum without settling.
- **Too small:** training is slow; the optimiser may get trapped in poor local minima or saddle points.

Scheduling resolves this by varying $\eta$ over the course of training: a larger rate during exploration and a smaller rate during refinement. The optimal schedule shape depends on the optimiser, model architecture, dataset size, and hardware constraints.

---

### Step Decay

The simplest schedule: multiply the learning rate by a factor $\gamma < 1$ every $s$ epochs or steps.

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

**Example:** $\eta_0 = 0.1$, $\gamma = 0.1$, $s = 30$ (epochs). Learning rate is $0.1$ for epochs 0--29, $0.01$ for epochs 30--59, $0.001$ for epochs 60--89.

Step decay is easy to implement and reason about. It was the dominant approach for ResNet-style image classification training (train for 90 epochs, decay at epochs 30 and 60). Its weakness is the hard transition: loss typically spikes briefly after each decay step as the network adjusts. Choosing $s$ and $\gamma$ requires careful empirical tuning.

PyTorch:
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
# or multi-step:
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[30, 60], gamma=0.1)
```

---

### Exponential Decay

A continuous version of step decay:

$$\eta_t = \eta_0 \cdot \gamma^t$$

After $T$ steps, $\eta_T = \eta_0 \cdot \gamma^T$. Setting $\gamma = ({\eta_{final}}/{\eta_0})^{1/T}$ gives exact control over the final learning rate. Exponential decay is smooth but the rate falls below useful magnitudes relatively early if $T$ is large.

---

### Cosine Annealing

Cosine annealing varies the learning rate between $\eta_{max}$ and $\eta_{min}$ following a cosine curve:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

where $t$ is the current step and $T$ is the total number of steps (or the period, in cyclic variants).

At $t = 0$: $\eta = \eta_{min} + (\eta_{max} - \eta_{min}) = \eta_{max}$.
At $t = T$: $\eta = \eta_{min} + 0 = \eta_{min}$.

The cosine shape has useful properties:
- Decays slowly at first (when the network is exploring) and quickly in the middle (efficient convergence) and slowly again near the end (fine-grained refinement into a minimum).
- No discontinuities; the gradient of the schedule is zero at both endpoints.

**Cosine Annealing with Warm Restarts (SGDR, Loshchilov & Hutter, 2017):** Periodically resets $\eta$ to $\eta_{max}$ to escape sharp minima:

$$T_i = T_0 \cdot T_{mult}^i$$

where $T_i$ is the period of the $i$-th restart and $T_{mult} \geq 1$ controls whether restarts become progressively longer. Warm restarts allow the model to explore multiple basins and often find flatter (better-generalising) minima.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=num_steps, eta_min=1e-6
)
# With restarts:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimiser, T_0=1000, T_mult=2, eta_min=1e-6
)
```

---

### Linear Warmup

In warmup, the learning rate starts near zero and increases linearly (or less commonly, exponentially) to the target $\eta_{max}$ over $W$ steps:

$$\eta_t = \eta_{max} \cdot \frac{t}{W}, \quad t \leq W$$

After step $W$, the main schedule (cosine, step decay, etc.) takes over.

**Why warmup is necessary for Adam and transformer training:**

1. **Cold second-moment estimates:** At the start of training, Adam's second moment $v_t \approx (1-\beta_2)g_t^2$ is small (undercorrected by the bias correction). The effective learning rate $\eta / (\sqrt{\hat{v}_t} + \epsilon)$ is therefore much larger than intended. Starting with a small $\eta$ during this phase prevents destructive early parameter updates.

2. **Large initial gradients:** At initialisation, gradients can be large and volatile, especially for deep transformers. A small initial learning rate dampens the effect of these outlier gradients.

3. **Distribution shift in early batches:** Early batches of data may not be representative of the full training distribution (if data is not thoroughly shuffled). Warmup prevents the model from making large, hard-to-reverse parameter changes based on the first few batches.

**Without warmup in transformer training,** loss spikes in the first few thousand steps are common, sometimes leading to divergence that requires training restarts.

---

### Linear Warmup + Cosine Decay (Standard Recipe)

The dominant schedule for transformer pre-training combines warmup and cosine decay:

$$\eta_t = \begin{cases}
\eta_{max} \cdot \dfrac{t}{W} & t \leq W \\[8pt]
\eta_{min} + \dfrac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\dfrac{\pi(t - W)}{T - W}\right)\right) & t > W
\end{cases}$$

Typical values: $W = 1$--$5\%$ of total steps $T$, $\eta_{min} = 0.1 \eta_{max}$ or $0$.

This schedule was used for GPT-2 (OpenAI), BERT (Google), and is the default in most HuggingFace training recipes.

---

### OneCycleLR

OneCycleLR (Smith & Touvron, 2019) is a cyclic schedule that trains with a single cycle consisting of:
1. A warmup phase where $\eta$ increases from $\eta_{max}/\text{div\_factor}$ to $\eta_{max}$.
2. An annealing phase where $\eta$ decreases from $\eta_{max}$ to $\eta_{max}/\text{final\_div\_factor}$.

The momentum (for SGD) is simultaneously cycled in the opposite direction: high momentum during annealing, lower momentum during warmup.

**Why it works:** The policy was found empirically to allow larger maximum learning rates (hence "Super-Convergence") while maintaining stability, because the rapid increase then decrease prevents the model from committing to sharp minima early. OneCycleLR uses annealing via cosine for both the LR increase and decrease phases.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser,
    max_lr=1e-3,
    total_steps=num_steps,
    pct_start=0.3,           # fraction of steps spent in warmup
    anneal_strategy='cos',   # 'cos' or 'linear'
    div_factor=25.0,         # initial lr = max_lr / div_factor
    final_div_factor=1e4,    # final lr = max_lr / (div_factor * final_div_factor)
)
```

OneCycleLR is popular for fine-tuning and small-to-medium vision tasks. It is less commonly used for large-scale LLM training where the cosine + warmup schedule is standard.

---

### Polynomial Decay

A generalisation of linear decay:

$$\eta_t = (\eta_0 - \eta_{end}) \left(1 - \frac{t}{T}\right)^{power} + \eta_{end}$$

With `power=1.0` this is linear decay; with `power=2.0` it is quadratic (decays more slowly initially). Used in some BERT implementations as an alternative to cosine.

---

### Choosing Warmup Duration

The warmup duration $W$ is problem-dependent:

| Setting | Typical $W$ (as fraction of $T$) |
|---|---|
| SGD from scratch (ResNet, CIFAR) | 0--1% (often no warmup) |
| SGD fine-tuning (pretrained CNN) | 0--1% |
| Adam, medium transformer, supervised | 1--5% |
| AdamW, large transformer from scratch (BERT-scale) | 1--4% |
| AdamW, very large LLM ($> 1$B params) | 0.1--1% of total steps |

For large models, warmup is measured in steps, not epochs: BERT-base uses 10,000 warmup steps out of 1,000,000 total (1%). GPT-3 used 375M tokens of warmup out of 300B (0.125%).

---

## Tier 1 -- Fundamentals

### Question F1
**What is the purpose of a learning rate schedule? Why not use a fixed learning rate throughout training?**

**Answer:**

A fixed learning rate must simultaneously satisfy two conflicting requirements: large enough to escape saddle points and cross loss barriers during early training, yet small enough to converge precisely into a minimum during late training.

A large fixed learning rate causes the optimiser to overshoot minima repeatedly. Once the training loss is near its minimum value, the gradient is small but the learning rate forces large steps that continually displace parameters away from the optimum, causing the loss to oscillate rather than converge. The test loss will also oscillate accordingly.

A small fixed learning rate avoids overshooting but makes early training extremely slow. The gradients in early training, when parameters are far from any minimum, are large and well-directed; taking tiny steps wastes this reliable gradient signal.

Scheduling solves this by starting with a larger learning rate (fast exploration) and reducing it over time (precise convergence). The intuition is analogous to simulated annealing: high "temperature" (large $\eta$) early to explore the loss landscape, low "temperature" (small $\eta$) late to settle into a minimum.

Empirically, cosine annealing and step decay both provide consistent improvements over a fixed learning rate on the same model. The improvement is not marginal: on ImageNet, the difference between a fixed rate and a step-decay schedule can be 5+ percentage points of top-1 accuracy with the same number of training epochs.

---

### Question F2
**Describe the linear warmup + cosine decay schedule. Write the formula for $\eta_t$ and sketch the shape.**

**Answer:**

For total training steps $T$ and warmup steps $W$:

$$\eta_t = \begin{cases}
\eta_{max} \cdot \dfrac{t}{W} & 0 \leq t \leq W \\[8pt]
\eta_{min} + \dfrac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\dfrac{\pi(t - W)}{T - W}\right)\right) & W < t \leq T
\end{cases}$$

**Sketch (qualitative):**

```
eta_max |          *
        |        *   *
        |      *       *
        |    *           *
        |  *               *
eta_min | *                  * . . . . . . 
        |_ _ _|_ _ _ _ _ _ _ _ _ _ _ _ _ _
             W                             T
              warmup       cosine decay
```

The warmup ramp is linear from 0 (or near-zero) to $\eta_{max}$. After step $W$, the cosine curve smoothly decreases to $\eta_{min}$. The cosine ends with zero slope (tangent horizontal at $t = T$), ensuring a smooth "landing" rather than a hard cutoff.

**Implementation in PyTorch** using `get_cosine_schedule_with_warmup` from HuggingFace:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimiser,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

Or manually:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
```

---

### Question F3
**Why is learning rate warmup particularly important when training transformers with Adam, but less critical for training a small CNN with SGD?**

**Answer:**

**Transformers with Adam:**

At the start of training, Adam's second moment $\hat{v}_t$ is initialised at zero and is corrected upward by the bias correction factor $1/(1 - \beta_2^t)$. Despite this correction, early estimates are noisy (based on very few gradient samples). The effective per-parameter learning rate $\eta_t^{eff} = \eta / (\sqrt{\hat{v}_t} + \epsilon)$ can be extremely large for parameters whose gradients have been small (since $\sqrt{\hat{v}_t}$ is small and $\epsilon$ dominates the denominator). Without warmup, these uncontrolled learning rates cause large, arbitrary parameter updates in the first few steps.

Transformers exacerbate this because:
- The architecture is deep with many interacting components (attention, layer norms, feed-forward layers).
- The initial gradient distribution is very non-uniform: embedding gradients are sparse, attention weight gradients are dense.
- Transformer training is sensitive to the scale of weight updates in the first few steps; a large initial step can put weights in a regime where the next gradient is very different, causing a feedback loop of instability.

**Small CNNs with SGD:**

SGD has no adaptive scaling, so there is no analogue of the "hot" effective learning rate from small $\hat{v}_t$. The update magnitude is simply $\eta \cdot |g|$. If $\eta$ is set conservatively (which it must be for SGD to be stable), the early updates are well-controlled. CNNs also have relatively uniform gradient scales across layers (batch normalisation helps equalise them), reducing the risk of any single layer making destructive early updates.

The conclusion: warmup is a compensating mechanism for Adam's second-moment cold start, and for the sensitivity of deep transformers to early perturbations. SGD on a small CNN does not have either of these problems.

---

## Tier 2 -- Intermediate

### Question I1
**You are training a 125M parameter language model for 300,000 steps. The peak learning rate is $3 \times 10^{-4}$, warmup is 3,000 steps, and you use cosine decay to $\eta_{min} = 3 \times 10^{-5}$. What is the learning rate at step 1,000, step 3,000, step 150,000, and step 300,000?**

**Answer:**

Parameters: $\eta_{max} = 3 \times 10^{-4}$, $\eta_{min} = 3 \times 10^{-5}$, $W = 3000$, $T = 300000$.

**Step 1,000** (warmup phase, $t \leq W$):

$$\eta_{1000} = \eta_{max} \cdot \frac{1000}{3000} = 3 \times 10^{-4} \times \frac{1}{3} = 1.0 \times 10^{-4}$$

**Step 3,000** (end of warmup, $t = W$):

$$\eta_{3000} = \eta_{max} \cdot \frac{3000}{3000} = 3 \times 10^{-4}$$

At exactly $t = W$, the warmup formula gives $\eta_{max}$. The cosine formula at $t = W$ gives:

$$\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos(0)\right) = \eta_{min} + (\eta_{max} - \eta_{min}) = \eta_{max}$$

Both formulas give $\eta_{max}$ at the transition point: the schedule is continuous.

**Step 150,000** (midpoint of cosine decay):

$$\text{progress} = \frac{150000 - 3000}{300000 - 3000} = \frac{147000}{297000} \approx 0.4949$$

$$\eta_{150000} = 3 \times 10^{-5} + \frac{1}{2}(3 \times 10^{-4} - 3 \times 10^{-5})\left(1 + \cos(\pi \times 0.4949)\right)$$

$$\cos(\pi \times 0.4949) = \cos(1.5543) \approx 0.0159$$

$$\eta_{150000} \approx 3 \times 10^{-5} + \frac{1}{2}(2.7 \times 10^{-4})(1.0159) \approx 3 \times 10^{-5} + 1.371 \times 10^{-4} \approx 1.67 \times 10^{-4}$$

Note: at exactly $t = (W + T)/2 = 151500$, $\text{progress} = 0.5$ and $\cos(\pi/2) = 0$, giving:

$$\eta_{151500} = 3 \times 10^{-5} + \frac{1}{2}(2.7 \times 10^{-4}) = 3 \times 10^{-5} + 1.35 \times 10^{-4} = 1.65 \times 10^{-4}$$

**Step 300,000** (end of schedule):

$$\text{progress} = \frac{300000 - 3000}{297000} = 1.0$$

$$\eta_{300000} = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi)) = \eta_{min} + 0 = 3 \times 10^{-5}$$

**Summary:**

| Step | Phase | $\eta$ |
|------|-------|--------|
| 1,000 | Warmup | $1.0 \times 10^{-4}$ |
| 3,000 | Warmup end | $3.0 \times 10^{-4}$ |
| 150,000 | Cosine decay | $\approx 1.67 \times 10^{-4}$ |
| 300,000 | End | $3.0 \times 10^{-5}$ |

---

### Question I2
**What is "Super-Convergence" and how does OneCycleLR enable it? What is the key empirical finding that makes this work?**

**Answer:**

Super-Convergence (Smith & Touvron, 2018) is the empirical observation that certain network/dataset combinations can be trained in $5$--$10\times$ fewer iterations than standard training recipes, using a single learning rate cycle that reaches a much higher peak learning rate than would normally be stable.

**The key empirical finding: the Learning Rate Range Test (LR Range Test).**

Smith's procedure: start training with a very small $\eta$ and increase it linearly over the course of a training run (e.g., from $10^{-7}$ to $1.0$ over 300 steps). Plot the training loss vs. $\eta$:

- Initially, as $\eta$ increases from near zero, the loss decreases (the model is learning faster).
- At some point, the loss reaches a minimum, then begins to increase as $\eta$ is too large and the training becomes unstable.
- The optimal range is approximately between the "start of fast decrease" and the "point before the loss starts rising significantly."

The maximum learning rate in OneCycleLR is set to this identified stable maximum (which is often $5$--$10\times$ higher than what intuition would suggest).

**Why OneCycleLR enables this:**

With a fixed large $\eta$, the optimiser overshoots and the training diverges. With OneCycleLR, the large $\eta$ is only encountered briefly (at the peak), and the network transitions through that region quickly, essentially using the large learning rate to escape poor basins. The subsequent annealing phase brings the network into a good minimum with the lower learning rate.

**The momentum interaction:** OneCycleLR simultaneously cycles momentum in the opposite direction to the learning rate. When $\eta$ is high, momentum is low ($\beta_1 \approx 0.85$), reducing the risk of large accumulated velocity causing instability. When $\eta$ is annealing downward, momentum increases back to $0.95$, helping the optimiser converge smoothly.

**When Super-Convergence applies:** It is most reliable on small to medium models with relatively well-conditioned loss landscapes (ResNets on CIFAR, FastAI-style vision models). It is less commonly applied to large transformer models, where cosine + warmup is more predictable.

---

### Question I3
**A team training a BERT-like model finds that the training loss spikes sharply at step 10,000, then recovers. The learning rate schedule has no warmup. What is the most likely cause and how would you fix it?**

**Answer:**

**Most likely cause: Adam's second-moment cold-start problem interacting with a step-decay or cosine schedule that begins with the full learning rate.**

Without warmup, Adam starts with $\hat{v}_0 \approx 0$ for all parameters. Despite bias correction, the denominator $\sqrt{\hat{v}_t} + \epsilon$ for parameters with small gradients is dominated by $\epsilon$ (default $10^{-8}$), making the effective learning rate $\eta / \epsilon = \eta \times 10^8$. This causes extremely large updates in the first few steps for any parameter whose gradient is small in absolute terms (e.g., later layers in a deep stack, or parameters corresponding to rare tokens in an embedding).

However, the spike occurring at step 10,000 (not step 1) suggests a different mechanism. More likely causes:

1. **Data pipeline issue:** If the dataset is not fully shuffled and a different data distribution appears around step 10,000 (e.g., a batch of unusually long sequences, or a domain shift in the data), the sudden change in gradient distribution causes the Adam second moment to momentarily underestimate the true gradient scale, leading to large steps.

2. **Gradient explosion from accumulated small gradients:** If a rare but high-magnitude gradient batch is encountered at step 10,000 and no gradient clipping is applied, a single large-gradient step can spike the loss.

3. **Learning rate schedule discontinuity:** If the schedule has a step decay that fires at step 10,000 and the new learning rate is too large for the current parameter scale, training destabilises.

4. **Missing gradient clipping:** Large gradient norms (from long sequences or unusual examples) combined with no clipping cause the loss to spike before recovering.

**Fixes in order of priority:**

```python
# 1. Add linear warmup (most important for Adam)
def lr_lambda(step):
    warmup = 10_000
    if step < warmup:
        return step / warmup
    # cosine annealing after warmup
    progress = (step - warmup) / (total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Verify data shuffling:
dataloader = DataLoader(dataset, shuffle=True, ...)

# 4. Monitor gradient norms during training
grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
writer.add_scalar('grad_norm', grad_norm, step)
```

The combination of warmup + gradient clipping resolves the vast majority of training spikes in transformer models.

---

## Tier 3 -- Advanced

### Question A1
**Derive the expected steady-state "noise" of SGD with a fixed learning rate versus cosine-annealed learning rate. Why do models generalise better with smaller final learning rates?**

**Answer:**

**SGD noise model.** Under standard assumptions (loss $\mathcal{L}$, true gradient $\nabla \mathcal{L}$, mini-batch gradient estimate $g_t = \nabla \mathcal{L} + \xi_t$ where $\xi_t$ is zero-mean noise with covariance $\Sigma$), SGD with fixed $\eta$ near a minimum can be approximated by a stochastic differential equation (SDE):

$$d\theta = -\nabla \mathcal{L}(\theta) \, dt + \sqrt{\eta \Sigma} \, dW_t$$

where $dW_t$ is a Wiener process. Near a minimum with Hessian $H$, the steady-state distribution of $\theta$ is Gaussian with covariance:

$$C \propto \eta H^{-1} \Sigma$$

The "temperature" of the steady-state is proportional to $\eta$: the larger the learning rate, the wider the distribution around the minimum.

**Consequence for flat vs. sharp minima.** For a sharp minimum (large eigenvalues of $H$), the $H^{-1}$ factor is small, but the learning-rate-dependent noise $\eta \Sigma$ still drives the parameter away from the minimum, causing the test loss to increase. For a flat minimum (small eigenvalues of $H$), the same noise has a smaller effect on test loss because the loss surface is flat. This is the mechanism behind the Keskar et al. (2017) observation that large-batch (low-noise) training converges to sharper minima with worse generalisation.

**Cosine decay effect.** By annealing $\eta \to \eta_{min}$ at the end of training, the steady-state covariance $C \propto \eta_{min} H^{-1} \Sigma$ shrinks, causing the parameter distribution to contract around the minimum. The model "cools" into a localised region around the minimum. Whether this is a sharp or flat minimum was determined during training, but the final small $\eta$ ensures the model stays near that minimum at deployment rather than wandering around it.

**Practical implication.** Training loss at step $T$ depends on $\eta_{min}$: if $\eta_{min}$ is too large, the training loss will not fully converge. If $\eta_{min} = 0$, the model converges to the exact minimum it found, which may be sharp. A non-zero $\eta_{min}$ (e.g., $\eta_{max}/10$) is a form of implicit regularisation. In practice, $\eta_{min} = 0.1 \eta_{max}$ is common for cosine decay; with longer training runs, $\eta_{min}$ can be set closer to 0.

---

### Question A2
**Explain the concept of "learning rate rewinding" in lottery ticket pruning. How does the choice of rewinding step affect the quality of the discovered subnetwork?**

**Answer:**

**Background: the Lottery Ticket Hypothesis (Frankle & Carlin, 2019).** The hypothesis states that large randomly initialised networks contain sparse subnetworks (winning tickets) that, when trained in isolation with their original initialisations, can match the full network's accuracy in the same or fewer training steps. Identifying these subnetworks via magnitude pruning followed by "rewinding" the unpruned weights to their initial values (or early training values) is the lottery ticket procedure.

**Learning rate rewinding (Frankle et al., 2020).** A practical refinement: instead of rewinding weights to their values at step 0 (random initialisation), rewind to their values at step $k$ (a small number of steps into training, e.g., 0.1%--1% of total training). The learning rate schedule is also reset to the value it had at step $k$, not to the initial learning rate.

**Why rewinding step $k$ matters:**

1. **Step $k = 0$ (original LTH):** Works well for small networks and simple tasks (LeNet on MNIST). For large networks on ImageNet, the sparse subnetwork trained from random initialisation often fails to match the full network, because modern training involves careful learning rate schedules and the random initialisation is far from any usable structure.

2. **Small $k$ (early in training, e.g., 100--500 steps):** The weights at step $k$ have had a few gradient steps and the gradient noise has "shaped" them slightly. This small amount of structure helps the subnetwork train stably when isolated. The learning rate at step $k$ is near the peak (if warmup is used), which gives the subnetwork aggressive early training.

3. **Large $k$ (late in training):** The weights at step $k$ are near-converged. Rewinding to this point and continuing only the surviving weights essentially continues training the pruned subnetwork. This finds structurally similar but potentially lower-quality tickets than early rewinding, because the starting point already encodes the solution and the subsequent training is refinement rather than discovery.

**The learning rate schedule interaction:** When rewinding weights to step $k$, it is critical to also rewind the learning rate schedule to step $k$ (not restart from step 0). If the schedule is restarted from step 0 while the weights are at step $k$ values, the high initial learning rate may disrupt the already-partially-trained weights, causing worse performance than expected. Rewinding the schedule ensures that the amount of learning rate "budget" given to the subnetwork matches the amount given to the full network from that same starting point.

**Practical guidance:**
- Use $k \approx 0.1\%$--$1\%$ of total training steps as the rewind point.
- Rewind both weights and learning rate schedule.
- Prune globally by magnitude (not per-layer), then rewind.
- Iterative magnitude pruning (prune 20%, rewind, retrain, repeat) finds better tickets than one-shot pruning at the same final sparsity.

---

### Question A3
**In large language model training, it is common to train with a cosine schedule to a fixed $T_{max}$ and then continue training beyond $T_{max}$ ("training beyond the schedule"). What happens to the learning rate and parameter updates, and is this practice recommended?**

**Answer:**

**What happens when training continues past the cosine schedule end:**

If the scheduler is not updated (e.g., `scheduler.step()` is called past $T_{max}$ for `CosineAnnealingLR`), PyTorch's default behaviour is to hold the learning rate at $\eta_{min}$ (the minimum). The schedule has reached its floor. The optimiser continues to update parameters with:

$$\theta_{t+1} = \theta_t - \frac{\eta_{min}}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The parameter updates are now with a fixed, small learning rate. This is similar to running SGD with a small constant learning rate: the model continues to improve, but slowly, and is now in a "fine-tuning at $\eta_{min}$" regime.

**Why teams extend training past $T_{max}$:**

In LLM pre-training, the amount of training data available is often not known in advance (or more data becomes available after the initial run). If a cosine schedule was designed for $T = 300,000$ steps but it is then decided to train for $T' = 500,000$ steps, naively continuing with the scheduler held at $\eta_{min}$ for the last 200,000 steps is wasteful: the learning rate is very small and most learning capacity is unused.

**Better approaches:**

1. **Extend the schedule:** Redefine $T_{max} = T'$ and recompute the schedule from the start. If the model is already at step 300,000 with the old schedule, one option is to create a new cosine schedule that treats step 300,000 as the new start of a second cosine half-cycle:

   $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{mid} - \eta_{min})\left(1 + \cos\left(\frac{\pi(t - T)}{T' - T}\right)\right), \quad T < t \leq T'$$

   where $\eta_{mid}$ is a reduced peak (e.g., $\eta_{max}/3$) to avoid disrupting the already-converged model.

2. **Warm restart:** If using `CosineAnnealingWarmRestarts`, the schedule restarts automatically with a new period, providing a natural mechanism to continue training.

3. **Chinchilla-style compute-optimal planning:** Before training begins, use the scaling laws (Hoffmann et al., 2022) to determine the compute-optimal training horizon $T$ for the model size and data budget. Set $T_{max}$ accordingly and do not extend; instead, train a larger model for the same compute budget if more capacity is available.

**Is extending past the schedule recommended?**

In practice, extending is common and effective when done carefully (option 1 above). Simply holding at $\eta_{min}$ for extended periods is wasteful but not harmful. The key insight from the Chinchilla paper is that the learning rate schedule should be designed to match the total training horizon from the outset; retrofitting a schedule after the fact is a compromise. For new projects, invest time in compute-optimal planning rather than ad hoc schedule extensions.
