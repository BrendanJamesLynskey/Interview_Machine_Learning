# SGD, Adam and Variants

## Prerequisites
- Calculus: partial derivatives, the chain rule
- Gradient descent update rule: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
- Mini-batch sampling and the concept of stochastic gradient noise
- Basic NumPy / PyTorch tensor operations

---

## Concept Reference

### Vanilla SGD

Stochastic Gradient Descent updates parameters using the gradient computed on a single mini-batch:

$$\theta_{t+1} = \theta_t - \eta \, g_t$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)$ is the mini-batch gradient and $\eta$ is the learning rate.

**Problems with vanilla SGD:**
- High variance in gradient estimates causes noisy, oscillating trajectories.
- Ravines (directions with small curvature but large gradient components) cause slow convergence.
- A single global learning rate must work for all parameters, even though gradients vary widely in scale across layers.
- No memory of past gradients: each step ignores the history that could improve direction.

---

### SGD with Momentum

Momentum maintains a running exponential moving average (EMA) of past gradients and uses that velocity to update parameters:

$$v_t = \beta v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

A common alternative formulation (used by PyTorch's `SGD` when `nesterov=False`):

$$v_t = \beta v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

With $\beta = 0.9$, each velocity is an EMA that downweights gradients from $k$ steps ago by $0.9^k$. Consistent gradients accumulate into a large velocity; oscillating gradients average out. This is analogous to a ball rolling down a hill that accelerates in flat regions and is dampened in steep ravines.

**Nesterov Accelerated Gradient (NAG):** Evaluates the gradient at the anticipated next position rather than the current position:

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \eta \beta v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

NAG looks ahead by one momentum step before computing the gradient, giving a more corrective update. In practice, PyTorch implements NAG as:

$$g_t^{nag} = g_t + \beta \, v_t$$
$$\theta_{t+1} = \theta_t - \eta \, g_t^{nag}$$

after updating $v_t$. NAG converges faster than plain momentum on convex objectives and is often the preferred SGD variant for fine-tuning.

---

### AdaGrad

AdaGrad adapts the learning rate per parameter by dividing by the square root of accumulated squared gradients:

$$G_t = G_{t-1} + g_t^2 \quad \text{(element-wise)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

Parameters that receive large gradients get a smaller effective learning rate; rarely updated parameters retain a larger effective rate. This is beneficial for sparse features (e.g., embeddings) but problematic for deep networks: $G_t$ is monotonically increasing, so learning rates shrink to near zero and training stalls.

---

### RMSProp

RMSProp replaces the accumulated sum with an exponential moving average, preventing the learning rate from decaying to zero:

$$v_t = \rho \, v_{t-1} + (1 - \rho) \, g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t$$

Typical $\rho = 0.99$. The moving average "forgets" old gradients, keeping the effective learning rate stable throughout training. RMSProp is effective for non-stationary objectives (RNNs) but has no bias correction and can still be sensitive to the initial learning rate.

---

### Adam (Adaptive Moment Estimation)

Adam combines momentum (first moment) with per-parameter adaptive learning rates (second moment), with bias correction to account for initialisation at zero:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment)}$$

Because $m_0 = v_0 = 0$, early estimates are biased towards zero. Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

After the first few steps, $\beta_1^t \approx 0$ and $\beta_2^t \approx 0$ (since $0.999^{1000} \approx 0.37$, bias correction remains meaningful for hundreds of steps).

**Why Adam converges faster in practice:**
- The denominator $\sqrt{\hat{v}_t}$ normalises the gradient by its recent RMS, reducing the effective step size in directions with high curvature.
- The numerator $\hat{m}_t$ provides smooth, low-variance gradient estimates.
- The result is an approximately normalised gradient with noise dampening.

**Adam's known failure modes:**
- Can converge to suboptimal minima compared to SGD on well-tuned image classification tasks.
- The adaptive learning rate can make Adam resistant to learning rate schedules unless learning rate decay is aggressive.
- Does not generalise as well as SGD+momentum in some vision tasks (this motivated AdamW).

---

### AdamW (Decoupled Weight Decay)

The standard Adam implementation adds L2 regularisation by including weight decay in the gradient:

$$g_t \leftarrow g_t + \lambda \theta_t \quad \text{(L2 in gradient)}$$

This is mathematically incorrect for adaptive optimisers. Because Adam scales the gradient by $1/\sqrt{\hat{v}_t}$, the effective weight decay per parameter becomes $\lambda / \sqrt{\hat{v}_t}$, not $\lambda$. Parameters with large gradient history are regularised less, regardless of their values.

AdamW fixes this by applying weight decay directly to the parameters, decoupled from the gradient update:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

The weight decay term $\lambda \theta_t$ is not passed through the adaptive scaling. This restores the intended L2 regularisation semantics. AdamW is now the default optimiser for transformer pre-training (BERT, GPT, ViT) and typically outperforms Adam with L2 on language and vision tasks.

**Typical AdamW hyperparameters for transformer training:** $\beta_1 = 0.9$, $\beta_2 = 0.95$--$0.999$, $\lambda = 0.01$--$0.1$, $\eta$ from a schedule.

---

### Gradient Accumulation

Mini-batch size is limited by GPU memory. Gradient accumulation simulates a larger effective batch size by accumulating gradients over $k$ forward/backward passes before applying one optimiser step:

```python
optimiser.zero_grad()
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / k   # scale loss to match effective batch
    loss.backward()                          # gradients accumulate in .grad buffers
    if (i + 1) % k == 0:
        optimiser.step()
        optimiser.zero_grad()
```

The effective batch size becomes $k \times B$ where $B$ is the per-step mini-batch size. This is mathematically equivalent to a true batch of size $k \times B$ when loss is the mean over the batch (hence the division by $k$ before `backward()`). If using gradient clipping, clip after accumulation, not per micro-step:

```python
if (i + 1) % k == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
    optimiser.zero_grad()
```

Gradient accumulation increases memory efficiency at the cost of $k \times$ the wall-clock time per effective batch. It is standard practice for large language model pre-training.

---

### Mixed-Precision Training (FP16 / BF16)

Training in full FP32 uses 4 bytes per parameter. Switching to FP16 (half precision) halves memory usage and accelerates matrix multiplications on GPUs with Tensor Cores. The main risks are:

1. **Underflow:** FP16 has a minimum representable value of $\approx 6 \times 10^{-5}$. Small gradients (especially early in training) flush to zero.
2. **Overflow:** FP16 maximum is $65504$. Loss values or intermediate activations can overflow to `inf` or `NaN`.

**Loss scaling** mitigates underflow by multiplying the loss by a scale factor $S$ before backward, then dividing gradients by $S$ before the optimiser step:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()          # backward in FP16 with scaled loss
scaler.unscale_(optimiser)             # divide grads by S before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimiser)                 # skip step if grads contain inf/NaN
scaler.update()                        # adjust S up/down based on overflow
```

**Master weights:** The optimiser states (parameters, momentum, variance) are kept in FP32. Forward and backward passes use FP16. Gradients are cast back to FP32 before accumulation into master weights. This preserves numerical precision for the parameter update while taking advantage of FP16 throughput.

**BF16 (BFloat16):** Uses the same exponent range as FP32 (8-bit exponent vs 5-bit for FP16) with reduced mantissa precision. BF16 does not overflow or underflow where FP32 does not, making it the preferred format for transformer training on hardware that supports it (Ampere+ GPUs, TPUs). Loss scaling is typically unnecessary with BF16.

---

## Tier 1 -- Fundamentals

### Question F1
**Explain the purpose of momentum in SGD. If $\beta = 0.9$ and the gradient has been consistently $g$ for many steps, what does the velocity $v_t$ converge to?**

**Answer:**

Momentum addresses two problems with vanilla SGD: high gradient variance and slow progress in flat directions.

With consistent gradient $g$ and $\beta = 0.9$, the velocity evolves as:

$$v_t = 0.9 \, v_{t-1} + g$$

At steady state ($v_t = v_{t-1} = v^*$):

$$v^* = 0.9 \, v^* + g \implies v^*(1 - 0.9) = g \implies v^* = \frac{g}{1 - \beta} = \frac{g}{0.1} = 10g$$

The steady-state velocity is $10\times$ the instantaneous gradient. In general, the terminal velocity is $g / (1 - \beta)$. This means that in directions where the gradient is consistent, momentum effectively multiplies the learning rate by $1/(1-\beta)$. For $\beta = 0.9$ this is a $10\times$ amplification; for $\beta = 0.99$ it is $100\times$.

In directions where the gradient oscillates in sign, the positive and negative contributions cancel out, keeping velocity near zero and reducing oscillation. This is the "dampening" effect of momentum in ravines or saddle points.

**Common mistake:** Confusing the momentum coefficient $\beta$ (how much of the past velocity to retain) with the learning rate $\eta$. Increasing $\beta$ makes the optimiser more "inertial" and can cause overshooting; it is not equivalent to increasing $\eta$.

---

### Question F2
**What is the bias correction term in Adam and why is it needed?**

**Answer:**

Adam initialises both moment estimates at zero: $m_0 = 0$, $v_0 = 0$. These zero initialisations cause systematic underestimation at early training steps.

At step $t = 1$ with $\beta_1 = 0.9$:

$$m_1 = 0.9 \times 0 + 0.1 \times g_1 = 0.1 \, g_1$$

The first-moment estimate $m_1$ is only $10\%$ of $g_1$, not a representative mean. Without correction, the initial learning rate would be effectively $10\times$ smaller than intended.

Bias correction scales each estimate by the inverse of the total "weight" that has accumulated:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

At $t = 1$: $\hat{m}_1 = m_1 / (1 - 0.9) = 0.1 g_1 / 0.1 = g_1$. The corrected estimate is exactly $g_1$, as desired.

As $t$ increases, $\beta_1^t \to 0$ and $1 - \beta_1^t \to 1$, so the correction factor approaches 1 and has no effect at large $t$. The correction is significant only during the warm-up phase (roughly the first 100 steps for $\beta_1 = 0.9$ and approximately the first 3000 steps for $\beta_1 = 0.999$, since $0.999^{1000} \approx 0.368$ so the bias remains substantial well past step 1000).

The same logic applies to $\hat{v}_t$: with $\beta_2 = 0.999$, the second moment requires roughly 1000 steps before it is representative.

---

### Question F3
**What is the difference between L2 regularisation and weight decay? Are they equivalent in Adam?**

**Answer:**

L2 regularisation adds a penalty term to the loss function:

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2$$

The gradient of the regularised loss is:

$$\nabla_\theta \mathcal{L}_{reg} = \nabla_\theta \mathcal{L} + \lambda \theta$$

Weight decay directly subtracts a fraction of the parameter at each update step:

$$\theta_{t+1} = (1 - \lambda) \theta_t - \eta g_t$$

For SGD, L2 regularisation and weight decay are exactly equivalent: the $\lambda \theta$ term in the gradient produces $(1 - \eta \lambda) \theta_t$ in the update, which is weight decay.

**For Adam, they are NOT equivalent.** Standard Adam applies the adaptive scaling $1/\sqrt{\hat{v}_t + \epsilon}$ to the entire gradient, including the L2 term:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\hat{m}_t + \lambda \theta_t)$$

The effective weight decay per parameter is $\lambda \eta / (\sqrt{\hat{v}_t} + \epsilon)$, which varies with the parameter's gradient history. Parameters with large gradients (large $\hat{v}_t$) are regularised much less than intended.

AdamW restores correctness by decoupling weight decay from the gradient:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

All parameters are decayed by the same factor $\eta \lambda$ regardless of their gradient history.

---

### Question F4
**Why is gradient accumulation needed and what is the key implementation detail to get it mathematically correct?**

**Answer:**

GPU memory limits the maximum batch size to perhaps 8--32 samples for large models. Many training recipes require effective batch sizes of 256--4096 (or larger for LLMs) to achieve stable training. Gradient accumulation allows $k$ micro-batches to contribute to a single parameter update, giving an effective batch size of $k \times B$ without requiring all samples to fit in memory simultaneously.

**The critical implementation detail:** The loss must be divided (or equivalently scaled) by $k$ before the backward pass.

If the loss is the mean over the batch and we compute `loss.backward()` for each micro-batch without scaling, then after $k$ steps the gradients in `.grad` represent the sum of $k$ per-micro-batch gradient estimates. This is equivalent to using the sum (not the mean) over $kB$ samples, which changes the effective learning rate by a factor of $k$. Dividing the loss by $k$ before each backward pass ensures the accumulated gradients represent the mean gradient over the full effective batch.

```python
# Correct: divide each micro-batch loss by k
loss = criterion(outputs, labels) / k
loss.backward()
```

An alternative is to divide by $k$ once after accumulation, using `grad /= k` before the optimiser step, but per-step division is cleaner when using mixed precision scalers.

---

## Tier 2 -- Intermediate

### Question I1
**Compare SGD with momentum and Adam across three practical training scenarios: (1) fine-tuning a pretrained vision model, (2) training a transformer from scratch, (3) training a small CNN from scratch. Which would you choose for each and why?**

**Answer:**

**(1) Fine-tuning a pretrained vision model (e.g., ResNet-50 on a new classification task):**

Prefer **SGD with momentum** ($\beta = 0.9$, $\eta \approx 10^{-3}$--$10^{-2}$, with cosine or step decay).

Rationale: The pretrained weights are already in a good region of parameter space. The goal is to refine them without large, erratic steps. SGD with a well-tuned learning rate schedule tends to find flatter, better-generalising minima for image classification tasks (this observation drove the "SGD generalises better than Adam" literature, e.g., Wilson et al., 2017). The lower adaptivity of SGD means it does not over-fit to the specific gradient magnitudes of the new task distribution.

**(2) Training a transformer from scratch (e.g., BERT-base):**

Use **AdamW** ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\lambda = 0.01$--$0.1$, $\eta = 10^{-4}$--$3 \times 10^{-4}$) with linear warmup followed by cosine or linear decay.

Rationale: Transformers have diverse parameter scales across layers (embedding tables, attention projections, feed-forward layers) and sparse gradient patterns (attention weights). Adam's per-parameter adaptivity is critical here: a global learning rate suitable for the feed-forward layers may be too large or too small for the embedding gradients. AdamW's decoupled weight decay prevents the L2 penalty from being absorbed into the gradient scaling. Linear warmup is essential to stabilise the second-moment estimates before taking large steps.

**(3) Small CNN from scratch (e.g., a 5-layer net on CIFAR-10):**

Either works, but **SGD with momentum** is a good default.

Rationale: With a small model and abundant data, the gradient estimates are relatively unbiased and the parameter scales are similar across layers. SGD generalises well here and the hyperparameter tuning is simpler (just $\eta$ and optionally $\beta$). Adam converges faster in terms of steps but may require more careful tuning of $\epsilon$ and learning rate decay to match SGD's final test accuracy. If training time is a concern and rapid prototyping is the goal, Adam is pragmatic.

---

### Question I2
**Derive the update rule for AdaGrad and explain why its learning rates monotonically decrease. What is the practical consequence for training deep networks?**

**Answer:**

Starting from the motivation: we want a larger learning rate for infrequently updated parameters and a smaller one for frequently updated parameters. Define the accumulated sum of squared gradients per parameter $i$:

$$G_{t,i} = \sum_{\tau=1}^{t} g_{\tau,i}^2$$

The AdaGrad update for parameter $i$:

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,i} + \epsilon}} g_{t,i}$$

The effective learning rate for parameter $i$ at step $t$ is:

$$\eta_{t,i}^{eff} = \frac{\eta}{\sqrt{G_{t,i} + \epsilon}}$$

**Why it monotonically decreases:** $G_{t,i}$ is a cumulative sum. Since $g_{\tau,i}^2 \geq 0$ for all $\tau$ and $i$, $G_{t,i}$ is non-decreasing. Adding each new squared gradient term increases (or leaves unchanged) $G_{t,i}$, and therefore $\sqrt{G_{t,i} + \epsilon}$ is non-decreasing, making $\eta_{t,i}^{eff}$ non-increasing.

**Practical consequence:** In a typical deep network trained for many epochs, every parameter receives gradients on virtually every step. After $T$ steps, $G_{T,i} \approx T \cdot \mathbb{E}[g_i^2]$, so $\eta_{t,i}^{eff} \propto 1/\sqrt{T}$. For large $T$, this decays to near zero. Training effectively stops long before convergence, regardless of the global learning rate $\eta$.

This behaviour is "useful" only for convex problems with a fixed data distribution and finite horizon (AdaGrad was designed for online learning), where the learning rate should decrease over time anyway. For non-convex deep network training, the perpetual decay is a fundamental flaw. RMSProp and Adam fix this by using an exponential moving average instead of a cumulative sum.

---

### Question I3
**Explain mixed-precision training with loss scaling. What is the purpose of the dynamic scale factor, and what happens when overflow is detected?**

**Answer:**

FP16 has a limited dynamic range: values smaller than $\approx 6 \times 10^{-5}$ (subnormals) or exactly zero, and a maximum of 65504. During backpropagation, gradients in the early layers of deep networks can be very small (vanishing gradients, especially before batch norm stabilises them) and would underflow to zero in FP16.

**Loss scaling** multiplies the scalar loss by a scale factor $S$ (typically $2^7$ to $2^{15}$) before calling `backward()`. By the chain rule, all gradients are also multiplied by $S$. This shifts the gradient distribution upward, preventing underflow of small gradients in FP16 arithmetic. Before the optimiser step, gradients are divided by $S$ to restore true magnitudes.

**Dynamic loss scaling:** PyTorch's `GradScaler` starts with a large $S$ and adjusts it:

- If any gradient contains `inf` or `NaN` (overflow), the optimiser step is **skipped** for that iteration, $S$ is halved (divided by 2), and training continues without a parameter update. The model is not corrupted because the step is skipped entirely.
- If no overflow occurs for `growth_interval` consecutive steps (default 2000), $S$ is multiplied by the growth factor (default 2), allowing larger values to be represented as gradients grow during training.

```
Overflow detection flow:
  scaler.unscale_(optimiser)    # divide .grad by S in-place, in FP32
  check for inf/NaN in .grad
  if inf/NaN found:
      scaler.update()           # reduce S, do NOT call optimiser.step()
  else:
      optimiser.step()          # safe to update
      scaler.update()           # potentially increase S
```

**Why master weights in FP32:** The optimiser state (Adam's $m_t$, $v_t$; SGD's $v_t$) accumulates small updates over thousands of steps. If stored in FP16, small updates (e.g., $\eta \cdot g \approx 10^{-6}$) would underflow or not be representable with sufficient precision. FP32 master weights ensure that small, consistent gradient signals are correctly accumulated into parameter updates, while FP16 is only used for the computationally intensive forward and backward passes.

---

### Question I4
**What is the "generalisation gap" between Adam and SGD, and what architectural or data-regime factors determine which optimiser generalises better?**

**Answer:**

The generalisation gap refers to the empirical observation that SGD with momentum often achieves lower test error than Adam on image classification benchmarks (e.g., ResNet on ImageNet, CIFAR), even when Adam converges to a lower training loss. Wilson et al. (2017) showed that adaptive methods can converge to sharper minima that generalise worse.

**Intuitive explanation:** Adam's per-parameter learning rate effectively normalises the gradient, making all directions of parameter space "equally flat" from the optimiser's perspective. This can cause Adam to explore and settle in sharp, narrow minima that have low training loss but high test error (the loss landscape around a sharp minimum is sensitive to perturbations). SGD's uniform scaling preserves the curvature information: it naturally takes smaller steps in sharp directions and larger steps in flat directions, tending towards flatter minima that generalise better.

**Factors that favour Adam:**
- **Sparse or heterogeneous gradients:** Transformers, embedding tables, and attention matrices have very different gradient scales. Adam's adaptivity is essential.
- **Large-scale language / multimodal pretraining:** The diversity of the data distribution and the depth of the model make uniform learning rates unworkable.
- **Fast convergence requirement:** Adam reaches competitive performance faster in terms of steps, which matters when compute is limited.

**Factors that favour SGD:**
- **Image classification with standard architectures (CNNs):** Gradients are relatively dense and homogeneous across layers. Fine-tuned SGD schedules match or beat Adam.
- **Small datasets with strong augmentation:** SGD's conservatism aligns with the need for regularisation.
- **When a carefully tuned scheduler is available:** SGD's generalisability is most apparent when combined with cosine annealing or step decay; without a schedule, Adam's robustness to learning rate choice is more valuable.

**Practical reconciliation:** For most modern workloads, AdamW with learning rate warmup and cosine decay is the dominant approach. The gap between SGD and Adam has narrowed with better learning rate schedules and weight decay. The choice matters most for small vision models where every fraction of accuracy counts.

---

## Tier 3 -- Advanced

### Question A1
**Derive the bias correction in Adam from first principles. Show that without correction, the expected value of the raw first moment $m_t$ is not $\mathbb{E}[g_t]$ at early steps, and derive the correction factor.**

**Answer:**

**Setting up the recurrence.** The first moment update is:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad m_0 = 0$$

Unrolling the recurrence over $t$ steps:

$$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

**Computing the expectation.** Taking expectations:

$$\mathbb{E}[m_t] = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \mathbb{E}[g_i]$$

Assuming that the expected gradient is stationary over the local window (i.e., $\mathbb{E}[g_i] \approx \mathbb{E}[g]$ for all $i$ in the window), we can factor it out:

$$\mathbb{E}[m_t] = (1 - \beta_1) \mathbb{E}[g] \sum_{i=1}^{t} \beta_1^{t-i}$$

Evaluating the geometric sum:

$$\sum_{i=1}^{t} \beta_1^{t-i} = \sum_{j=0}^{t-1} \beta_1^{j} = \frac{1 - \beta_1^t}{1 - \beta_1}$$

Therefore:

$$\mathbb{E}[m_t] = (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} \cdot \mathbb{E}[g] = (1 - \beta_1^t) \mathbb{E}[g]$$

**The bias.** For the estimate $m_t$ to be an unbiased estimate of $\mathbb{E}[g]$, we need the factor $(1 - \beta_1^t)$ to equal 1. At step $t = 1$:

$$(1 - 0.9^1) = 0.1$$

The raw $m_1$ has expected value $0.1 \, \mathbb{E}[g]$, a $10\times$ underestimate. At $t = 10$: $(1 - 0.9^{10}) \approx 0.651$, still a significant bias.

**Correcting the bias.** Dividing by the bias factor restores the unbiased estimate:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \implies \mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$$

The same argument applies to the second moment $v_t$, yielding $\hat{v}_t = v_t / (1 - \beta_2^t)$.

**When bias correction matters.** At $t = 1000$ with $\beta_1 = 0.9$: $0.9^{1000} \approx 2.66 \times 10^{-46} \approx 0$, so the correction has no effect. With $\beta_2 = 0.999$: $0.999^{1000} \approx 0.368$, so the second-moment bias correction remains significant for the first ~3000 steps. This means that for transformer pre-training where the second moment takes thousands of steps to warm up, bias correction is non-trivially important throughout the early phase of training.

---

### Question A2
**Explain the Adam "sign update" interpretation. How does this relate to the Lion optimiser, and what are the trade-offs of sign-based updates?**

**Answer:**

**Adam as a sign update (approximate).** When the gradient signal is large relative to noise (high SNR), $\hat{m}_t \approx g_t$ and $\sqrt{\hat{v}_t} \approx |g_t|$. In this regime:

$$\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \approx \frac{g_t}{|g_t|} = \text{sign}(g_t)$$

Adam's update approaches the sign of the gradient, with magnitude approximately $\eta$ regardless of the gradient scale. This is why Adam is sometimes described as "approximate sign gradient descent": the update step size is approximately uniform across parameters, controlled only by $\eta$ rather than by the true gradient magnitude.

**The Lion optimiser (Chen et al., 2023).** Lion (Evolved Sign Momentum) formalises this observation. Its update is:

$$c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(interpolate momentum and current grad)}$$
$$\theta_{t+1} = \theta_t - \eta \left( \text{sign}(c_t) + \lambda \theta_t \right) \quad \text{(sign update + weight decay)}$$
$$m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t \quad \text{(update momentum for next step)}$$

Critically: $\beta_1 < \beta_2$ (e.g., $\beta_1 = 0.9$, $\beta_2 = 0.99$), so the "query direction" $c_t$ is more responsive to the current gradient than the stored momentum $m_t$.

**Trade-offs of sign-based updates:**

Advantages:
- Memory efficiency: Lion stores only one momentum vector (vs Adam's two). On a model with $N$ parameters, Adam requires $3N$ floats of state (parameters, $m$, $v$); Lion requires $2N$ (parameters, $m$).
- Computational simplicity: no square root, no division, no second moment.
- Empirically matches or exceeds AdamW quality on vision transformers and large language models while being $\sim 1.5\times$ more memory-efficient.

Disadvantages:
- The learning rate must be tuned differently from Adam. Because the update magnitude is always $\eta$ per step (before weight decay), a Lion learning rate is typically $3$--$10\times$ smaller than the equivalent AdamW learning rate for the same model.
- The weight decay coefficient also interacts differently: since the sign of the weight decay term is added to the sign update, the effective regularisation strength differs.
- Less well-understood theoretically. The convergence guarantees for non-convex objectives are less mature than for Adam.
- Can be less robust on tasks with high gradient noise (very small batches), where the sign of the gradient estimate is frequently wrong.

**Practical guidance:** Lion is a strong alternative to AdamW for large vision and language models when memory is the bottleneck. For reinforcement learning, small-batch regimes, or tasks where gradient noise is high, AdamW remains more reliable.

---

### Question A3
**Describe gradient accumulation in the context of distributed data parallel (DDP) training. What is the interaction between gradient accumulation steps and the `no_sync()` context manager in PyTorch?**

**Answer:**

**DDP gradient synchronisation.** In PyTorch DDP, each GPU computes gradients on its local mini-batch. After each `loss.backward()` call, DDP performs an AllReduce operation across all participating GPUs to average the gradients. This ensures all GPUs take the same optimiser step (equivalent to a single large-batch update).

AllReduce is expensive: it requires ring or tree communication with latency proportional to $2(N-1)/N \times \text{data\_size}$ per parameter. For a model with billions of parameters, AllReduce can dominate training time.

**Interaction with gradient accumulation.** When using $k$ accumulation steps, only the final micro-batch step needs to synchronise gradients. The intermediate $k-1$ steps should accumulate locally without communication:

```python
model = DDP(model)
optimiser.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    # Use no_sync() for all but the last accumulation step
    context = model.no_sync() if (i + 1) % k != 0 else contextlib.nullcontext()

    with context:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels) / k
        scaler.scale(loss).backward()   # local accumulation, no AllReduce

    if (i + 1) % k == 0:
        # AllReduce happens automatically during this backward() because
        # no no_sync() context is active
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad()
```

**What `no_sync()` does.** Normally, DDP hooks into the autograd graph to trigger AllReduce as each parameter's gradient is computed (bucket-based, overlapping communication with backward computation). `no_sync()` disables these hooks for the duration of its context. Gradients accumulate locally in `.grad` buffers without any inter-GPU communication.

When the final micro-batch's `backward()` runs without `no_sync()`, DDP performs AllReduce on the fully accumulated gradients (which are the sum of $k$ micro-batch gradients, equivalent to a full-batch gradient before averaging). The AllReduce averages across GPUs, giving the correct mean gradient over all $k \times B \times \text{num\_GPUs}$ samples.

**Effective batch size calculation:**

$$B_{eff} = B_{micro} \times k \times N_{GPUs}$$

When scaling $B_{eff}$, it is common to apply the linear scaling rule: multiply $\eta$ by $B_{eff}/B_{base}$, where $B_{base}$ is the batch size used in the original hyperparameter recipe. This maintains the effective per-sample learning rate.

**Common mistakes:**
- Forgetting `no_sync()` and performing AllReduce on every micro-step: wastes $k-1$ communication rounds per optimiser step, severely reducing training throughput.
- Clipping gradients per micro-step rather than after full accumulation: clips to a tighter norm than intended, reducing the effective update magnitude.
- Not dividing the loss by $k$: effective batch gradient is $k$ times too large, requiring the learning rate to be reduced by $k$ to compensate (implicit and easy to miss).
