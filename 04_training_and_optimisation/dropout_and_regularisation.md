# Dropout and Regularisation

## Prerequisites
- Overfitting and the bias-variance trade-off
- L1 and L2 norms; the concept of a penalty term in the loss
- Forward and backward pass through a dense layer
- PyTorch `nn.Module` and training loop

---

## Concept Reference

### Overfitting and the Goal of Regularisation

A model overfits when it achieves low training loss but high validation/test loss -- it has memorised the training data rather than learning generalisable patterns. The gap between training and validation loss is the overfit signal. Regularisation techniques add inductive biases or constraints that penalise complexity, encourage robust representations, and reduce this gap.

---

### Weight Decay (L2 Regularisation)

Weight decay adds a penalty term to the loss proportional to the squared $\ell_2$ norm of the parameters:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2} \sum_j \theta_j^2$$

The gradient of this penalised loss:

$$\frac{\partial \mathcal{L}_{total}}{\partial \theta_j} = \frac{\partial \mathcal{L}_{task}}{\partial \theta_j} + \lambda \theta_j$$

The SGD update becomes:

$$\theta_j \leftarrow \theta_j - \eta \frac{\partial \mathcal{L}_{task}}{\partial \theta_j} - \eta \lambda \theta_j = (1 - \eta\lambda)\theta_j - \eta \frac{\partial \mathcal{L}_{task}}{\partial \theta_j}$$

Each update shrinks the parameter by a factor $(1 - \eta\lambda)$ regardless of the gradient, hence the name "weight decay." Large weights are penalised more, encouraging the model to spread weight across many small connections rather than relying on a few large ones.

**L2 vs L1 (Lasso):**

$$\mathcal{L}_{L1} = \mathcal{L}_{task} + \lambda \sum_j |\theta_j|$$

The L1 gradient is $\text{sign}(\theta_j)$, a constant magnitude push towards zero. L1 regularisation encourages exact sparsity (many parameters become exactly zero) because the penalty has equal cost per unit of $|\theta_j|$ regardless of magnitude. L2 encourages small but non-zero weights. L1 is used when feature selection or model compression is desired; L2 is the standard for deep learning.

**Decoupled weight decay (AdamW):** In Adam with L2, the penalty $\lambda \theta_j$ enters the gradient and is scaled by $1/\sqrt{\hat{v}_j}$. This weakens the effective regularisation for parameters with large gradient history. AdamW applies weight decay directly to the parameters without going through the gradient scaling, restoring the intended regularisation strength.

---

### Dropout

Dropout (Srivastava et al., 2014) randomly sets activations to zero during training with probability $p$ (the "dropout rate"), and scales the surviving activations by $1/(1-p)$ to maintain the expected activation magnitude (inverted dropout):

$$\text{Dropout}(x_i) = \begin{cases} 0 & \text{with probability } p \\ x_i / (1-p) & \text{with probability } 1-p \end{cases}$$

At inference, Dropout is disabled and all activations pass through unchanged. The $1/(1-p)$ scaling during training ensures that $\mathbb{E}[\text{Dropout}(x_i)] = x_i$, so the expected activation is the same at train and inference time.

**Why Dropout works -- the ensemble interpretation:**

With $n$ units and Dropout rate $p$, there are $2^n$ possible sub-networks (thinned networks) corresponding to all possible dropout masks. Training with Dropout can be seen as training an exponentially large ensemble of these sub-networks, with shared weights. At inference, the full network with all units active is an approximate geometric mean of all sub-network predictions (under the scaling assumption). This ensemble averaging provides variance reduction and improved generalisation.

**Why Dropout works -- the co-adaptation interpretation:**

Without Dropout, units can develop complex co-dependencies: unit A learns to correct the mistakes of unit B. Such co-adaptation leads to brittle representations. Dropout prevents co-adaptation by forcing each unit to be useful independently, as any unit may be dropped at any time. The result is a more distributed, redundant representation.

**Dropout rates by architecture type:**

| Context | Typical $p$ |
|---|---|
| Dense layers in CNNs | 0.4--0.5 |
| Classifier head (pre-logit) | 0.5 |
| Transformer residual stream | 0.1 |
| Transformer attention weights | 0.0--0.1 |
| Input/embedding Dropout | 0.1--0.2 |

Higher dropout in feed-forward layers; lower in transformer attention to preserve the attention pattern's expressiveness.

---

### DropPath (Stochastic Depth)

DropPath (Larsson et al., 2017; Huang et al., 2016 for stochastic depth) applies Dropout at the level of entire residual blocks rather than individual activations. An entire block is randomly skipped with probability $p$:

$$x_{l+1} = \begin{cases} x_l & \text{with probability } p \quad \text{(block dropped)} \\ x_l + F_l(x_l) & \text{with probability } 1-p \end{cases}$$

This is equivalent to randomly setting the depth of the network: a 12-layer network may effectively become a 6-layer network for a given forward pass. DropPath is the primary regulariser in Vision Transformers (ViT, DeiT, Swin) and is more effective than element-wise Dropout for transformer architectures because:
- It reduces memory and compute for dropped blocks.
- The residual path is always preserved, maintaining gradient flow.
- The effective depth stochasticity acts as a stronger regulariser than element-wise activation noise for large models.

---

### Early Stopping

Early stopping monitors a validation metric (typically validation loss or accuracy) during training and halts training when the metric stops improving. A patience parameter $P$ specifies how many epochs to wait after the last improvement before stopping.

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(max_epochs):
    train(model)
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

model.load_state_dict(torch.load('best_model.pt'))
```

Early stopping implicitly regularises by finding the point on the training trajectory where the model has the best bias-variance trade-off -- enough training to learn the signal, not so much that it memorises noise.

**Trade-off:** Early stopping requires a held-out validation set, reducing the data available for training. When data is scarce, training for the full number of epochs and using other regularisation techniques may be preferable.

---

### Label Smoothing

Standard cross-entropy uses a one-hot target: probability 1 for the correct class, 0 for all others. Label smoothing distributes a small probability mass $\epsilon$ uniformly across all $K$ classes:

$$y_k^{smooth} = \begin{cases} 1 - \epsilon + \epsilon/K & k = \text{correct class} \\ \epsilon / K & k \neq \text{correct class} \end{cases}$$

The smoothed cross-entropy loss:

$$\mathcal{L}_{LS} = (1 - \epsilon) \mathcal{L}_{CE} + \frac{\epsilon}{K} \sum_k \log p_k$$

The second term is the cross-entropy with the uniform distribution, penalising overconfident predictions.

**Why label smoothing helps:**
- Prevents the model from driving the logit of the correct class to $+\infty$ and all others to $-\infty$. Hard targets create an unbounded optimisation objective; label smoothing provides a finite lower bound.
- Improves calibration: the model's output probabilities are more reflective of true class uncertainty.
- Acts as a regulariser for the logit space, preventing the model from becoming overconfident on training examples.

**Typical values:** $\epsilon = 0.1$ for image classification (used in Inception-v4, ViT, DeiT); $\epsilon = 0.1$ for machine translation; lower values ($\epsilon = 0.01$--$0.05$) when the training labels are known to be high quality.

**Caveat:** Label smoothing hurts performance when temperature scaling or distillation is used, because the teacher's soft labels already encode class uncertainty. Applying label smoothing on top of soft targets can conflate the uncertainty signal.

---

### Stochastic Weight Averaging (SWA)

SWA (Izmailov et al., 2018) averages model weights over the last $k$ training steps (or epochs) of a cosine-annealed training run. The averaged weights often correspond to a flatter minimum than any individual checkpoint:

$$\theta_{SWA} = \frac{1}{k} \sum_{i=T-k}^{T} \theta_i$$

After SWA, BatchNorm running statistics must be recomputed by making a forward pass through the training data with the SWA weights. SWA is a simple, low-overhead technique that typically improves validation accuracy by 0.5--1% without additional training.

---

## Tier 1 -- Fundamentals

### Question F1
**Explain what Dropout does during training and inference. Why is the $1/(1-p)$ scaling applied during training (inverted dropout)?**

**Answer:**

**During training:** Each activation $x_i$ is independently set to zero with probability $p$ and retained with probability $(1-p)$. The retained activations are scaled up by $1/(1-p)$:

$$\text{Dropout}(x_i) = \begin{cases} 0 & \text{w.p. } p \\ x_i / (1-p) & \text{w.p. } 1-p \end{cases}$$

**During inference (eval mode):** The Dropout layer is a no-op. All activations pass through unchanged.

**Why the $1/(1-p)$ scaling?**

Without scaling, the expected output of a Dropout layer is:

$$\mathbb{E}[\text{Dropout}(x_i)] = p \cdot 0 + (1-p) \cdot x_i = (1-p) x_i$$

This is $(1-p)$ times the input magnitude. At inference, with no Dropout, the output is $x_i$. The inference output is larger by a factor of $1/(1-p)$, which systematically shifts the activation distribution that subsequent layers were trained on. For a network with several Dropout layers, this compounding mismatch would break inference.

The $1/(1-p)$ scale factor during training corrects the expected value:

$$\mathbb{E}[(1-p) \cdot 0 / (1-p) + (1-p) \cdot x_i/(1-p)] = x_i$$

Now $\mathbb{E}[\text{Dropout}(x_i)] = x_i$, matching the inference output. The subsequent layers see consistent expected activation magnitudes in both modes.

The alternative (non-inverted dropout) scales at inference instead: multiply activations by $(1-p)$ during inference. This is mathematically equivalent but requires the scale factor to be applied at inference, increasing inference cost. Inverted Dropout is standard in PyTorch because it moves the scaling cost to training (acceptable) and leaves inference fast and scale-factor-free.

---

### Question F2
**What is weight decay and how does it affect the loss landscape? Why is it applied to weight matrices but typically not to bias terms or layer norm parameters?**

**Answer:**

Weight decay adds an $\ell_2$ penalty to the total loss:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2} \|\theta\|^2$$

**Effect on the loss landscape:**

The penalty $\frac{\lambda}{2}\|\theta\|^2$ adds a quadratic bowl centred at the origin to the task loss. This:

1. Tilts the loss surface so that the gradient always has a component pointing towards the origin.
2. Constrains the optimal parameters to a ball of radius $\propto 1/\sqrt{\lambda}$. Larger $\lambda$ = tighter ball = more regularisation.
3. Encourages distributed representations: a single weight of 10 incurs the same penalty as 100 weights of 1 (since $10^2 = 100 = 100 \times 1^2$). The model prefers spreading the explanatory power across many small weights.

**Why not apply weight decay to biases?**

Bias parameters control the offset of an activation (e.g., $\text{ReLU}(Wx + b)$). Penalising $b^2$ would push biases towards zero, preventing neurons from learning the appropriate threshold for their activation. The expressive role of biases is different from that of weight matrices: weights capture feature interactions; biases capture intercepts. There is no regularisation benefit from shrinking biases -- a well-generalising model may legitimately need large bias terms.

**Why not apply weight decay to LayerNorm / BatchNorm parameters?**

The $\gamma$ (scale) and $\beta$ (shift) parameters of normalisation layers have a special meaning: they control the overall magnitude and offset of the normalised representation. Penalising $\gamma^2$ would shrink the scale towards zero, counteracting the normalisation layer's learned re-scaling. The number of normalisation parameters ($2 \times d_{model}$ for LayerNorm) is tiny compared to the weight matrices ($d_{model}^2$ for each attention projection), so regularising them has negligible effect on model complexity anyway.

**Standard practice in transformers (e.g., GPT, BERT):**

Apply weight decay only to weight matrices (linear layers, embedding tables). Exclude biases, LayerNorm $\gamma$/$\beta$, and BatchNorm parameters.

```python
decay_params = [p for n, p in model.named_parameters()
                if p.ndim >= 2 and 'norm' not in n]
no_decay_params = [p for n, p in model.named_parameters()
                   if p.ndim < 2 or 'norm' in n]
optimiser = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4)
```

---

### Question F3
**What is label smoothing? Write the smoothed target distribution for a 5-class problem where the true label is class 2, using $\epsilon = 0.1$.**

**Answer:**

Label smoothing replaces the one-hot target with a soft distribution that assigns most of the probability to the correct class and a small amount uniformly to all classes.

With $\epsilon = 0.1$ and $K = 5$ classes, the smoothed target for the correct class ($k=2$):

$$y_k^{smooth} = \begin{cases} 1 - \epsilon + \epsilon/K = 1 - 0.1 + 0.1/5 = 0.92 & k = 2 \text{ (correct)} \\ \epsilon/K = 0.1/5 = 0.02 & k \neq 2 \end{cases}$$

**Smoothed target vector:** $[0.02, 0.02, 0.92, 0.02, 0.02]$ (for classes 0--4 with correct class 2).

**Verify:** $0.92 + 4 \times 0.02 = 0.92 + 0.08 = 1.0$. The probabilities sum to 1.

**Why the "leakage" to other classes helps:**

With a hard target of $[0, 0, 1, 0, 0]$, the cross-entropy loss $-\log p_2$ is minimised by driving $p_2 \to 1$, which requires the logit for class 2 to approach $+\infty$ while all other logits approach $-\infty$. This creates a large magnitude gradient norm even near the optimum.

With a smoothed target of $[0.02, 0.02, 0.92, 0.02, 0.02]$, the cross-entropy is $-0.92 \log p_2 - 4 \times 0.02 \log p_{k \neq 2}$. The loss is minimised at a finite logit value: the model should assign $p_2 = 0.92$ and $p_{k \neq 2} = 0.02$, not $p_2 \to 1$. This prevents over-confident output probabilities and improves calibration.

---

## Tier 2 -- Intermediate

### Question I1
**Compare Dropout, weight decay, and early stopping as regularisation techniques. For each, describe what it constrains and when it is most effective.**

**Answer:**

**Dropout**

What it constrains: the co-adaptation of neurons. Each neuron must be independently useful because any of its co-activating partners may be dropped.

Mechanism: stochastic sparsity in activations. The model trains on an implicit ensemble of sub-networks.

Most effective when:
- The network is large enough to overfit (many parameters relative to data).
- The task involves dense predictions where many features should contribute independently (image classification, language modelling).
- Placed after wide fully-connected layers or within transformer feed-forward blocks.

Less effective when:
- The network is small and already capacity-constrained.
- Convolutional layers with shared kernels (Dropout applied to feature maps is less effective because spatially correlated activations mean dropping one position is correlated with dropping neighbours; DropBlock is preferred).

**Weight decay (L2 regularisation)**

What it constrains: the magnitude of parameters. Forces the model to use many small weights rather than few large ones. Geometrically, constrains the solution to lie within an $\ell_2$ ball.

Mechanism: adds a gradient component towards the origin at every step, regardless of the task loss.

Most effective when:
- The model has many parameters with potential for extreme magnitudes (e.g., attention weight matrices in transformers).
- Combined with AdamW (decoupled) to ensure uniform regularisation across all parameters.
- The data distribution has high signal-to-noise ratio (weight decay is not needed to prevent the model from memorising noise -- it is needed to prevent the model from over-specialising).

Less effective when:
- The optimal weights are sparse (L1 is better) or concentrated in a few parameters.
- Over-applied: too large $\lambda$ causes under-fitting.

**Early stopping**

What it constrains: training duration. Prevents the model from reaching a high-complexity regime of the optimisation trajectory.

Mechanism: uses a held-out validation set as a proxy for generalisation; stops when validation loss stops improving.

Most effective when:
- There is a clear U-shaped validation loss curve (validation loss decreases then increases as training continues).
- The model has sufficient capacity to memorise training data if trained too long.
- Used in combination with a validation set that is representative of the test distribution.

Less effective when:
- The validation loss is monotonically decreasing throughout training (the model has not yet overfitted; more training would help). Early stopping wastes this opportunity.
- The validation set is small and noisy, causing premature stopping due to fluctuations in validation loss.
- The training and validation distributions differ (early stopping based on a misrepresentative validation loss may stop at the wrong point).

---

### Question I2
**Derive the effect of L2 regularisation on the gradient. If $\lambda = 0.01$, $\eta = 0.001$, and the current weight $\theta = 5.0$, what fraction of $\theta$ is retained after one update step (ignoring the task loss gradient)?**

**Answer:**

The L2 regularised loss:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2}\theta^2$$

The gradient with respect to $\theta$:

$$\frac{\partial \mathcal{L}_{total}}{\partial \theta} = \frac{\partial \mathcal{L}_{task}}{\partial \theta} + \lambda\theta$$

The SGD update step (ignoring the task loss gradient, i.e., setting $\partial \mathcal{L}_{task}/\partial\theta = 0$):

$$\theta_{new} = \theta - \eta \lambda \theta = \theta(1 - \eta\lambda)$$

With $\eta = 0.001$ and $\lambda = 0.01$:

$$1 - \eta\lambda = 1 - 0.001 \times 0.01 = 1 - 10^{-5} = 0.99999$$

The weight is reduced to $99.999\%$ of its current value in one step, independent of the actual gradient.

With $\theta = 5.0$:

$$\theta_{new} = 5.0 \times 0.99999 = 4.99995$$

After $k$ steps: $\theta_k = \theta_0 (1 - \eta\lambda)^k$. After 100,000 steps:

$$\theta_{100000} = 5.0 \times (0.99999)^{100000} = 5.0 \times e^{100000 \times \ln(0.99999)} \approx 5.0 \times e^{-1} \approx 1.84$$

The weight decays exponentially towards zero over many steps. The rate of decay depends on $\eta\lambda$: with $\eta = 0.001$ and $\lambda = 0.01$, the "weight decay half-life" is approximately $\ln(2) / (0.001 \times 0.01) \approx 69{,}300$ steps.

**Intuition:** For a large model trained for 300,000 steps, the decay factor per step of $0.99999$ is mild but cumulative. The effective regularisation strength is determined by $\eta\lambda$, not $\lambda$ alone -- halving $\eta$ and doubling $\lambda$ gives the same per-step decay. This is why AdamW's decoupling matters: with Adam's adaptive $\eta_{eff}^{(j)} = \eta / (\sqrt{\hat{v}_j} + \epsilon)$ per parameter, the effective per-step decay would be $\eta_{eff}^{(j)} \lambda$ which varies across parameters.

---

### Question I3
**Explain the DropPath (stochastic depth) technique. How is it different from standard Dropout and why is it preferred for Vision Transformers?**

**Answer:**

**Standard Dropout:** Randomly zeroes individual activations within a layer. For a feature vector $x \in \mathbb{R}^d$, each element $x_i$ is independently zeroed with probability $p$.

**DropPath (Stochastic Depth):** Randomly drops entire residual blocks. For a residual block $F_l$ in a residual network:

$$x_{l+1} = \begin{cases} x_l & \text{with probability } p_l \quad \text{(whole block dropped)} \\ x_l + F_l(x_l) & \text{with probability } 1-p_l \end{cases}$$

The dropout decision is per-sample within the batch: some samples in the batch may have block $l$ active while others have it dropped.

**Why DropPath is preferred for Vision Transformers:**

1. **Residual architecture compatibility:** ViT consists of stacked transformer blocks, each with a residual connection. DropPath preserves the residual path (identity) when a block is dropped, ensuring gradient flow is never broken. Element-wise Dropout on the residual stream would partially corrupt the residual, creating a noisier gradient signal without the clean "skip the block" semantics.

2. **Stronger regularisation per application:** Dropping an entire block removes a full self-attention layer or feed-forward layer, which is a much larger perturbation than dropping a fraction of individual activations. This stronger perturbation is necessary for ViT because:
   - ViTs lack the inductive bias of convolutions (translation equivariance) and can overfit more easily, especially with smaller datasets.
   - The larger perturbation forces more robust attention patterns.

3. **Linear depth scaling:** DropPath rate is often linearly scaled with block depth: $p_l = p_{max} \times l/L$ where $l$ is the layer index and $L$ the total depth. Deeper layers (closer to the output) have higher drop probability. This is intuitive: early layers capture low-level features that are essential; later layers capture high-level task-specific features that benefit from more regularisation.

4. **Efficient implementation:** Dropping an entire block requires masking the block output for selected samples but does not require computing the block for those samples if the mask is applied before the forward pass. In practice, DeiT and Swin implementations compute all blocks but multiply the output by the mask, relying on PyTorch's autograd to handle the zero-gradient case efficiently.

```python
import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, ...) to broadcast over spatial/sequence dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask.float() / keep_prob  # scale to preserve expectation
```

---

## Tier 3 -- Advanced

### Question A1
**Derive the relationship between early stopping with SGD and L2 regularisation. Under what conditions are they equivalent, and where does the equivalence break down?**

**Answer:**

**The equivalence (Bishop, 1995; Goodfellow et al., 2016).**

Consider a quadratic loss near a local minimum $\theta^*$:

$$\mathcal{L}(\theta) \approx \mathcal{L}(\theta^*) + \frac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$$

where $H$ is the Hessian (positive definite near a minimum). The gradient:

$$\nabla \mathcal{L}(\theta) \approx H(\theta - \theta^*)$$

**SGD trajectory:** Starting from $\theta_0 = 0$ (typical initialisation), after $t$ gradient steps with step size $\eta$:

$$\theta_t = \theta^* - (I - \eta H)^t (\theta^* - \theta_0) = \theta^* - (I - \eta H)^t \theta^*$$

In the eigenbasis of $H$ with eigenvalues $\lambda_j$ and eigenvectors $v_j$, expanding $\theta^* = \sum_j c_j v_j$:

$$\theta_t = \sum_j c_j [1 - (1 - \eta \lambda_j)^t] v_j$$

**L2-regularised solution:** With weight decay $\lambda_{wd}$:

$$\theta_{L2}^* = (H + \lambda_{wd} I)^{-1} H \theta^* = \sum_j \frac{\lambda_j}{\lambda_j + \lambda_{wd}} c_j v_j$$

**The correspondence:** Comparing the early-stopping solution $\theta_t$ and the L2 solution $\theta_{L2}^*$, the coefficient of $v_j$ in $\theta_t$ is:

$$1 - (1 - \eta \lambda_j)^t$$

and in $\theta_{L2}^*$ is:

$$\frac{\lambda_j}{\lambda_j + \lambda_{wd}}$$

Setting these equal and solving for $\lambda_{wd}$:

$$\lambda_{wd} \approx \frac{\lambda_j}{t \eta \lambda_j} = \frac{1}{t\eta} \quad \text{(for small } \eta\lambda_j \text{, so } (1-\eta\lambda_j)^t \approx e^{-\eta\lambda_j t}\text{)}$$

The effective weight decay is $\lambda_{wd} \approx 1/(t\eta)$: more training steps $\to$ smaller effective regularisation, corresponding to less L2 penalty. **Fewer training steps $\leftrightarrow$ more weight decay.**

**Where the equivalence breaks down:**

1. **Non-quadratic loss:** The equivalence requires the loss to be well-approximated by a quadratic. Deep neural networks have highly non-convex, non-quadratic losses. The direction in which SGD travels is not a simple function of the Hessian eigenvectors, and the early-stopping trajectory does not correspond to any simple weight-decay constraint.

2. **Mini-batch noise:** SGD with stochastic gradients does not follow the exact gradient descent path assumed in the derivation. The gradient noise introduces a stochastic exploration component that L2 regularisation does not replicate.

3. **Momentum:** SGD with momentum changes the trajectory further, as it accumulates past gradients. The equivalence is derived for gradient descent, not momentum SGD.

4. **Adaptive optimisers:** Adam's parameter-wise learning rates mean that each parameter has a different effective step size, breaking the simple relationship between training steps and effective weight decay.

5. **Interaction with normalisation:** BatchNorm and LayerNorm change the effective loss landscape (by decoupling the scale of parameters from their effect on outputs). Early stopping interacts with this normalisation in ways that weight decay does not replicate.

**Practical implication:** The equivalence is a useful conceptual tool for understanding why both techniques regularise, but it should not be taken as a reason to prefer one over the other. In practice, L2 weight decay is preferred because it provides explicit, interpretable control over regularisation strength and does not require holding out a validation set.

---

### Question A2
**Explain the "Dropout as Bayesian Approximation" interpretation (Gal & Ghahramani, 2016). What does this interpretation imply about obtaining uncertainty estimates from a trained Dropout network?**

**Answer:**

**The Bayesian Deep Learning connection.**

Gal and Ghahramani (2016) showed that a neural network with Dropout applied before every weight layer is mathematically equivalent to a deep Gaussian Process when the network is treated as performing variational inference in a Bayesian framework.

Specifically, optimising the standard Dropout objective (cross-entropy loss + L2 weight decay) is equivalent to maximising the Evidence Lower BOund (ELBO) for a specific Bayesian model where:
- The prior over weights is a mixture of two Gaussians (with one component at zero, corresponding to the "dropped" case).
- The variational approximate posterior is a product of Bernoulli distributions over which weights are active.

The key insight: each random Dropout mask corresponds to a different sample from the approximate posterior over model parameters $q(\theta)$. A forward pass with a specific mask gives the prediction of a specific $\theta$ sampled from $q$.

**Monte Carlo Dropout for uncertainty estimation (MC Dropout).**

If we leave Dropout active at inference and perform $T$ stochastic forward passes, we obtain $T$ predictions $\{y_1, \ldots, y_T\}$:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^T y_t \quad \text{(predictive mean)}$$

$$\text{Var}[\hat{y}] \approx \frac{1}{T}\sum_{t=1}^T y_t^2 - \hat{y}^2 \quad \text{(predictive variance)}$$

The predictive variance is an approximation to the epistemic uncertainty (model uncertainty): high variance means the model's weights are uncertain about this input. This is distinct from aleatoric uncertainty (inherent noise in the data), which is not captured by Dropout variance.

```python
model.train()   # keep Dropout active at inference

T = 50
predictions = torch.stack([model(x) for _ in range(T)])  # (T, batch, classes)
mean = predictions.mean(dim=0)    # (batch, classes)
variance = predictions.var(dim=0)  # (batch, classes)
```

**Practical limitations:**

1. **Dropout is a coarse approximation.** The theoretical equivalence relies on specific architectural choices (Dropout before every weight layer, specific L2 regularisation). Standard architectures (Dropout only in the classifier head, batch norm, residual connections) break the correspondence. The uncertainty estimates are heuristic rather than calibrated posteriors.

2. **Underestimates out-of-distribution uncertainty.** MC Dropout uncertainty reflects variance within the model's learned representations. For inputs far outside the training distribution, the network may produce confident predictions (low variance) because the Dropout masks still activate known features. True Bayesian uncertainty should increase for OOD inputs, but MC Dropout often does not.

3. **Computational cost.** $T$ forward passes at inference is $T\times$ the standard inference cost. For production systems, this is often unacceptable. Cheaper uncertainty estimates (conformal prediction, temperature scaling) are preferred in practice.

4. **Training-time Dropout rate matters.** Using a different Dropout rate at inference than at training (e.g., using $p_{inference} \neq p_{training}$) is not supported by the Bayesian interpretation. The same rate must be used for the uncertainty estimate to be meaningful.

**When MC Dropout is useful:** As a simple, low-engineering-overhead baseline for uncertainty estimation in research settings. It is especially useful for active learning (identifying high-uncertainty samples for labelling) and for anomaly detection in classification.
