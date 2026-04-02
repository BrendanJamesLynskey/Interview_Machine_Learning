# Batch Normalisation

## Prerequisites
- Forward pass through a deep neural network
- Layer-wise activations and the concept of covariate shift
- Basic statistics: mean, variance, normalisation
- Familiarity with CNN and transformer architectures

---

## Concept Reference

### The Problem: Internal Covariate Shift

As a deep network trains, the distribution of each layer's inputs changes as the parameters of the preceding layers change. Ioffe and Szegedy (2015) called this "internal covariate shift." Its consequences:

- Later layers must continuously adapt to shifting input distributions, slowing learning.
- Saturating activations (sigmoid, tanh) receive inputs with drifting means and variances, causing the gradient to vanish in the saturated region.
- The learning rate must be kept small to prevent the distribution shift in one layer from cascading through the network.

Normalisation layers address this by explicitly controlling the statistics of activations at each layer, decoupling the scale of the output of one layer from the sensitivity of the next.

---

### Batch Normalisation (BatchNorm)

For a mini-batch of $m$ samples with activations $\{x_1, \ldots, x_m\}$ at a given layer, Batch Normalisation:

**1. Compute batch statistics:**

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**2. Normalise:**

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

**3. Scale and shift (learnable affine transform):**

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ (scale) and $\beta$ (shift) are learnable parameters, one per feature dimension. Initially $\gamma = 1$, $\beta = 0$, so the layer starts as a pure normaliser. The affine transform allows the network to learn to undo the normalisation if it is not beneficial for a particular layer.

**Running statistics for inference:** During training, BatchNorm uses the mini-batch statistics $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$. These are not available at inference time (a single sample has no batch to average over). PyTorch maintains exponential moving averages:

$$\mu_{running} \leftarrow (1 - \alpha) \mu_{running} + \alpha \mu_\mathcal{B}$$
$$\sigma^2_{running} \leftarrow (1 - \alpha) \sigma^2_{running} + \alpha \sigma_\mathcal{B}^2$$

with default momentum $\alpha = 0.1$. At inference, the running statistics are used:

$$\hat{x} = \frac{x - \mu_{running}}{\sqrt{\sigma^2_{running} + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

This is the **train/eval mode distinction**: `model.train()` uses per-batch statistics; `model.eval()` uses the stored running statistics. Forgetting to call `model.eval()` before inference is a common bug.

**Where BatchNorm is applied in CNNs:** Before (or after) the activation function, over the $(N, C, H, W)$ tensor. The statistics are computed over $(N, H, W)$ for each channel $C$ independently. $\gamma$ and $\beta$ are per-channel parameters of shape $(C,)$.

---

### Layer Normalisation (LayerNorm)

Layer Normalisation computes statistics across the feature dimension for each sample independently:

$$\mu^{(i)} = \frac{1}{H} \sum_{j=1}^{H} x_j^{(i)}, \qquad \sigma^{(i)2} = \frac{1}{H} \sum_{j=1}^{H} \left(x_j^{(i)} - \mu^{(i)}\right)^2$$

$$\hat{x}_j^{(i)} = \frac{x_j^{(i)} - \mu^{(i)}}{\sqrt{\sigma^{(i)2} + \epsilon}}, \qquad y_j^{(i)} = \gamma_j \hat{x}_j^{(i)} + \beta_j$$

where $H$ is the hidden dimension and $i$ indexes the sample. Critically: **LayerNorm statistics do not depend on the batch dimension.** Each sample is normalised independently using its own feature statistics.

**Consequences:**
- LayerNorm works correctly with batch size 1 (or variable batch sizes). BatchNorm requires batch size $\geq 2$ (and typically $\geq 16$ for stable statistics).
- LayerNorm has no running statistics to maintain. Inference mode is identical to training mode (no `eval()` mode distinction for LayerNorm statistics).
- LayerNorm is applied to each token in a transformer independently of the other tokens in the batch or sequence.

**Standard transformer placement:** LayerNorm is applied before the multi-head attention and feed-forward sublayers (Pre-LN), or after (Post-LN). Pre-LN is now dominant (used in GPT-2, LLaMA, Mistral):

```
# Pre-LN (dominant):
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-LN (original transformer):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

Pre-LN is more stable to train (especially without warmup) because the residual path is not normalised, preserving gradient flow through the skip connections.

---

### RMS Normalisation (RMSNorm)

RMSNorm (Zhang & Sennrich, 2019) simplifies LayerNorm by removing the mean subtraction step. It normalises purely by the root mean square:

$$\text{RMS}(x) = \sqrt{\frac{1}{H}\sum_{j=1}^{H} x_j^2}$$

$$\hat{x}_j = \frac{x_j}{\text{RMS}(x) + \epsilon}, \qquad y_j = \gamma_j \hat{x}_j$$

Note: **no bias parameter $\beta$** (no shift term). The only learnable parameters are the scale $\gamma$.

**Why RMSNorm?**
- Re-centering (mean subtraction) in LayerNorm is often redundant in practice, as the model learns to use the $\beta$ parameters to shift the distribution.
- Removing the mean computation saves $\approx 7\%$--$15\%$ of the LayerNorm computation.
- Empirically matches LayerNorm quality on language model benchmarks at reduced cost.

RMSNorm is used in LLaMA, LLaMA-2, Mistral, Falcon, and other large language models. It is the preferred normalisation layer for modern transformer variants where inference efficiency matters.

---

### Instance Normalisation and Group Normalisation

**Instance Normalisation (InstanceNorm):** Computes statistics over $(H, W)$ for each sample and each channel separately. Equivalent to BatchNorm with batch size 1. Used primarily in style transfer (each image is normalised independently, preserving style statistics).

**Group Normalisation (GroupNorm):** Divides channels into $G$ groups and computes statistics over $(H, W)$ and the channels within each group, for each sample independently. GroupNorm with $G = 1$ reduces to LayerNorm (for CNNs); with $G = C$ (one channel per group) it reduces to InstanceNorm. GroupNorm was designed for computer vision tasks where the batch size is constrained by memory (object detection, segmentation), where BatchNorm's statistics become unreliable.

---

### Comparison Table

| Property | BatchNorm | LayerNorm | RMSNorm | GroupNorm |
|---|---|---|---|---|
| Statistics over | Batch dimension | Feature dimension | Feature dimension (RMS only) | Group + spatial |
| Depends on batch size? | Yes | No | No | No |
| Running stats at inference? | Yes | No | No | No |
| Train/eval mode distinction? | Yes | No | No | No |
| Primary use case | CNNs | Transformers, NLP | Large LLMs | Vision, small batches |
| Learnable params | $\gamma, \beta$ per feature | $\gamma, \beta$ per feature | $\gamma$ per feature | $\gamma, \beta$ per group |

---

### Train vs. Eval Mode: Key Implications

The `model.train()` / `model.eval()` call in PyTorch affects BatchNorm (and Dropout) behaviour. For LayerNorm and RMSNorm, there is no difference between train and eval modes in terms of normalisation statistics.

**BatchNorm in train mode:**
- Uses per-mini-batch $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ for normalisation.
- Updates running statistics with the batch statistics.
- The normalised output is slightly different from the "true" normalisation the inference-time running statistics provide, introducing a form of stochastic regularisation (the network must be robust to slight distribution noise).

**BatchNorm in eval mode:**
- Uses frozen running statistics $\mu_{running}$ and $\sigma^2_{running}$.
- Running statistics are not updated.
- The output is deterministic for a given input.

**Common bugs:**
1. **Evaluating with `model.train()`:** Each evaluation mini-batch is normalised by its own statistics rather than the global running statistics. If the evaluation set has a different mean/variance than the training set (e.g., a different image brightness distribution), the per-batch normalisation in train mode may give misleadingly higher accuracy than the true inference accuracy. The running stats capture the training distribution; eval mode uses them correctly.

2. **Frozen parameters but not frozen running stats:** When fine-tuning with some layers frozen (`param.requires_grad = False`), the running statistics of BatchNorm layers can still be updated if the model is in `train()` mode. This can corrupt a well-calibrated pretrained BatchNorm with the statistics of the new fine-tuning dataset. Fix: call `model.eval()` for all BatchNorm layers whose statistics should be frozen.

3. **Small batch sizes with BatchNorm:** With batch size 2--4, $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are noisy estimates. The running statistics will take many steps to converge, and training may be unstable. Switch to GroupNorm or LayerNorm for small-batch regimes.

---

## Tier 1 -- Fundamentals

### Question F1
**What does Batch Normalisation compute? Write the forward pass formula and explain what each learnable parameter does.**

**Answer:**

For a feature with values $\{x_1, \ldots, x_m\}$ across a mini-batch:

**Step 1 -- Batch statistics:**
$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$

**Step 2 -- Normalise:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

After this step, $\hat{x}_i$ has approximately zero mean and unit variance across the batch.

**Step 3 -- Affine transform:**
$$y_i = \gamma \hat{x}_i + \beta$$

**What the parameters do:**
- $\gamma$ (scale / weight in PyTorch): allows the normalised activations to be rescaled. If the optimal standard deviation for the next layer is 2.0 rather than 1.0, the network learns $\gamma = 2.0$. Without $\gamma$, BatchNorm would force all features to unit variance, which may not be optimal.
- $\beta$ (shift / bias in PyTorch): allows the normalised activations to be offset. Without $\beta$, BatchNorm would force all features to zero mean, preventing the network from using a non-zero optimal activation mean for subsequent layers.

Together, $\gamma$ and $\beta$ allow BatchNorm to learn the "best" scale and shift for each feature. They are initialised to $\gamma = 1$ and $\beta = 0$ so that at the start of training BatchNorm performs pure normalisation with no learned transformation.

**Common mistake:** Confusing $\gamma$/$\beta$ with the weight/bias of a preceding linear layer. They are separate, BN-specific parameters. A linear layer followed by BatchNorm should typically not include a bias (the $\beta$ of BatchNorm provides an equivalent shift): `nn.Linear(in, out, bias=False)` followed by `nn.BatchNorm1d(out)`.

---

### Question F2
**Why does calling `model.eval()` matter for BatchNorm? What goes wrong if you forget to call it before inference?**

**Answer:**

In `train()` mode, BatchNorm normalises each input using the statistics of the current mini-batch ($\mu_\mathcal{B}$, $\sigma_\mathcal{B}^2$). In `eval()` mode, it uses the running statistics accumulated during training ($\mu_{running}$, $\sigma^2_{running}$).

**What goes wrong if `model.eval()` is omitted:**

1. **Non-deterministic outputs:** Normalising by batch statistics means the output for the same input depends on what other samples are in the batch. A sample presented alone (batch size 1) will produce different outputs than the same sample in a batch of 32, because the batch mean and variance differ.

2. **Distribution mismatch:** If the test/inference distribution differs from the training distribution (which is common), the per-batch statistics computed from a test mini-batch will not match the running statistics that represent the training distribution. The BatchNorm layers will normalise test inputs to unit variance over the test batch rather than to the training distribution. This breaks the calibration of the subsequent affine transform ($\gamma$, $\beta$), which was learned relative to training-distribution statistics.

3. **Inflated evaluation metrics:** During training it is common to compute a "train accuracy" using forward passes in training mode. The per-batch normalisation introduces noise that acts as a regulariser, making training accuracy artificially lower (the model has to perform well despite variable normalisation statistics). Evaluation in train mode similarly adds noise to the metrics, making it harder to compare evaluation accuracy across different checkpoints.

**Correct pattern:**

```python
model.eval()
with torch.no_grad():
    outputs = model(test_inputs)
    # ...

model.train()   # restore train mode after evaluation
```

The `torch.no_grad()` context additionally disables gradient tracking, saving memory and computation during inference.

---

### Question F3
**Why is LayerNorm preferred over BatchNorm for transformers?**

**Answer:**

**Batch size independence.** Transformers process variable-length sequences and are often trained with small batch sizes (1--8 per GPU before gradient accumulation). BatchNorm requires batch size $\geq 2$ to compute meaningful statistics; with batch size 1, $\sigma_\mathcal{B}^2 = 0$ and the normalisation is undefined (or degenerate). LayerNorm normalises over the feature dimension of each token independently, so it works identically regardless of batch size.

**Autoregressive generation.** At inference time, transformer language models generate one token at a time, giving a batch size of 1 per sequence. BatchNorm cannot be used here at all without relying on running statistics that may not reflect the statistics of individual tokens. LayerNorm handles this naturally since each token's LayerNorm is independent of the batch.

**Sequence padding.** In NLP, sequences in a batch are padded to the same length. Padded positions have artificial activations (zeros or masked values) that would corrupt batch statistics if included. LayerNorm normalises each token separately and is unaffected by padding in other positions.

**Residual stream normalisation.** In the Pre-LN transformer, LayerNorm is applied before sublayers and the output is added to the residual. The normalised value of each token depends only on that token's features, not on the other tokens in the batch. This is the correct inductive bias: the attention mechanism itself handles cross-token interactions; normalisation should not introduce spurious batch-level dependencies.

**No running statistics to maintain.** Deploying a transformer does not require careful calibration of running statistics or ensuring that the evaluation data distribution matches the training distribution for BatchNorm purposes. LayerNorm eliminates this maintenance burden.

---

## Tier 2 -- Intermediate

### Question I1
**Derive the backward pass through Batch Normalisation. Specifically, compute $\partial \mathcal{L} / \partial x_i$ given the upstream gradient $\partial \mathcal{L} / \partial y_i$.**

**Answer:**

Forward pass: $y_i = \gamma \hat{x}_i + \beta$ where $\hat{x}_i = (x_i - \mu_\mathcal{B}) / \sqrt{\sigma_\mathcal{B}^2 + \epsilon}$.

**Step 1: Gradient with respect to $\gamma$ and $\beta$:**

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i, \qquad \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial y_i}$$

**Step 2: Gradient with respect to $\hat{x}_i$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \gamma$$

**Step 3: Gradient with respect to $\sigma_\mathcal{B}^2$ (note: $\sigma^{-3}$ shorthand $s = (\sigma_\mathcal{B}^2 + \epsilon)^{1/2}$):**

$$\frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left(-\frac{1}{2}(x_i - \mu_\mathcal{B})(\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}\right)$$

**Step 4: Gradient with respect to $\mu_\mathcal{B}$:**

$$\frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left(-\frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}\right) + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \left(-\frac{2}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})\right)$$

The second term is zero because $\sum_{i=1}^m (x_i - \mu_\mathcal{B}) = 0$ by definition of $\mu_\mathcal{B}$.

$$\frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} = -\frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i}$$

**Step 5: Gradient with respect to $x_i$ (the full expression):**

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

**Compact form** (substituting the expressions above):

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{m \sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \left[ m \frac{\partial \mathcal{L}}{\partial y_i} - \sum_{j=1}^m \frac{\partial \mathcal{L}}{\partial y_j} - \hat{x}_i \sum_{j=1}^m \frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j \right]$$

**Key insight:** The gradient $\partial \mathcal{L}/\partial x_i$ depends on all $m$ elements of the batch (through the summation terms). This is why BatchNorm introduces inter-sample dependencies during the backward pass, which cannot be trivially parallelised per sample. This dependency also means that the gradient of one sample is implicitly regularised by the other samples in the batch, a form of stochastic regularisation that contributes to BatchNorm's generalisation benefit.

---

### Question I2
**Explain why BatchNorm helps with the vanishing gradient problem. Trace the mechanism through a deep network with sigmoid activations.**

**Answer:**

**The vanishing gradient mechanism (without BatchNorm).**

The sigmoid activation $\sigma(x) = 1/(1+e^{-x})$ has a maximum derivative of $0.25$ (at $x=0$) and approaches zero for $|x| \gg 0$. By the chain rule, the gradient from the output to layer $l$ is:

$$\frac{\partial \mathcal{L}}{\partial W_l} \propto \prod_{k=l}^{L} \sigma'(z_k) \cdot W_k$$

where $z_k$ are the pre-activation values at layer $k$. If the activations at layer $k$ are in the saturated region (large $|z_k|$), then $\sigma'(z_k) \approx 0$ and the product of derivatives collapses towards zero. With 10 layers each contributing $\sigma'(z_k) \leq 0.25$, the product is at most $0.25^{10} \approx 10^{-6}$: gradients vanish.

**How BatchNorm breaks this.**

After the linear transformation $z = Wx + b$, BatchNorm normalises $z$ to have zero mean and unit variance across the batch:

$$\hat{z} = \frac{z - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

Regardless of how large or small $z$ is, $\hat{z}$ is centred at zero. The sigmoid of a zero-mean, unit-variance input has derivative:

$$\sigma'(\hat{z}) \approx \sigma'(0) = 0.25 \quad \text{(for most of the distribution)}$$

BatchNorm actively prevents the pre-activation values from drifting into the saturated regime by re-centering them before each activation. As long as $\gamma$ is not too large (which would rescale $\hat{z}$ into the saturated region), the sigmoid derivatives remain near their maximum throughout training.

**Additionally:** The gradient flowing backwards through the BatchNorm itself is well-conditioned. The normalisation ensures that the upstream gradient $\partial \mathcal{L} / \partial \hat{z}_i$ is not multiplied by extreme values when passed back as $\partial \mathcal{L} / \partial z_i$ (the $1/\sqrt{\sigma_\mathcal{B}^2 + \epsilon}$ factor is $O(1)$ rather than the layer's unnormalised variance which could be very small or very large).

**Practical implication:** BatchNorm allowed training of much deeper networks with sigmoid/tanh activations. Before BatchNorm, very deep networks required careful weight initialisation (Xavier/He) and often used ReLU specifically to avoid vanishing gradients. BatchNorm provides a more direct fix: it actively corrects the distribution rather than relying on carefully calibrated initial weights.

---

### Question I3
**What is the difference between Pre-LN and Post-LN transformer architectures? Why has Pre-LN become dominant, and what trade-off does it introduce?**

**Answer:**

**Post-LN (original transformer, Vaswani et al., 2017):**

```
x -> [Attention] -> add x -> [LayerNorm] -> [FFN] -> add -> [LayerNorm] -> output
```

$$x_{l+1} = \text{LN}(x_l + \text{FFN}(\text{LN}(x_l + \text{Attn}(x_l))))$$

**Pre-LN (GPT-2 and most modern transformers):**

```
x -> [LayerNorm] -> [Attention] -> add x -> [LayerNorm] -> [FFN] -> add x -> output
```

$$x_{l+1} = x_l + \text{FFN}(\text{LN}(x_l + \text{Attn}(\text{LN}(x_l))))$$

**Why Pre-LN is more stable:**

In Post-LN, the gradient flowing back through the network must pass through the LayerNorm at each layer boundary. LayerNorm has a normalisation effect on the gradient, but the residual $x_l$ is not on a "clean" residual path -- it is passed through LayerNorm before being passed to the next layer. This means the gradient through the residual connection is transformed by LayerNorm, which can amplify or suppress gradient components depending on the statistics of the activations.

In practice, Post-LN requires careful learning rate warmup because without warmup, the gradients through the deep stack are poorly conditioned at initialisation. Training instabilities and divergence are more frequent in Post-LN without warmup.

In Pre-LN, the residual path (the $x_l +$ terms) is completely unaffected by LayerNorm. The gradient can flow directly through the skip connections from the output back to any earlier layer without passing through any normalisation. This is analogous to the highway network / ResNet argument for skip connections: the residual path provides a "gradient highway." Pre-LN is much more robust to the absence of warmup and generally trains more stably.

**The trade-off with Pre-LN:**

Pre-LN can underfit relative to Post-LN with sufficient warmup, because:
- The residual stream in Pre-LN can have growing magnitudes over depth. Without normalisation on the output of each block, the residual contributions can compound, and the signal at depth $L$ may be dominated by the raw residual rather than the learned transformations.
- Post-LN normalises the representation at each layer boundary, keeping magnitudes controlled and potentially enabling the network to learn more complex transformations per layer.

**Practical guidance:** Use Pre-LN for training stability, especially for large models without extensive hyperparameter tuning. If final model quality is critical and training stability can be ensured (with careful warmup and gradient clipping), Post-LN may achieve marginally better performance. The GPT family (Pre-LN) and the LLaMA family (Pre-LN with RMSNorm) have demonstrated that Pre-LN is sufficient for state-of-the-art results at scale.

---

## Tier 3 -- Advanced

### Question A1
**Explain Batch Renormalisation (Batch Renorm). What problem does standard BatchNorm have with small batches or non-i.i.d. mini-batches, and how does Batch Renorm fix it?**

**Answer:**

**The problem with standard BatchNorm at small batch sizes.**

Standard BatchNorm normalises activations by the mini-batch statistics $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$. At inference, it uses the running statistics $\mu_{running}$ and $\sigma^2_{running}$ which are exponential moving averages of the training batch statistics.

When the batch size is small (e.g., $m = 2$), the batch statistics are very noisy estimates of the true population statistics. The running averages converge slowly and may not accurately represent the true mean and variance of the feature distribution. Consequently:

- The normalisation during training (using noisy $\mu_\mathcal{B}$) and during inference (using inaccurate running stats) become systematically different.
- The model is trained to produce outputs that are correct when normalised by noisy statistics, but at inference the deterministic running stats produce a different normalised value, degrading performance.

Additionally, in distributed training or reinforcement learning, mini-batches may be drawn from different workers with different data distributions (non-i.i.d.). Each worker has a different $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$, and the running statistics computed per-worker diverge, causing the model to behave differently on different workers' data.

**Batch Renormalisation (Ioffe, 2017).**

Batch Renorm introduces a correction that keeps the training normalisation consistent with the running statistics. The normalised value is:

$$\hat{x}_i^{renorm} = \frac{x_i - \mu_\mathcal{B}}{\sigma_\mathcal{B}} \cdot \frac{\sigma_\mathcal{B}}{\sigma_{running}} + \frac{\mu_\mathcal{B} - \mu_{running}}{\sigma_{running}}$$

Defining $r = \sigma_\mathcal{B} / \sigma_{running}$ (clipped to $[1/r_{max}, r_{max}]$) and $d = (\mu_\mathcal{B} - \mu_{running}) / \sigma_{running}$ (clipped to $[-d_{max}, d_{max}]$):

$$\hat{x}_i^{renorm} = r \hat{x}_i + d$$

where $r$ and $d$ are treated as **constants** for the backward pass (stop-gradient). The forward pass uses the batch statistics (as in standard BN) but is corrected to be closer to what the running-statistics normalisation would give. The affine transform then learns:

$$y_i = \gamma \hat{x}_i^{renorm} + \beta$$

The clipping of $r$ and $d$ prevents the correction from being too large at the start of training (when running stats are inaccurate). Clipping limits are annealed from zero (no correction, equivalent to standard BN) to the final values during training.

**Practical result:** Batch Renorm allows BatchNorm-like training stability even with batch sizes of $m = 2$--$4$, and is more robust to non-i.i.d. batches. The model trained with Batch Renorm has smaller train/inference discrepancy because the running statistics more accurately represent the normalisation seen during training.

---

### Question A2
**In a model that uses both BatchNorm and Dropout, what is the interaction between these two layers, and why can the train/inference performance gap be larger than expected?**

**Answer:**

**Train/inference discrepancy from Dropout.**

In training, Dropout randomly zeroes a fraction $p$ of activations. The expected value of a Dropout-masked activation is $(1-p)x$, and PyTorch compensates by scaling the surviving activations by $1/(1-p)$ (inverted dropout). The variance of the surviving activations after Dropout is:

$$\text{Var}[x \cdot \text{mask}] = (1-p)\text{Var}[x] + p(1-p)\mathbb{E}[x]^2$$

This is higher than the variance of $x$ alone when $p > 0$ (Dropout increases variance because some activations are zero and others are amplified by $1/(1-p)$). At inference, Dropout is disabled, and the variance of the activation is simply $\text{Var}[x]$ -- lower than during training.

**Interaction with BatchNorm.**

BatchNorm placed after Dropout computes $\sigma_\mathcal{B}^2$ from the Dropout-masked activations. Because Dropout increases variance, the training-time $\sigma_\mathcal{B}^2$ is larger than the inference-time variance of the same activations (without Dropout). The running variance $\sigma^2_{running}$ accumulates the inflated training variance.

At inference, Dropout is disabled, so the pre-BatchNorm activations have lower variance. But BatchNorm uses the running statistics (which were computed with Dropout-inflated variance) to normalise them:

$$\hat{x}_{inference} = \frac{x - \mu_{running}}{\sqrt{\sigma^2_{running} + \epsilon}}$$

Since $\sigma^2_{running}$ overestimates the true inference variance, $\hat{x}_{inference}$ is under-normalised: the denominator is too large, so the normalised values are smaller in magnitude than expected. The subsequent $\gamma$ and $\beta$ parameters were learned relative to the training normalisation scale. At inference, the smaller $|\hat{x}|$ values produce different outputs than the trained affine transform expects.

**Recommended fixes:**

1. **Place BatchNorm before Dropout:** Apply BN to the raw (non-masked) activations, then apply Dropout to the normalised output. The BN statistics are now based on the clean activation distribution, not the Dropout-modified one.

   ```python
   nn.Linear(in, out, bias=False),
   nn.BatchNorm1d(out),   # normalise raw activations
   nn.ReLU(),
   nn.Dropout(p=0.5),     # dropout after normalisation + activation
   ```

2. **Do not combine BatchNorm and Dropout in the same block** if the model is small enough that LayerNorm + Dropout is feasible (transformers already use this pattern).

3. **Use a small Dropout rate or use DropPath** (stochastic depth), which applies Dropout at the entire layer level rather than element-wise. DropPath has a different statistical interaction with BatchNorm than element-wise Dropout.

**The broader principle:** Any stochastic regularisation during training that changes the activation distribution will cause a train/inference gap for BatchNorm's running statistics. Techniques like Mixup (which changes label distributions) and input noise augmentation that affects activation means/variances should be considered carefully in conjunction with BatchNorm.
