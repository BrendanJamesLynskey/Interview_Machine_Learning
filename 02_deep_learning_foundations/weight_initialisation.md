# Weight Initialisation

## Prerequisites
- Probability: variance, expectation, normal distribution
- Linear algebra: matrix multiplication, eigenvalues (conceptual)
- Activation functions: ReLU, tanh, sigmoid (see `activation_functions.md`)
- Backpropagation and vanishing/exploding gradients (see `backpropagation_derivation.md`)

---

## Concept Reference

### Why Initialisation Matters

Neural network training is a non-convex optimisation problem. The initial weights determine:

1. **Whether gradients flow at all.** If weights are too small, activations and gradients shrink to zero exponentially with depth (vanishing). If too large, activations explode, causing overflow or gradient explosion.

2. **Symmetry breaking.** If all weights are initialised identically, all neurons in a layer compute the same function and receive the same gradient. They remain identical throughout training. The network behaves as if it has one neuron per layer, not $n$.

3. **Which local minimum is found.** Different initialisations lead gradient descent to different solutions. Good initialisations tend to find flatter, more generalisable minima.

### The Failure of Zero and Constant Initialisation

**Zero initialisation:**

If $\mathbf{W}^{[l]} = \mathbf{0}$, then $\mathbf{z}^{[l]} = \mathbf{b}^{[l]}$ for all inputs. All neurons in layer $l$ receive the same input and compute the same activation. The gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^{\top}$ is the same for every neuron in layer $l$. After the update, all weights in a row of $\mathbf{W}^{[l]}$ remain equal. The network's representational capacity is reduced to that of a single neuron per layer.

**Constant non-zero initialisation ($\mathbf{W}^{[l]} = c\mathbf{I}$):**

Same problem. All neurons remain symmetric throughout training.

**Why biases can be zero:** Biases do not suffer from the symmetry problem, because the gradient $\frac{\partial \mathcal{L}}{\partial b_j^{[l]}} = \delta_j^{[l]}$ depends on $j$ even when the weights are symmetric. In practice, biases are initialised to 0 (or small constants for specific activations, see below).

### Variance Analysis: The Core Requirement

To understand proper initialisation, we analyse how the variance of activations and gradients propagates through the network.

**Setting:** Layer $l$ with $n_{l-1}$ inputs, weights $W_{jk}^{[l]} \overset{\text{iid}}{\sim} \mathcal{D}(0, \sigma_w^2)$, inputs $a_k^{[l-1]}$ i.i.d. with mean 0 and variance $\text{Var}[a^{[l-1]}]$.

**Pre-activation variance (forward pass):**

$$z_j^{[l]} = \sum_{k=1}^{n_{l-1}} W_{jk}^{[l]} a_k^{[l-1]}$$

Since $W_{jk}$ and $a_k$ are independent and zero-mean:

$$\text{Var}\!\left[z_j^{[l]}\right] = \sum_{k=1}^{n_{l-1}} \text{Var}[W_{jk}^{[l]}] \cdot \text{Var}[a_k^{[l-1]}] = n_{l-1} \sigma_w^2 \cdot \text{Var}\!\left[a^{[l-1]}\right]$$

**For the variance to be preserved across layers:** We need $\text{Var}[z^{[l]}] = \text{Var}[a^{[l-1]}]$, which requires:

$$n_{l-1} \sigma_w^2 = 1 \implies \sigma_w^2 = \frac{1}{n_{l-1}}$$

This is the **fan-in** condition.

**Backward pass variance:**

For the gradient $\boldsymbol{\delta}^{[l]} = (\mathbf{W}^{[l+1]})^{\top} \boldsymbol{\delta}^{[l+1]} \odot g'(\mathbf{z}^{[l]})$:

$$\text{Var}\!\left[\delta^{[l]}\right] \approx n_{l+1} \sigma_w^2 \cdot (g'(0))^2 \cdot \text{Var}\!\left[\delta^{[l+1]}\right]$$

For the variance to be preserved: $n_{l+1} \sigma_w^2 (g'(0))^2 = 1$, which requires:

$$\sigma_w^2 = \frac{1}{n_{l+1} (g'(0))^2}$$

This is the **fan-out** condition. The forward and backward conditions generally disagree unless $n_{l-1} = n_{l+1}$.

### Xavier / Glorot Initialisation

**Proposed by:** Xavier Glorot and Yoshua Bengio, 2010 (AISTATS).

**Designed for:** Tanh and sigmoid activations (where $g'(0) \approx 1$ for tanh, $g'(0) = 0.25$ for sigmoid).

**Derivation:**

The forward condition gives $\sigma_w^2 = \frac{1}{n_{\text{in}}}$.

The backward condition gives $\sigma_w^2 = \frac{1}{n_{\text{out}}}$.

Glorot and Bengio compromise by averaging the two conditions:

$$\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

**Uniform variant:** Instead of Gaussian, draw from a uniform distribution that matches this variance. Recall $\text{Var}[\mathcal{U}(-a, a)] = \frac{a^2}{3}$. Setting $\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}}$:

$$a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

So weights are drawn from $\mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$.

**Why it works for tanh:**

For tanh: $\tanh'(0) = 1$. The linearisation $a^{[l]} \approx z^{[l]}$ holds near zero (since $\tanh(z) \approx z$ for small $z$). The variance preserved by $\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ is a good approximation to the true variance (which involves the non-linearity's effect more carefully).

**Why it does not work for ReLU:**

ReLU zeros out half the neurons (those with negative pre-activations). The effective variance after ReLU is approximately half the pre-activation variance:

$$\text{Var}\!\left[\text{ReLU}(z)\right] = \frac{1}{2}\text{Var}[z]$$

(for zero-mean Gaussian $z$). Xavier initialisation does not account for this factor of $\frac{1}{2}$, causing activations to shrink by $\frac{1}{2}$ per layer, leading to vanishing.

### He / Kaiming Initialisation

**Proposed by:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2015 (ICCV, "Delving Deep into Rectifiers").

**Designed for:** ReLU and its variants.

**Derivation:**

For ReLU, the expected value of the activation is:

$$\mathbb{E}\!\left[\text{ReLU}(z)\right] = \mathbb{E}[z \cdot \mathbf{1}[z > 0]] = \frac{1}{2}\mathbb{E}[z \mid z > 0] \cdot P(z > 0)$$

For zero-mean Gaussian $z$: $P(z > 0) = \frac{1}{2}$ and $\mathbb{E}[z \mid z > 0] = \sqrt{\frac{2}{\pi}} \sigma_z$. But for the variance calculation:

$$\text{Var}\!\left[\text{ReLU}(z)\right] = \mathbb{E}\!\left[\text{ReLU}(z)^2\right] - \left(\mathbb{E}\!\left[\text{ReLU}(z)\right]\right)^2$$

For zero-mean $z \sim \mathcal{N}(0, \sigma_z^2)$:

$$\mathbb{E}\!\left[\text{ReLU}(z)^2\right] = \int_0^\infty z^2 \cdot \frac{1}{\sqrt{2\pi}\sigma_z} e^{-z^2/(2\sigma_z^2)} dz = \frac{\sigma_z^2}{2}$$

$$\left(\mathbb{E}[\text{ReLU}(z)]\right)^2 = \left(\frac{\sigma_z}{\sqrt{2\pi}}\right)^2 = \frac{\sigma_z^2}{2\pi}$$

$$\text{Var}[\text{ReLU}(z)] = \frac{\sigma_z^2}{2} - \frac{\sigma_z^2}{2\pi} = \sigma_z^2\left(\frac{1}{2} - \frac{1}{2\pi}\right) \approx 0.432 \sigma_z^2$$

He et al. use the simpler bound (ignoring the squared mean, which is a second-order term):

$$\text{Var}\!\left[\text{ReLU}(z)\right] \approx \frac{1}{2} \text{Var}[z]$$

Substituting into the forward variance condition $\text{Var}[a^{[l]}] = \text{Var}[a^{[l-1]}]$:

$$\frac{1}{2} \text{Var}[z^{[l]}] = \text{Var}[a^{[l-1]}]$$

$$\frac{1}{2} n_{\text{in}} \sigma_w^2 \cdot \text{Var}[a^{[l-1]}] = \text{Var}[a^{[l-1]}]$$

$$\sigma_w^2 = \frac{2}{n_{\text{in}}}$$

**He initialisation:**

$$W \sim \mathcal{N}\!\left(0,\ \frac{2}{n_{\text{in}}}\right)$$

or equivalently $W = \mathcal{N}(0, 1) \cdot \sqrt{\frac{2}{n_{\text{in}}}}$.

**Fan-out variant:** Using the backward pass condition gives $\sigma_w^2 = \frac{2}{n_{\text{out}}}$. The fan-in variant (above) is more commonly used in practice and is PyTorch's default for `torch.nn.Linear` with ReLU.

**For Leaky ReLU** with negative slope $a$:

The factor $\frac{1}{2}$ becomes $\frac{1}{2}(1 + a^2)$ (accounting for the $a$-scaled negative side):

$$\sigma_w^2 = \frac{2}{(1 + a^2) n_{\text{in}}}$$

For $a = 0$ this reduces to standard He initialisation.

### Summary of Initialisation Schemes

| Method | Formula | Designed for | Notes |
|---|---|---|---|
| Zero init | $W = 0$ | Nothing | Symmetry breaking failure |
| Random uniform | $\mathcal{U}(-0.1, 0.1)$ | Small networks (historical) | Not principled |
| Xavier (Normal) | $\mathcal{N}(0, \frac{2}{n_{\text{in}}+n_{\text{out}}})$ | Tanh, sigmoid | Geometric mean of fan-in, fan-out |
| Xavier (Uniform) | $\mathcal{U}(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}})$ | Tanh, sigmoid | Equivalent variance |
| He (fan-in) | $\mathcal{N}(0, \frac{2}{n_{\text{in}}})$ | ReLU | Forward pass stability |
| He (fan-out) | $\mathcal{N}(0, \frac{2}{n_{\text{out}}})$ | ReLU | Backward pass stability |
| Orthogonal init | Random orthogonal matrix | RNNs, very deep | Preserves gradient norms exactly |

### Bias Initialisation

**Default: zero.** Biases do not contribute to the symmetry problem (unlike weights), so initialising to 0 is standard and correct.

**Exceptions:**

1. **ReLU + BatchNorm:** The batch norm layer makes the bias redundant (it is absorbed into the batch norm's $\beta$ parameter). PyTorch's `BatchNorm` disables the affine bias if the preceding linear layer has `bias=False`.

2. **Output layer biases for classification:** Initialising output biases to $\log(\text{class frequency}) - \log(K)$ (prior log-probabilities) can speed up convergence by starting the softmax output close to the marginal class distribution rather than uniform.

3. **LSTM forget gates:** Often initialised to 1.0 or 2.0 to encourage the network to remember information at the start of training (prevents early vanishing through the forget gate).

---

## Tier 1 -- Fundamentals

### Question F1
**Why does zero-weight initialisation fail for neural networks? What goes wrong mathematically?**

**Answer:**

Zero initialisation creates a **symmetry problem** that permanently prevents the network from learning rich representations.

**The mathematical argument:**

With $\mathbf{W}^{[l]} = \mathbf{0}$ and biases $\mathbf{b}^{[l]} = \mathbf{0}$:

Forward pass at layer $l$:
$$\mathbf{z}^{[l]} = \mathbf{0} \cdot \mathbf{a}^{[l-1]} + \mathbf{0} = \mathbf{0}$$
$$\mathbf{a}^{[l]} = g(\mathbf{0}) = \mathbf{c}^{[l]}$$

All neurons output the same constant $g(0)$ regardless of the input (for ReLU: 0; for sigmoid: 0.5; for tanh: 0).

**Gradient update:**

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} \cdot a_k^{[l-1]}$$

If all neurons in layer $l-1$ output the same value, then for any two neurons $j_1, j_2$ in layer $l$:

$$\frac{\partial \mathcal{L}}{\partial W_{j_1 k}^{[l]}} = \delta_{j_1}^{[l]} \cdot a_k^{[l-1]} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial W_{j_2 k}^{[l]}} = \delta_{j_2}^{[l]} \cdot a_k^{[l-1]}$$

We need to show $\delta_{j_1}^{[l]} = \delta_{j_2}^{[l]}$. By the recurrence and the fact that all neurons in every layer have identical activations (by induction), all neurons in any given layer will have the same error signal. Therefore all weight gradients $W_{j_1 k}^{[l]}$ and $W_{j_2 k}^{[l]}$ are equal, and after the update, all weights remain equal.

**Permanent damage:** Unlike a saddle point that training can escape, zero initialisation traps the network in a subspace where all neurons in each layer are permanently identical. The network effectively has one neuron per layer regardless of the specified width. This cannot be escaped by gradient descent alone.

**Key insight:** Random initialisation breaks symmetry by giving each neuron a different starting point. Even small random differences result in different gradients, which then cause the neurons to diverge and specialise.

---

### Question F2
**State Xavier and He initialisation. For a layer with 512 input neurons and 256 output neurons, compute the standard deviation for weights under each scheme. When should you use each?**

**Answer:**

**Xavier (Glorot) initialisation:**

$$\sigma_w = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

For $n_{\text{in}} = 512$, $n_{\text{out}} = 256$:

$$\sigma_w^{\text{Xavier}} = \sqrt{\frac{2}{512 + 256}} = \sqrt{\frac{2}{768}} = \sqrt{0.002604} \approx 0.05103$$

**He initialisation (fan-in):**

$$\sigma_w = \sqrt{\frac{2}{n_{\text{in}}}}$$

For $n_{\text{in}} = 512$:

$$\sigma_w^{\text{He}} = \sqrt{\frac{2}{512}} = \sqrt{0.003906} \approx 0.06250$$

**When to use each:**

| Scheme | Use when |
|---|---|
| Xavier | Tanh or sigmoid activations; symmetric activations where $g'(0) \approx 1$ |
| He | ReLU, Leaky ReLU, PReLU; any activation that zeroes out negative pre-activations |

**Why He gives a larger standard deviation here:**

He is derived knowing that ReLU zeros out half the neurons, halving the effective variance. To compensate for this halving, the initial weight variance is doubled relative to Xavier. He's $\sigma_w^2 = \frac{2}{n_{\text{in}}} = \frac{1}{n_{\text{in}}} \times 2$ whereas Xavier uses approximately $\frac{1}{n_{\text{in}}} \times \frac{2n_{\text{in}}}{n_{\text{in}} + n_{\text{out}}}$, which is approximately $\frac{1}{n_{\text{in}}}$ when $n_{\text{in}} \approx n_{\text{out}}$.

```python
import torch
import torch.nn as nn
import math

# PyTorch built-in initialisers
layer = nn.Linear(512, 256)

# Xavier (used by default for linear layers in PyTorch)
nn.init.xavier_normal_(layer.weight)
print(f"Xavier std: {math.sqrt(2 / (512 + 256)):.5f}")

# He / Kaiming
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
print(f"He std:     {math.sqrt(2 / 512):.5f}")

# Verify empirically
print(f"Xavier weight std: {layer.weight.std().item():.5f}")  # approx 0.051
```

---

### Question F3
**What is the role of the activation function's derivative at zero in the derivation of weight initialisation schemes? Why does Xavier need a correction factor for sigmoid but not tanh?**

**Answer:**

The variance analysis for weight initialisation requires an approximation of how the activation function scales the variance as it maps pre-activations to activations. Near zero input, a differentiable activation $g$ can be linearised:

$$g(z) \approx g(0) + g'(0) \cdot z$$

If $g(0) = 0$ (zero-centred activation), then $g(z) \approx g'(0) \cdot z$ and:

$$\text{Var}[g(z)] \approx (g'(0))^2 \text{Var}[z]$$

For the variance to be preserved through the activation:

$$\sigma_w^2 = \frac{1}{n_{\text{in}} (g'(0))^2}$$

**For tanh:**

$\tanh(0) = 0$ and $\tanh'(0) = 1$. So:

$$\sigma_w^2 = \frac{1}{n_{\text{in}} \cdot 1^2} = \frac{1}{n_{\text{in}}}$$

The balanced Xavier formula $\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ is approximately $\frac{1}{n_{\text{in}}}$ when layers have similar sizes. **No correction factor is needed.**

**For sigmoid:**

$\sigma(0) = 0.5 \neq 0$ and $\sigma'(0) = 0.25$. The linearisation is:

$$\sigma(z) \approx 0.5 + 0.25 z$$

The mean of the activation is 0.5 (non-zero), and the variance scales by $(0.25)^2$:

$$\text{Var}[\sigma(z)] \approx (0.25)^2 \text{Var}[z] = \frac{1}{16} \text{Var}[z]$$

For variance preservation: $\sigma_w^2 = \frac{16}{n_{\text{in}}}$.

Xavier initialisation uses $\sigma_w^2 \approx \frac{1}{n_{\text{in}}}$, which is 16 times too small. **A correction factor of 16 is needed** but is not explicitly included in the standard Xavier formula.

**Why Xavier still works somewhat for sigmoid in shallow networks:** The $g(0) = 0.5$ offset issue (non-zero-centred) is often more problematic than the variance mismatch. In practice, sigmoid hidden units are rarely used in modern deep networks regardless of initialisation -- the vanishing gradient problem dominates. Xavier was primarily designed for tanh and works best there.

---

## Tier 2 -- Intermediate

### Question I1
**Derive the He initialisation variance from first principles for a ReLU-activated layer. Show the full calculation of $\text{Var}[\text{ReLU}(z)]$ for $z \sim \mathcal{N}(0, \sigma^2)$.**

**Answer:**

We want to find $\text{Var}[\text{ReLU}(z)]$ for $z \sim \mathcal{N}(0, \sigma^2)$.

**Step 1: Compute $\mathbb{E}[\text{ReLU}(z)^2]$**

$$\mathbb{E}[\text{ReLU}(z)^2] = \mathbb{E}[z^2 \cdot \mathbf{1}[z > 0]] = \int_0^{\infty} z^2 \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-z^2/(2\sigma^2)} dz$$

Substitute $u = z^2$, or use the symmetry of $z^2$ about 0:

$$= \frac{1}{2} \mathbb{E}[z^2] = \frac{1}{2} \sigma^2$$

since $\int_0^\infty z^2 \cdot \frac{1}{\sqrt{2\pi}\sigma}e^{-z^2/(2\sigma^2)} dz = \frac{1}{2}\int_{-\infty}^\infty z^2 \cdot \frac{1}{\sqrt{2\pi}\sigma}e^{-z^2/(2\sigma^2)} dz = \frac{1}{2}\sigma^2$.

**Step 2: Compute $\mathbb{E}[\text{ReLU}(z)]$**

$$\mathbb{E}[\text{ReLU}(z)] = \mathbb{E}[z \cdot \mathbf{1}[z > 0]] = \int_0^{\infty} z \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-z^2/(2\sigma^2)} dz$$

Let $u = \frac{z^2}{2\sigma^2}$, $du = \frac{z}{\sigma^2} dz$:

$$= \frac{\sigma}{\sqrt{2\pi}} \int_0^{\infty} e^{-u} du = \frac{\sigma}{\sqrt{2\pi}}$$

**Step 3: Compute $\text{Var}[\text{ReLU}(z)]$**

$$\text{Var}[\text{ReLU}(z)] = \mathbb{E}[\text{ReLU}(z)^2] - \left(\mathbb{E}[\text{ReLU}(z)]\right)^2 = \frac{\sigma^2}{2} - \frac{\sigma^2}{2\pi} = \sigma^2 \left(\frac{1}{2} - \frac{1}{2\pi}\right)$$

Numerically: $\frac{1}{2} - \frac{1}{2\pi} \approx 0.5 - 0.159 = 0.341$.

**Step 4: Derive He initialisation**

He et al. use the approximation $\text{Var}[\text{ReLU}(z)] \approx \frac{\sigma^2}{2}$ (ignoring the $\frac{1}{2\pi}$ correction, which is a second-order effect).

For the forward variance to be preserved across layer $l$:

$$\text{Var}[a^{[l]}] = \text{Var}[a^{[l-1]}]$$

$$\frac{1}{2} \text{Var}[z^{[l]}] = \text{Var}[a^{[l-1]}]$$

$$\frac{1}{2} \cdot n_{\text{in}} \sigma_w^2 \cdot \text{Var}[a^{[l-1]}] = \text{Var}[a^{[l-1]}]$$

$$\frac{n_{\text{in}} \sigma_w^2}{2} = 1 \implies \sigma_w^2 = \frac{2}{n_{\text{in}}}$$

**Note on the approximation:** The exact formula would use $\frac{1}{2} - \frac{1}{2\pi} \approx 0.341$ instead of $\frac{1}{2} = 0.5$, giving $\sigma_w^2 = \frac{1}{0.341 \cdot n_{\text{in}}} \approx \frac{2.93}{n_{\text{in}}}$. The difference is small and the $\frac{2}{n_{\text{in}}}$ formula dominates the literature. Empirically, the exact factor makes negligible difference because the variance analysis is itself an approximation (it assumes activations are Gaussian, which is only true at initialisation).

---

### Question I2
**Modern deep learning relies heavily on batch normalisation, which makes the network more robust to weight initialisation. Does this mean initialisation no longer matters? Argue both sides with specific examples.**

**Answer:**

**Argument that initialisation matters less with batch norm:**

Batch normalisation re-centres and re-scales the pre-activations at every layer on every forward pass:

$$\hat{z}_j = \frac{z_j - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

Regardless of how $\mathbf{z}$ was computed, $\hat{\mathbf{z}}$ has approximately zero mean and unit variance. This eliminates:

1. **Vanishing/exploding activations.** Even with constant (non-zero) initialisation, batch norm will rescale activations to a usable range after the first forward pass.
2. **Dead ReLU from poor initialisation.** If batch norm precedes the activation, the normalised pre-activations are near zero, ensuring roughly 50% of ReLUs are active initially.
3. **Sensitivity to weight scale.** Multiplying all weights by a constant factor is corrected by batch norm's normalisation, making the network scale-invariant.

**Evidence:** Batch norm was explicitly introduced (Ioffe and Szegedy, 2015) partly to reduce sensitivity to initialisation, allowing higher learning rates and less careful hyperparameter tuning.

**Argument that initialisation still matters with batch norm:**

1. **Batch norm does not fix symmetry breaking.** Batch norm normalises each neuron's pre-activation independently. If all neurons within a layer are still identical (as with zero or constant initialisation), they receive the same gradient and remain identical. Batch norm cannot break this symmetry. Zero initialisation still fails.

2. **Training dynamics at the start.** In the first few iterations before batch statistics are reliable (especially with small batch sizes), the normalisation is noisy. Good initialisation ensures the initial forward pass is numerically stable and gradients are reasonable even before batch norm has converged.

3. **Without batch norm (transformers, small batch sizes).** Transformers use layer normalisation (not batch norm) or no normalisation. Models trained on single examples (online learning) cannot use batch norm. In these cases, good initialisation remains critical. For example, GPT-type models use specific modified He/Xavier initialisation scaled by $\frac{1}{\sqrt{2L}}$ (where $L$ is the depth) to prevent residual stream growth.

4. **Very deep networks near the initialisation.** Even with batch norm, if weights are initialised with very large variance, the batch statistics may be driven by outlier neurons in the first layer, causing unstable batch norm statistics that take many iterations to stabilise.

**Conclusion:** Batch normalisation reduces but does not eliminate the importance of initialisation. For standard architectures with batch norm, He/Xavier initialisation is the safe default, and it largely works even without careful tuning. For non-standard architectures, large models, or architectures without batch norm (transformers, some GANs), initialisation remains a critical design decision.

---

## Tier 3 -- Advanced

### Question A1
**Transformers use a modified weight initialisation. GPT-2 initialises residual projection layers with $\mathcal{N}(0, \frac{0.02}{\sqrt{2L}})$ where $L$ is the number of layers. Derive from first principles why depth causes variance growth in residual networks and why this specific factor corrects it.**

**Answer:**

**Variance growth in residual networks:**

A residual network adds the block output to the input stream:

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + F^{(l)}(\mathbf{x}^{(l)})$$

After $L$ blocks, the output is:

$$\mathbf{x}^{(L)} = \mathbf{x}^{(0)} + \sum_{l=1}^{L} F^{(l)}(\mathbf{x}^{(l)})$$

If the residual functions $F^{(l)}$ are approximately independent and each contributes variance $V_F$, then:

$$\text{Var}\!\left[\mathbf{x}^{(L)}\right] = \text{Var}\!\left[\mathbf{x}^{(0)}\right] + L \cdot V_F$$

Variance grows **linearly** with depth $L$. For large $L$ (GPT-3 has $L = 96$), this can increase activation variance by a factor of $L$ compared to a single-block network.

**Where the variance comes from in a transformer block:**

Each transformer block has two sub-layers, each contributing a residual:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Attn}(\mathbf{x})$$
$$\mathbf{x} \leftarrow \mathbf{x} + \text{FFN}(\mathbf{x})$$

So there are $2L$ residual additions in total for $L$ transformer layers. The residual stream variance after all layers:

$$\text{Var}[\mathbf{x}^{(2L)}] \approx \text{Var}[\mathbf{x}^{(0)}] + 2L \cdot V_F$$

**The correction: scale projection weights by $\frac{1}{\sqrt{2L}}$**

The projection layers (the output projection of attention and the second linear in FFN) are initialised with standard deviation $\sigma = \frac{c}{\sqrt{2L}}$ where $c$ is a base constant (0.02 in GPT-2).

For a linear projection with $n$ inputs and this initialisation, the output variance is:

$$\text{Var}\!\left[W\mathbf{h}\right] = n \cdot \frac{c^2}{2L} \cdot \text{Var}[h]$$

The total variance after $2L$ residual additions:

$$\text{Var}[\mathbf{x}^{(2L)}] \approx \text{Var}[\mathbf{x}^{(0)}] + 2L \cdot n \cdot \frac{c^2}{2L} \cdot \text{Var}[h] = \text{Var}[\mathbf{x}^{(0)}] + n c^2 \text{Var}[h]$$

The $2L$ factors cancel. The total variance is now **independent of depth** $L$, controlled only by the model width and the constant $c$.

**Intuition:** Without the correction, a 96-layer model has 96 times the residual variance of a 1-layer model. The $\frac{1}{\sqrt{2L}}$ scaling keeps the residual contributions per layer small enough that their sum over $2L$ layers remains bounded. This is analogous to how a sum of $N$ independent variables with variance $\frac{1}{N}$ each has total variance 1.

**In practice:**

```python
# GPT-2 initialisation (simplified)
import torch.nn as nn
import math

def init_weights(module, n_layers):
    if isinstance(module, nn.Linear):
        # Standard weight init for most layers
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_residual_projection(module, n_layers):
    """Special init for residual projection layers (c_proj in attn and FFN)."""
    nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
```

**Connection to $\mu$P (Maximal Update Parameterisation):**

Greg Yang and colleagues generalised this idea in $\mu$P (2022), deriving principled scaling rules for all weights as a function of model width and depth. The GPT-2 initialisation is a special case of a broader principle: parameters whose contributions accumulate across depth should be initialised with variance inversely proportional to the depth of accumulation.
