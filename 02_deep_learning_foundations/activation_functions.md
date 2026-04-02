# Activation Functions

## Prerequisites
- Calculus: derivatives, limits, piecewise functions
- Neural network forward pass and backpropagation (see `backpropagation_derivation.md`)
- Understanding of vanishing/exploding gradients

---

## Concept Reference

### Why Activation Functions Are Necessary

Without non-linear activation functions, composing any number of linear layers produces only a linear transformation (see `neural_network_basics.md`). Non-linear activations allow networks to:

1. Represent non-linear decision boundaries
2. Build hierarchical, compositional feature representations
3. Approximate any continuous function (Universal Approximation Theorem)

The choice of activation function directly affects:
- Gradient magnitude during backpropagation (vanishing/exploding gradients)
- Computational efficiency
- The set of functions the network can represent
- Training speed and stability

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Derivation of derivative:**

$$\sigma'(z) = \frac{d}{dz}\left(1 + e^{-z}\right)^{-1} = -\left(1 + e^{-z}\right)^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$$

$$= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot (1 - \sigma(z))$$

**Properties:**

| Property | Value |
|---|---|
| Range | $(0, 1)$ |
| Max derivative | $0.25$ at $z = 0$ |
| Saturates | Yes, for $\|z\| \gg 0$ |
| Zero-centred output | No (always positive) |
| Computationally expensive | Yes (exponential) |

**When to use:** Output layer of binary classifiers (where the probabilistic interpretation is needed). Almost never in hidden layers of modern networks.

**Problems:**
1. **Vanishing gradients:** $\sigma'(z) \leq 0.25$, so gradients shrink by at least $4\times$ per layer.
2. **Saturated gradients:** For $|z| > 5$, $\sigma'(z) \approx 0$. A neuron in saturation passes nearly zero gradient upstream regardless of the loss.
3. **Non-zero-centred:** Outputs are always positive. Gradients w.r.t. the weight matrix of the next layer are $\boldsymbol{\delta}^{[l+1]} (\mathbf{a}^{[l]})^{\top}$. If all $a_j^{[l]} > 0$, all weight gradients have the same sign as $\boldsymbol{\delta}^{[l+1]}$. This forces weight updates to all be positive or all be negative (zig-zag dynamics in weight space).

### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

**Derivative:**

$$\tanh'(z) = 1 - \tanh^2(z)$$

**Properties:**

| Property | Value |
|---|---|
| Range | $(-1, 1)$ |
| Max derivative | $1.0$ at $z = 0$ |
| Saturates | Yes, for $\|z\| \gg 0$ |
| Zero-centred output | Yes |
| Computationally expensive | Yes |

**Advantage over sigmoid:** Zero-centred output resolves the zig-zag problem. Still suffers from vanishing gradients at saturation (max derivative is 1.0 at $z=0$, but tends to 0 for large $|z|$).

**When to use:** LSTM and GRU gates (where bounded output is desirable), sometimes in RNNs. Rare in modern feedforward networks.

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Derivative:**

$$\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

(Undefined at $z = 0$; in practice, frameworks set the subgradient to 0 at $z = 0$.)

**Properties:**

| Property | Value |
|---|---|
| Range | $[0, +\infty)$ |
| Derivative for $z > 0$ | Exactly 1 (no gradient shrinkage) |
| Saturates | Only on the negative side |
| Zero-centred output | No |
| Computationally cheap | Yes (single comparison and multiply) |
| Sparse activations | Yes ($\sim$50% neurons active for random inputs) |

**Why ReLU transformed deep learning:**

For $z > 0$, $\text{ReLU}'(z) = 1$. Gradients pass through active ReLU neurons without attenuation. This eliminates vanishing gradients on the positive side and enabled training of much deeper networks than sigmoid/tanh.

**The Dead ReLU Problem:**

A ReLU neuron with $z \leq 0$ has zero gradient. If a neuron's pre-activation is consistently negative for all training examples, the weight gradient is always zero and the neuron never updates -- it is "dead." Causes include:

1. **Large learning rates** causing weights to jump to a regime where all inputs produce negative pre-activations.
2. **Negative biases** initialised too large.
3. **Weight initialisation** producing large negative activations early in training.

Once dead, a ReLU neuron cannot recover -- there is no gradient signal to move it back to the positive regime. Dead neurons waste capacity and can be a significant problem in deep networks.

```
Diagnosing dead ReLUs:
  Monitor the fraction of activations == 0 per layer.
  If > 50% of a layer's neurons are always zero across training batches,
  many of those neurons are likely dead.
  
  In PyTorch:
  dead_fraction = (activations == 0).float().mean()
```

**Mitigation:**
- Leaky ReLU or Parametric ReLU (non-zero gradient for $z < 0$)
- Careful learning rate scheduling (learning rate warmup)
- He initialisation (keeps initial activations in a reasonable range)
- Batch normalisation (keeps pre-activations near zero where gradient flows)

### Leaky ReLU and Parametric ReLU

**Leaky ReLU:**

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

with a fixed $\alpha$ (typically 0.01).

**Derivative:**

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$$

No neuron is ever truly dead: even for negative pre-activations, a small gradient $\alpha$ still flows.

**Parametric ReLU (PReLU):**

Identical to Leaky ReLU, but $\alpha$ is a learnable parameter (different per channel). The network learns how much signal to pass for negative activations.

### ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

**Derivative:**

$$\text{ELU}'(z) = \begin{cases} 1 & z > 0 \\ \text{ELU}(z) + \alpha & z \leq 0 \end{cases}$$

**Advantage:** Smooth at $z = 0$ (unlike ReLU, which has a kink). Mean activations pushed towards zero (reduces internal covariate shift without batch norm). **Disadvantage:** Slower to compute than ReLU due to the exponential.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(X \leq z), \quad X \sim \mathcal{N}(0, 1)$$

where $\Phi$ is the standard normal CDF. A widely used approximation:

$$\text{GELU}(z) \approx 0.5z \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715z^3\right)\right)\right)$$

**Derivative (exact):**

$$\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)$$

where $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ is the standard normal PDF.

**Properties:**

- Smooth everywhere (infinitely differentiable)
- Non-monotonic: has a slight dip below 0 for small negative values
- Stochastic interpretation: $\text{GELU}(z) = \mathbb{E}[X \cdot z]$ where $X \sim \text{Bernoulli}(\Phi(z))$ -- it stochastically gates the input
- Used in BERT, GPT-2, GPT-3, ViT, and most modern transformers

**Intuition:** A neuron with a high activation $z$ is likely to be important (high probability $\Phi(z)$), so it passes through nearly unchanged. A neuron with a very negative activation $z$ is unlikely to be relevant and is gated to near zero. The gating is smooth and data-dependent.

### Swish

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

A special case of the SiLU (Sigmoid Linear Unit). Discovered by neural architecture search (Ramachandran et al. 2017).

**Derivative:**

$$\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z)\left(1 + z(1 - \sigma(z))\right)$$

**Properties:**

- Smooth, non-monotonic (slight dip below 0 for small negative values)
- Unbounded above: $\text{Swish}(z) \to \infty$ as $z \to \infty$
- Bounded below: approaches 0 as $z \to -\infty$
- $\text{Swish}(0) = 0$, $\text{Swish}'(0) = 0.5$
- Used in EfficientNet, MobileNetV3, and many modern efficient networks

**Swish vs. GELU:** Closely related. GELU uses the Gaussian CDF as the gate; Swish uses the sigmoid. In practice they perform similarly. Swish is simpler to implement and slightly cheaper to compute.

### Softmax

For a vector $\mathbf{z} \in \mathbb{R}^K$ (multi-class logits):

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties:**
- Output is a probability distribution: all entries positive, sum to 1
- Used exclusively at the output layer of multi-class classifiers
- Amplifies differences between logits: the largest logit dominates

**Numerical stability:** Computing $e^{z_k}$ for large $z_k$ causes overflow. The standard fix:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}, \quad c = \max_j z_j$$

Subtracting $c$ does not change the result (the constant cancels), but prevents overflow.

**Jacobian of softmax:**

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i \left(\delta_{ij} - \text{softmax}(\mathbf{z})_j\right)$$

where $\delta_{ij}$ is the Kronecker delta. This is the $(i,j)$ entry of the Jacobian matrix $\mathbf{J} \in \mathbb{R}^{K \times K}$.

When combined with cross-entropy loss, the Jacobian-vector product simplifies to $\hat{\mathbf{y}} - \mathbf{y}$ (analogous to the sigmoid + binary cross-entropy case).

### Comparison Table

| Activation | Range | Saturates | Dead neurons? | Zero-centred | Use case |
|---|---|---|---|---|---|
| Sigmoid | $(0,1)$ | Yes (both ends) | No | No | Binary output |
| Tanh | $(-1,1)$ | Yes (both ends) | No | Yes | LSTM gates, RNNs |
| ReLU | $[0,\infty)$ | Negative side | Yes | No | Deep networks (default) |
| Leaky ReLU | $\mathbb{R}$ | No | No | No | When dead ReLU is a problem |
| ELU | $(-\alpha,\infty)$ | Negative side (softly) | No | Approximately | Smooth variant of ReLU |
| GELU | $\approx (-0.17,\infty)$ | No | No | No | Transformers, BERT, GPT |
| Swish | $\approx (-0.28,\infty)$ | No | No | No | EfficientNet, MobileNet |
| Softmax | $(0,1)^K$, sums to 1 | N/A | N/A | N/A | Multi-class output only |

---

## Tier 1 -- Fundamentals

### Question F1
**What is the vanishing gradient problem and which activation functions cause it? Explain using the backpropagation recurrence.**

**Answer:**

The vanishing gradient problem occurs when gradients shrink exponentially as they are propagated backwards through a deep network, making early layers receive near-zero gradient signals and therefore fail to learn.

**Mechanism via the backpropagation recurrence:**

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]} \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

For a scalar path through the network, the gradient at layer 1 is proportional to a product of activation derivatives over all $L-1$ layers:

$$\left|\frac{\partial \mathcal{L}}{\partial z^{[1]}}\right| \propto \prod_{l=1}^{L-1} \left|g'^{[l]}(z^{[l]})\right| \cdot \left|W^{[l+1]}\right|$$

**Activations that cause vanishing gradients:**

1. **Sigmoid:** $\sigma'(z) \leq 0.25$. The gradient is multiplied by at most 0.25 per layer. After 10 layers: $(0.25)^{10} \approx 10^{-6}$.

2. **Tanh:** $\tanh'(z) \leq 1.0$, with maximum at $z = 0$. For inputs in the saturated region ($|z| \gg 1$), $\tanh'(z) \approx 0$.

Both sigmoid and tanh **saturate**: when inputs are large in magnitude, the output is nearly flat and the gradient nearly zero. A neuron that receives a strongly positive or strongly negative pre-activation contributes almost nothing to the gradient of earlier layers.

**Why ReLU largely solves this:**

For active neurons ($z > 0$): $\text{ReLU}'(z) = 1$. The gradient passes through with no attenuation. A chain of active ReLU neurons does not cause vanishing gradients. The remaining issue (dead neurons) is a different problem.

---

### Question F2
**What is the dead ReLU problem? Give two scenarios that cause it and two methods to mitigate it.**

**Answer:**

A **dead ReLU neuron** is one whose pre-activation $z^{[l]}_j = \mathbf{W}^{[l]}_j \mathbf{a}^{[l-1]} + b^{[l]}_j \leq 0$ for every input in the training set. Since $\text{ReLU}'(z) = 0$ for $z \leq 0$, the gradient $\frac{\partial \mathcal{L}}{\partial z^{[l]}_j} = 0$ for all training examples. The weight update is zero, so the neuron never changes and remains dead indefinitely.

**Scenario 1: Large learning rate causing weight explosion**

Suppose a weight update step moves the weight vector $\mathbf{W}^{[l]}_j$ by a large amount in a direction that makes most training examples produce a negative pre-activation. This can happen when the gradient is large (confident wrong prediction) and the learning rate is high. After such a step, the neuron fires for very few (or no) training examples, eliminating its gradient for subsequent updates.

**Scenario 2: Negative bias initialisation**

If biases are initialised with a large negative value, the pre-activation $z = \mathbf{w}^{\top} \mathbf{a} + b$ is dominated by $b < 0$ for many inputs, pushing the neuron into the $z \leq 0$ region. If the gradient needed to push $b$ positive is zero (because the neuron is dead), the bias remains large and negative.

**Mitigation 1: Leaky ReLU**

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ 0.01z & z \leq 0 \end{cases}$$

For negative pre-activations, the gradient is $0.01 \neq 0$. The neuron still receives a small but non-zero gradient, allowing it to recover. The recovery is slow (small gradient), but the neuron is not permanently dead.

**Mitigation 2: Learning rate warmup**

Start training with a very small learning rate and gradually increase it over the first few hundred or thousand iterations. This prevents large initial weight updates from immediately killing neurons. The network first moves to a stable region of the loss landscape before taking larger steps.

**Other mitigations:** He initialisation (appropriate scale for ReLU networks), batch normalisation (keeps pre-activations near zero where ReLU is active), ELU or GELU activations.

---

### Question F3
**Why is tanh generally preferred over sigmoid for hidden layers, even though both saturate? When might you still use sigmoid in a hidden layer?**

**Answer:**

**Tanh is preferred for three reasons:**

1. **Zero-centred output.** Tanh outputs are in $(-1, 1)$, centred around 0. Sigmoid outputs are in $(0, 1)$, always positive.

   For a layer receiving tanh activations, the weight gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l+1]}} = \boldsymbol{\delta}^{[l+1]} (\mathbf{a}^{[l]})^{\top}$ has elements that can be positive or negative regardless of the sign of $\boldsymbol{\delta}^{[l+1]}$. With sigmoid activations, all elements of $\mathbf{a}^{[l]}$ are positive, so all weight gradients in $\mathbf{W}^{[l+1]}$ must have the same sign as the corresponding $\delta^{[l+1]}$ element. This sign constraint means the gradient update can only move in an "orthant" of weight space, causing zig-zag oscillations in gradient descent.

2. **Stronger gradient at the origin.** $\tanh'(0) = 1$ vs $\sigma'(0) = 0.25$. A tanh neuron near zero passes gradients four times more strongly than a sigmoid neuron near zero.

3. **Symmetric non-linearity.** Tanh produces a symmetric function around zero, which can be advantageous for representing features with both positive and negative components.

**When you might still use sigmoid in a hidden layer:**

1. **Output gates in LSTMs/GRUs.** Gate activations need to be in $(0, 1)$ to represent a probability of passing information. Tanh is used for content (cells) but sigmoid for the gates themselves.

2. **Binary auxiliary outputs within a network.** If an intermediate layer must produce a binary probability (e.g., an attention mask or a confidence score), sigmoid is appropriate.

3. **Compatibility with pretrained models.** If fine-tuning a model that was trained with sigmoid hidden activations, maintaining the same architecture may be preferred.

---

## Tier 2 -- Intermediate

### Question I1
**GELU is used in almost every modern transformer. Explain its probabilistic interpretation and why it is preferred over ReLU in transformer architectures.**

**Answer:**

**Probabilistic interpretation:**

GELU can be derived as a stochastic regulariser. Consider multiplying the input $z$ by a Bernoulli random variable $m$ where $P(m = 1) = P(X \leq z)$ for $X \sim \mathcal{N}(0,1)$. The expected output is:

$$\mathbb{E}[mz] = z \cdot P(X \leq z) = z \cdot \Phi(z) = \text{GELU}(z)$$

**Interpretation:** GELU gates the input based on how likely it is to be a "useful" activation, where usefulness is measured by comparison to a standard normal distribution. Inputs with a large positive value are almost certainly useful ($\Phi(z) \approx 1$) and pass through unchanged. Inputs with a large negative value are almost certainly not useful ($\Phi(z) \approx 0$) and are zeroed out. The gating is **soft and smooth**, not the hard threshold of ReLU.

This is also related to dropout (which multiplies by $\text{Bernoulli}(p)$), but GELU makes the keep probability input-dependent rather than fixed.

**Why GELU is preferred in transformers over ReLU:**

1. **Smoothness.** ReLU has a non-differentiable kink at $z = 0$. GELU is infinitely differentiable everywhere. This matters in transformers where second-order optimisation methods and careful gradient scaling are used.

2. **Non-monotonicity.** GELU has a slight dip below zero near $z \approx -0.17$. This means a neuron can produce a small negative output even for a non-negative input, giving it richer representational capacity than ReLU, which is always non-negative for positive inputs.

3. **No dead neuron problem.** The GELU gradient is never exactly zero (unlike ReLU for $z < 0$). Neurons cannot permanently die.

4. **Empirical performance.** In transformer benchmarks, GELU consistently outperforms ReLU and Swish. The original BERT paper used GELU based on empirical results on NLP tasks. Subsequent models (GPT, RoBERTa, T5) adopted it and found no reason to switch back.

5. **Interaction with layer normalisation.** Transformers universally use layer normalisation, which keeps pre-activations near zero where GELU and ReLU behave similarly. But GELU's smooth behaviour for larger activations provides a safety net when normalisation is imperfect.

---

### Question I2
**Swish was discovered by neural architecture search rather than designed from first principles. Write out its derivative and explain why its non-monotonic behaviour might be advantageous over a strictly monotonic activation like ReLU.**

**Answer:**

**Swish definition and derivative:**

$$\text{Swish}(z) = z \cdot \sigma(z)$$

$$\text{Swish}'(z) = \sigma(z) + z \cdot \sigma'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z))$$

$$= \sigma(z)\left(1 + z(1 - \sigma(z))\right) = \sigma(z)(1 + z - z\sigma(z))$$

At key points:
- $z = 0$: $\text{Swish}'(0) = 0.5$
- $z \to +\infty$: $\text{Swish}'(z) \to 1$ (like ReLU in the positive regime)
- $z \to -\infty$: $\text{Swish}'(z) \to 0$ (but always positive, unlike ReLU which is 0)

The minimum of Swish is at approximately $z \approx -1.28$, where $\text{Swish}(z) \approx -0.28$.

**Why non-monotonicity might be advantageous:**

1. **Richer function class.** A monotonic function maps each input to a unique output. A non-monotonic function can map two different inputs to the same output and can produce non-monotonic input-output relationships within a single neuron. This potentially increases the expressiveness of each hidden unit.

2. **Self-gating behaviour near zero.** For small positive inputs, Swish is $z \cdot \sigma(z) \approx 0.5z$ -- it attenuates low-confidence activations. For large positive inputs, $\sigma(z) \approx 1$ and it behaves like identity. The network can implicitly learn to "filter" small activations.

3. **Smooth gradient landscape.** The slight dip below zero creates a smoother loss landscape compared to ReLU, which has a discrete boundary at $z = 0$ creating a ridge in the loss surface. Smooth landscapes are easier to navigate with gradient descent.

4. **Consistent gradient flow.** The derivative is always positive (unlike ReLU, which is 0 for $z < 0$), preventing dead neurons while still attenuating negative activations (unlike Leaky ReLU, which passes negative activations with constant slope).

**Caveat:** The advantages of non-monotonicity are empirical and not fully theoretically understood. The discovery by NAS reflects the difficulty of designing activation functions from first principles -- what works is often dataset and architecture dependent.

---

### Question I3
**Derive the numerical stability issue in the softmax function and show how the subtract-max trick resolves it without changing the mathematical result.**

**Answer:**

**The overflow problem:**

For logits $\mathbf{z} = [z_1, \dots, z_K]$, softmax computes $e^{z_k}$ for each $k$. If any $z_k > 709$ (approximately), $e^{z_k}$ overflows a 32-bit float (maximum value $\approx 3.4 \times 10^{38}$, and $e^{709} \approx 8.2 \times 10^{307}$ overflows float32 at $e^{88.7}$). Even without overflow, large values cause significant floating-point precision loss.

**The underflow problem:**

If all logits are very negative (e.g., $\mathbf{z} = [-1000, -1001, -999]$), all $e^{z_k}$ underflow to 0.0 in float32, making the denominator 0 and producing `NaN`.

**The subtract-max trick:**

Let $c = \max_k z_k$. Then:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{z_k - c} \cdot e^c}{\sum_j e^{z_j - c} \cdot e^c} = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}$$

The $e^c$ factor cancels exactly in numerator and denominator. The result is mathematically identical.

**Why this resolves both problems:**

- The shifted values $z_k - c$ are all $\leq 0$ (since $c = \max_k z_k$). So $e^{z_k - c} \in (0, 1]$.
- The maximum shifted value is 0, so $e^0 = 1$, and at least one term in the numerator is exactly 1.
- All exponentials are in $(0, 1]$: no overflow.
- The denominator is at least 1: no underflow.

```python
import numpy as np

def stable_softmax(z):
    """Numerically stable softmax using the subtract-max trick."""
    z_shifted = z - np.max(z)          # shift: all values <= 0
    exp_z = np.exp(z_shifted)          # all in (0, 1]
    return exp_z / np.sum(exp_z)

# Example that would overflow without the trick:
z_large = np.array([1000.0, 1001.0, 999.0])
print(stable_softmax(z_large))  # [0.2689, 0.7311, 0.0999] (approximately)

# Verify: naive softmax would produce inf/inf = NaN
# np.exp(1000.0) = inf in float32
```

**Note on log-softmax:** Computing $\log(\text{softmax}(\mathbf{z}))$ is even more numerically treacherous. The standard trick:

$$\log \text{softmax}(\mathbf{z})_k = z_k - c - \log \sum_j e^{z_j - c}$$

This avoids ever computing the raw softmax probabilities (which might underflow for non-maximum logits before taking the log). PyTorch's `F.log_softmax` and `F.cross_entropy` use this internally.

---

## Tier 3 -- Advanced

### Question A1
**Analyse the gradient dynamics of a deep sigmoid network at initialisation. Show mathematically that if weights are drawn from $\mathcal{N}(0, \sigma_w^2)$, the variance of the pre-activations either explodes or vanishes exponentially with depth, and that there is no single $\sigma_w$ that stabilises both activations and gradients simultaneously for sigmoid.**

**Answer:**

**Forward pass variance:**

Consider layer $l$ with $n$ inputs. The pre-activation is $z_j^{[l]} = \sum_{k=1}^n W_{jk}^{[l]} a_k^{[l-1]}$. Assuming $W_{jk}^{[l]} \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_w^2)$ and $a_k^{[l-1]}$ are independent with mean 0 and variance $\text{Var}[a^{[l-1]}]$:

$$\text{Var}\!\left[z^{[l]}\right] = n \cdot \sigma_w^2 \cdot \text{Var}\!\left[a^{[l-1]}\right]$$

For the activations $a^{[l]} = \sigma(z^{[l]})$, we need $\text{Var}[a^{[l]}]$ in terms of $\text{Var}[z^{[l]}]$.

For small-variance inputs around zero, we can linearise: $\sigma(z) \approx \sigma(0) + \sigma'(0) z = 0.5 + 0.25 z$.

$$\text{Var}\!\left[a^{[l]}\right] \approx (0.25)^2 \text{Var}\!\left[z^{[l]}\right]$$

Substituting back:

$$\text{Var}\!\left[z^{[l]}\right] = n \sigma_w^2 \cdot (0.25)^2 \text{Var}\!\left[z^{[l-1]}\right] = \frac{n \sigma_w^2}{16} \text{Var}\!\left[z^{[l-1]}\right]$$

For stability: $\frac{n \sigma_w^2}{16} = 1 \implies \sigma_w^2 = \frac{16}{n}$, so $\sigma_w = \frac{4}{\sqrt{n}}$.

**Backward pass variance:**

The gradient recurrence $\boldsymbol{\delta}^{[l]} = (\mathbf{W}^{[l+1]})^{\top} \boldsymbol{\delta}^{[l+1]} \odot \sigma'(\mathbf{z}^{[l]})$ gives a variance:

$$\text{Var}\!\left[\delta^{[l]}\right] = n \cdot \sigma_w^2 \cdot (\sigma'(z))^2_{\text{avg}} \cdot \text{Var}\!\left[\delta^{[l+1]}\right]$$

Near $z \approx 0$ with a linearised $\sigma'(z) \approx 0.25$:

$$\text{Var}\!\left[\delta^{[l]}\right] \approx n \sigma_w^2 \cdot (0.25)^2 \cdot \text{Var}\!\left[\delta^{[l+1]}\right]$$

For stability: $n \sigma_w^2 (0.25)^2 = 1 \implies \sigma_w^2 = \frac{16}{n}$

**The key observation:**

For a sigmoid network near zero, the forward and backward stability conditions give the same requirement: $\sigma_w = \frac{4}{\sqrt{n}}$.

**However, the sigmoid mean is not zero.** The sigmoid centred at zero has $\mathbb{E}[\sigma(z)] = 0.5$, not 0. Initialising with $\sigma_w = \frac{4}{\sqrt{n}}$ will push pre-activations into the saturated regime after the first few layers, because the non-zero mean of sigmoid activations builds up. Once in saturation, $\sigma'(z) \approx 0$ rather than 0.25, and both the forward and backward variance analysis break down completely -- variances collapse to near zero.

**The fundamental incompatibility:** The linear approximation holds only near $z = 0$. Sigmoid is not zero-centred ($\mathbb{E}[\sigma(z)] = 0.5$), so pre-activations systematically drift away from zero across layers, rendering the linear approximation invalid. No fixed $\sigma_w$ stabilises a deep sigmoid network for all depths simultaneously.

**Contrast with tanh:** $\mathbb{E}[\tanh(z)] = 0$ for zero-mean inputs (tanh is an odd function). Xavier initialisation ($\sigma_w = \sqrt{1/n}$ or $\sqrt{2/(n_{in} + n_{out})}$) approximately stabilises tanh networks because the zero-mean property keeps the linear approximation valid for more layers. This is why tanh was historically preferred and why Xavier initialisation was designed for tanh.

---

### Question A2
**The SwiGLU activation, used in LLaMA and PaLM, is defined as $\text{SwiGLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}, \mathbf{b}, \mathbf{c}) = \text{Swish}(\mathbf{x}\mathbf{W} + \mathbf{b}) \odot (\mathbf{x}\mathbf{V} + \mathbf{c})$. Explain the gating mechanism, its connection to LSTM gates, and why SwiGLU outperforms a standard FFN layer with the same parameter budget.**

**Answer:**

**The gating mechanism:**

SwiGLU is a **gated linear unit (GLU)** variant. It splits the FFN computation into two parallel linear projections:

- **Gate path:** $\mathbf{g} = \text{Swish}(\mathbf{x}\mathbf{W} + \mathbf{b})$ -- produces a soft gate in $(-0.28, \infty)$
- **Value path:** $\mathbf{v} = \mathbf{x}\mathbf{V} + \mathbf{c}$ -- linear feature computation

The output is their element-wise product: $\text{SwiGLU} = \mathbf{g} \odot \mathbf{v}$.

**Intuition:** The gate path learns which features to pass (Swish output near 1 = pass, near 0 = suppress). The value path learns what values to propagate. The network can adaptively suppress irrelevant features without relying on the downstream weight matrix to cancel them out. This is analogous to attention: the gate provides a data-dependent selection mechanism.

**Connection to LSTM gates:**

In an LSTM, the forget gate $\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$ controls how much of the cell state to retain. It is also a learned, data-dependent gate applied to a value (the cell state). SwiGLU applies the same principle -- data-dependent gating of a learned projection -- within a single feedforward layer rather than across time steps.

The difference: LSTM gates are constrained to $(0,1)$ by the sigmoid, enabling exact forgetting ($g = 0$) or exact retention ($g = 1$). SwiGLU's Swish gate is unbounded above and has a small negative region, providing richer signal but without a hard "closed" state.

**Why SwiGLU outperforms standard FFN with the same parameter budget:**

A standard FFN in a transformer has two weight matrices: an up-projection $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$ and a down-projection $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, giving $2 \cdot d \cdot d_{\text{ff}}$ parameters.

SwiGLU uses three matrices: $\mathbf{W}, \mathbf{V} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and the down-projection, giving $3 \cdot d \cdot d_{\text{ff}}$ parameters with the same $d_{\text{ff}}$. To match parameter count, SwiGLU networks use $d_{\text{ff}} = \frac{2}{3} \cdot 4d$ (the transformer convention is $d_{\text{ff}} = 4d$), giving $3 \cdot d \cdot \frac{8d}{3} = 8d^2$ parameters -- the same as the standard FFN.

At matched parameter count, SwiGLU empirically outperforms ReLU and GELU-activated FFNs on language modelling (Noam Shazeer, 2020; LLaMA, 2023). The reasons are:

1. **Multiplicative interactions.** $\mathbf{g} \odot \mathbf{v}$ creates multiplicative interactions between two independent linear projections, increasing expressiveness beyond what a single projection followed by a pointwise non-linearity can achieve.

2. **Smoother gradient flow.** The Swish gate has a non-zero, smooth gradient everywhere. Combined with the linear value path, the overall gradient is well-conditioned.

3. **Implicit regularisation.** The gate can learn to zero out noisy features, effectively providing a learned dropout mechanism that adapts to each input.
