# Backpropagation Derivation

## Prerequisites
- Calculus: multivariate chain rule, partial derivatives
- Linear algebra: matrix-vector products, Jacobians, transposes
- Neural network forward pass notation (see `neural_network_basics.md`)

---

## Concept Reference

### The Chain Rule

The chain rule is the mathematical engine of backpropagation. For scalar functions:

If $y = f(u)$ and $u = g(x)$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

For a composition of $k$ functions $y = f_k(f_{k-1}(\cdots f_1(x) \cdots))$:

$$\frac{dy}{dx} = \frac{dy}{df_k} \cdot \frac{df_k}{df_{k-1}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

For multivariate functions: if $z = f(\mathbf{u})$ where $\mathbf{u} \in \mathbb{R}^n$ and each $u_i = g_i(\mathbf{x})$:

$$\frac{\partial z}{\partial x_j} = \sum_{i=1}^{n} \frac{\partial z}{\partial u_i} \frac{\partial u_i}{\partial x_j}$$

In matrix form (vector-to-scalar chain rule):

$$\frac{\partial z}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{u}}{\partial \mathbf{x}}\right)^{\!\top} \frac{\partial z}{\partial \mathbf{u}}$$

where $\frac{\partial \mathbf{u}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times m}$ is the Jacobian of $\mathbf{u}$ with respect to $\mathbf{x} \in \mathbb{R}^m$.

### Computational Graphs

A computational graph is a directed acyclic graph (DAG) where:
- **Nodes** represent values (scalars, vectors, or tensors)
- **Edges** represent functional dependencies: an edge from $a$ to $b$ means $b$ is computed from $a$
- **Forward pass** traverses the graph from inputs to outputs, computing values
- **Backward pass** traverses the graph from outputs to inputs, computing gradients

**Example: computing $L = (wx + b - y)^2$**

```
Forward graph:

x ──┐
    ├─► mul(w, x) ──┐
w ──┘               ├─► add(wx, b) ──┐
                b ──┘                ├─► sub(wx+b, y) ──► square ──► L
                                y ──┘
```

Each node stores its forward output for use during the backward pass. Modern frameworks (PyTorch, JAX) build this graph dynamically as operations are performed.

### The Delta (Error Signal) Notation

For a network with loss $\mathcal{L}$, define the **error signal** at layer $l$ as:

$$\boldsymbol{\delta}^{[l]} \triangleq \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \in \mathbb{R}^{n_l}$$

This is the gradient of the loss with respect to the pre-activation at layer $l$. It measures how much each pre-activation contributes to the loss.

The error signal is the central quantity in backpropagation. Once we have $\boldsymbol{\delta}^{[l]}$, the gradients for the parameters are immediate:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^{\!\top}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$

### Full Backpropagation Derivation

Consider an $L$-layer network with:

- Pre-activations: $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
- Activations: $\mathbf{a}^{[l]} = g^{[l]}\!\left(\mathbf{z}^{[l]}\right)$ (applied element-wise)
- Loss: $\mathcal{L} = \mathcal{L}\!\left(\mathbf{a}^{[L]}, \mathbf{y}\right)$

**Step 1: Output layer error signal**

At the output layer, we need $\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}}$.

By the chain rule:

$$\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot g'^{[L]}\!\left(\mathbf{z}^{[L]}\right)$$

where $\odot$ denotes element-wise (Hadamard) multiplication. The term $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}}$ is the gradient of the loss with respect to the final layer's activations, and $g'^{[L]}$ is the derivative of the output activation function.

**Special case (cross-entropy loss + sigmoid output):** The combined gradient simplifies beautifully. If $\mathcal{L} = -y \log \hat{y} - (1-y) \log(1-\hat{y})$ and $\hat{y} = \sigma(z^{[L]})$, then:

$$\delta^{[L]} = \hat{y} - y$$

**Step 2: Backpropagation recurrence**

For any hidden layer $l < L$, we want to express $\boldsymbol{\delta}^{[l]}$ in terms of $\boldsymbol{\delta}^{[l+1]}$.

By the chain rule, the loss depends on $\mathbf{z}^{[l]}$ only through $\mathbf{a}^{[l]}$, which in turn affects $\mathbf{z}^{[l+1]}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]}$$

**Derivation of this step:** The $(j)$-th component of $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}}$ is:

$$\frac{\partial \mathcal{L}}{\partial a_j^{[l]}} = \sum_{k=1}^{n_{l+1}} \frac{\partial \mathcal{L}}{\partial z_k^{[l+1]}} \cdot \frac{\partial z_k^{[l+1]}}{\partial a_j^{[l]}}$$

Since $z_k^{[l+1]} = \sum_j W_{kj}^{[l+1]} a_j^{[l]} + b_k^{[l+1]}$, we have $\frac{\partial z_k^{[l+1]}}{\partial a_j^{[l]}} = W_{kj}^{[l+1]}$.

Therefore:

$$\frac{\partial \mathcal{L}}{\partial a_j^{[l]}} = \sum_{k=1}^{n_{l+1}} \delta_k^{[l+1]} W_{kj}^{[l+1]} = \left[\left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]}\right]_j$$

In vector form: $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]}$

Now applying the chain rule through the activation function:

$$\boldsymbol{\delta}^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right) = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]} \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

This is the backpropagation recurrence. It propagates error signals backwards through the network.

**Step 3: Parameter gradients from error signals**

Given $\boldsymbol{\delta}^{[l]}$, the parameter gradients follow directly.

**Weight gradient:** $\mathcal{L}$ depends on $\mathbf{W}^{[l]}$ only through $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$.

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} \cdot \frac{\partial z_j^{[l]}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} \cdot a_k^{[l-1]}$$

In matrix form:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^{\!\top}$$

This is an outer product: $\boldsymbol{\delta}^{[l]} \in \mathbb{R}^{n_l}$ and $\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1}}$, so $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} \in \mathbb{R}^{n_l \times n_{l-1}}$, matching $\mathbf{W}^{[l]}$.

**Bias gradient:**

$$\frac{\partial \mathcal{L}}{\partial b_j^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} \cdot \frac{\partial z_j^{[l]}}{\partial b_j^{[l]}} = \delta_j^{[l]} \cdot 1 = \delta_j^{[l]}$$

In vector form: $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$

### Complete Backpropagation Algorithm

```
Forward pass:
  Cache z^[l] and a^[l] for all layers l = 1, ..., L

Backward pass:
  1. Compute δ^[L] = ∂L/∂a^[L] ⊙ g'^[L](z^[L])
  
  2. For l = L-1, L-2, ..., 1:
       δ^[l] = (W^[l+1])^T · δ^[l+1]  ⊙  g'^[l](z^[l])
  
  3. For each l = 1, ..., L:
       ∂L/∂W^[l] = δ^[l] · (a^[l-1])^T
       ∂L/∂b^[l] = δ^[l]

Weight update (gradient descent):
  W^[l] ← W^[l] - η · ∂L/∂W^[l]
  b^[l] ← b^[l] - η · ∂L/∂b^[l]
```

### Batched Backpropagation

For a mini-batch of $m$ samples, the forward pass computes $\mathbf{Z}^{[l]} \in \mathbb{R}^{n_l \times m}$ and $\mathbf{A}^{[l]} \in \mathbb{R}^{n_l \times m}$.

The error signal becomes a matrix $\boldsymbol{\Delta}^{[l]} \in \mathbb{R}^{n_l \times m}$ (one error vector per sample).

The batched recurrence is identical in form:

$$\boldsymbol{\Delta}^{[l]} = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\Delta}^{[l+1]} \odot g'^{[l]}\!\left(\mathbf{Z}^{[l]}\right)$$

The weight gradient is now **averaged** over the batch:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \boldsymbol{\Delta}^{[l]} \left(\mathbf{A}^{[l-1]}\right)^{\!\top}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \boldsymbol{\delta}_i^{[l]} = \frac{1}{m} \boldsymbol{\Delta}^{[l]} \mathbf{1}$$

where $\mathbf{1} \in \mathbb{R}^m$ is a vector of ones. Dividing by $m$ gives the gradient of the mean loss across the batch.

### Why Backpropagation is Efficient

A naive approach to computing $\frac{\partial \mathcal{L}}{\partial \theta_i}$ for each parameter $\theta_i$ by finite differences requires two forward passes per parameter: $O(P)$ forward passes for $P$ parameters. This is prohibitive.

Backpropagation computes all $P$ gradients in:
- One forward pass: $O\!\left(\sum_l n_l n_{l-1}\right)$
- One backward pass: $O\!\left(\sum_l n_l n_{l-1}\right)$ (same asymptotic cost)

The backward pass costs approximately $2\text{--}3\times$ a forward pass in practice (due to the additional matrix transposes and stored activations). The total cost is $O(P)$, not $O(P^2)$.

This efficiency is not specific to neural networks. Backpropagation is simply **reverse-mode automatic differentiation**, which efficiently computes the gradient of any scalar output with respect to all inputs of a computational graph. It exploits the chain rule to share intermediate computations.

---

## Tier 1 -- Fundamentals

### Question F1
**State the chain rule for scalar functions and explain how it applies to computing the gradient of a loss function through a two-layer network.**

**Answer:**

The chain rule states: if $y = f(g(x))$, then $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$.

For a two-layer network with loss $\mathcal{L}$, the computation is:

$$\mathbf{x} \xrightarrow{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}} \mathbf{z}^{[1]} \xrightarrow{g^{[1]}} \mathbf{a}^{[1]} \xrightarrow{\mathbf{W}^{[2]}, \mathbf{b}^{[2]}} \mathbf{z}^{[2]} \xrightarrow{g^{[2]}} \hat{y} \xrightarrow{} \mathcal{L}$$

To compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}}$, we apply the chain rule through the entire computation chain:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}$$

Each factor:
- $\frac{\partial \mathcal{L}}{\partial \hat{y}}$: derivative of loss w.r.t. prediction (depends on loss type)
- $\frac{\partial \hat{y}}{\partial z^{[2]}} = g'^{[2]}(z^{[2]})$: derivative of output activation
- $\frac{\partial z^{[2]}}{\partial \mathbf{a}^{[1]}} = \mathbf{W}^{[2]}$: the weight matrix of the second layer
- $\frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} = \text{diag}(g'^{[1]}(\mathbf{z}^{[1]}))$: diagonal Jacobian of element-wise activation
- $\frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}} = \mathbf{x}^{\top}$ (in the relevant sense)

The key insight is that we compute these factors starting from the loss and working backwards, accumulating the chain. The error signal $\boldsymbol{\delta}^{[l]}$ is the accumulated product up to layer $l$, avoiding redundant computation.

---

### Question F2
**What does it mean to "cache" values during the forward pass? Why is this necessary for backpropagation?**

**Answer:**

During the forward pass, the network computes both pre-activations $\mathbf{z}^{[l]}$ and post-activations $\mathbf{a}^{[l]}$ for each layer $l$. "Caching" means storing these intermediate values in memory rather than discarding them.

Backpropagation requires these cached values because:

1. **The activation derivative $g'^{[l]}(\mathbf{z}^{[l]})$** needs $\mathbf{z}^{[l]}$ (the pre-activation). For ReLU, $g'(z) = \mathbf{1}[z > 0]$; to evaluate this at the correct point we need the $\mathbf{z}^{[l]}$ computed during the forward pass.

2. **The weight gradient $\boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^{\top}$** needs $\mathbf{a}^{[l-1]}$ (the input activations to layer $l$). Without caching, we would need to recompute the entire forward pass from layer 0 to layer $l-1$ for every layer we want to update.

**Memory cost of caching:**

For a network with $L$ layers, width $w$, and batch size $m$, the total memory for cached activations is $O(Lwm)$. For large networks and batch sizes, this is the dominant memory cost during training (not the parameters themselves). This is why gradient checkpointing exists: it trades compute for memory by recomputing activations on demand during the backward pass rather than storing them all.

---

### Question F3
**Write the backpropagation equations for the output layer of a binary classifier with sigmoid activation and binary cross-entropy loss. Show that the gradient has a remarkably simple form.**

**Answer:**

**Loss:** Binary cross-entropy

$$\mathcal{L} = -y \log \hat{y} - (1-y) \log(1-\hat{y})$$

**Output:** $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$

**Gradient of loss with respect to $\hat{y}$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Derivative of sigmoid:**

$$\frac{d\hat{y}}{dz} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

**Combining by chain rule:**

$$\delta = \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{d\hat{y}}{dz}$$

$$= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

$$= -y(1-\hat{y}) + (1-y)\hat{y}$$

$$= -y + y\hat{y} + \hat{y} - y\hat{y}$$

$$= \hat{y} - y$$

**The remarkable result:** $\delta = \hat{y} - y$. The gradient of the output pre-activation is simply the prediction error. This is not a coincidence -- it arises because cross-entropy is the natural (conjugate) loss for the sigmoid activation, derived from the exponential family. The $\hat{y}(1-\hat{y})$ in the sigmoid derivative exactly cancels the denominators in the cross-entropy gradient.

**Practical significance:** When $\hat{y} = 0.99$ and $y = 0$, the gradient is $0.99$ -- large, indicating the network made a confident wrong prediction and needs a large update. The gradient is never saturated (stuck near zero) regardless of how confident the wrong prediction is, unlike the MSE + sigmoid combination.

---

## Tier 2 -- Intermediate

### Question I1
**Derive the gradient of the loss with respect to the weight matrix $\mathbf{W}^{[l]}$ and explain the outer product structure. Why must the gradient have the same shape as the weight matrix?**

**Answer:**

We want $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ where $\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$.

The pre-activation is $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, so $z_j^{[l]} = \sum_k W_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]}$.

The loss depends on $W_{jk}^{[l]}$ only through $z_j^{[l]}$ (the $j$-th pre-activation):

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} \cdot \frac{\partial z_j^{[l]}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} \cdot a_k^{[l-1]}$$

Arranging this into a matrix, where the $(j,k)$ entry is $\delta_j^{[l]} \cdot a_k^{[l-1]}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^{\!\top}$$

This is the **outer product** of $\boldsymbol{\delta}^{[l]} \in \mathbb{R}^{n_l}$ and $\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1}}$, yielding a matrix in $\mathbb{R}^{n_l \times n_{l-1}}$.

**Why it must match the shape of $\mathbf{W}^{[l]}$:**

The gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ must have the same shape as $\mathbf{W}^{[l]}$ because a gradient update $\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ requires element-wise subtraction, which requires identical shapes.

Conceptually: $\mathbf{W}^{[l]}$ has $n_l \times n_{l-1}$ scalar parameters, and each scalar parameter has exactly one partial derivative, so the gradient is a matrix of partial derivatives with the same dimensions.

**Interpretation of the outer product:**

$\delta_j^{[l]}$ encodes how much the loss changes per unit change in the $j$-th output's pre-activation. $a_k^{[l-1]}$ is the activation that multiplied $W_{jk}^{[l]}$ in the forward pass. Their product gives the sensitivity of the loss to $W_{jk}^{[l]}$: if the input was large ($a_k^{[l-1]}$ large) and the error signal is large ($\delta_j^{[l]}$ large), the weight contributed substantially to the error and deserves a large update.

---

### Question I2
**Explain the vanishing gradient problem from first principles using the backpropagation recurrence. For a sigmoid-activated network of depth $L$, bound the magnitude of the gradient at the first layer.**

**Answer:**

The backpropagation recurrence is:

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]}\right)^{\!\top} \boldsymbol{\delta}^{[l+1]} \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

For a scalar path from the output to layer 1, the gradient passes through $L-1$ weight matrices and $L-1$ activation derivatives. Consider the $(i,j)$ path (the influence of $z_j^{[1]}$ on $z_i^{[L]}$ through a specific neuron path):

$$\frac{\partial \mathcal{L}}{\partial z_j^{[1]}} \propto \prod_{l=1}^{L-1} W_{\pi(l), \pi(l+1)}^{[l+1]} \cdot g'^{[l]}\!\left(z_{\pi(l)}^{[l]}\right)$$

**Sigmoid derivative bound:**

The sigmoid derivative is $\sigma'(z) = \sigma(z)(1-\sigma(z))$. Since $\sigma(z) \in (0,1)$, by AM-GM inequality:

$$\sigma(z)(1-\sigma(z)) \leq \left(\frac{\sigma(z) + (1-\sigma(z))}{2}\right)^2 = \frac{1}{4}$$

with maximum at $z = 0$. So $\sigma'(z) \leq 0.25$ for all $z$.

After $L-1$ layers, the product of activation derivatives is bounded:

$$\prod_{l=1}^{L-1} g'^{[l]}\!\left(z^{[l]}\right) \leq \left(\frac{1}{4}\right)^{L-1}$$

For a 10-layer sigmoid network, this is at most $(0.25)^9 \approx 3.8 \times 10^{-6}$ -- six orders of magnitude smaller than the gradient at the output layer.

**Combined with weight magnitudes:**

If weights are initialised with standard deviation $\sigma_w$, the spectral norm of $\mathbf{W}^{[l]}$ is typically $O(\sigma_w \sqrt{n})$. If $\sigma_w \sqrt{n} < 1$ (common for small initialisation), gradients shrink exponentially with depth.

**Practical consequence:** With sigmoid activations and 10+ layers, gradients at the early layers are effectively zero. The weights in early layers barely update, meaning the network cannot learn useful low-level features. This was a major barrier to training deep networks before ReLU and residual connections.

---

### Question I3
**Describe the difference between forward-mode and reverse-mode automatic differentiation. Which does backpropagation use, and why is it the correct choice for neural networks?**

**Answer:**

Both modes mechanically apply the chain rule to a computational graph. They differ in the order of accumulation.

**Forward-mode automatic differentiation:**

For each input variable $x_i$, a forward sweep computes the directional derivative $\frac{\partial f_j}{\partial x_i}$ for all outputs $f_j$. To compute the full Jacobian $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ (shape $n_{\text{out}} \times n_{\text{in}}$), one forward sweep is needed per input dimension $x_i$.

Cost: $O(n_{\text{in}})$ forward passes for the full Jacobian.

**Reverse-mode automatic differentiation (backpropagation):**

For each scalar output $y$, a single backward sweep computes $\frac{\partial y}{\partial x_i}$ for all inputs $x_i$. To compute the full Jacobian, one backward sweep is needed per output dimension.

Cost: $O(n_{\text{out}})$ backward passes for the full Jacobian.

**Why reverse mode is correct for neural networks:**

In neural network training, we have:
- $n_{\text{in}} = P$ = number of parameters, potentially billions
- $n_{\text{out}} = 1$ = the scalar loss $\mathcal{L}$

Forward mode would require $P$ forward passes to compute $\frac{\partial \mathcal{L}}{\partial \theta_i}$ for all parameters. This is completely impractical.

Reverse mode requires only **1 backward pass** to compute all $P$ gradients. This is the core reason why neural network training is feasible at all.

**Summary:**

| Mode | Cost (full Jacobian) | Best for |
|---|---|---|
| Forward | $O(n_{\text{in}})$ sweeps | Few inputs, many outputs (e.g., Jacobian-vector products) |
| Reverse | $O(n_{\text{out}})$ sweeps | Many inputs, few outputs (e.g., gradient of a scalar loss) |

For a loss function $\mathcal{L} : \mathbb{R}^P \to \mathbb{R}$, reverse mode reduces the cost from $O(P)$ forward sweeps to a single backward sweep of the same order as one forward pass.

---

## Tier 3 -- Advanced

### Question A1
**Derive the backpropagation equations for a batch normalisation layer embedded within a deeper network. Treat batch norm as a differentiable operation and derive $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, $\frac{\partial \mathcal{L}}{\partial \gamma}$, and $\frac{\partial \mathcal{L}}{\partial \beta}$.**

**Answer:**

**Batch normalisation forward pass** for a batch $\mathbf{x} \in \mathbb{R}^{m}$ (single feature, $m$ samples):

$$\mu = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ and $\beta$ are learnable scale and shift parameters, and $\varepsilon$ is a small constant for numerical stability.

**Computing $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$:**

Since $y_i = \gamma \hat{x}_i + \beta$:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i}$$

**Computing $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ (the difficult part):**

Define $\frac{\partial \mathcal{L}}{\partial y_i} \equiv \mathrm{d}y_i$ (received from upstream). We must propagate through the normalisation.

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \mathrm{d}y_i \cdot \gamma$$

Define $\nu = \sqrt{\sigma^2 + \varepsilon}$ for brevity. Then $\hat{x}_i = \frac{x_i - \mu}{\nu}$.

**Gradient through $\nu$:**

$$\frac{\partial \mathcal{L}}{\partial \nu} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \nu} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left(-\frac{x_i - \mu}{\nu^2}\right)$$

**Gradient through $\sigma^2$:**

$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = \frac{\partial \mathcal{L}}{\partial \nu} \cdot \frac{\partial \nu}{\partial \sigma^2} = \frac{\partial \mathcal{L}}{\partial \nu} \cdot \frac{1}{2\nu}$$

**Gradient through $\mu$:**

$$\frac{\partial \mathcal{L}}{\partial \mu} = \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \left(-\frac{1}{\nu}\right) + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{-2}{m} \sum_{i=1}^m (x_i - \mu)$$

**Final gradient with respect to $x_i$:**

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\nu} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{m} + \frac{\partial \mathcal{L}}{\partial \mu} \cdot \frac{1}{m}$$

**Why this is more complex than a plain linear layer:**

In a plain linear layer, the $j$-th output depends only on the $j$-th input (for element-wise operations). In batch normalisation, each $\hat{x}_i$ depends on the entire batch through $\mu$ and $\sigma^2$. This creates cross-sample dependencies in the backward pass: the gradient at sample $i$ depends on all other samples in the batch. This is why batch normalisation behaves differently at test time (where we use running statistics instead of batch statistics) and why it is incompatible with batch size 1.

---

### Question A2
**Modern deep learning frameworks implement backpropagation through automatic differentiation on dynamic computational graphs. Describe the key data structures and algorithms involved. How does PyTorch's autograd system avoid computing unnecessary gradients?**

**Answer:**

**Key data structures:**

1. **Tensor:** A multi-dimensional array that optionally carries a `grad_fn` attribute pointing to the Function that created it, and a `requires_grad` flag indicating whether gradients should accumulate on it.

2. **Function (Node):** Represents one operation in the graph. Each Function stores:
   - `saved_tensors`: the inputs (or values needed for the backward computation) cached from the forward pass
   - `next_functions`: pointers to the Functions that produced the inputs (the edges of the graph)
   - A `backward()` method that computes the local Jacobian-vector product (gradient of this operation's output w.r.t. its inputs, multiplied by the upstream gradient)

3. **Gradient accumulation buffers:** Each leaf Tensor with `requires_grad=True` has a `.grad` attribute. When multiple paths in the graph flow through a leaf, gradients are summed (added to `.grad`).

**Backward pass algorithm (topological order):**

```python
# Conceptual implementation of loss.backward()
# PyTorch uses a C++ engine, but the algorithm is:

from collections import deque

def backward(loss):
    # loss.grad_fn is the last operation node
    # Topological sort via BFS on the dynamic graph
    
    queue = deque([(loss.grad_fn, torch.tensor(1.0))])  # seed gradient = 1.0
    grad_map = {}  # accumulate gradients at each node
    
    while queue:
        node, upstream_grad = queue.popleft()
        
        # Compute this node's contribution to gradients of its inputs
        local_grads = node.backward(upstream_grad)   # list of gradients, one per input
        
        for (next_fn, _), local_grad in zip(node.next_functions, local_grads):
            if next_fn is not None:
                if next_fn in grad_map:
                    grad_map[next_fn] = grad_map[next_fn] + local_grad
                else:
                    grad_map[next_fn] = local_grad
                    queue.append((next_fn, grad_map[next_fn]))
```

**How PyTorch avoids unnecessary gradients:**

1. **`requires_grad=False` short-circuits the graph.** If an input tensor has `requires_grad=False`, PyTorch does not create a `grad_fn` for operations on it. The computational graph is not built for those operations, saving both memory (no cached intermediates) and compute.

2. **`torch.no_grad()` context manager.** Disables gradient tracking globally within its scope. All operations produce tensors with `requires_grad=False` and no `grad_fn`. Inference is typically wrapped in `torch.no_grad()` to halve memory usage and speed up computation.

3. **Gradient checkpointing (`torch.utils.checkpoint`).** Trades compute for memory: intermediate activations are not cached during the forward pass. During the backward pass, the relevant segment is re-run to recompute the needed intermediates just-in-time. This allows training larger models at the cost of approximately 30% extra computation.

4. **`retain_graph=False` (default).** After `loss.backward()` completes, the computational graph is freed. The stored tensors in each `grad_fn` are released. If you call `loss.backward()` twice without `retain_graph=True`, the second call fails because the graph has been deallocated. This is the default because in standard training you accumulate gradients, update weights, and then build a new graph for the next iteration.
