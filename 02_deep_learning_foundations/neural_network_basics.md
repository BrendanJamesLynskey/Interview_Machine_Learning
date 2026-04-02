# Neural Network Basics

## Prerequisites
- Linear algebra: matrix multiplication, dot products, vector norms
- Calculus: partial derivatives, chain rule
- Probability: conditional probability, expectation
- Python/NumPy: array operations, broadcasting

---

## Concept Reference

### The Perceptron

The perceptron is the simplest model of an artificial neuron, proposed by Rosenblatt in 1958. It takes a real-valued input vector, computes a weighted sum, adds a bias, and passes the result through a step function.

For an input $\mathbf{x} \in \mathbb{R}^n$, weights $\mathbf{w} \in \mathbb{R}^n$, and bias $b \in \mathbb{R}$:

$$z = \mathbf{w}^{\top}\mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

$$\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

The perceptron learning rule updates weights when a prediction is wrong:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$$
$$b \leftarrow b + \eta (y - \hat{y})$$

where $\eta > 0$ is the learning rate, $y$ is the true label, and $\hat{y}$ is the prediction.

**The fundamental limitation:** A single perceptron can only learn linearly separable functions. It cannot represent XOR. This observation (Minsky and Papert, 1969) temporarily killed neural network research until multilayer networks revived the field.

### Multilayer Perceptron (MLP)

An MLP is a feedforward network with at least one hidden layer. With non-linear activations, an MLP can represent non-linear decision boundaries.

**Layer notation convention:**

- $L$ = total number of layers (input layer is layer 0, output is layer $L$)
- $n^{[l]}$ = number of units in layer $l$
- $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ = weight matrix for layer $l$
- $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$ = bias vector for layer $l$
- $\mathbf{a}^{[l]} \in \mathbb{R}^{n^{[l]}}$ = activation vector at layer $l$
- $\mathbf{a}^{[0]} = \mathbf{x}$ = the input

**Forward pass equations for layer $l$:**

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

where $g^{[l]}$ is the activation function at layer $l$, applied element-wise.

**Example: Two-layer MLP (one hidden layer)**

```
Input:   x ∈ R^{n_0}
Hidden:  z^[1] = W^[1] x + b^[1],   a^[1] = relu(z^[1])   in R^{n_1}
Output:  z^[2] = W^[2] a^[1] + b^[2],   a^[2] = sigma(z^[2])  in R^{n_2}
```

Parameter count:
- Layer 1: $n_1 \times n_0$ weights + $n_1$ biases
- Layer 2: $n_2 \times n_1$ weights + $n_2$ biases
- Total: $n_1(n_0 + 1) + n_2(n_1 + 1)$

### Universal Approximation Theorem

**Informal statement:** A feedforward network with a single hidden layer containing a sufficient number of neurons, and a non-polynomial (Borel measurable) activation function, can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision.

**Formal statement (Cybenko 1989, Hornik 1991):** Let $\varphi : \mathbb{R} \to \mathbb{R}$ be a non-constant, bounded, continuous function (a sigmoid-like squashing function). Then for any $f \in C([0,1]^n)$ (continuous function on the unit hypercube) and any $\varepsilon > 0$, there exist $N \in \mathbb{N}$, weights $\alpha_i, w_{ij}$, and biases $\theta_i$ such that:

$$\left| f(\mathbf{x}) - \sum_{i=1}^{N} \alpha_i \, \varphi\!\left(\mathbf{w}_i^{\top} \mathbf{x} - \theta_i\right) \right| < \varepsilon \quad \forall \mathbf{x} \in [0,1]^n$$

**What the theorem does NOT say:**

| Common Misconception | Reality |
|---|---|
| We can find the approximating network by training | The theorem is an existence proof only. Gradient descent may not find the approximating parameters. |
| A single hidden layer is sufficient in practice | $N$ may need to be exponentially large. Depth is far more efficient. |
| The theorem tells us how to set the architecture | It says nothing about the number of neurons needed for a given error tolerance and function. |
| Any activation function works | Only non-polynomial activations. A purely linear network collapses to a single linear transform regardless of depth. |

**The deeper version (depth):** Depth exponentially increases expressivity. Some functions representable by a depth-$k$ network of polynomial size require exponentially many neurons in a depth-$(k-1)$ network. This is why deep networks dominate in practice.

### The Forward Pass

The forward pass computes $\hat{\mathbf{y}}$ from input $\mathbf{x}$ by sequentially applying each layer's affine transform and non-linearity.

**Batch forward pass (vectorised over $m$ samples):**

Let $\mathbf{X} \in \mathbb{R}^{n_0 \times m}$ be the input matrix (each column is one sample).

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

The bias $\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$ is broadcast across the $m$ columns (NumPy broadcasting).

$$\mathbf{A}^{[l]} = g^{[l]}\!\left(\mathbf{Z}^{[l]}\right) \quad \text{applied element-wise}$$

The output of the final layer $\mathbf{A}^{[L]}$ is the prediction $\hat{\mathbf{Y}}$.

**Computational complexity of one forward pass:**

For a layer with $n_l$ output units and $n_{l-1}$ input units, the matrix multiply $\mathbf{W}^{[l]} \mathbf{A}^{[l-1]}$ costs $O(n_l \cdot n_{l-1} \cdot m)$ operations. The total cost is $O\!\left(m \sum_{l=1}^{L} n_l \cdot n_{l-1}\right)$.

---

## Tier 1 -- Fundamentals

### Question F1
**What is a perceptron and what is its fundamental limitation? Why do we need multiple layers?**

**Answer:**

A perceptron computes $\hat{y} = \text{step}(\mathbf{w}^{\top}\mathbf{x} + b)$. It is a linear binary classifier: its decision boundary is always a hyperplane $\mathbf{w}^{\top}\mathbf{x} + b = 0$.

The fundamental limitation is that it can only solve linearly separable problems. XOR is the canonical counterexample:

```
Input (x1, x2)  |  XOR output
(0, 0)          |  0
(0, 1)          |  1
(1, 0)          |  1
(1, 1)          |  0
```

No single line (or hyperplane in higher dimensions) can separate the 0s from the 1s here. The two 1s are at diagonally opposite corners.

Adding a hidden layer solves this. An MLP can represent XOR with two hidden units:

```
h1 = relu(x1 + x2 - 0.5)      # fires if at least one input is active
h2 = relu(x1 + x2 - 1.5)      # fires only if both inputs are active
output = step(h1 - h2 - 0.5)  # XOR = (at least one) AND NOT (both)
```

More generally, multiple layers with non-linear activations allow the network to compose non-linear transformations, building up complex feature representations layer by layer. Depth allows a hierarchical representation: early layers detect simple features (edges), later layers combine them into complex structures (objects).

---

### Question F2
**Write out the forward pass equations for a three-layer MLP with ReLU hidden activations and sigmoid output. Specify the dimensions of every matrix and vector.**

**Answer:**

Network: input $\mathbf{x} \in \mathbb{R}^{d}$, hidden layer 1 with $h_1$ units, hidden layer 2 with $h_2$ units, output layer with 1 unit (binary classification).

**Layer 1 (hidden):**

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$

- $\mathbf{W}^{[1]} \in \mathbb{R}^{h_1 \times d}$, $\mathbf{b}^{[1]} \in \mathbb{R}^{h_1}$, $\mathbf{z}^{[1]} \in \mathbb{R}^{h_1}$

$$\mathbf{a}^{[1]} = \text{ReLU}\!\left(\mathbf{z}^{[1]}\right) = \max\!\left(0, \mathbf{z}^{[1]}\right) \in \mathbb{R}^{h_1}$$

**Layer 2 (hidden):**

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

- $\mathbf{W}^{[2]} \in \mathbb{R}^{h_2 \times h_1}$, $\mathbf{b}^{[2]} \in \mathbb{R}^{h_2}$, $\mathbf{z}^{[2]} \in \mathbb{R}^{h_2}$

$$\mathbf{a}^{[2]} = \text{ReLU}\!\left(\mathbf{z}^{[2]}\right) \in \mathbb{R}^{h_2}$$

**Layer 3 (output):**

$$z^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + b^{[3]}$$

- $\mathbf{W}^{[3]} \in \mathbb{R}^{1 \times h_2}$, $b^{[3]} \in \mathbb{R}$, $z^{[3]} \in \mathbb{R}$

$$\hat{y} = \sigma\!\left(z^{[3]}\right) = \frac{1}{1 + e^{-z^{[3]}}} \in (0, 1)$$

**Common mistake:** Confusing the weight matrix orientation. With the convention $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, the weight matrix has shape (output units, input units). Some textbooks use the transpose convention -- always check which convention a source uses before reading its derivations.

---

### Question F3
**What does the Universal Approximation Theorem guarantee, and what does it not guarantee? Why do deep networks outperform wide shallow networks in practice despite the theorem?**

**Answer:**

**What it guarantees:** For any continuous function on a compact domain and any target approximation error $\varepsilon > 0$, there exists a one-hidden-layer network that approximates it within $\varepsilon$.

**What it does not guarantee:**

1. **Learnability.** The theorem is an existence result. It says a network with the right weights can approximate the function. It says nothing about whether gradient descent will find those weights, or how much data is needed to learn them.

2. **Efficiency.** The required number of hidden units $N$ may be exponentially large in the input dimension. A single hidden layer approximating a function like parity over $d$ bits needs $O(2^d)$ neurons.

3. **Generalisation.** A network that fits the training data may not generalise. The theorem says nothing about sample complexity or the bias-variance trade-off.

**Why depth outperforms width:**

1. **Exponential expressivity.** A function that requires exponentially many neurons in a depth-2 network can be represented with polynomially many neurons in a deeper network. The classic example is computing the parity function: depth $O(\log n)$ with $O(n \log n)$ neurons suffices.

2. **Compositional structure.** Natural data (images, language, audio) has hierarchical structure. A cat is made of parts, which are made of edges. Deep networks can learn to compose simple functions, matching this structure efficiently.

3. **Smoother loss landscapes in practice.** Empirically, deeper networks with modern regularisation (batch norm, residual connections) converge to better solutions more reliably than very wide shallow networks.

4. **Parameter efficiency.** A deep network with $k$ layers of width $w$ has $O(kw^2)$ parameters but exponentially greater representational capacity than a two-layer network with the same parameter budget.

---

### Question F4
**How many parameters does a fully-connected MLP with input size 784, hidden layers of size [512, 256], and output size 10 have? Show your calculation.**

**Answer:**

Parameters are the weights and biases in each layer.

**Layer 1: input (784) -> hidden (512)**

- Weights: $784 \times 512 = 401,408$
- Biases: $512$
- Subtotal: $401,920$

**Layer 2: hidden (512) -> hidden (256)**

- Weights: $512 \times 256 = 131,072$
- Biases: $256$
- Subtotal: $131,328$

**Layer 3: hidden (256) -> output (10)**

- Weights: $256 \times 10 = 2,560$
- Biases: $10$
- Subtotal: $2,570$

**Total: $401,920 + 131,328 + 2,570 = 535,818$ parameters**

```python
import numpy as np

layers = [784, 512, 256, 10]
total = 0
for i in range(1, len(layers)):
    weights = layers[i - 1] * layers[i]
    biases  = layers[i]
    print(f"Layer {i}: {layers[i-1]}x{layers[i]} = {weights} weights + {biases} biases")
    total += weights + biases

print(f"Total parameters: {total}")
# Layer 1: 784x512 = 401408 weights + 512 biases
# Layer 2: 512x256 = 131072 weights + 256 biases
# Layer 3: 256x10  = 2560   weights + 10  biases
# Total parameters: 535818
```

**Practical implication:** This is a modest network by modern standards. GPT-3 has 175 billion parameters. The parameter count of an MLP scales quadratically with width but only linearly with depth for fixed width -- another reason depth is preferred over width for a fixed parameter budget.

---

## Tier 2 -- Intermediate

### Question I1
**Explain why a deep network with linear activations everywhere is equivalent to a single-layer linear model, regardless of depth. What does this imply about activation function choice?**

**Answer:**

Consider a depth-$L$ network where every activation function $g^{[l]}(z) = z$ (identity). The forward pass becomes:

$$\mathbf{a}^{[L]} = \mathbf{W}^{[L]} \mathbf{a}^{[L-1]} + \mathbf{b}^{[L]}$$
$$= \mathbf{W}^{[L]} \left(\mathbf{W}^{[L-1]} \mathbf{a}^{[L-2]} + \mathbf{b}^{[L-1]}\right) + \mathbf{b}^{[L]}$$
$$= \mathbf{W}^{[L]} \mathbf{W}^{[L-1]} \cdots \mathbf{W}^{[1]} \mathbf{x} + \text{(combined bias term)}$$

The product of matrices $\mathbf{W}^{[L]} \cdots \mathbf{W}^{[1]}$ is just another matrix $\mathbf{W}_{\text{eff}}$ of the same shape as a single layer connecting the input to the output. No matter how many layers you add, the composed transform is still a single affine function:

$$\mathbf{a}^{[L]} = \mathbf{W}_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}$$

**Implication:** Non-linear activations are not optional -- they are the mechanism by which depth adds representational capacity. Without non-linearity, a network with 100 layers is identical in representational power to logistic regression.

**Choosing an activation function:**
- Output layer: sigmoid (binary classification), softmax (multiclass), linear (regression)
- Hidden layers: ReLU and its variants (GELU, Swish) for nearly all modern deep networks
- Avoid sigmoid/tanh in deep hidden layers due to vanishing gradients

---

### Question I2
**A fully-connected MLP takes a 32x32 RGB image as input. What preprocessing steps are required before the image can be fed into the first layer? What is the computational disadvantage of using an MLP for images compared to a CNN?**

**Answer:**

**Preprocessing steps:**

1. **Flatten the input.** The image must be reshaped from shape $(3, 32, 32)$ (channels, height, width) to a 1D vector of length $3 \times 32 \times 32 = 3072$.

2. **Normalise pixel values.** Raw pixel values are integers in $[0, 255]$. Scale to $[0, 1]$ (divide by 255) or standardise to zero mean and unit variance:
   $$x_{\text{norm}} = \frac{x - \mu_{\text{channel}}}{\sigma_{\text{channel}}}$$
   where $\mu$ and $\sigma$ are computed over the training set per channel.

3. **Type conversion.** Convert to float32 (not uint8) before matrix operations.

```python
# PyTorch preprocessing for a batch of images
import torch
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),                             # uint8 [H,W,C] -> float32 [C,H,W] in [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet channel means
                std=[0.229, 0.224, 0.225]),   # ImageNet channel stds
    T.Lambda(lambda x: x.flatten())           # [3, 32, 32] -> [3072]
])
```

**Computational disadvantage of MLP vs CNN:**

The first MLP layer maps $\mathbb{R}^{3072} \to \mathbb{R}^{h_1}$, requiring $3072 \times h_1$ weights. For $h_1 = 4096$, that is $\sim$12.6 million weights in the first layer alone. Each of these weights connects one pixel to one hidden unit with no assumption about spatial locality.

A CNN's first layer instead uses a $3 \times 3 \times 3$ convolutional kernel (27 weights per filter, shared across all spatial positions). 64 such filters have only $27 \times 64 = 1,728$ learnable parameters, regardless of the image size.

**Deeper problems with MLPs for images:**

- **No translation invariance.** A learned feature at position (5,5) does not automatically generalise to position (10,10). Each position has its own weights.
- **No locality inductive bias.** Neighbouring pixels are highly correlated, but an MLP treats all pairs of input pixels equally.
- **Poor scaling.** A 224x224 image needs a $150,528$-dimensional input vector. The first layer alone would have $> 600$ million parameters for a moderate-width hidden layer.

---

### Question I3
**Describe the representational difference between a network of depth $d$ with width $w$ versus a network of depth 1 with width $w^d$. In terms of expressivity, which is more powerful?**

**Answer:**

**Depth-$d$, width-$w$ network:**

The depth-$d$ network composes $d$ non-linear functions, each of width $w$. The key property is that **composition allows reuse of computed features**. A neuron in layer 3 can receive as input a non-linear combination of outputs from layer 2, which themselves are non-linear combinations of layer 1 outputs. This recursive composition is the source of deep networks' power.

Parameter count: approximately $d \cdot w^2$ (ignoring biases, for fully connected layers).

**Depth-1, width-$w^d$ network:**

This network has one hidden layer with $w^d$ neurons. Each neuron independently maps the raw input to a single scalar. There is no hierarchical composition.

Parameter count: approximately $n_{\text{input}} \cdot w^d + w^d \cdot n_{\text{output}}$ -- exponentially larger for large $d$.

**Which is more expressive?**

The deep network is more expressive per parameter. Theoretical results (e.g., Montufar et al. 2014) show that a depth-$d$ ReLU network with $w$ units per layer can produce $O\!\left(\left(\frac{w}{n_{\text{input}}}\right)^{(d-1) n_{\text{input}}} w^{n_{\text{input}}}\right)$ linear regions in the input space, which is exponential in depth. The shallow network with the same parameter budget can achieve exponentially fewer linear regions.

**Practical implication:** You should prefer depth over width when budget-constrained. A 10-layer network with 256 units per layer outperforms a 2-layer network with 256,000 units per layer on most practical tasks, using a fraction of the parameters.

---

## Tier 3 -- Advanced

### Question A1
**Derive the expression for the output of a single neuron with a sigmoid activation in terms of a log-linear model. What is the probabilistic interpretation? Why is this the canonical choice for binary classification?**

**Answer:**

Consider one neuron with weights $\mathbf{w}$, bias $b$, and sigmoid activation:

$$\hat{y} = \sigma(\mathbf{w}^{\top} \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^{\top}\mathbf{x} + b)}}$$

**Log-linear connection:**

Define the log-odds (logit) of the positive class:

$$\log \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}^{\top} \mathbf{x} + b$$

Taking the inverse of the logit function:

$$P(y=1 \mid \mathbf{x}) = \frac{e^{\mathbf{w}^{\top}\mathbf{x}+b}}{1 + e^{\mathbf{w}^{\top}\mathbf{x}+b}} = \frac{1}{1+e^{-(\mathbf{w}^{\top}\mathbf{x}+b)}} = \sigma(\mathbf{w}^{\top}\mathbf{x}+b)$$

The sigmoid output is the probability of the positive class under a log-linear model. The sigmoid neuron is exactly a logistic regression model.

**Why it is canonical for binary classification:**

1. **Probabilistic output.** $\hat{y} \in (0,1)$, which can be directly interpreted as $P(y=1 \mid \mathbf{x})$.

2. **Maximum likelihood training.** If we assume the data-generating process is $y \sim \text{Bernoulli}(\hat{y})$, then the negative log-likelihood is the binary cross-entropy loss:
   $$\mathcal{L} = -y \log \hat{y} - (1-y) \log(1 - \hat{y})$$
   Minimising this loss is equivalent to maximising the likelihood, giving well-founded statistical interpretation.

3. **Decision boundary.** The decision boundary $\hat{y} = 0.5$ corresponds to $\mathbf{w}^{\top}\mathbf{x} + b = 0$, a hyperplane. The sigmoid provides soft (probabilistic) classification rather than a hard threshold.

4. **Gradient properties.** $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which combines cleanly with cross-entropy to give a simple gradient: $\partial \mathcal{L} / \partial z = \hat{y} - y$. This is one of the most aesthetically clean results in gradient-based learning.

**Common interview trap:** "Why not use MSE with sigmoid output for classification?" MSE with sigmoid gives a non-convex loss surface with vanishing gradients when $\hat{y}$ is far from 0.5, making training slow. Cross-entropy (the correct log-likelihood) gives gradients proportional to the prediction error regardless of how confident the wrong prediction is.

---

### Question A2
**Prove that the composition of two affine functions is itself affine. Then explain why this means that, without non-linear activations, increasing network depth cannot increase the hypothesis class beyond linear functions.**

**Answer:**

**Proof that composition of affine functions is affine:**

An affine function $f : \mathbb{R}^n \to \mathbb{R}^m$ has the form $f(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^m$.

Let $f(\mathbf{x}) = \mathbf{A}_1 \mathbf{x} + \mathbf{b}_1$ with $\mathbf{A}_1 \in \mathbb{R}^{p \times n}$ and $g(\mathbf{y}) = \mathbf{A}_2 \mathbf{y} + \mathbf{b}_2$ with $\mathbf{A}_2 \in \mathbb{R}^{m \times p}$.

$$g(f(\mathbf{x})) = \mathbf{A}_2 (\mathbf{A}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$
$$= \mathbf{A}_2 \mathbf{A}_1 \mathbf{x} + \mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2$$
$$= \underbrace{\mathbf{A}_2 \mathbf{A}_1}_{\mathbf{A}_{\text{eff}}} \mathbf{x} + \underbrace{\mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2}_{\mathbf{b}_{\text{eff}}}$$

This is affine with $\mathbf{A}_{\text{eff}} = \mathbf{A}_2 \mathbf{A}_1 \in \mathbb{R}^{m \times n}$ and $\mathbf{b}_{\text{eff}} = \mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2 \in \mathbb{R}^m$. $\square$

**Consequence for networks without non-linearities:**

By induction on the depth $L$: the composition of $L$ affine layer transformations is a single affine function from $\mathbb{R}^{n_0}$ to $\mathbb{R}^{n_L}$.

The hypothesis class of a depth-$L$ purely linear network is exactly:

$$\mathcal{H}_{\text{linear}} = \left\{ \mathbf{x} \mapsto \mathbf{A}\mathbf{x} + \mathbf{b} \mid \mathbf{A} \in \mathbb{R}^{n_L \times n_0},\ \mathbf{b} \in \mathbb{R}^{n_L} \right\}$$

This is identical to the hypothesis class of a single-layer (depth-1) linear model. Adding layers adds parameters that must be jointly trained, but the representable function set does not expand. In fact, the effective weight matrix $\mathbf{A}_{\text{eff}}$ is constrained to be a product of matrices with intermediate ranks, which is a strictly smaller set than all matrices of shape $\mathbb{R}^{n_L \times n_0}$ unless the intermediate widths are at least $\min(n_0, n_L)$.

**Practical takeaway:** Every hidden layer must have a non-linear activation. Even one linear hidden layer in an otherwise non-linear network reduces its effective depth by one for the purposes of function composition.
