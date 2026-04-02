# Problem 01: Backpropagation by Hand

**Topic:** Backpropagation, chain rule, gradient computation

**Difficulty:** Fundamental to Intermediate

**Time to solve:** 20--35 minutes

---

## Problem Statement

Consider the following small fully-connected network for binary classification:

```
Input:   x = [x1, x2] = [1.0, 0.5]
Hidden:  1 neuron, sigmoid activation
Output:  1 neuron, sigmoid activation
True label: y = 1
Loss: Binary cross-entropy
```

**Network parameters (initial values):**

```
W1 = [w11, w12] = [0.5, -0.3]   (weights from input to hidden)
b1 = 0.1                         (bias of hidden neuron)

w2 = 0.8                         (weight from hidden to output)
b2 = -0.2                        (bias of output neuron)
```

**Your tasks:**

1. Perform the complete forward pass. Compute $z^{[1]}$, $a^{[1]}$, $z^{[2]}$, $\hat{y}$, and $\mathcal{L}$.
2. Derive and compute the gradient $\frac{\partial \mathcal{L}}{\partial z^{[2]}}$ (output layer error signal $\delta^{[2]}$).
3. Derive and compute $\frac{\partial \mathcal{L}}{\partial w_2}$ and $\frac{\partial \mathcal{L}}{\partial b_2}$.
4. Derive and compute $\frac{\partial \mathcal{L}}{\partial z^{[1]}}$ (hidden layer error signal $\delta^{[1]}$).
5. Derive and compute $\frac{\partial \mathcal{L}}{\partial w_{11}}$, $\frac{\partial \mathcal{L}}{\partial w_{12}}$, and $\frac{\partial \mathcal{L}}{\partial b_1}$.
6. Perform one step of gradient descent with learning rate $\eta = 0.5$. Write out the updated parameter values.

---

## Full Solution

### Step 1: Forward Pass

**Hidden layer pre-activation:**

$$z^{[1]} = w_{11} x_1 + w_{12} x_2 + b_1 = (0.5)(1.0) + (-0.3)(0.5) + 0.1$$
$$= 0.5 - 0.15 + 0.1 = 0.45$$

**Hidden layer activation (sigmoid):**

$$a^{[1]} = \sigma(z^{[1]}) = \frac{1}{1 + e^{-0.45}}$$

$e^{-0.45} \approx 0.6376$

$$a^{[1]} = \frac{1}{1 + 0.6376} = \frac{1}{1.6376} \approx 0.6106$$

**Output layer pre-activation:**

$$z^{[2]} = w_2 \cdot a^{[1]} + b_2 = (0.8)(0.6106) + (-0.2)$$
$$= 0.4885 - 0.2 = 0.2885$$

**Output (predicted probability):**

$$\hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-0.2885}}$$

$e^{-0.2885} \approx 0.7492$

$$\hat{y} = \frac{1}{1.7492} \approx 0.5717$$

**Binary cross-entropy loss** (true label $y = 1$):

$$\mathcal{L} = -y \log \hat{y} - (1-y) \log(1-\hat{y}) = -1 \cdot \log(0.5717) - 0 \cdot \log(0.4283)$$
$$= -\log(0.5717) \approx 0.5585$$

**Forward pass summary:**

```
x1 = 1.0,   x2 = 0.5
z^[1] = 0.45,   a^[1] = 0.6106
z^[2] = 0.2885,  y_hat = 0.5717
Loss = 0.5585
```

---

### Step 2: Output Layer Error Signal

For sigmoid output + binary cross-entropy, the combined gradient (derived in `backpropagation_derivation.md`) is:

$$\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \hat{y} - y = 0.5717 - 1 = -0.4283$$

**Verification via chain rule:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = -\frac{1}{0.5717} + 0 = -1.7491$$

$$\frac{d\hat{y}}{dz^{[2]}} = \sigma'(z^{[2]}) = \hat{y}(1-\hat{y}) = 0.5717 \times 0.4283 = 0.2449$$

$$\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{d\hat{y}}{dz^{[2]}} = (-1.7491)(0.2449) \approx -0.4283 \checkmark$$

---

### Step 3: Output Layer Parameter Gradients

**Gradient with respect to $w_2$:**

$$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial w_2} = \delta^{[2]} \cdot a^{[1]} = (-0.4283)(0.6106) \approx -0.2616$$

**Gradient with respect to $b_2$:**

$$\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial b_2} = \delta^{[2]} \cdot 1 = -0.4283$$

**Interpretation:** Both gradients are negative. A gradient descent step will increase $w_2$ and $b_2$ (subtract the negative gradient), which will increase $z^{[2]}$, increasing $\hat{y}$ towards the target $y = 1$. This makes intuitive sense.

---

### Step 4: Hidden Layer Error Signal

Apply the backpropagation recurrence:

$$\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial z^{[1]}} = \left(\frac{\partial z^{[2]}}{\partial a^{[1]}}\right) \cdot \delta^{[2]} \cdot \sigma'(z^{[1]})$$

**Gradient through the weight $w_2$:**

$$\frac{\partial z^{[2]}}{\partial a^{[1]}} = w_2 = 0.8$$

**Sigmoid derivative at hidden layer:**

$$\sigma'(z^{[1]}) = a^{[1]}(1 - a^{[1]}) = 0.6106 \times (1 - 0.6106) = 0.6106 \times 0.3894 \approx 0.2378$$

**Hidden error signal:**

$$\delta^{[1]} = w_2 \cdot \delta^{[2]} \cdot \sigma'(z^{[1]}) = (0.8)(-0.4283)(0.2378) \approx -0.08152$$

---

### Step 5: Hidden Layer Parameter Gradients

**Gradient with respect to $w_{11}$:**

$$\frac{\partial \mathcal{L}}{\partial w_{11}} = \delta^{[1]} \cdot x_1 = (-0.08152)(1.0) = -0.08152$$

**Gradient with respect to $w_{12}$:**

$$\frac{\partial \mathcal{L}}{\partial w_{12}} = \delta^{[1]} \cdot x_2 = (-0.08152)(0.5) = -0.04076$$

**Gradient with respect to $b_1$:**

$$\frac{\partial \mathcal{L}}{\partial b_1} = \delta^{[1]} \cdot 1 = -0.08152$$

**Observation:** All hidden layer gradients are also negative. The gradient descent step will increase all weights in the hidden layer, which will increase $a^{[1]}$, which in turn increases $z^{[2]}$, pushing $\hat{y}$ towards 1. This is consistent with the target.

---

### Step 6: Gradient Descent Update ($\eta = 0.5$)

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

**Output layer:**

$$w_2 \leftarrow 0.8 - 0.5 \times (-0.2616) = 0.8 + 0.1308 = 0.9308$$

$$b_2 \leftarrow -0.2 - 0.5 \times (-0.4283) = -0.2 + 0.2142 = 0.0142$$

**Hidden layer:**

$$w_{11} \leftarrow 0.5 - 0.5 \times (-0.08152) = 0.5 + 0.04076 = 0.5408$$

$$w_{12} \leftarrow -0.3 - 0.5 \times (-0.04076) = -0.3 + 0.02038 = -0.2796$$

$$b_1 \leftarrow 0.1 - 0.5 \times (-0.08152) = 0.1 + 0.04076 = 0.1408$$

**Updated parameters:**

```
w11 = 0.5408  (was 0.5000)
w12 = -0.2796  (was -0.3000)
b1  = 0.1408  (was 0.1000)
w2  = 0.9308  (was 0.8000)
b2  = 0.0142  (was -0.2000)
```

---

### Verification: Post-Update Forward Pass

Let us verify the loss decreased after one step.

$$z^{[1]}_{\text{new}} = (0.5408)(1.0) + (-0.2796)(0.5) + 0.1408 = 0.5408 - 0.1398 + 0.1408 = 0.5418$$

$$a^{[1]}_{\text{new}} = \sigma(0.5418) \approx 0.6323$$

$$z^{[2]}_{\text{new}} = (0.9308)(0.6323) + 0.0142 = 0.5885 + 0.0142 = 0.6027$$

$$\hat{y}_{\text{new}} = \sigma(0.6027) \approx 0.6464$$

$$\mathcal{L}_{\text{new}} = -\log(0.6464) \approx 0.4364$$

The loss decreased from $0.5585$ to $0.4364$. The update moved in the correct direction.

---

## NumPy Reference Implementation

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)

# Network parameters
params = {
    'W1': np.array([[0.5, -0.3]]),   # shape (1, 2): 1 hidden unit, 2 inputs
    'b1': np.array([0.1]),            # shape (1,)
    'W2': np.array([[0.8]]),          # shape (1, 1): 1 output unit, 1 hidden unit
    'b2': np.array([-0.2]),           # shape (1,)
}

x = np.array([[1.0], [0.5]])          # shape (2, 1)
y = np.array([[1.0]])                 # shape (1, 1)

# ---- Forward pass ----
z1 = params['W1'] @ x + params['b1'].reshape(-1, 1)   # (1, 1)
a1 = sigmoid(z1)                                        # (1, 1)
z2 = params['W2'] @ a1 + params['b2'].reshape(-1, 1)   # (1, 1)
y_hat = sigmoid(z2)                                     # (1, 1)
loss = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

print(f"z1     = {z1[0,0]:.4f}")
print(f"a1     = {a1[0,0]:.4f}")
print(f"z2     = {z2[0,0]:.4f}")
print(f"y_hat  = {y_hat[0,0]:.4f}")
print(f"loss   = {loss[0,0]:.4f}")

# ---- Backward pass ----
# Output layer
delta2 = y_hat - y                                       # (1, 1): d_Loss/d_z2
dW2    = delta2 @ a1.T                                   # (1, 1)
db2    = delta2.sum(axis=1)                              # (1,)

# Hidden layer
delta1 = (params['W2'].T @ delta2) * sigmoid_deriv(z1)  # (1, 1)
dW1    = delta1 @ x.T                                    # (1, 2)
db1    = delta1.sum(axis=1)                              # (1,)

print(f"\ndelta2  = {delta2[0,0]:.5f}")
print(f"dW2     = {dW2[0,0]:.5f}")
print(f"db2     = {db2[0]:.5f}")
print(f"delta1  = {delta1[0,0]:.5f}")
print(f"dW1     = {dW1[0,0]:.5f}, {dW1[0,1]:.5f}")
print(f"db1     = {db1[0]:.5f}")

# ---- Gradient descent step ----
lr = 0.5
params['W1'] -= lr * dW1
params['b1'] -= lr * db1
params['W2'] -= lr * dW2
params['b2'] -= lr * db2

print(f"\nUpdated W1: {params['W1']}")
print(f"Updated b1: {params['b1']}")
print(f"Updated W2: {params['W2']}")
print(f"Updated b2: {params['b2']}")
```

**Expected output:**

```
z1     = 0.4500
a1     = 0.6106
z2     = 0.2885
y_hat  = 0.5717
loss   = 0.5585

delta2  = -0.42830
dW2     = -0.26155
db2     = -0.42830
delta1  = -0.08154
dW1     = -0.08154, -0.04077
db1     = -0.08154

Updated W1: [[ 0.5408 -0.2796]]
Updated b1: [0.1408]
Updated W2: [[0.9308]]
Updated b2: [0.0142]
```

---

## Key Takeaways

1. **The forward pass must be computed and cached first.** Backpropagation requires the stored values of $z^{[l]}$ and $a^{[l]}$ at every layer.

2. **The sigmoid + BCE gradient is simply $\hat{y} - y$**, not the full product of $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ and $\sigma'(z^{[2]})$ separately. The cancellation simplifies computation significantly.

3. **Gradients flow backwards through weights via the transpose.** The factor $w_2$ appears in $\delta^{[1]}$ because $z^{[2]} = w_2 a^{[1]} + b_2$, so $\frac{\partial z^{[2]}}{\partial a^{[1]}} = w_2$.

4. **Weight gradients are outer products** of the error signal and the preceding layer's activations. The weight $W_{jk}$ connects input $k$ to output $j$, so its gradient is $\delta_j \cdot a_k$.

5. **A negative gradient means increase the weight.** If $\frac{\partial \mathcal{L}}{\partial w} < 0$, then $w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$ increases $w$, which increases the pre-activation, increasing the loss's sensitivity to changes in the loss-reducing direction.

---

## Common Mistakes

**Mistake 1: Wrong dimension for the weight gradient**

The gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ must have the same shape as $\mathbf{W}^{[l]}$. In this example, $\mathbf{W}^{[1]} \in \mathbb{R}^{1 \times 2}$ and the gradient must also be $\mathbb{R}^{1 \times 2}$.

Incorrect: computing $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \delta^{[1]} \cdot \mathbf{x}$ (inner product, scalar) instead of $\delta^{[1]} \cdot \mathbf{x}^{\top}$ (outer product, matrix).

**Mistake 2: Forgetting the activation derivative**

The hidden error signal requires the factor $\sigma'(z^{[1]})$. A common error is to compute $\delta^{[1]} = w_2 \cdot \delta^{[2]}$ without multiplying by $\sigma'(z^{[1]})$. This misses the chain rule term through the activation function.

**Mistake 3: Using post-activation instead of pre-activation in the derivative**

The sigmoid derivative is evaluated at the **pre-activation** $z^{[1]}$, not at $a^{[1]}$. While $\sigma'(z) = \sigma(z)(1-\sigma(z)) = a^{[1]}(1-a^{[1]})$ (both are correct), the distinction matters for non-standard activations.

**Mistake 4: Confusing the direction of the weight matrix transpose**

The backward recurrence uses $(\mathbf{W}^{[l+1]})^{\top}$, not $\mathbf{W}^{[l+1]}$. In this example: $\delta^{[1]}$ requires $w_2$ (a scalar, so transpose is trivial), but in multi-neuron layers the transpose is essential.
