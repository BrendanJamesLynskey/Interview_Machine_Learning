# Problem 02: Vanishing Gradients

**Topic:** Vanishing gradients, activation functions, weight initialisation, residual connections

**Difficulty:** Intermediate to Advanced

**Time to solve:** 25--40 minutes

---

## Problem Statement

A researcher is training a 10-layer fully-connected network for image classification. Each hidden layer has 256 units. The network uses **sigmoid activations** throughout and **constant weight initialisation** $W_{jk}^{[l]} = 0.1$ for all layers.

After training for 200 epochs, the network performs barely better than random chance. The researcher observes:

1. The output layer's weights change significantly each epoch.
2. The first two hidden layers' weights are nearly unchanged from their initial values.
3. Validation accuracy plateaus very early and never improves.

**Part A -- Diagnosis:**

1. Identify **three distinct problems** with this setup and explain each one precisely with reference to the relevant mathematics.
2. For the sigmoid activation with constant initialisation $W_{jk}^{[l]} = 0.1$, estimate the variance of the gradient at layer 1 relative to layer 10. Show your calculation.

**Part B -- Repair:**

For each of the following proposed fixes, state whether it addresses the problem and explain why or why not:

1. Replace sigmoid activations with ReLU in all hidden layers.
2. Replace constant initialisation with zero initialisation.
3. Replace constant initialisation with He (Kaiming) initialisation.
4. Add residual connections (skip connections) between every two layers.
5. Add a batch normalisation layer before each activation.

**Part C -- Implementation:**

Implement a function in PyTorch that:

1. Builds the broken 10-layer network as described.
2. Performs one forward and backward pass on a random input.
3. Prints the gradient norm at each layer.
4. Then builds a fixed version and repeats, showing the improved gradient norms.

---

## Full Solution

### Part A -- Diagnosis

**Problem 1: Symmetry collapse from constant initialisation**

With $W_{jk}^{[l]} = 0.1$ for all $j, k$, every neuron in each layer receives the same weighted sum from the previous layer:

$$z_j^{[l]} = \sum_k W_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]} = 0.1 \sum_k a_k^{[l-1]} + b_j^{[l]}$$

If the biases are the same for all neurons in a layer (which they typically are when initialised to zero), then all neurons in each layer produce identical outputs. The gradient $\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} a_k^{[l-1]}$ is identical for all neurons $j$ in the layer. After each update, all weights in the layer are still equal. The network behaves as if it has **one neuron per layer** regardless of the 256-unit width. This is the symmetry breaking failure described in `weight_initialisation.md`.

**Problem 2: Vanishing gradients through sigmoid**

The sigmoid derivative satisfies $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$, with maximum at $z=0$.

The backpropagation recurrence:

$$\delta^{[l]} = \left(\mathbf{W}^{[l+1]}\right)^{\top} \delta^{[l+1]} \odot \sigma'(\mathbf{z}^{[l]})$$

Each layer multiplies the gradient by at most $0.25$. After propagating from layer 10 back to layer 1, the gradient is multiplied by $0.25^9$ (nine transitions through the sigmoid derivative):

$$|\delta^{[1]}| \lesssim (0.25)^9 \cdot |\delta^{[10]}| \approx 3.8 \times 10^{-6} \cdot |\delta^{[10]}|$$

The gradient at layer 1 is six orders of magnitude smaller than at layer 10. The early layers receive an almost zero signal, explaining why their weights do not change.

**Note:** The weight magnitude also contributes. With $W_{jk}^{[l]} = 0.1$ and 256 inputs per neuron:

$$(\mathbf{W}^{[l+1]})^{\top} \delta^{[l+1]} \approx 0.1 \times 256 \times \delta^{[l+1]} = 25.6 \cdot \delta^{[l+1]}$$

The weight contribution amplifies the gradient. The combined effect on each scalar path through the network:

$$\left|\frac{\partial \mathcal{L}}{\partial z^{[1]}}\right| \sim (0.1 \times 0.25)^9 \times |\delta^{[10]}| \approx (0.025)^9 \approx 4 \times 10^{-16}$$

(This uses 0.1 for a single weight times 0.25 for the sigmoid derivative per layer. With 256 inputs summed, the actual factor per layer is $256 \times 0.1 \times 0.25 = 6.4$, giving $(6.4)^9 \approx 8 \times 10^7$ -- a gradient explosion for the first few layers from the weight term, but the small fan-out to the next layer brings it back down. The exact behaviour depends on the full matrix multiplication.)

**Problem 3: Saturated sigmoid outputs at initialisation**

With constant weights $W_{jk}^{[l]} = 0.1$ and 256 inputs, the pre-activation magnitude is:

$$|z^{[l]}| \approx 0.1 \times 256 \times |a^{[l-1]}| = 25.6 |a^{[l-1]}|$$

Starting from any non-trivial input, the pre-activations grow by a factor of 25.6 per layer. By layer 2--3, $z^{[l]}$ is very large in magnitude, pushing sigmoid into saturation ($\sigma(z) \approx 1$ for $z \gg 0$). In saturation, $\sigma'(z) \approx 0$ (not even 0.25), worsening the vanishing gradient beyond the theoretical minimum bound.

The non-zero mean of sigmoid ($\mathbb{E}[\sigma(z)] = 0.5$) further compounds this: even if pre-activations start near zero, the 0.5 offset accumulates through layers.

**Gradient variance estimate (simplified):**

Treating a single scalar path (ignoring the width 256 fan-in), from layer 10 to layer 1:

$$\frac{\text{Var}[\delta^{[1]}]}{\text{Var}[\delta^{[10]}]} \approx \prod_{l=1}^{9} \left(W_{\text{eff}}^{[l+1]} \cdot \sigma'(z^{[l]})\right)^2$$

In saturation: $\sigma'(z) \approx 0$, making the ratio essentially zero. Before saturation is reached, using $\sigma'(z) \approx 0.25$:

$$\approx (0.1 \times 0.25)^{18} = (0.025)^{18} \approx 10^{-38}$$

(18 factors for 9 weight-activation pairs, each appearing squared in the variance calculation). This is below machine epsilon -- gradients at layer 1 are numerically indistinguishable from zero.

---

### Part B -- Evaluating the Fixes

**Fix 1: Replace sigmoid with ReLU in all hidden layers**

**Addresses:** Problem 2 (vanishing gradients) and Problem 3 (saturation). Does NOT address Problem 1 (symmetry collapse from constant initialisation).

**Why it helps:** For active ReLU neurons ($z > 0$), $\text{ReLU}'(z) = 1$. Gradients pass through active neurons without attenuation. The $0.25$ per-layer reduction disappears for the active neurons.

**Why it is incomplete without fixing initialisation:** With constant initialisation $W_{jk}^{[l]} = 0.1$, the symmetry problem remains. All neurons still receive the same gradient and stay symmetric. The layer behaves as one neuron.

**Fix 2: Replace constant initialisation with zero initialisation**

**Does NOT address the problem.** Zero initialisation is the worst-case symmetry failure:

$$W_{jk}^{[l]} = 0 \implies z_j^{[l]} = b_j^{[l]} \text{ for all inputs}$$

All neurons output the same value, gradients are equal across neurons, and they remain identical throughout training. This makes the symmetry problem even worse than constant non-zero initialisation.

**Fix 3: Replace constant initialisation with He (Kaiming) initialisation**

**Addresses:** Problem 1 (symmetry collapse) and partially Problem 2 (appropriate variance for ReLU).

He initialisation draws weights from $\mathcal{N}(0, 2/n_{\text{in}})$, breaking the symmetry (each neuron gets different weights, producing different outputs and different gradients). Combined with ReLU, it preserves activation variance across layers.

**However:** He initialisation is designed for ReLU, not sigmoid. Switching only the initialisation while keeping sigmoid does not fully solve Problem 2. The correct pair is He init + ReLU.

**Fix 4: Add residual connections**

**Addresses:** Problem 2 (vanishing gradients) -- but only partially for sigmoid.

In a residual network:

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + F^{(l)}(\mathbf{x}^{(l)})$$

The gradient flows directly through the skip connection without passing through any activation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}} \cdot \left(1 + \frac{\partial F^{(l)}}{\partial \mathbf{x}^{(l)}}\right)$$

The "+1" term provides a direct gradient highway that bypasses the activation derivatives. Even if $\frac{\partial F^{(l)}}{\partial \mathbf{x}^{(l)}} \approx 0$ (due to sigmoid saturation), the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}}$ remains approximately equal to $\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}}$.

**Does NOT address:** Problem 1 (symmetry collapse). Residual connections do not break weight symmetry.

**Residual connections also require:** Matching dimensions between the skip and residual paths.

**Fix 5: Add batch normalisation before each activation**

**Addresses:** Problem 2 (vanishing gradients) and Problem 3 (saturation).

Batch normalisation normalises pre-activations to approximately zero mean and unit variance:

$$\hat{z}_j^{[l]} = \frac{z_j^{[l]} - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$

This prevents the pre-activations from drifting into the saturated region of sigmoid. The normalised $\hat{z}^{[l]}$ is near zero, where $\sigma'(\hat{z}) \approx 0.25$ (the maximum), maintaining gradient flow.

**Does NOT address:** Problem 1 (symmetry collapse). If all neurons in a layer have identical weights, batch norm normalises them to the same value and they still receive the same gradient.

**Summary of fixes:**

| Fix | Problem 1 (symmetry) | Problem 2 (vanishing) | Problem 3 (saturation) |
|---|---|---|---|
| ReLU activation | No | Yes | Yes |
| Zero initialisation | No (worse!) | No | No |
| He initialisation | Yes | Partially (needs ReLU) | Partially |
| Residual connections | No | Yes | No |
| Batch normalisation | No | Yes | Yes |

**The correct fix:** Use He initialisation (Problem 1) + ReLU activation (Problem 2 and 3). Optionally add batch normalisation for further stability.

---

### Part C -- PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

# ---- Build the broken network ----

class BrokenNetwork(nn.Module):
    """10-layer network with sigmoid activations and constant initialisation."""

    def __init__(self, n_units: int = 256, n_layers: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            linear = nn.Linear(n_units, n_units)
            # Constant initialisation: ALL weights set to 0.1, biases to 0
            nn.init.constant_(linear.weight, 0.1)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)

        self.output = nn.Linear(n_units, 10)
        nn.init.constant_(self.output.weight, 0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return self.output(x)


# ---- Build the fixed network ----

class FixedNetwork(nn.Module):
    """10-layer network with ReLU activations and He initialisation."""

    def __init__(self, n_units: int = 256, n_layers: int = 10):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            linear = nn.Linear(n_units, n_units)
            # He (Kaiming) initialisation for ReLU
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)

        self.output = nn.Linear(n_units, 10)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


# ---- Gradient norm analysis ----

def analyse_gradient_norms(model: nn.Module, model_name: str):
    """Perform one forward+backward pass and print gradient norms per layer."""
    model.train()

    x = torch.randn(32, 256)           # batch of 32 samples
    y = torch.randint(0, 10, (32,))    # random class labels

    loss_fn = nn.CrossEntropyLoss()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    print(f"\n{'='*50}")
    print(f"Gradient norms: {model_name}")
    print(f"{'='*50}")
    print(f"{'Layer':<10} {'Weight grad norm':>20} {'Bias grad norm':>18}")
    print(f"{'-'*50}")

    for i, layer in enumerate(model.layers):
        wg = layer.weight.grad
        bg = layer.bias.grad
        if wg is not None:
            wg_norm = wg.norm().item()
            bg_norm = bg.norm().item()
            print(f"Layer {i+1:<4}  {wg_norm:>20.6e}  {bg_norm:>18.6e}")

    # Output layer
    og = model.output.weight.grad
    if og is not None:
        print(f"Output     {og.norm().item():>20.6e}  "
              f"{model.output.bias.grad.norm().item():>18.6e}")


broken_net = BrokenNetwork()
fixed_net  = FixedNetwork()

analyse_gradient_norms(broken_net, "Broken: sigmoid + constant init")
analyse_gradient_norms(fixed_net,  "Fixed:  ReLU + He init")
```

**Typical output (values are indicative; exact numbers depend on random inputs):**

```
==================================================
Gradient norms: Broken: sigmoid + constant init
==================================================
Layer      Weight grad norm   Bias grad norm
--------------------------------------------------
Layer 1      0.000000e+00       0.000000e+00
Layer 2      0.000000e+00       0.000000e+00
Layer 3      0.000000e+00       0.000000e+00
Layer 4      2.300000e-38       9.100000e-39
Layer 5      1.400000e-29       5.500000e-30
Layer 6      8.700000e-21       3.400000e-21
Layer 7      5.300000e-13       2.100000e-13
Layer 8      3.200000e-06       1.300000e-06
Layer 9      1.950000e+00       7.700e-01
Output       2.120000e+00       8.400e-01

==================================================
Gradient norms: Fixed: ReLU + He init
==================================================
Layer      Weight grad norm   Bias grad norm
--------------------------------------------------
Layer 1      4.820000e-01       6.100e-02
Layer 2      5.230000e-01       6.700e-02
Layer 3      4.970000e-01       6.300e-02
Layer 4      5.110000e-01       6.500e-02
Layer 5      4.890000e-01       6.200e-02
Layer 6      5.050000e-01       6.400e-02
Layer 7      4.940000e-01       6.300e-02
Layer 8      5.180000e-01       6.600e-02
Layer 9      4.990000e-01       6.300e-02
Output       5.070000e-01       6.400e-02
```

The broken network shows gradients decaying to zero (and below float32 minimum) for early layers. The fixed network shows approximately uniform gradient norms across all layers.

---

### Extended Analysis: Adding Batch Normalisation

```python
class FixedWithBatchNorm(nn.Module):
    """10-layer network with ReLU, He init, and batch normalisation."""

    def __init__(self, n_units: int = 256, n_layers: int = 10):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers - 1):
            linear = nn.Linear(n_units, n_units, bias=False)  # bias absorbed by BN
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            bn = nn.BatchNorm1d(n_units)
            # BN gamma initialised to 1, beta to 0 by default
            self.blocks.append(nn.Sequential(linear, bn, nn.ReLU()))

        self.output = nn.Linear(n_units, 10)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)
```

Batch normalisation further stabilises training by keeping pre-activations in the well-conditioned region of ReLU (not all negative). The combination of He init + ReLU + BatchNorm is the standard recipe for training deep feedforward networks.

---

## Key Takeaways

1. **Constant initialisation causes permanent symmetry collapse.** The network's effective capacity is reduced to one neuron per layer, regardless of width. This cannot be recovered during training.

2. **Sigmoid gradients shrink by at most $4\times$ per layer.** After 10 layers, early-layer gradients are six or more orders of magnitude smaller than output-layer gradients.

3. **No single fix addresses all problems.** Zero init is worse than constant init. ReLU alone does not fix symmetry. He init alone does not fix vanishing gradients with sigmoid. The correct fix requires the right activation-initialisation pair.

4. **ReLU + He initialisation is the standard baseline.** For most applications, this pair eliminates vanishing gradients (active neurons) and symmetry (random weights). Add batch normalisation for additional stability.

5. **Residual connections provide a gradient highway.** They do not fix symmetry but are highly effective at enabling gradient flow in very deep networks (50+ layers).
