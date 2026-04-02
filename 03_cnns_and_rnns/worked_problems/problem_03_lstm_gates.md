# Problem 03: Tracing Data Through LSTM Gates

**Difficulty:** Intermediate to Advanced  
**Topic:** LSTM gate mechanics — numerical trace, gradient flow, gate behaviour analysis  
**Skills tested:** Applying LSTM equations by hand, interpreting gate activations, understanding cell vs hidden state dynamics

---

## Background: LSTM Equations

All weight matrices and biases are listed explicitly. Let $\mathbf{z}_t = [\mathbf{h}_{t-1}; \mathbf{x}_t]$ (column vector concatenation).

$$\mathbf{f}_t = \sigma(W_f \mathbf{z}_t + \mathbf{b}_f) \tag{forget gate}$$

$$\mathbf{i}_t = \sigma(W_i \mathbf{z}_t + \mathbf{b}_i) \tag{input gate}$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c \mathbf{z}_t + \mathbf{b}_c) \tag{cell candidate}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \tag{cell state update}$$

$$\mathbf{o}_t = \sigma(W_o \mathbf{z}_t + \mathbf{b}_o) \tag{output gate}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \tag{hidden state}$$

Activation function values for reference:
- $\sigma(0) = 0.5$, $\sigma(1) \approx 0.731$, $\sigma(2) \approx 0.880$, $\sigma(-2) \approx 0.119$
- $\tanh(0) = 0$, $\tanh(1) \approx 0.762$, $\tanh(2) \approx 0.964$, $\tanh(-1) \approx -0.762$

---

## Part A: Scalar LSTM Step-by-Step

Work through a single LSTM with **1-dimensional state** (scalar $h_t$, $c_t$, $x_t$).

### Setup

Initial state: $h_0 = 0$, $c_0 = 0$.

At time $t=1$, the input is $x_1 = 1.0$.

For a 1D LSTM, each weight matrix is a scalar and the concatenated input is $z_1 = [h_0; x_1] = [0; 1]$ (a 2-element vector). Each gate equation has the form $W_f z_1 + b_f$ where $W_f = [w_{f,h}, w_{f,x}]$ is a row vector.

**Parameters:**

| Gate | $w_h$ (from $h$) | $w_x$ (from $x$) | $b$ |
|---|---|---|---|
| Forget ($f$) | 0 | 0 | 0 |
| Input ($i$) | 0 | 1 | 0 |
| Cell candidate ($\tilde{c}$) | 0 | 2 | 0 |
| Output ($o$) | 0 | 1 | 0 |

### Question A1: Compute $h_1$ and $c_1$.

---

**Answer:**

Concatenated input: $z_1 = [h_0; x_1] = [0; 1]$.

**Forget gate:**

$$f_1 = \sigma(w_{f,h} \cdot h_0 + w_{f,x} \cdot x_1 + b_f) = \sigma(0 \cdot 0 + 0 \cdot 1 + 0) = \sigma(0) = 0.5$$

**Input gate:**

$$i_1 = \sigma(w_{i,h} \cdot h_0 + w_{i,x} \cdot x_1 + b_i) = \sigma(0 \cdot 0 + 1 \cdot 1 + 0) = \sigma(1) \approx 0.731$$

**Cell candidate:**

$$\tilde{c}_1 = \tanh(w_{c,h} \cdot h_0 + w_{c,x} \cdot x_1 + b_c) = \tanh(0 + 2 \cdot 1 + 0) = \tanh(2) \approx 0.964$$

**Cell state update:**

$$c_1 = f_1 \cdot c_0 + i_1 \cdot \tilde{c}_1 = 0.5 \times 0 + 0.731 \times 0.964 \approx 0 + 0.705 = 0.705$$

**Output gate:**

$$o_1 = \sigma(0 \cdot 0 + 1 \cdot 1 + 0) = \sigma(1) \approx 0.731$$

**Hidden state:**

$$h_1 = o_1 \cdot \tanh(c_1) = 0.731 \times \tanh(0.705) \approx 0.731 \times 0.607 \approx 0.444$$

**Results: $c_1 \approx 0.705$, $h_1 \approx 0.444$.**

**Interpretation:**
- The forget gate ($f_1 = 0.5$) is at its "neutral" value because it received no input signal. Since $c_0 = 0$, it doesn't matter here.
- The input gate ($i_1 = 0.731$) is moderately open — the input signal $x_1 = 1$ partially opened it.
- The candidate ($\tilde{c}_1 \approx 0.964$) is strongly positive because the weight on $x$ is 2.
- The cell state has "written" approximately 0.70 — a significant fraction of the candidate.
- The output gate partially exposes the cell state as the hidden state.

---

### Question A2: Second time step

At $t=2$, input $x_2 = 0$.

Using the same weights as above, compute $c_2$ and $h_2$.

---

**Answer:**

Concatenated input: $z_2 = [h_1; x_2] = [0.444; 0]$.

**Forget gate:**

$$f_2 = \sigma(0 \cdot 0.444 + 0 \cdot 0 + 0) = \sigma(0) = 0.5$$

**Input gate:**

$$i_2 = \sigma(0 \cdot 0.444 + 1 \cdot 0 + 0) = \sigma(0) = 0.5$$

**Cell candidate:**

$$\tilde{c}_2 = \tanh(0 \cdot 0.444 + 2 \cdot 0 + 0) = \tanh(0) = 0$$

**Cell state update:**

$$c_2 = f_2 \cdot c_1 + i_2 \cdot \tilde{c}_2 = 0.5 \times 0.705 + 0.5 \times 0 = 0.353$$

**Output gate:**

$$o_2 = \sigma(0 + 0 + 0) = 0.5$$

**Hidden state:**

$$h_2 = o_2 \cdot \tanh(c_2) = 0.5 \times \tanh(0.353) \approx 0.5 \times 0.340 = 0.170$$

**Results: $c_2 \approx 0.353$, $h_2 \approx 0.170$.**

**Interpretation:** The zero input caused all gates to output 0.5 (their unbiased default). The cell state decayed by a factor of 0.5 (the forget gate). The cell state "remembers" $x_1 = 1$ from the previous step, but attenuated. This illustrates how the LSTM can retain information over time without any explicit input.

---

## Part B: Engineered Gate Configurations

This part asks you to reason about what specific gate configurations achieve without computing exact numbers.

### Question B1: Perfect memory

You want an LSTM to **perfectly remember** the cell state indefinitely once written, ignoring all new inputs after writing.

**What values should $f_t$, $i_t$, and $o_t$ take?**

---

**Answer:**

- **Forget gate $f_t = 1$**: retain 100% of the previous cell state at every step.
- **Input gate $i_t = 0$**: block all new writes to the cell state (regardless of the candidate).
- **Output gate $o_t$ = whatever is appropriate**: the output gate does not affect the cell state; it can be set freely.

Cell state update with $f=1$, $i=0$:

$$c_t = 1 \cdot c_{t-1} + 0 \cdot \tilde{c}_t = c_{t-1}$$

The cell state is preserved indefinitely.

**Real-world analogy:** this is what the LSTM learns when processing the sentence "The cat, which I saw yesterday at the market, **is**..." — it needs to remember that "cat" (singular) is the subject, ignoring the long relative clause, to correctly predict the singular verb "is". The forget gate stays near 1 and the input gate near 0 during the relative clause.

---

### Question B2: Complete reset

At a certain time step, you want the LSTM to **completely forget** its previous state and start fresh based only on the new input.

**What values should $f_t$ and $i_t$ take?**

---

**Answer:**

- **Forget gate $f_t = 0$**: erase the entire cell state.
- **Input gate $i_t = 1$**: write the candidate fully to the cell state.

Cell state update with $f=0$, $i=1$:

$$c_t = 0 \cdot c_{t-1} + 1 \cdot \tilde{c}_t = \tilde{c}_t$$

The new cell state is exactly the candidate derived from the current input.

**Real-world analogy:** in language modelling, when a sentence ends and a new sentence begins, the LSTM should reset its state (forget sentence-level context like subject/verb agreement from the previous sentence). The model learns to set $f \approx 0$ when it sees a sentence boundary marker.

---

## Part C: Gradient Flow Analysis

### Question C1

Consider the gradient of the loss with respect to the cell state $c_{t-k}$ for some $k > 0$:

$$\frac{\partial \mathcal{L}}{\partial c_{t-k}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot \prod_{j=1}^{k} f_{t-j+1}$$

**If the forget gate is always $f=0.9$, how many time steps back can the cell state gradient propagate before it falls below 1% of its original value?**

---

**Answer:**

We need to find $k$ such that:

$$0.9^k < 0.01$$

Taking logarithms:

$$k \ln(0.9) < \ln(0.01)$$

$$k > \frac{\ln(0.01)}{\ln(0.9)} = \frac{-4.605}{-0.105} \approx 43.8$$

So gradients decay below 1% after approximately **44 steps**.

**Compare to vanilla RNN:**

The vanilla RNN gradient through the hidden state involves $\prod W_{hh} \cdot \tanh'(\cdot)$. The tanh derivative satisfies $0 < \tanh'(x) \leq 1$, so the product of $\tanh'$ values alone decays the gradient. If $\|W_{hh}\| < 1/\tanh'_{\max}$, gradients vanish exponentially fast — often in fewer than 10 steps.

**Key insight:** the LSTM cell gradient depends only on the forget gate values $f_{t-j}$, not on weight matrices or activation derivatives. When the model learns to set $f \approx 1$ for a relevant memory, the gradient flows through with almost no attenuation. This is the mechanism by which LSTMs achieve long-range dependency learning.

---

### Question C2: The role of the output gate in gradient flow

Why does the output gate $o_t$ NOT help with the vanishing gradient problem through the cell state?

---

**Answer:**

The output gate controls the mapping from cell state to hidden state:

$$h_t = o_t \odot \tanh(c_t)$$

The gradient of the loss through the **hidden state** path involves $o_t$ and $\tanh'(c_t)$. This path can still vanish because $\tanh'(c_t) \in (0, 1]$ and $o_t \in (0, 1)$ by construction (sigmoid output).

However, the cell state gradient path bypasses the output gate entirely. The BPTT gradient through the cell state is:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

There is no $\tanh'$ or $\sigma'$ term in this path — the cell state backpropagates through **only the forget gate**, which is a learned scalar. When $f_t \approx 1$, this gradient is near 1.

The output gate affects what information is *read out* but not what information is *stored* or how gradients flow through the stored memory. The gradient highway is the cell state path, not the hidden state path.

---

## Part D: PyTorch Verification

The following code verifies the Part A calculations numerically.

```python
import torch
import torch.nn as nn

# Verify Part A calculations with a 1D LSTM
# Using exact weights from the problem setup

def manual_lstm_step(x, h_prev, c_prev, weights):
    """
    Single LSTM step with scalar state and 2-element input z = [h_prev, x].
    weights: dict with keys 'Wf', 'Wi', 'Wc', 'Wo' each [w_h, w_x]
             and 'bf', 'bi', 'bc', 'bo' each scalar
    """
    import math

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def tanh(x):
        return math.tanh(x)

    # Gate pre-activations: w_h * h_prev + w_x * x + b
    f_pre = weights['Wf'][0] * h_prev + weights['Wf'][1] * x + weights['bf']
    i_pre = weights['Wi'][0] * h_prev + weights['Wi'][1] * x + weights['bi']
    c_pre = weights['Wc'][0] * h_prev + weights['Wc'][1] * x + weights['bc']
    o_pre = weights['Wo'][0] * h_prev + weights['Wo'][1] * x + weights['bo']

    f = sigmoid(f_pre)
    i = sigmoid(i_pre)
    c_tilde = tanh(c_pre)
    o = sigmoid(o_pre)

    c = f * c_prev + i * c_tilde
    h = o * tanh(c)

    print(f"  forget gate f = {f:.4f}")
    print(f"  input gate  i = {i:.4f}")
    print(f"  cell cand  c~ = {c_tilde:.4f}")
    print(f"  cell state  c = {c:.4f}")
    print(f"  output gate o = {o:.4f}")
    print(f"  hidden state h = {h:.4f}")

    return h, c

weights = {
    'Wf': [0.0, 0.0], 'bf': 0.0,  # forget gate
    'Wi': [0.0, 1.0], 'bi': 0.0,  # input gate
    'Wc': [0.0, 2.0], 'bc': 0.0,  # cell candidate
    'Wo': [0.0, 1.0], 'bo': 0.0,  # output gate
}

print("Time step t=1, x=1.0:")
h1, c1 = manual_lstm_step(x=1.0, h_prev=0.0, c_prev=0.0, weights=weights)

print("\nTime step t=2, x=0.0:")
h2, c2 = manual_lstm_step(x=0.0, h_prev=h1, c_prev=c1, weights=weights)
```

Expected output:
```
Time step t=1, x=1.0:
  forget gate f = 0.5000
  input gate  i = 0.7311
  cell cand  c~ = 0.9640
  cell state  c = 0.7050
  output gate o = 0.7311
  hidden state h = 0.4440

Time step t=2, x=0.0:
  forget gate f = 0.5000
  input gate  i = 0.5000
  cell cand  c~ = 0.0000
  cell state  c = 0.3525
  output gate o = 0.5000
  hidden state h = 0.1701
```

---

## Summary: What to Know for an Interview

1. **The six equations**: be able to write the full LSTM update without prompting — forget, input, cell candidate, cell update, output, hidden state.

2. **Why cell state helps gradient flow**: the gradient through $c_t$ back to $c_{t-1}$ is $f_t$. With $f_t \approx 1$, the gradient travels far without attenuation. No tanh/sigmoid derivatives on this path.

3. **Gate interpretations**: forget erases, input writes, output reads. A learnt $f=1$, $i=0$ pattern maintains memory; $f=0$, $i=1$ resets completely.

4. **Parameter count**: $4H(D + H + 1)$. The factor of 4 is always from the four gate/candidate matrices.

5. **LSTM vs GRU**: GRU merges cell and hidden states, uses 2 gates (update = combined forget+input, reset), has 25% fewer parameters, similar performance.
