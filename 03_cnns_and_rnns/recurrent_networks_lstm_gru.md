# Recurrent Networks: LSTM and GRU

A complete reference for recurrent architectures covering LSTM gates, GRU simplifications, backpropagation through time, and the vanishing gradient problem. Organised by interview difficulty tier.

---

## Table of Contents

- [Fundamentals](#fundamentals)
- [Intermediate](#intermediate)
- [Advanced](#advanced)
- [Common Mistakes](#common-mistakes)
- [Quick Reference](#quick-reference)

---

## Fundamentals

### The Vanilla RNN

A recurrent neural network (RNN) processes a sequence $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$ by maintaining a hidden state $\mathbf{h}_t$ that summarises all information seen so far.

**Update equations:**

$$\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y$$

The same weight matrices $W_{hh}$, $W_{xh}$, $W_{hy}$ are shared across all time steps — analogous to weight sharing in CNNs.

**Problem:** the vanilla RNN struggles to capture long-range dependencies. Gradients of the loss with respect to early time steps decay exponentially with sequence length, making it impossible for the network to learn that an event at step $t=1$ matters for the output at step $t=100$.

### Long Short-Term Memory (LSTM)

The LSTM (Hochreiter & Schmidhuber, 1997) solves the long-range dependency problem by introducing:
1. A **cell state** $\mathbf{c}_t$ — a separate memory that runs along the sequence and is designed to change slowly and smoothly.
2. Three **gates** that learn to control what information flows into, out of, and is discarded from the cell state.

The LSTM has two pieces of state: the cell state $\mathbf{c}_t$ and the hidden state $\mathbf{h}_t$.

### The Three LSTM Gates

All gates use a sigmoid activation to produce values in $[0, 1]$, interpreted as "how much to let through."

Let $\mathbf{z}_t = [\mathbf{h}_{t-1}; \mathbf{x}_t]$ (concatenation of previous hidden state and current input).

**Forget gate** — what to erase from the cell state:

$$\mathbf{f}_t = \sigma(W_f \mathbf{z}_t + \mathbf{b}_f)$$

$f_t \approx 0$: discard the stored memory.
$f_t \approx 1$: keep the stored memory unchanged.

**Input gate** — what new information to write to the cell state:

$$\mathbf{i}_t = \sigma(W_i \mathbf{z}_t + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c \mathbf{z}_t + \mathbf{b}_c)$$

$\mathbf{i}_t$ decides *how much* to write; $\tilde{\mathbf{c}}_t$ is the candidate content to write (values in $[-1, 1]$).

**Cell state update** — combine forget and input:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

The $\odot$ is element-wise multiplication. This is the key equation: the cell state is updated by scaling the old state and adding new content. Crucially, this is an **additive update** — the gradient of $\mathbf{c}_t$ with respect to $\mathbf{c}_{t-1}$ is $\mathbf{f}_t$, which can be close to 1, enabling gradients to flow back many steps without vanishing.

**Output gate** — what portion of the cell state to expose as the hidden state:

$$\mathbf{o}_t = \sigma(W_o \mathbf{z}_t + \mathbf{b}_o)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

$\tanh(\mathbf{c}_t)$ squashes the cell state into $[-1, 1]$; $\mathbf{o}_t$ gates how much of it to expose.

### Intuitive Role of Each Gate

**Forget gate:** "What do I no longer need to remember from before?" 
Example: in language modelling, when the model sees a new subject, the forget gate should erase the previous subject's gender/number information stored in the cell state.

**Input gate:** "What new information is worth remembering?" 
Example: storing the tense of the current verb so future agreement can be enforced.

**Output gate:** "What aspect of what I know do I need to output right now?"
Example: different hidden states might be appropriate for predicting the next word versus generating a summary.

---

## Intermediate

### LSTM Parameter Count

For hidden size $H$ and input size $D$:
- 4 weight matrices for the input: $4 \times H \times D$
- 4 weight matrices for the hidden state: $4 \times H \times H$
- 4 bias vectors: $4 \times H$

$$\text{Total params} = 4H(D + H + 1)$$

The factor of 4 comes from the 4 separate parameter sets (forget, input, cell candidate, output).

**Example:** $D=128$, $H=256$: $4 \times 256 \times (128 + 256 + 1) = 4 \times 256 \times 385 = 394{,}240$ parameters.

### Gated Recurrent Unit (GRU)

The GRU (Cho et al., 2014) simplifies the LSTM by merging the cell state and hidden state, and using only two gates.

**Update gate** — controls how much the hidden state should update (combines forget and input):

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_z)$$

**Reset gate** — controls how much of the previous hidden state to use when computing the candidate:

$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_r)$$

**Candidate hidden state:**

$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden state update:**

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Note the elegant $1 - \mathbf{z}_t$ coupling: when the update gate is 1, the hidden state is replaced entirely with the candidate; when it is 0, the previous hidden state is preserved (skip/copy). This is a single interpolation, versus LSTM's separate forget and input gates.

**GRU parameter count:**

$$\text{Total params} = 3H(D + H + 1)$$

GRU has 25% fewer parameters than an LSTM with the same $H$ and $D$.

### When to Use LSTM vs GRU

In practice their performance is similar. Rules of thumb:

- **GRU** when you want fewer parameters, faster training, or are working with smaller datasets.
- **LSTM** when you have very long sequences where the explicit separation of cell and hidden states is beneficial, or when you need the output gate to decouple what is stored from what is output.
- Most large-scale sequence modelling tasks (language, speech) have moved to Transformers, so LSTM vs GRU is now mainly relevant for embedded/streaming applications and tasks requiring strict causality.

### Backpropagation Through Time (BPTT)

Training an RNN requires computing gradients with respect to parameters shared across time steps. This is done by unrolling the network for $T$ steps and applying standard backpropagation to the unrolled computation graph — this is **BPTT**.

The gradient of the loss $\mathcal{L} = \sum_t \mathcal{L}_t$ with respect to $W_{hh}$ involves summing contributions from every time step:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hh}}$$

Each term requires backpropagating through the hidden state recurrence:

$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_k} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \prod_{j=k+1}^{t} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$$

The product of Jacobians $\prod_{j=k+1}^{t} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$ is what causes the gradient to vanish or explode over long sequences.

**Truncated BPTT:** instead of unrolling the full sequence, gradients are propagated back only $k$ steps. This is the standard training approach for long sequences — hidden states from the previous segment are treated as fixed (detached from the computation graph) and used as the initial state for the next segment.

---

## Advanced

### Vanishing and Exploding Gradients in RNNs

For a vanilla RNN, the Jacobian of the hidden state recurrence is:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(1 - \mathbf{h}_{t-1}^2) \cdot W_{hh}$$

(for tanh activation, where $\text{diag}(1 - \mathbf{h}^2)$ is the derivative of tanh)

For a sequence of length $T$, the product of $T-1$ such Jacobians is taken. Let $\lambda_{\max}$ be the largest singular value of $W_{hh}$ scaled by the tanh derivative.

- If $\lambda_{\max} < 1$: gradients shrink exponentially with sequence length — **vanishing gradients**.
- If $\lambda_{\max} > 1$: gradients grow exponentially — **exploding gradients**.

**Exploding gradients** are easy to detect (NaN losses) and easy to fix: **gradient clipping** rescales the gradient vector when its norm exceeds a threshold $\tau$:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{\tau}{\|\mathbf{g}\|}\right)$$

**Vanishing gradients** are harder to solve because there is no obvious signal. Solutions:
1. **LSTM/GRU**: additive cell state updates keep gradients from vanishing through the cell path.
2. **Gradient clipping**: helps with exploding only.
3. **Orthogonal initialisation** of $W_{hh}$: keeps singular values near 1 at initialisation.
4. **Attention mechanisms**: allow the output to directly attend to any input position, bypassing the recurrence entirely.

### Why the LSTM Cell State Mitigates Vanishing Gradients

The gradient of the loss through the cell state is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_{t-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \cdot \mathbf{f}_t$$

This is a multiplicative operation, but with $\mathbf{f}_t$ learned rather than fixed. The key is that when the forget gate is close to 1 (the model wants to remember), the gradient flows through essentially unchanged — no repeated squashing through tanh or sigmoid derivatives.

The hidden state path still suffers from vanishing gradients (it goes through sigmoid and tanh), but the **cell state provides an alternative gradient highway** that can maintain signal over many steps.

This is the same principle as residual connections in CNNs: an additive or near-identity path gives gradients a route that avoids repeated multiplicative attenuation.

### Peephole Connections

Standard LSTM gates see only $\mathbf{h}_{t-1}$ and $\mathbf{x}_t$, not $\mathbf{c}_{t-1}$ directly. Peephole connections add $\mathbf{c}_{t-1}$ (and $\mathbf{c}_t$ for the output gate) as an additional input:

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}; \mathbf{x}_t; \mathbf{c}_{t-1}] + \mathbf{b}_f)$$

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}; \mathbf{x}_t; \mathbf{c}_{t-1}] + \mathbf{b}_i)$$

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}; \mathbf{x}_t; \mathbf{c}_t] + \mathbf{b}_o)$$

Peepholes allow the gates to make more precise decisions (e.g., the forget gate can directly inspect the current cell value). They give modest improvements in some tasks (time series, speech) at the cost of $3H$ extra parameters.

### Bidirectional RNNs

A standard RNN only processes the sequence left to right, so the hidden state at time $t$ only sees $\mathbf{x}_1, \ldots, \mathbf{x}_t$.

A **bidirectional RNN** runs two RNNs: one forward and one backward. The outputs are concatenated:

$$\overrightarrow{\mathbf{h}}_t = \text{RNN}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = \text{RNN}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t]$$

Each position's representation now has access to the full context — both what came before and what comes after. Bidirectional LSTMs (BiLSTMs) are the standard choice for NLP tasks like named entity recognition and relation extraction where the full input sequence is available at once.

**Cannot be used for autoregressive generation**: the backward pass requires knowing future tokens, so BiLSTMs are encoder-only models.

### Stacked (Multi-Layer) RNNs

A stacked RNN feeds the output of one RNN layer as the input to the next. With $L$ layers, depth increases the capacity to learn hierarchical temporal abstractions — analogous to depth in feedforward networks.

Practical notes:
- Dropout is applied to the *vertical* connections between layers (not to the recurrent connections) to avoid corrupting the temporal information flow. Zoneout drops hidden state *updates* rather than hidden state values, which is more appropriate for RNNs.
- 2-4 layers is typical; beyond 4 layers returns diminish sharply.

---

## Common Mistakes

1. **Saying LSTM "solves" vanishing gradients**: the cell state provides an alternative path but the hidden state path can still vanish. Gradients can still vanish over very long sequences.

2. **Confusing the cell state and hidden state**: $\mathbf{c}_t$ is the long-term memory, $\mathbf{h}_t$ is the working memory exposed to subsequent layers/outputs. Only $\mathbf{h}_t$ is used as input to the output layer; $\mathbf{c}_t$ is internal.

3. **Getting the GRU update equation wrong**: the update is $\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t$, not a direct replacement. When $\mathbf{z}_t = 0$ the hidden state is unchanged.

4. **Forgetting truncated BPTT detaches gradients**: the hidden state carried over from the previous segment does not propagate gradients back into it. This is intentional but means very long-range dependencies spanning multiple segments are not learned.

5. **Applying dropout to recurrent connections naively**: standard dropout on recurrent connections degrades performance because it disrupts temporal information flow. Variational dropout (same mask at every step) or Zoneout should be used instead.

---

## Quick Reference

**LSTM equations summary:**

$$\mathbf{f}_t = \sigma(W_f[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget)}$$

$$\mathbf{i}_t = \sigma(W_i[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input)}$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_c) \quad \text{(cell candidate)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell update)}$$

$$\mathbf{o}_t = \sigma(W_o[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden state)}$$

**GRU equations summary:**

$$\mathbf{z}_t = \sigma(W_z[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_z) \quad \text{(update)}$$

$$\mathbf{r}_t = \sigma(W_r[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_r) \quad \text{(reset)}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h[\mathbf{r}_t\odot\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_h) \quad \text{(candidate)}$$

$$\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t \quad \text{(hidden state)}$$

| Property | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| States | $\mathbf{h}_t$ | $\mathbf{h}_t, \mathbf{c}_t$ | $\mathbf{h}_t$ |
| Gates | 0 | 3 (forget, input, output) | 2 (update, reset) |
| Params vs RNN | 1x | 4x | 3x |
| Long-range deps | Poor | Good | Good |
| Training speed | Fast | Slow | Medium |
