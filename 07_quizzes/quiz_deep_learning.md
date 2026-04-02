# Quiz: Deep Learning

## Instructions

15 multiple-choice questions covering neural network fundamentals, backpropagation,
activation functions, loss functions, CNNs, RNNs, and attention mechanisms. Each question
has exactly one correct answer. Work through all questions before checking the answer key.

Difficulty distribution: Questions 1-5 Fundamentals, Questions 6-11 Intermediate,
Questions 12-15 Advanced.

---

## Questions

### Q1 (Fundamentals)

The Universal Approximation Theorem states that a feedforward neural network with:

- A) Infinitely many layers and a linear activation can approximate any continuous function.
- B) A single hidden layer of sufficient width and a non-linear activation function can
     approximate any continuous function on a compact domain to arbitrary precision.
- C) Exactly three hidden layers and ReLU activations can exactly represent any function.
- D) Polynomial activation functions can approximate any function with a fixed number
     of neurons.

---

### Q2 (Fundamentals)

In backpropagation, the gradient of the loss with respect to a weight W in layer l is
computed using:

- A) The chain rule applied forward through the network from input to output.
- B) The chain rule applied backward from the loss through all downstream layers.
- C) Finite difference approximation of the gradient.
- D) The transpose of the weight matrix in the next layer.

---

### Q3 (Fundamentals)

What is the "dying ReLU" problem?

- A) ReLU activations saturate at +1 for large positive inputs, halting gradient flow.
- B) Neurons with ReLU activation receive large negative pre-activations and output zero,
     resulting in zero gradients that prevent any future updates to that neuron's weights.
- C) ReLU networks cannot be trained with standard SGD because the gradient is undefined
     at zero.
- D) ReLU functions cause vanishing gradients in all layers due to their bounded output.

---

### Q4 (Fundamentals)

In a convolutional layer with input H x W x C_in, kernel K x K x C_in x C_out,
padding P, and stride S, what is the output spatial size (H_out, assuming H = W)?

- A) floor((H - K) / S) + 1
- B) floor((H + 2P - K) / S) + 1
- C) H * K / S
- D) (H - K + P) / S + 1

---

### Q5 (Fundamentals)

Which of the following activation functions is most commonly used in the hidden layers
of modern deep neural networks and why?

- A) Sigmoid: bounded output [0,1] ensures stable gradients throughout training.
- B) Tanh: zero-centred output reduces the zig-zagging gradient updates seen with sigmoid.
- C) ReLU: computationally cheap, does not saturate for positive inputs (avoiding
     vanishing gradients), and is empirically very effective.
- D) Softmax: normalises activations to a probability distribution for stable training.

---

### Q6 (Intermediate)

Batch Normalisation normalises activations within a mini-batch. Which of the following
correctly describes what happens at INFERENCE time with Batch Normalisation?

- A) Activations are normalised using the mean and variance of the current inference batch.
- B) The BN layer is removed and replaced with the identity function.
- C) Activations are normalised using running mean and running variance statistics
     accumulated during training.
- D) The BN layer uses the global mean and variance of the entire training set,
     recomputed from scratch at the start of each inference run.

---

### Q7 (Intermediate)

In an LSTM cell, which gates control the flow of information into and out of the cell
state C_t?

- A) The input gate controls what new information enters; the output gate controls what
     is read out; the forget gate removes old cell state content.
- B) The forget gate controls new information entering; the input gate controls
     discarding old state; the output gate modulates the hidden state.
- C) The input gate alone controls all information flow; the forget and output gates
     are regularisation mechanisms.
- D) The cell state is not gated; it receives the full input and hidden state each step.

---

### Q8 (Intermediate)

ResNet introduced skip connections. The key insight is that it is easier to learn:

- A) The full mapping F(x) = x, because identity mappings have zero loss.
- B) The residual F(x) = H(x) - x rather than the full desired mapping H(x), because
     if the optimal transformation is near-identity, F(x) can be driven to near-zero,
     which is easier to learn.
- C) Large weights in deep networks because skip connections amplify gradients.
- D) Non-linear functions by bypassing the non-linear activation layers.

---

### Q9 (Intermediate)

Cross-entropy loss is preferred over MSE loss for classification tasks because:

- A) Cross-entropy is always smaller than MSE, leading to faster convergence.
- B) Cross-entropy combined with softmax produces a gradient proportional to the
     prediction error (p - y), avoiding the vanishing gradient problem that arises when
     MSE is paired with sigmoid/softmax output activations.
- C) MSE cannot represent multi-class problems while cross-entropy can.
- D) Cross-entropy loss is convex while MSE is non-convex for classification.

---

### Q10 (Intermediate)

Xavier/Glorot uniform initialisation scales initial weights to maintain activation
variance across layers. Its formula for a layer with fan_in inputs and fan_out outputs is:

- A) W ~ Uniform(-sqrt(6 / fan_in), sqrt(6 / fan_in))
- B) W ~ Uniform(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
- C) W ~ Normal(0, 1 / fan_in)
- D) W ~ Normal(0, sqrt(2 / fan_in))

---

### Q11 (Intermediate)

In multi-head attention, why are Q, K, V projected to dimension d_k = d_model / n_heads
rather than using the full d_model for each head?

- A) Smaller dimensions make the attention weights easier to interpret visually.
- B) Multiple heads at d_model / n_heads have the same total computational cost as a
     single head at d_model, while allowing the model to attend to different
     representation subspaces simultaneously.
- C) Smaller Q and K dimensions ensure dot products never overflow floating-point range.
- D) The projections reduce memory usage by eliminating redundant input features.

---

### Q12 (Advanced)

In scaled dot-product attention, scores are divided by sqrt(d_k) before softmax.
Without this scaling, what problem arises as d_k becomes large?

- A) The dot products grow in magnitude, pushing softmax into a saturated region with
     very small gradients, causing slow or stalled learning.
- B) The dot products become too small, causing numerical underflow in the softmax.
- C) The Q and K matrices develop rank-1 structure, collapsing attention to one token.
- D) The gradient of the loss with respect to Q and K becomes undefined.

---

### Q13 (Advanced)

For a sequence of length T = 1024, d_model = 512, n_heads = 8: which operation has
higher computational complexity, a 1D convolution (K=3, C_in=C_out=512) or self-attention,
and what is the dominant asymptotic term?

- A) Convolution dominates: O(T * K * C^2).
- B) Self-attention dominates: O(T^2 * d_model) due to the T x T attention matrix.
- C) They have identical complexity at all sequence lengths.
- D) Convolution dominates for T > 512; attention dominates for T < 512.

---

### Q14 (Advanced)

The vanishing gradient problem in sigmoid networks arises because the sigmoid derivative
sigma'(z) = sigma(z)(1 - sigma(z)) has a maximum value of 0.25. In a network of depth L,
the gradient at layer 1 is bounded by:

- A) (0.25)^L, which decays exponentially to zero as L increases.
- B) 0.25 * L, which grows linearly but stays manageable.
- C) 0, because the sigmoid derivative is identically zero everywhere.
- D) 1, because gradients are normalised by Batch Normalisation in all modern networks.

---

### Q15 (Advanced)

Dropout with rate p zeroes activations during training and scales remaining ones by
1/(1-p). At test time no dropout is applied. What property does the 1/(1-p) scaling
preserve?

- A) The L2 norm of the weight matrix is constant between training and test.
- B) The expected activation value at test time (all neurons active) equals the expected
     activation value during training (with random neuron zeroing).
- C) Activations are bounded to [0, 1] during training to prevent exploding activations.
- D) The model sees exactly p fraction of its neurons during every forward pass.

---

## Answer Key

| Q  | Answer | Difficulty    |
|----|--------|---------------|
| 1  | B      | Fundamentals  |
| 2  | B      | Fundamentals  |
| 3  | B      | Fundamentals  |
| 4  | B      | Fundamentals  |
| 5  | C      | Fundamentals  |
| 6  | C      | Intermediate  |
| 7  | A      | Intermediate  |
| 8  | B      | Intermediate  |
| 9  | B      | Intermediate  |
| 10 | B      | Intermediate  |
| 11 | B      | Intermediate  |
| 12 | A      | Advanced      |
| 13 | B      | Advanced      |
| 14 | A      | Advanced      |
| 15 | B      | Advanced      |

---

## Detailed Explanations

### Q1 - Answer: B

The Universal Approximation Theorem (Cybenko 1989, Hornik 1991) states that a single
hidden layer with sufficient neurons and a non-constant, bounded, continuous activation
function can approximate any continuous function on a compact subset of R^n to arbitrary
precision.

- **A** is wrong: linear activations make the network equivalent to a single linear
  transformation regardless of depth.
- **C** is wrong: "exact" representation is not what UAT states; it is approximation
  to epsilon precision with sufficient width.
- **D** is wrong: polynomial activations have universal approximation properties in
  specific senses but are not the standard statement of UAT.

---

### Q2 - Answer: B

Backpropagation applies the chain rule **backwards** from the loss L through the network.
The gradient at layer l depends on gradients at layer l+1 (downstream):
```
dL/dW_l = dL/dA_{l+1} * dA_{l+1}/dZ_l * dZ_l/dW_l
```
Computation starts from dL/d(output), which is trivial, and propagates backwards.

- **A** is wrong: the chain rule in backprop is applied backwards, not forwards.
- **C** describes numerical gradient checking, not backpropagation.
- **D** is a partial description of one step in the chain (the error signal involves
  W_{l+1}^T) but is not the complete definition.

---

### Q3 - Answer: B

The dying ReLU problem: if a neuron receives a large negative pre-activation, its ReLU
output is permanently zero. The gradient of ReLU for negative inputs is also zero, so no
gradient flows back and the weights cannot recover. Once dead, a neuron stays dead.
Causes include large learning rates and poor weight initialisation.

- **A** describes sigmoid saturation. ReLU does not saturate for positive inputs.
- **C** is wrong: the gradient is zero (not undefined) for negative inputs. In practice
  the subgradient convention assigns 0 at the kink, and training works fine.
- **D** is wrong: ReLU is unbounded for positive inputs and does not cause vanishing
  gradients in active neurons.

---

### Q4 - Answer: B

The output size formula:
```
H_out = floor((H_in + 2*P - K) / S) + 1
```
Example: H=32, K=3, P=1, S=1: H_out = floor((32 + 2 - 3) / 1) + 1 = 32. Same size.
Example: H=32, K=3, P=0, S=2: H_out = floor((32 - 3) / 2) + 1 = 15.

- **A** is the P=0 special case.
- **C** is dimensionally incorrect and does not match any standard formula.
- **D** omits the floor and uses P instead of 2P.

---

### Q5 - Answer: C

ReLU became dominant because: (1) its gradient is 1 for positive inputs (no vanishing);
(2) it is computationally trivial (a comparison); (3) it induces sparse activations;
(4) it consistently outperforms sigmoid and tanh empirically on deep networks.

- **A** is wrong: sigmoid saturates at 0 and 1 (gradient near zero), causing vanishing
  gradients in deep networks.
- **B** is partially correct but tanh also saturates, suffering from vanishing gradients.
- **D** is wrong: softmax is an output activation for multi-class output, not for hidden
  layers. Applying softmax to hidden layers creates unnecessary competition.

---

### Q6 - Answer: C

During training, BN uses per-batch statistics AND maintains exponential moving averages
(running_mean, running_var) with a momentum parameter. At inference, it uses these
running statistics. This makes inference deterministic (independent of batch composition).

- **A** describes training mode. Inference on a single sample would give meaningless
  statistics if per-batch normalisation were used.
- **B** is wrong: BN is kept at inference but uses stored statistics.
- **D** is computationally infeasible for large datasets and is not what BN does.

---

### Q7 - Answer: A

LSTM gate functions:
- **Input gate** i_t: sigmoid; controls what new information enters the cell state.
- **Forget gate** f_t: sigmoid; controls what fraction of C_{t-1} to retain.
- **Output gate** o_t: sigmoid; controls what is exposed from cell state to h_t.
- **Cell input** g_t: tanh; the new candidate values to add to cell state.

Update: C_t = f_t * C_{t-1} + i_t * g_t; h_t = o_t * tanh(C_t).

- **B** has the roles of forget and input gates swapped.
- **C** is wrong: each gate performs a distinct and essential function.
- **D** is wrong: the cell state is the most carefully controlled part of the LSTM.

---

### Q8 - Answer: B

He et al. (2016): reformulate the layer as H(x) = F(x) + x. If the optimal H(x) ≈ x
(near-identity), then F(x) = H(x) - x ≈ 0, which is easy to learn (small weights).
Without skip connections, the network must learn that all weights produce an identity
output, which is much harder. Skip connections also provide a direct gradient path,
mitigating vanishing gradients.

- **A** is wrong: the residual F(x), not the full mapping, needs to be driven toward 0.
- **C** is wrong: skip connections ensure gradient flow but do not amplify in a harmful way.
- **D** is wrong: skip connections add identity paths, not bypass activations.

---

### Q9 - Answer: B

With MSE + sigmoid output, the gradient includes sigma'(z) which saturates near 0 and 1.
Confident wrong predictions (sigma near 0 or 1) produce near-zero gradients, causing
very slow correction.

The cross-entropy + softmax gradient simplifies to:
```
dL/dz_k = p_k - y_k
```
Large when confidently wrong (p_k ≈ 1 but y_k = 0), small when correct. No saturation.

- **A** is wrong: the relative magnitude depends on the specific predictions.
- **C** is wrong: MSE can be applied to multi-class problems as multi-output regression.
- **D** is wrong: both are non-convex when combined with a neural network.

---

### Q10 - Answer: B

Xavier initialisation: W ~ Uniform(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out)).

Derived assuming linear activations; requires that Var(activations) and Var(gradients)
are preserved across layers. The denominator fan_in + fan_out balances both conditions.

- **A** is LeCun uniform initialisation (fan_in only).
- **C** is LeCun normal initialisation.
- **D** is Kaiming/He normal initialisation for ReLU (factor 2 because ReLU discards
  half the activations, doubling the required variance).

---

### Q11 - Answer: B

With d_model = 512, n_heads = 8, d_k = 64:
- Total parameters for Q projection (all heads): 8 * (512 * 64) = 262,144 = 512 * 512.
- This equals the parameter count of a single head at d_model.

So multi-head attention is NOT more expensive than single-head attention of the same
total size. The benefit is that 8 different heads can each learn to attend to different
types of relationships (positional, syntactic, semantic) independently.

- **A** is wrong: interpretability is a post-hoc observation, not the design motivation.
- **C** is wrong: float32 handles d_k = 512 dot products without overflow.
- **D** is wrong: the projection changes the subspace, it does not remove redundancy per se.

---

### Q12 - Answer: A

q^T k is a sum of d_k products of unit-variance random variables. Its variance scales as
d_k (sum of d_k unit-variance terms). For large d_k, the dot products have large standard
deviation (sqrt(d_k)), pushing softmax inputs into the saturated regime.

Softmax saturation: one logit >> others -> softmax output approaches one-hot. The gradient
of softmax at saturation is near zero. Dividing by sqrt(d_k) normalises the variance back
to 1 and keeps softmax in its sensitive region.

- **B** is wrong: the problem is large values causing saturation, not small values.
- **C** is wrong: rank collapse is a distinct phenomenon (related to over-smoothing in
  attention, not scaling).
- **D** is wrong: the gradient is defined; the problem is it becomes very small.

---

### Q13 - Answer: B

Convolution FLOPs: O(T * K * C_in * C_out) = O(T * 3 * 512 * 512) = O(T).

Self-attention:
- QKV projections: O(T * d_model^2) = O(T) for fixed d_model.
- Attention matrix: O(T^2 * d_model) -- each of T query positions attends to T keys,
  each of dimension d_model.

At T = 1024: attention matrix computation ~ T^2 * 512 ≈ 5.4 * 10^8 multiplications,
which dominates both the projection O(T) term and the convolution O(T) term.

- **A** is wrong: convolution is O(T) which is dominated by attention's O(T^2) term.
- **C** is wrong: they differ asymptotically (T vs T^2).
- **D** has the crossover direction wrong.

---

### Q14 - Answer: A

Sigmoid derivative maximum = 0.25 at z = 0. For L layers, each contributing at most
0.25 to the chain rule product:
```
|dL/dW_1| <= (max sigma')^L = (0.25)^L
```
At L = 10: (0.25)^10 ≈ 10^{-6}. At L = 20: ≈ 10^{-12}. Training the first layers
becomes effectively impossible. This is the vanishing gradient problem.

- **B** is wrong: the bound is multiplicative (exponential), not additive (linear).
- **C** is wrong: sigmoid derivative is nonzero (maximum 0.25), just small.
- **D** is wrong: Batch Normalisation addresses activation scale but does not guarantee
  gradient flow through sigmoid saturated regions.

---

### Q15 - Answer: B

With dropout rate p, training expected activation for a neuron with value a:
```
E[activation during training] = (1-p) * (a / (1-p)) + p * 0 = a
```
Test time (no dropout): activation = a.

Both are equal to a. The 1/(1-p) scaling ensures the expected activation matches between
training (where dropout reduces effective capacity) and test time (full capacity).

Without scaling, training-time expected activation would be (1-p)*a, which is smaller
than test-time activation a, causing a distribution mismatch that degrades test accuracy.

- **A** is wrong: dropout does not directly constrain weight norms.
- **C** is wrong: 1/(1-p) can exceed 1 (e.g., p=0.5 gives 2x scale), expanding not
  bounding activations.
- **D** is wrong: exactly p fraction of neurons are zeroed, so 1-p fraction remain active.
