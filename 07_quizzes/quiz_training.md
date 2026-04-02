# Quiz: Training

## Instructions

15 multiple-choice questions covering optimisers, learning rate scheduling, regularisation,
data augmentation, batch normalisation, and training best practices. Each question has
exactly one correct answer. Work through all questions before checking the answer key.

Difficulty distribution: Questions 1-5 Fundamentals, Questions 6-11 Intermediate,
Questions 12-15 Advanced.

---

## Questions

### Q1 (Fundamentals)

In mini-batch gradient descent, the gradient is computed on a random subset (batch) of
training data. Compared to full-batch gradient descent, mini-batch gradient descent:

- A) Always converges to a better solution because the noise helps escape local minima.
- B) Is faster per iteration and introduces gradient noise that can help escape sharp
     local minima, but produces a noisier estimate of the true gradient.
- C) Requires more memory because all samples must be loaded for each gradient step.
- D) Guarantees convergence to the global minimum for any loss function.

---

### Q2 (Fundamentals)

L2 weight decay in a neural network adds a penalty `lambda * sum(w_i^2)` to the loss.
The effect on the gradient update is:

- A) The gradient is multiplied by (1 - lambda) at each step.
- B) A term `2 * lambda * w` is added to the gradient, shrinking weights toward zero
     at each step.
- C) The learning rate is scaled by lambda at each step.
- D) The loss function becomes non-differentiable at w=0.

---

### Q3 (Fundamentals)

What is the purpose of a learning rate warm-up period at the beginning of training?

- A) To allow the model weights to stabilise before the loss function is applied.
- B) To gradually increase the learning rate from a small value, preventing large
     gradient updates early in training when weight estimates are unreliable.
- C) To increase the batch size gradually so GPU memory is used efficiently.
- D) To apply stronger regularisation at the start of training when overfitting risk is
     highest.

---

### Q4 (Fundamentals)

Dropout is set to rate p=0.5 during training. At test time, which of the following
is CORRECT?

- A) Dropout is applied with rate p=0.25 at test time (half the training rate).
- B) No dropout is applied; all neurons are active and no scaling is needed because
     the 1/(1-p) scaling was already applied during training.
- C) Dropout is applied with the same rate p=0.5 to reduce overconfidence.
- D) Dropout is only applied to the first layer at test time.

---

### Q5 (Fundamentals)

Data augmentation reduces overfitting primarily because:

- A) It increases the effective training set size by generating new, valid training
     examples, exposing the model to more diverse transformations of the input.
- B) It adds regularisation to the loss function in the form of an extra penalty term.
- C) It reduces the number of parameters in the model, reducing its capacity to overfit.
- D) It changes the model architecture by removing layers during training.

---

### Q6 (Intermediate)

Adam optimiser uses two moment estimates. Which of the following correctly describes
the first moment (m) and second moment (v)?

- A) m is the exponential moving average of gradients (biased estimate of mean);
     v is the exponential moving average of squared gradients (biased estimate of
     uncentred variance).
- B) m is the sum of all past gradients; v is the sum of all past squared gradients.
- C) m is the gradient at the current step; v is the gradient variance across the batch.
- D) m is the parameter value; v is the learning rate for that parameter.

---

### Q7 (Intermediate)

Cosine annealing learning rate schedule reduces the learning rate as:

    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

What is the learning rate at t=0 and at t=T?

- A) lr(0) = lr_min,  lr(T) = lr_max
- B) lr(0) = lr_max,  lr(T) = lr_min
- C) lr(0) = (lr_max + lr_min) / 2,  lr(T) = lr_min
- D) lr(0) = lr_max,  lr(T) = (lr_max + lr_min) / 2

---

### Q8 (Intermediate)

During training, you observe that training loss decreases smoothly but validation loss
first decreases then increases. Which regularisation technique would you NOT expect
to help?

- A) Adding L2 weight decay to the loss.
- B) Increasing the dropout rate.
- C) Adding more training data or augmentation.
- D) Increasing the model's depth and width.

---

### Q9 (Intermediate)

Gradient clipping by norm is applied when:
`if ||g|| > threshold: g = g * threshold / ||g||`

This ensures the gradient vector's L2 norm does not exceed `threshold`. When is gradient
clipping most commonly needed?

- A) In feedforward networks with ReLU activations, because ReLU can produce unbounded
     gradient magnitudes.
- B) In recurrent networks (LSTMs, GRUs) or very deep networks, where gradients can
     explode due to the long chain of multiplications through time or layers.
- C) In networks trained with Adam, because Adam's adaptive learning rates can cause
     individual parameter updates to be too large.
- D) When batch size is large, because large batches have more stable gradients that
     must be clipped to match single-sample gradient scale.

---

### Q10 (Intermediate)

Label smoothing replaces one-hot labels y with smoothed labels y_smooth:

    y_smooth = (1 - eps) * y + eps / K

where K is the number of classes and eps is a small constant (e.g., 0.1). What is the
PRIMARY benefit of label smoothing?

- A) It speeds up convergence by making the cross-entropy loss smaller.
- B) It prevents the model from becoming overconfident in its predictions, improving
     calibration and generalisation.
- C) It reduces the number of classes effectively, making the classification task easier.
- D) It is equivalent to increasing the learning rate by a factor of 1/(1-eps).

---

### Q11 (Intermediate)

You are training a ResNet-50 from scratch on a 50,000-image dataset. Which of the
following training strategies is most appropriate?

- A) Train with a very large learning rate (0.1) and no warm-up to converge quickly.
- B) Use transfer learning: load ImageNet pre-trained weights and fine-tune all layers
     with a small learning rate (1e-4 to 1e-3).
- C) Train with batch size 1 to get the most accurate gradient estimate per step.
- D) Use a fixed learning rate for the entire training run with no scheduling.

---

### Q12 (Advanced)

The "sharp minima" hypothesis suggests that solutions found by SGD (small batch) tend
to generalise better than those found by large-batch training. Which explanation best
captures the theoretical reasoning?

- A) SGD uses fewer samples per step and therefore trains more slowly, giving the model
     more epochs to explore the loss landscape.
- B) Small-batch SGD introduces higher gradient noise, which acts as an implicit
     regulariser that biases the optimiser toward flat minima. Flat minima have a wider
     basin of attraction and generalise better than sharp minima because a small shift
     in weights causes a smaller increase in loss.
- C) Large batches compute more accurate gradients, causing the model to overfit the
     gradient direction and memorise training samples.
- D) Small batches require more iterations per epoch, allowing more weight updates
     and therefore better coverage of the parameter space.

---

### Q13 (Advanced)

In AdamW (Adam with decoupled weight decay), weight decay is applied to the parameters
directly rather than as a gradient penalty. What problem does this solve compared to
Adam with L2 regularisation?

- A) It prevents the weight decay term from being inadvertently scaled by the adaptive
     learning rate (the second moment estimate), which would effectively reduce the
     regularisation strength for parameters with large gradients.
- B) It makes the optimiser converge faster by separating the loss gradient from the
     regularisation term.
- C) It avoids the need for bias correction in the moment estimates.
- D) It ensures that the weight decay is applied identically to both biases and weights.

---

### Q14 (Advanced)

You are training a large model and notice that the gradient norm explodes early in
training, then stabilises. Which of the following interventions is LEAST likely to
solve the exploding gradient problem?

- A) Reduce the initial learning rate.
- B) Use gradient clipping with an appropriate threshold.
- C) Apply Xavier or Kaiming weight initialisation to ensure initial activations and
     gradients have appropriate variance.
- D) Increase the batch size.

---

### Q15 (Advanced)

The "linear scaling rule" for learning rate with large batch training states that if you
multiply the batch size by k, you should multiply the learning rate by k. This is derived
from:

- A) The requirement that the total number of weight updates per epoch remains constant.
- B) The requirement that the expected parameter update per epoch is approximately
     constant: with k times more samples per batch, the gradient is k times more
     accurate, so each step is k times larger and the learning rate must compensate
     by scaling k times.
- C) The requirement that the gradient variance per sample is preserved, which requires
     scaling the learning rate proportionally to batch size.
- D) The requirement that training loss at the end of the first epoch is the same
     regardless of batch size.

---

## Answer Key

| Q  | Answer | Difficulty    |
|----|--------|---------------|
| 1  | B      | Fundamentals  |
| 2  | B      | Fundamentals  |
| 3  | B      | Fundamentals  |
| 4  | B      | Fundamentals  |
| 5  | A      | Fundamentals  |
| 6  | A      | Intermediate  |
| 7  | B      | Intermediate  |
| 8  | D      | Intermediate  |
| 9  | B      | Intermediate  |
| 10 | B      | Intermediate  |
| 11 | B      | Intermediate  |
| 12 | B      | Advanced      |
| 13 | A      | Advanced      |
| 14 | D      | Advanced      |
| 15 | C      | Advanced      |

---

## Detailed Explanations

### Q1 - Answer: B

Mini-batch gradient descent computes the gradient on a subset of data (batch), which is:
- Faster per iteration than full-batch (only B samples to process, not N).
- Noisier than full-batch: the batch gradient is an unbiased but high-variance estimate
  of the true gradient.
- The noise is often beneficial: it helps escape sharp local minima and saddle points
  that would trap full-batch gradient descent.

- **A** is wrong: mini-batch does not always converge to a better solution. The noise
  can also prevent convergence to the exact optimum; learning rate decay is needed.
- **C** is wrong: mini-batch requires less memory (only B samples loaded at once).
- **D** is wrong: convergence to the global minimum is not guaranteed for non-convex
  losses regardless of the method.

---

### Q2 - Answer: B

L2 regularisation modifies the loss:
```
L_total = L_data + lambda * sum(w_i^2)
dL_total/dw = dL_data/dw + 2 * lambda * w
```
The update becomes: `w <- w - lr * (grad + 2*lambda*w)`.

This is equivalent to `w <- w * (1 - 2*lr*lambda) - lr * grad`. Each step, the weight
is scaled by a factor slightly less than 1 (shrunk toward zero) before the gradient step.

- **A** describes the multiplicative form of weight decay, which is equivalent to L2 only
  for standard SGD -- not for adaptive optimisers. This is the AdamW insight (Q13).
- **C** is wrong: the learning rate is not scaled by lambda.
- **D** is wrong: the L2 penalty sum(w^2) is smooth and differentiable everywhere
  including at w=0.

---

### Q3 - Answer: B

At the start of training, weights are randomly initialised and far from their optimal
values. A large learning rate applied immediately can cause wildly large parameter
updates that destabilise training (loss spikes, or divergence). Warm-up gradually
increases the learning rate from a small initial value over the first few epochs,
allowing the optimiser to make conservative early steps while weights are uncertain.

- **A** is wrong: weights are updated at every step; "stabilising before loss is applied"
  is not a meaningful concept.
- **C** describes batch size warmup, which is a different (less common) technique.
- **D** is wrong: strong regularisation at the start is not the purpose of warm-up.

---

### Q4 - Answer: B

With inverted dropout (the standard implementation), activations that survive (are not
zeroed) are scaled by 1/(1-p) during training. This ensures that the expected activation
value during training equals the expected value at test time (when all neurons are active
and no scaling is applied). At test time, no dropout and no scaling is needed.

- **A** is wrong: standard dropout is not applied at test time at all.
- **C** is wrong: applying dropout at test time introduces randomness in predictions and
  is only done in specific cases (e.g., Monte Carlo dropout for uncertainty estimation).
- **D** is wrong: dropout is disabled entirely at test time in standard usage.

---

### Q5 - Answer: A

Data augmentation creates valid variants of training examples (flips, crops, colour
jitter, rotations) that the model might encounter in deployment. This effectively
multiplies the training set size, exposing the model to more diverse inputs and making
it harder to memorise specific training instances.

- **B** is wrong: augmentation does not add a penalty term to the loss function.
- **C** is wrong: augmentation does not change model architecture or reduce parameters.
- **D** is wrong: augmentation does not remove layers.

---

### Q6 - Answer: A

Adam maintains:
```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t         (EMA of gradient)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (EMA of squared gradient)
```
Both are biased toward zero at t=0. Bias-corrected:
```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```
Update: `w -= lr * m_hat / (sqrt(v_hat) + eps)`.

- **B** is wrong: cumulative sums would grow without bound; EMA has a finite memory.
- **C** is wrong: m is a smoothed estimate across time, not the current gradient alone.
- **D** is wrong: neither m nor v is the parameter value or the learning rate itself.

---

### Q7 - Answer: B

At t=0: cos(pi*0/T) = cos(0) = 1.
```
lr(0) = lr_min + 0.5 * (lr_max - lr_min) * (1 + 1) = lr_min + (lr_max - lr_min) = lr_max
```

At t=T: cos(pi*T/T) = cos(pi) = -1.
```
lr(T) = lr_min + 0.5 * (lr_max - lr_min) * (1 - 1) = lr_min + 0 = lr_min
```

The schedule starts at lr_max and cosine-anneals down to lr_min over T steps.

- **A** has the values swapped.
- **C** would require cos = 0 at t=0, which is only true at t=T/2.
- **D** would require cos = 0 at t=T, which is only true at t=T/2.

---

### Q8 - Answer: D

Training loss decreasing while validation loss increases is classic overfitting (high
variance). The model has excess capacity relative to the data.

- **A** L2 weight decay reduces overfitting by penalising large weights.
- **B** Higher dropout reduces overfitting by preventing co-adaptation of neurons.
- **C** More data or augmentation reduces overfitting by providing more signal.
- **D** WRONG: Increasing model depth and width INCREASES capacity and would worsen
  overfitting. This is the correct answer -- it would NOT help.

---

### Q9 - Answer: B

Gradient explosion is most common in:
- **RNNs**: gradients flow through T time steps, and repeated multiplication by the
  weight matrix can cause exponential growth if the singular values exceed 1.
- **Very deep networks**: the product of L Jacobians can grow if each has eigenvalues > 1.

Gradient clipping truncates the gradient vector when its norm exceeds a threshold,
preventing catastrophically large parameter updates.

- **A** is wrong: ReLU does not cause gradient explosion in feedforward networks (its
  derivative is 0 or 1, neither of which amplifies gradients).
- **C** is wrong: Adam normalises per-parameter step sizes but does not inherently cause
  gradient explosion; clipping is rarely needed with Adam.
- **D** is wrong: large batches produce more stable gradients, which reduces the need
  for clipping.

---

### Q10 - Answer: B

Hard one-hot labels encourage the model to assign probability 1 to the correct class
and 0 to all others. This causes the logits (and therefore weights) to grow without
bound during training ("overconfident" behaviour). Overconfident models have poor
calibration (their probability estimates do not match empirical frequencies).

Label smoothing prevents this: the target is now (1-eps, eps/(K-1), ...) instead of
(1, 0, 0, ...). The model can no longer achieve zero cross-entropy loss by being
infinitely confident. This acts as a soft regulariser and typically improves
generalisation and calibration by a small but consistent margin.

- **A** is wrong: label smoothing increases the minimum achievable loss, potentially
  slowing convergence slightly.
- **C** is wrong: label smoothing does not reduce the number of classes.
- **D** is wrong: there is no direct equivalence to learning rate scaling.

---

### Q11 - Answer: B

For a 50,000-image dataset, training ResNet-50 from scratch is challenging because the
dataset is relatively small (ImageNet has ~1.2M images). Transfer learning is the
appropriate choice:
1. Load ImageNet pre-trained weights (already encode rich visual features).
2. Fine-tune all layers with a small learning rate to adapt to the target task.
3. Optionally freeze early layers initially if the dataset is very small.

- **A** is wrong: lr=0.1 without warm-up on a fine-tuning task will cause catastrophic
  forgetting of pre-trained features.
- **C** is wrong: batch size 1 is extremely slow, has very high gradient variance, and
  interacts poorly with BatchNorm (per-batch statistics from a single sample are useless).
- **D** is wrong: a fixed learning rate without scheduling typically leads to slower
  convergence and suboptimal final accuracy compared to scheduling.

---

### Q12 - Answer: B

The sharp-vs-flat minima generalisation theory (Hochreiter & Schmidhuber 1997, Keskar
et al. 2017): sharp minima have large positive curvature in many directions. A small
perturbation of the weights (as naturally occurs between training and test distribution)
causes a large increase in loss. Flat minima are less sensitive to weight perturbations
and tend to generalise better.

Small-batch SGD has higher gradient noise, which acts as an implicit regulariser
preferring flat minima. Large batches have lower noise and tend to converge to sharper
minima near the starting point.

- **A** is wrong: more epochs are not the mechanism -- noise during training is.
- **C** is wrong: "memorising the gradient direction" is not a meaningful mechanism.
- **D** is wrong: more weight updates do not directly cause flat-minima preference.

---

### Q13 - Answer: A

Adam with L2 regularisation: the gradient includes the L2 penalty term `2*lambda*w`.
Adam then divides this gradient by the adaptive scaling factor sqrt(v_hat). For parameters
with large gradients, v_hat is large, so the weight decay term `2*lambda*w / sqrt(v_hat)`
is reduced. Parameters with large gradients receive less effective regularisation.

AdamW separates weight decay:
```
w_t+1 = w_t - lr * m_hat / (sqrt(v_hat) + eps)   [gradient step]
w_t+1 *= (1 - lr * lambda)                          [weight decay applied directly]
```
The weight decay is independent of the adaptive scaling, preserving the intended
regularisation strength for all parameters regardless of gradient magnitude.

- **B** is wrong: decoupling does not primarily affect convergence speed.
- **C** is wrong: bias correction is still needed and still applied.
- **D** is wrong: AdamW intentionally does NOT apply decay to biases in most
  implementations (biases are excluded from weight decay).

---

### Q14 - Answer: D

Increasing batch size has no direct effect on gradient explosion. Gradient explosion
is caused by large Jacobians in the chain rule product (due to weight initialisation,
architecture, or learning rate), not by batch size.

The other three options directly address explosion:
- **A** Reducing learning rate reduces the size of each parameter update, limiting damage
  from one explosive gradient.
- **B** Gradient clipping directly truncates gradients that exceed the threshold.
- **C** Good weight initialisation (Xavier/Kaiming) ensures activations and gradients
  start with appropriate scale, preventing early explosion.

- **D** is correct as the LEAST likely to help: batch size does not affect the gradient
  chain rule computation or initial weight scales.

---

### Q15 - Answer: C

The linear scaling rule derivation (Goyal et al., 2017):

With batch size B and learning rate lr, one SGD step updates parameters by:
```
w <- w - lr * (1/B) * sum_{i=1}^{B} g_i
```
The gradient variance per step is `sigma^2 / B` (variance of the mean of B samples).

For a fair comparison when scaling batch size from B to kB:
- The gradient signal (mean) stays the same.
- The gradient variance decreases by k.
- To produce parameter updates of the same variance (same effective step distribution),
  the learning rate should scale by k.

This preserves the statistical properties of each update step.

- **A** is partially related but not the precise derivation. The rule is about gradient
  variance, not the number of updates.
- **B** describes an incorrect mechanism: the gradient mean (not accuracy) stays constant
  with more samples; variance decreases.
- **D** is an empirical observation that sometimes holds, not the derivation.
