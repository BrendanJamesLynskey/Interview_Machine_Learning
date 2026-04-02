# Quantisation and Pruning

## Prerequisites
- Neural network training and inference
- Floating-point number representation (FP32, FP16, INT8)
- Basic understanding of hardware compute and memory bandwidth

---

## Concept Reference

### Why Model Compression Matters

A ResNet-50 has ~25 million parameters stored as FP32 (32-bit floats), consuming ~100 MB
of memory. A BERT-base has ~110 million parameters (~440 MB). Deploying these models on
edge devices (microcontrollers, mobile phones) or serving them at high throughput in data
centres creates hard constraints on:

- **Memory footprint:** Model weights must fit in device memory (RAM or SRAM).
- **Inference latency:** Compute time per input must meet real-time requirements.
- **Energy consumption:** Battery-powered devices have strict power budgets.
- **Throughput:** A server must handle thousands of requests per second cost-effectively.

Model compression techniques reduce these costs, ideally with minimal accuracy loss.

---

### Quantisation

Quantisation reduces the numerical precision of weights and/or activations from FP32 to a
lower-precision format (FP16, BF16, INT8, INT4).

#### Why Lower Precision Helps

- **Memory:** INT8 uses 4x less memory than FP32. INT4 uses 8x less.
- **Compute:** INT8 arithmetic is faster than FP32 on most hardware (SIMD units, tensor
  cores, dedicated INT8 engines on NPUs). NVIDIA tensor cores execute INT8 at ~4x the
  throughput of FP32.
- **Bandwidth:** Smaller weights reduce the memory bandwidth bottleneck, which dominates
  inference of large models.

#### Linear (Uniform) Quantisation

The core operation maps a floating-point value x to a k-bit integer:

```
x_quant = round((x - zero_point) / scale)

scale     = (x_max - x_min) / (2^k - 1)
zero_point = round(-x_min / scale)

Dequantise: x_approx = x_quant * scale + zero_point
```

For symmetric quantisation around zero, `zero_point = 0`:
```
scale     = max(|x|) / (2^(k-1) - 1)
x_quant   = round(x / scale)
x_approx  = x_quant * scale
```

**Quantisation error:** The approximation error per element is bounded by `scale/2`.
Finer scale (smaller range) reduces error but clips values outside the range.

#### Post-Training Quantisation (PTQ)

Quantise a trained FP32 model without retraining. Requires a small **calibration dataset**
(a few hundred representative samples) to determine the range (scale and zero-point) for
each layer's weights and activations.

```
Calibration steps:
1. Run calibration data through the FP32 model.
2. Collect activation statistics (min, max, or percentiles) at each layer.
3. Compute scale and zero_point for each layer.
4. Replace FP32 operations with quantised equivalents.
```

**PTQ variants:**
- **Weight-only quantisation:** Quantise weights to INT8 (or INT4); keep activations
  in FP16 or FP32. Simple and effective; no calibration data needed for weights (can
  use per-channel min/max of the weight tensor directly).
- **Full integer quantisation (weights + activations):** Requires calibration. Achieves
  maximum hardware speedup on INT8 accelerators.
- **Dynamic quantisation:** Quantise weights statically but compute activation ranges
  dynamically at inference time. Higher accuracy, slightly lower speedup.

**Accuracy impact:** PTQ typically causes less than 1 % accuracy drop for INT8 on
well-trained models. INT4 PTQ often causes noticeable accuracy drops and may require
quantisation-aware training.

#### Quantisation-Aware Training (QAT)

Simulate quantisation during the forward pass of training using "fake quantisation"
nodes. Gradients flow through these nodes via the straight-through estimator (the
rounding function's gradient is approximated as 1).

```
Forward:  x_fake_quant = round(x / scale) * scale
Backward: dx_fake_quant/dx = 1  (straight-through estimator)
```

The model learns to be robust to quantisation noise, typically recovering most of the
accuracy lost by PTQ, especially at INT4 or lower.

**QAT is preferred when:**
- PTQ accuracy loss is unacceptable (common at INT4 and below).
- The model has sensitive layers (e.g., attention softmax, batch norm statistics).
- Maximum accuracy at a given bit-width is required.

**QAT requires:** Access to training data and a training compute budget. Typically
requires only 10-20 % of the original training compute for fine-tuning to recover accuracy.

---

### Knowledge Distillation

Distillation trains a small "student" model to mimic a large "teacher" model, transferring
the teacher's knowledge beyond what the labels alone provide.

#### Why Distillation Works

Hard labels (one-hot vectors) contain only class identity information. The teacher's
**soft logits** contain richer information: class similarities, confidence structure, and
the relative probabilities of confusable classes.

Example: a teacher's softmax outputs [0.8, 0.15, 0.05] for classes [cat, dog, bird]
reveals that the image looks somewhat like a dog, which the hard label [1, 0, 0] does not.

#### The Distillation Loss

```
L_distill = alpha * L_CE(soft_student, soft_teacher, T)
           + (1 - alpha) * L_CE(student_logits, hard_labels)

soft outputs: softmax(logits / T)  where T is temperature
```

**Temperature T:** Flattens the softmax distribution (T > 1), preventing very confident
teacher predictions from acting like hard labels and making the similarity structure more
informative. T = 2-5 is common.

**alpha:** Weight on the distillation loss. alpha = 0.7-0.9 is typical.

#### Feature-Level Distillation

Beyond output logits, intermediate feature maps can also be distilled. The student is
trained to produce feature maps similar (in terms of L2 distance or attention maps) to
the teacher's. This is especially useful when student and teacher architectures differ
significantly.

---

### Pruning

Pruning identifies and removes parameters (weights, neurons, attention heads, or entire
layers) that contribute little to the model's output.

#### Unstructured Pruning

Remove individual weights below a threshold. Results in sparse weight tensors.

```
mask[i,j] = 1 if |W[i,j]| > threshold else 0
W_pruned   = W * mask
```

**Magnitude pruning** is the simplest criterion: remove weights closest to zero.
Iterative magnitude pruning (prune a fraction, retrain, repeat) is more effective than
one-shot pruning.

**Challenge:** Sparse matrix arithmetic is only faster than dense on hardware with
dedicated sparse compute units (e.g., NVIDIA A100 Ampere sparse tensor cores at 2:4
sparsity). Generic hardware executes sparse matrices at the same speed as dense matrices
because memory access patterns are irregular.

#### Structured Pruning

Remove entire structures: neurons, convolutional filters, attention heads, or Transformer
layers. The resulting network has reduced dense weight matrices and accelerates on all
hardware without sparse compute support.

```
Filter pruning: remove the k-th convolutional filter by deleting row k from the
weight tensor W of shape [out_channels, in_channels, kH, kW].
The corresponding feature map channel is also removed from the next layer's input.
```

**Pruning criteria for structured pruning:**
- **L1-norm of filter weights:** Filters with small L1 norm have little activation.
- **Activation statistics:** Filters whose output activations are small on a calibration
  set contribute little.
- **Taylor expansion:** Estimate the change in loss from removing a unit using first-order
  Taylor expansion of the loss with respect to activations.

#### Lottery Ticket Hypothesis (LeCun et al., revisited by Frankle & Carlin 2019)

A randomly initialised dense network contains small subnetworks ("winning tickets") that,
when trained in isolation from their original initialisation, match the full network's
accuracy. This suggests that large networks are over-parameterised and the redundancy
is exploited during optimisation.

Practical implication: prune-then-retrain-from-scratch (using the original initialisation
of the surviving weights) can recover accuracy that magnitude pruning without re-initialisation loses.

---

### Combined Workflows in Practice

**Typical edge deployment pipeline:**
```
1. Train FP32 model to target accuracy.
2. Apply structured pruning to reduce FLOPs by 2-4x (remove filters/heads).
3. Fine-tune pruned model to recover accuracy.
4. Apply PTQ (INT8) for final deployment.
5. Evaluate on target hardware (latency, power, accuracy).
6. If accuracy is insufficient, apply QAT on the pruned model.
```

**Knowledge distillation + quantisation:**
Train a compact student model (via distillation) and then apply PTQ to the student.
The student is smaller, making INT8 quantisation error proportionally smaller relative
to the model's capacity.

---

## Interview Questions by Difficulty

### Fundamentals

**Q1.** What is the difference between FP32, FP16, and INT8 in terms of range and
precision? Why does quantisation to INT8 often hurt accuracy?

**Answer:**

- **FP32:** 32 bits (1 sign, 8 exponent, 23 mantissa). Range ~1.2e-38 to 3.4e38.
  ~7 decimal digits of precision. Standard training format.
- **FP16:** 16 bits (1 sign, 5 exponent, 10 mantissa). Range ~6e-8 to 65504.
  ~3 decimal digits of precision. Can overflow for large activations; gradient
  underflow is an issue during training (use BF16 or loss scaling).
- **INT8:** 8 bits, integer. Range [-128, 127] (signed) or [0, 255] (unsigned).
  256 discrete levels. No floating-point exponent, so all values are equally spaced.

**Why INT8 hurts accuracy:** The fixed 256-level grid cannot represent the continuous
distribution of FP32 weights and activations exactly. Quantisation error is bounded by
`scale/2` per element, where scale covers the full dynamic range. Layers with large
weight ranges (large scale) have coarse grids and high error. Outlier activations (a
few very large values) force a large scale, wasting precision on small values.
Outlier-robust calibration methods (e.g., GPTQ, SmoothQuant) address this.

---

**Q2.** What is knowledge distillation and why can a small student model trained with
distillation outperform one trained only on hard labels?

**Answer:**

Knowledge distillation trains a student model to match the output distribution of a
teacher (large, high-accuracy) model, in addition to or instead of matching the hard
ground-truth labels.

The student benefits because:
1. **Soft labels carry more information.** The teacher's probability distribution over
   all classes reveals inter-class similarity structure (e.g., "this image is 80 % cat,
   15 % dog" tells the student that cats and dogs look similar).
2. **Richer training signal.** Each training example now provides a full probability
   vector rather than a one-hot target, reducing the number of training samples needed
   to achieve good generalisation.
3. **Implicit data augmentation.** The teacher's uncertainty captures aleatoric ambiguity
   in the data that hard labels do not express.

The result is that a student trained with distillation often matches the accuracy of a
model 2-5x larger trained on hard labels alone.

---

### Intermediate

**Q3.** Describe the process of post-training INT8 quantisation for a convolutional
neural network. What is per-channel quantisation and why does it improve accuracy?

**Answer:**

**PTQ process:**
1. Start with a trained FP32 model.
2. Collect a calibration dataset (100-1000 representative images).
3. Run calibration data through the model, collecting min/max (or percentile) statistics
   for the weight and activation tensors at each layer.
4. Compute per-layer (or per-channel) scale and zero_point from these statistics.
5. Quantise weight tensors to INT8 offline (stored as INT8, dequantised to FP32 for
   compute on hardware that lacks INT8 kernels, or kept as INT8 for hardware with INT8
   matrix multiply units).
6. At inference, activations are quantised dynamically or with static scales from
   calibration.

**Per-channel quantisation:** Instead of one scale per weight tensor, compute a separate
scale for each output channel (each convolutional filter). Different filters can have
vastly different weight ranges; a single global scale would either clip large-weight
filters or waste precision on small-weight filters. Per-channel scales use the dynamic
range of each channel optimally.

Per-channel quantisation for weights is standard in TensorFlow Lite and PyTorch's
`torch.quantization` and typically reduces INT8 accuracy loss from 1-3 % to under 0.5 %
compared to per-tensor weight quantisation.

---

### Advanced

**Q4.** Explain the straight-through estimator (STE) used in quantisation-aware training.
Why is it necessary and what are its theoretical limitations?

**Answer:**

**Why STE is necessary:** Quantisation involves a rounding function:
```
x_q = round(x / scale) * scale
```
The rounding operation has zero gradient almost everywhere (gradient of the floor function
is 0 except at integers where it is undefined). Standard backpropagation through this
operation would produce zero gradients, making the model untrainable.

**The STE:** Approximate the gradient of the rounding operation as 1 within the
quantisation range and 0 outside it:
```
d(x_q)/dx = 1  if  x_min <= x <= x_max  else  0
```
This "passes the gradient straight through" the quantise-dequantise operation, allowing
upstream layers to receive meaningful gradients.

**Theoretical limitations:**
1. **Biased gradient estimate:** The STE is not the true gradient of the quantised loss.
   It is a heuristic that works empirically but is theoretically unjustified.
2. **Gradient-quantisation mismatch:** The model is optimised as if the quantisation
   did not exist, but evaluated as if it does. This inconsistency means convergence
   is not guaranteed, and learning can be unstable at very low bit-widths (1-2 bits).
3. **Scale parameter gradients:** The scale and zero_point are also parameters that
   affect accuracy. Computing gradients through them is non-trivial and various
   methods (LSQ, PACT) propose learnable scale parameters with their own STE variants.

Despite these limitations, STE-based QAT is widely used and produces state-of-the-art
results for INT8 and INT4 quantisation in practice.

---

**Q5.** A colleague proposes removing all attention heads from a Transformer that have
low L1-norm weights after training on the primary task, then fine-tuning the pruned model.
Identify any flaws in this approach and describe a more principled method.

**Answer:**

**Flaws in L1-norm head pruning:**

1. **L1-norm of attention weights is not a reliable importance proxy.** An attention head
   with small weights but large values in certain key positions may be critical for
   specific input patterns (e.g., coreference resolution in NLP). L1-norm captures average
   magnitude, not functional importance.

2. **One-shot pruning ignores interactions between heads.** Removing one head changes
   the gradient landscape for remaining heads. Iterative pruning (prune a few heads,
   fine-tune, repeat) is more reliable.

3. **No calibration data used.** Importance should be measured on a representative
   calibration set, not just from weight magnitudes at rest.

**More principled approach: Taylor-expansion importance scoring**

Estimate the change in loss from removing head h:
```
Importance(h) = |delta_L| ≈ |sum over x: (activation_h(x) * grad_h(x))|
```
This first-order Taylor approximation measures how much the loss would change if the
head's output were zeroed. Computed on a calibration set, it accounts for both weight
values and the gradient signal, making it sensitive to heads the model relies on
regardless of their weight magnitude.

**Full workflow:**
1. Compute Taylor importance scores for all heads on a calibration set.
2. Sort heads by importance (ascending). Prune the bottom k %.
3. Fine-tune the pruned model for a short period to recover accuracy.
4. Repeat steps 1-3 iteratively until the target compression ratio is reached.
5. Apply PTQ to the final pruned model for additional speedup.
