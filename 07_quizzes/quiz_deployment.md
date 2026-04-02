# Quiz: Deployment

## Instructions

15 multiple-choice questions covering evaluation metrics, cross validation, model
selection, quantisation, pruning, ONNX, and inference runtimes. Each question has
exactly one correct answer. Work through all questions before checking the answer key
at the end.

Difficulty distribution: Questions 1-5 Fundamentals, Questions 6-11 Intermediate,
Questions 12-15 Advanced.

---

## Questions

### Q1 (Fundamentals)

A binary classifier is tested on a dataset with 1000 samples (50 positive, 950
negative). The model predicts 40 samples as positive, of which 35 are truly positive.
What is the precision of this classifier?

- A) 35/50 = 0.700
- B) 35/40 = 0.875
- C) 35/(35 + 15) = 0.700
- D) (35 + 950)/1000 = 0.985

---

### Q2 (Fundamentals)

You train a fraud detection model and observe AUC-ROC = 0.97 on a test set where only
0.1 % of transactions are fraudulent. A colleague says the model is excellent. What is
the most important additional metric to examine?

- A) Training accuracy, to check for data leakage.
- B) AUC-PR (area under the precision-recall curve), because AUC-ROC can be optimistic
     when the dataset is highly imbalanced.
- C) The macro F1 score across all classes, to check class balance.
- D) The cross-entropy loss on the test set, to verify calibration.

---

### Q3 (Fundamentals)

In stratified k-fold cross validation, "stratified" means:

- A) The folds are ordered by sample index rather than shuffled.
- B) Each fold preserves the original class distribution of the full dataset.
- C) Each fold is drawn independently with replacement (bootstrap sampling).
- D) The folds are sorted so that the hardest examples appear in the last fold.

---

### Q4 (Fundamentals)

Post-training quantisation (PTQ) converts a trained FP32 model to INT8. Compared to the
FP32 model, the INT8 model:

- A) Always has identical accuracy because quantisation is a lossless operation.
- B) Uses approximately 4x less memory for weights and typically runs faster on hardware
     with INT8 compute units, at the cost of a small accuracy drop.
- C) Uses 2x less memory because INT8 has 2 bytes rather than 4 bytes per value.
- D) Must be retrained from scratch on the original dataset before deployment.

---

### Q5 (Fundamentals)

ONNX (Open Neural Network Exchange) is best described as:

- A) A training framework developed by Microsoft that competes with PyTorch and
     TensorFlow.
- B) An open standard intermediate representation for neural network models that allows
     models trained in one framework to be deployed in another.
- C) A hardware-specific inference runtime for NVIDIA GPUs.
- D) A pruning algorithm that removes redundant neurons before deployment.

---

### Q6 (Intermediate)

You compare two classifiers using 5-fold cross validation. Classifier A achieves mean
F1 = 0.72 ± 0.15 and classifier B achieves mean F1 = 0.74 ± 0.02. Which classifier
should you prefer and why?

- A) Classifier A, because it achieved the highest individual fold score.
- B) Classifier B, because its mean is higher and its standard deviation is much
     lower, indicating more stable and reliable performance across folds.
- C) Classifier A, because a higher standard deviation means the model has higher
     capacity and will generalise better.
- D) Neither; you need to retrain both on the full dataset and compare on a held-out
     test set before drawing any conclusions.

---

### Q7 (Intermediate)

You are building a churn prediction model using 36 months of historical customer data.
You want to use cross validation. Which CV strategy is correct?

- A) Standard 5-fold CV with random shuffling, because shuffling removes temporal
     autocorrelation.
- B) Walk-forward (expanding window) CV where each training set uses only data prior
     to the validation period, to prevent training on future data.
- C) Leave-one-out CV to maximise the number of training samples in each fold.
- D) Stratified 5-fold CV, stratifying by the churn label to preserve class balance
     across folds.

---

### Q8 (Intermediate)

Bayesian optimisation is preferred over random search for hyperparameter tuning when:

- A) You have more than 10 hyperparameters to tune, because Bayesian optimisation
     scales well to high dimensions.
- B) Each model evaluation is cheap (seconds), so you can afford a large number of
     random evaluations.
- C) Each model evaluation is expensive (hours or days) and you have a small total
     budget of evaluations (20-100), making sample efficiency critical.
- D) You need a fully parallelisable search, because Bayesian optimisation is
     inherently parallel.

---

### Q9 (Intermediate)

Knowledge distillation trains a small "student" model to match a large "teacher" model.
The temperature parameter T in the distillation loss:

- A) Controls the learning rate of the student during distillation training.
- B) Scales the student's weights to match the teacher's parameter magnitude.
- C) Flattens the teacher's softmax distribution (T > 1), making inter-class similarity
     information in the soft labels more pronounced and easier for the student to learn.
- D) Determines the number of layers transferred from teacher to student.

---

### Q10 (Intermediate)

A TensorRT engine (.plan file) built on an NVIDIA A100 GPU is moved to an NVIDIA T4 GPU
for serving. What is the most likely outcome?

- A) The engine runs correctly on the T4 with a slight accuracy drop due to different
     tensor core precision.
- B) The engine fails to load or produces incorrect results because TensorRT plans are
     compiled for a specific GPU architecture and are not portable across SM generations.
- C) The engine runs correctly but at the speed of FP32, because INT8 tensor cores
     differ between A100 and T4.
- D) TensorRT automatically recompiles the plan for the T4 at load time.

---

### Q11 (Intermediate)

Structured pruning differs from unstructured pruning in that:

- A) Structured pruning removes individual scalar weights below a threshold; unstructured
     pruning removes entire filters or neurons.
- B) Structured pruning removes entire structures (filters, neurons, attention heads),
     yielding dense weight matrices that accelerate inference on standard hardware;
     unstructured pruning removes individual weights, producing sparse tensors that
     require special hardware to accelerate.
- C) Structured pruning requires retraining from scratch, while unstructured pruning
     does not.
- D) Structured pruning is only applicable to convolutional networks; unstructured
     pruning works for all network types.

---

### Q12 (Advanced)

Nested cross validation is used to:

- A) Shuffle data in a nested manner to avoid correlations between folds.
- B) Produce an unbiased estimate of the generalisation error of a full
     "hyperparameter-tune-and-train" pipeline, by separating the evaluation loop
     (outer CV) from the model-selection loop (inner CV).
- C) Evaluate models on multiple datasets simultaneously by nesting dataset loops
     inside fold loops.
- D) Reduce the variance of a single k-fold CV estimate by running multiple rounds
     of CV nested inside each other.

---

### Q13 (Advanced)

Quantisation-Aware Training (QAT) uses the straight-through estimator (STE) to compute
gradients through the rounding operation. What is the STE's fundamental limitation?

- A) The STE requires twice the memory of standard backpropagation because it stores
     both the quantised and full-precision versions of every activation.
- B) The STE produces a biased gradient estimate because it approximates the zero
     gradient of the rounding function as 1, creating an inconsistency between the
     forward (quantised) and backward (approximate unquantised) passes.
- C) The STE only works for INT8 quantisation; it cannot be applied to INT4 or lower
     bit-widths.
- D) The STE computes exact gradients for the scale and zero-point parameters but
     approximate gradients for the weights.

---

### Q14 (Advanced)

In an ONNX export of a PyTorch model, `dynamic_axes` is set only for the batch dimension.
During deployment, a client sends requests with variable sequence lengths. What happens?

- A) ONNX Runtime automatically infers the correct shape from the input tensor at
     runtime and executes without error.
- B) The export fails immediately with a shape error because all axes must be dynamic
     if any axis is dynamic.
- C) The model runs correctly for the specific sequence length used during export but
     raises a shape error or produces incorrect results for any other sequence length,
     because that dimension was not marked as dynamic.
- D) ONNX Runtime pads shorter sequences to the exported length and truncates longer
     ones, preserving accuracy within a few percentage points.

---

### Q15 (Advanced)

You train an XGBoost classifier on a fraud detection dataset resampled to 50/50 class
balance via random oversampling of the minority class. The model achieves precision = 0.72
and recall = 0.79 at the default threshold on the balanced validation set. At deployment
the real fraud rate is 0.2 %. Which problem will you most likely encounter, and what is
the correct remedy?

- A) The model will miss too many frauds (low recall) because it was trained on an
     unrealistically easy balanced dataset. Remedy: collect more real fraud examples.
- B) The model will flag far too many legitimate transactions (very low precision at
     deployment) because the decision boundary was learned for a 50 % fraud prior and
     will be too aggressive when the true prior is 0.2 %. Remedy: recalibrate predicted
     probabilities using Platt scaling or isotonic regression on a validation set with
     the true class distribution, then set the operating threshold to match the
     operational flag-rate budget.
- C) The model's predicted probabilities are perfectly calibrated because balancing the
     training data corrects the class prior. No remediation is needed.
- D) The model will have high variance because the oversampled minority examples are
     duplicates. Remedy: use SMOTE instead of random oversampling to generate diverse
     synthetic examples.

---

## Answer Key

| Q  | Answer | Difficulty    |
|----|--------|---------------|
| 1  | B      | Fundamentals  |
| 2  | B      | Fundamentals  |
| 3  | B      | Fundamentals  |
| 4  | B      | Fundamentals  |
| 5  | B      | Fundamentals  |
| 6  | B      | Intermediate  |
| 7  | B      | Intermediate  |
| 8  | C      | Intermediate  |
| 9  | C      | Intermediate  |
| 10 | B      | Intermediate  |
| 11 | B      | Intermediate  |
| 12 | B      | Advanced      |
| 13 | B      | Advanced      |
| 14 | C      | Advanced      |
| 15 | B      | Advanced      |

---

## Detailed Explanations

### Q1 - Answer: B

Precision measures the fraction of predicted positives that are truly positive:

```
Precision = TP / (TP + FP)
          = 35 / (35 + 5)   [40 predicted positive - 35 TP = 5 FP]
          = 35 / 40
          = 0.875
```

- **A** computes recall (TP / actual positives = 35/50 = 0.70), not precision.
- **C** arrives at the same numeric value as A by different reasoning and is wrong for
  the same reason -- it confuses FN with FP. FN = 50 - 35 = 15; this is the false
  negative count, not the false positive count.
- **D** computes accuracy: (TP + TN) / total = (35 + 950) / 1000 = 0.985.

---

### Q2 - Answer: B

On a 0.1 % positive prevalence dataset, the ROC curve is inflated by the large number of
true negatives. A small absolute number of false positives produces a negligible FPR
because the denominator (FP + TN) is dominated by TN. A model can look excellent on the
ROC curve while generating thousands of false positives in practice.

AUC-PR does not involve TN at all. It measures ranking quality among the positive class.
A random classifier has AUC-PR equal to the positive prevalence (0.001 here). An AUC-PR
of 0.05 represents a 50x improvement; an AUC-PR of 0.50 would be outstanding. This is
far more informative than AUC-ROC for this imbalanced setting.

- **A** is wrong: training accuracy would detect data leakage, but the question is about
  evaluating the model's real-world utility, not detecting a training error.
- **C** is wrong: macro F1 is relevant for multi-class problems. For binary fraud detection
  the per-class F1 (for the positive class) and AUC-PR are more informative.
- **D** is wrong: cross-entropy loss is useful for calibration assessment but less directly
  actionable for operational decision-making than AUC-PR at the operating point.

---

### Q3 - Answer: B

Stratified k-fold ensures each fold has (approximately) the same proportion of each class
as the full dataset. If the dataset has 10 % positives, every fold will have ~10 %
positives. This prevents folds with zero minority-class samples on small or imbalanced
datasets, stabilises per-fold metric estimates, and makes the CV estimate more reliable.

- **A** describes sequential (sorted) splitting, not stratification.
- **C** describes bootstrap sampling (used in bagging), not k-fold CV.
- **D** is not a standard method and does not describe stratification.

---

### Q4 - Answer: B

INT8 uses 8 bits per value, FP32 uses 32 bits. The ratio is 32/8 = 4, so INT8 uses
approximately 4x less memory for the weight tensors. Hardware with INT8 tensor cores
(e.g., NVIDIA T4, A100) can execute INT8 matrix multiplications at significantly higher
throughput than FP32 (up to 4x on NVIDIA Turing tensor cores). The accuracy drop is
typically less than 1 % for well-calibrated models quantised to INT8 with PTQ.

- **A** is wrong: quantisation maps a continuous range to a discrete grid, which
  introduces rounding error and is a lossy approximation.
- **C** is wrong: INT8 uses 1 byte (8 bits), not 2 bytes. The correct ratio is 4x, not 2x.
- **D** is wrong: PTQ does not require retraining. It uses a small calibration dataset to
  determine quantisation scales, but the original weights are not updated.

---

### Q5 - Answer: B

ONNX is an open specification (originally developed by Microsoft and Facebook/Meta) that
defines a common graph format for neural network models. It solves the N framework ×
M hardware compatibility problem: any framework that can export to ONNX can be deployed
on any runtime that supports ONNX loading.

- **A** is wrong: ONNX is a format standard, not a training framework.
- **C** is wrong: ONNX Runtime (ORT) is an inference runtime that runs ONNX models, but
  ONNX itself is the format specification, not a runtime. TensorRT is the NVIDIA GPU
  inference SDK.
- **D** is wrong: ONNX has nothing to do with pruning.

---

### Q6 - Answer: B

The standard deviation of cross-validation scores estimates the variability of a model's
performance across different data splits. Classifier A has mean = 0.72 ± 0.15, meaning
individual fold scores ranged widely (possibly 0.57 to 0.87). This high variance suggests
the model is sensitive to which data ends up in the training set -- an unstable model.
Classifier B has mean = 0.74 ± 0.02, slightly higher mean with very low variance:
consistently reliable performance regardless of the fold.

For deployment, stability is often as important as mean performance. A model that
occasionally achieves 0.87 but can drop to 0.57 is less trustworthy than one that
consistently scores around 0.74.

- **A** is wrong: the highest single-fold score is not a reliable model selection criterion;
  it is a noisy estimate subject to the specific data in that fold.
- **C** is wrong: high standard deviation is a sign of instability (high variance), not
  higher capacity.
- **D** is wrong: comparing on a held-out test set is best practice for final evaluation,
  but the CV comparison already gives meaningful evidence that B is preferable.

---

### Q7 - Answer: B

Time series data has temporal ordering. Using standard k-fold with shuffling allows a
model to train on data from month 30 and predict month 10 -- effectively training on
the future. This constitutes data leakage and produces optimistic CV estimates that do
not reflect real deployment performance.

Walk-forward CV (also called expanding window or time-series split) ensures the training
set always precedes the validation set in time, matching the real deployment scenario
exactly. A gap equal to the prediction horizon should be added between the last training
observation and the first validation observation.

- **A** is wrong: shuffling time series data is a fundamental error that introduces leakage.
- **C** is wrong: LOO-CV is computationally expensive and, more critically, does not
  respect temporal ordering.
- **D** is wrong: stratification by churn label does not address temporal leakage.

---

### Q8 - Answer: C

Bayesian optimisation builds a surrogate model (Gaussian Process or TPE) of the objective
function and uses it to intelligently select the next configuration to evaluate. Each
observation updates the surrogate, enabling directed exploration. This is valuable when
each evaluation (training a full model) costs hours or days and only 20-100 total
evaluations are feasible.

When evaluations are cheap (seconds), you can run thousands of random evaluations, which
sample the space more densely than the sequential Bayesian approach can with 100 directed
evaluations.

- **A** is wrong: Bayesian optimisation scales poorly to high-dimensional spaces because
  Gaussian Process inference is O(N^3) in evaluations and the acquisition function
  maximisation becomes hard in many dimensions. Tree Parzen Estimator (TPE) handles
  more dimensions but still suffers in extreme high-dimensional settings.
- **B** is wrong: when evaluations are cheap, random search with a large budget is
  preferred because it is simpler and achieves good coverage.
- **D** is wrong: Bayesian optimisation is inherently sequential. Each step uses the
  full history of previous observations, making parallelisation non-trivial.

---

### Q9 - Answer: C

In knowledge distillation, the teacher produces soft label vectors via softmax. Without
temperature scaling, a well-trained teacher assigns probability ~0.999 to the correct
class and ~0.001 to all others -- effectively a hard label with no inter-class information.

Dividing logits by T > 1 before softmax flattens the distribution:
```
soft_output = softmax(logits / T)
```
At T = 4, a teacher producing logits [10, 2, 1] gives soft outputs far more similar
to [0.97, 0.02, 0.01] vs [0.9999, 0.00005, 0.00005] at T=1. The inter-class structure
(class 1 is somewhat similar to class 2 but not class 3) is visible to the student,
providing a richer training signal than hard labels.

- **A** is wrong: temperature does not affect the learning rate.
- **B** is wrong: temperature is applied to the logits before softmax, not to the weights.
- **D** is wrong: which layers to transfer is a separate architectural decision unrelated
  to temperature.

---

### Q10 - Answer: B

TensorRT's engine build process benchmarks CUDA kernel implementations for the exact
GPU SM architecture (compute capability) and selects the fastest. The resulting `.plan`
file contains kernel code and bindings specific to that GPU architecture. An A100
(sm_80) plan cannot run on a T4 (sm_75) because the selected kernels may use
instructions or tensor core formats unavailable on the T4.

In practice, loading a mismatched plan either raises an explicit error ("incompatible
plan") or produces garbage output. TensorRT does not auto-recompile at load time.

- **A** is wrong: the issue is not precision accuracy but binary incompatibility.
- **C** is wrong: the plan fails outright rather than falling back gracefully to FP32.
- **D** is wrong: TensorRT does not silently rebuild at load time.

---

### Q11 - Answer: B

Unstructured pruning sets individual weights to zero based on a criterion (e.g.,
small magnitude). The resulting weight tensors remain the same shape but become sparse.
Generic hardware (CPUs, standard GPUs) executes dense and sparse matrix multiplications
with the same code paths -- no speedup unless the hardware has dedicated sparse compute
units (e.g., NVIDIA A100 2:4 structured sparsity).

Structured pruning removes entire rows or columns: deleting a convolutional filter
removes a complete output channel, and the next layer's corresponding input channel is
also removed. The resulting weight tensors are smaller and dense. Standard matrix
multiply kernels can exploit this immediately on any hardware.

- **A** has the definitions reversed.
- **C** is wrong: both unstructured and structured pruning typically require fine-tuning
  (not full retraining from scratch) to recover accuracy.
- **D** is wrong: structured pruning applies to any architecture with removable units
  (Transformer attention heads, fully-connected neurons, etc.).

---

### Q12 - Answer: B

Without nesting, if you use the same k-fold CV to both select hyperparameters and report
performance, the reported score is optimistically biased. The best configuration was
selected because it happened to score highest on those specific folds; a new dataset would
not necessarily yield the same score.

Nested CV separates these two concerns:
- **Inner CV** (k_inner folds on the outer training set): selects hyperparameters.
- **Outer CV** (k_outer folds on the full data): estimates generalisation performance
  of the complete pipeline (select-and-train), using data that was never involved in
  hyperparameter selection.

The outer CV mean is an unbiased estimate of how well the pipeline generalises.

- **A** describes a data shuffling strategy, not nested CV.
- **C** is not a standard technique; nested CV does not involve multiple datasets.
- **D** describes repeated k-fold CV, which reduces variance of the estimate but does
  not solve the selection bias problem.

---

### Q13 - Answer: B

The rounding function `round(x)` has a derivative of zero almost everywhere (it is a
staircase function). Applying standard backpropagation through it would zero out all
gradients upstream, making the model untrainable.

The STE approximates the gradient as 1 within the quantisation range:
```
d(round(x))/dx ≈ 1  if  x_min <= x <= x_max
```
This allows gradients to flow through as if the rounding did not exist. However, this
means the model is optimised as though it is unquantised but evaluated as though it is
quantised -- an inconsistency. Convergence is not theoretically guaranteed and can be
unstable at very low bit-widths (1-2 bits).

- **A** is wrong: while dual storage of quantised and full-precision values does increase
  memory usage, this is an implementation cost, not the "fundamental limitation" of the STE.
- **C** is wrong: STE is applied at 4-bit, 2-bit, and even 1-bit quantisation (binary
  networks). It becomes less stable but is not restricted to INT8.
- **D** is wrong: scale and zero-point gradient computation is separately handled by
  methods like LSQ or PACT; the standard STE does not produce exact gradients for scale.

---

### Q14 - Answer: C

When `torch.onnx.export` traces a model, it records the exact execution graph for the
shapes present in the dummy input. Dimensions not marked as dynamic are baked into the
graph as constants. The ONNX runtime uses those constant shapes for optimisations,
memory allocation, and operator shape inference.

If the sequence length dimension is not in `dynamic_axes`, the graph has a fixed sequence
length equal to the value used during export. Providing a different sequence length at
runtime causes a shape mismatch when the input tensor is fed to the first operation
that depends on that dimension.

- **A** is wrong: ONNX Runtime does not auto-infer new shapes for dimensions not marked
  as dynamic. Static dimensions are hardcoded.
- **B** is wrong: each axis is independently dynamic or static. You can mark only the
  batch dimension as dynamic without errors.
- **D** is wrong: ONNX Runtime does not silently pad or truncate inputs to match the
  exported shape.

---

### Q15 - Answer: B

Training on a 50/50 resampled dataset teaches the model's decision boundary that fraud
and non-fraud are equally likely. At the default threshold of 0.5, the model flags
transactions where its score exceeds the point where P(fraud) = P(not fraud) in the
training distribution. In the real 0.2 % prevalence world, that threshold is far too
low -- the model treats any moderately suspicious transaction as likely fraudulent,
flagging a very large fraction of legitimate transactions (extremely low precision at
deployment).

The remedy is probability recalibration: fit a calibration function (Platt scaling /
isotonic regression) on a held-out validation set with the real 0.2 % prevalence. Then
sweep the threshold on the calibrated probabilities to find the operating point that
satisfies the analyst capacity constraint (e.g., flag only the top-K highest-scoring
transactions per day).

- **A** is wrong: oversampling does not impair recall; it typically improves recall.
  The problem is a miscalibrated decision threshold, not missed fraud.
- **C** is wrong: training on a resampled 50/50 dataset distorts the prior. The model
  is not calibrated for the 0.2 % deployment prior.
- **D** is wrong: while SMOTE generates more diverse synthetic samples and reduces
  overfitting to duplicates, it does not fix the fundamental miscalibration caused by
  the 50/50 training prior. Threshold recalibration is the correct remedy regardless
  of whether SMOTE or random oversampling was used.
