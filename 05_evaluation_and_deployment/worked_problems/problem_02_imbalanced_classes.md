# Problem 02: Handling Class Imbalance

**Topic:** Strategies for training and evaluating classifiers on imbalanced datasets  
**Difficulty:** Fundamentals (Parts A-B), Intermediate (Parts C-D), Advanced (Part E-F)  
**Prerequisites:** `metrics_precision_recall_f1.md`, `cross_validation.md`

---

## Problem Statement

You are building a fraud detection system for a payment processor. The training dataset
has 1,000,000 transactions, of which 5,000 (0.5 %) are fraudulent. Initial experiments
with a gradient boosted tree (XGBoost) trained on the raw dataset produce:

```
Test set (stratified split, 100,000 transactions, 500 fraud):
  Accuracy : 99.51 %
  Precision: 0.12
  Recall   : 0.65
  F1       : 0.20
```

The product team considers this unacceptable. Your task is to improve the model.

---

### Part A (Fundamentals)

Explain why 99.51 % accuracy corresponds to such poor F1. What does precision = 0.12 tell
you about the model's behaviour in practice?

---

### Part B (Fundamentals)

List and briefly describe four distinct strategies for handling class imbalance. For each,
state one advantage and one limitation.

---

### Part C (Intermediate)

Your colleague proposes using **SMOTE** (Synthetic Minority Oversampling TEchnique) to
rebalance the training set. Explain:
(a) How SMOTE generates synthetic samples.
(b) Why SMOTE should only be applied to the **training** fold during cross validation
    and never to the validation or test set.
(c) One failure mode of SMOTE on high-dimensional or sparse feature spaces.

---

### Part D (Intermediate)

Instead of resampling, you decide to use **focal loss** as the training objective.
(a) Write the focal loss formula and explain each term.
(b) Explain intuitively why focal loss addresses class imbalance differently from
    cross-entropy with class weights.
(c) Given gamma=2 and alpha=0.25, compute the focal loss for a sample where the model
    predicts p=0.95 for the correct (fraudulent) class.

---

### Part E (Advanced)

You apply random oversampling (duplicate minority samples until 50/50 balance) and retrain.
The new test metrics are:

```
Precision: 0.61
Recall   : 0.82
F1       : 0.70
```

However, during deployment the model flags 12 % of all transactions as fraud (the product
team's budget allows investigating only 0.5 %). Identify the problem and propose a
threshold calibration strategy.

---

### Part F (Advanced)

Design a complete experiment to select among three imbalance-handling strategies:
random oversampling, SMOTE, and focal loss. Specify the evaluation protocol, metrics,
and what you would report to stakeholders.

---

## Solutions

### Part A Solution

**Why 99.51 % accuracy with poor F1:**

With 500 fraud cases in 100,000 test samples (0.5 % prevalence), a model that **always
predicts legitimate** achieves accuracy = 99.5 % while catching zero fraud. The model
here (99.51 %) is only marginally better than this trivial baseline.

Accuracy is dominated by correct negative predictions (TN). The 99,500 legitimate
transactions are nearly all predicted correctly -- these contribute 99.5 percentage points
to accuracy. The model's ability to detect fraud contributes only ~0.01 additional
percentage points. Accuracy is therefore a nearly useless signal on this dataset.

**What precision = 0.12 means in practice:**

```
Precision = TP / (TP + FP) = 0.12
```

For every 100 transactions the model flags as fraudulent, only 12 are genuinely fraudulent.
The other 88 are legitimate transactions that have been incorrectly blocked or sent to
investigators. This creates two operational problems:
1. **Analyst fatigue:** 88 % of flagged alerts are false alarms; analysts will quickly
   lose confidence in the system and start ignoring alerts.
2. **Customer experience:** If flags result in blocking transactions, 88 % of blocks
   affect legitimate customers, causing churn and support costs.

Recall = 0.65 means 35 % of actual fraud is being missed -- also poor.

---

### Part B Solution

Four strategies for class imbalance:

**1. Oversampling (Random or SMOTE)**
- *How:* Increase the number of minority class samples either by duplicating existing
  ones (random) or generating synthetic neighbours (SMOTE).
- *Advantage:* No data is discarded; the model sees more minority examples.
- *Limitation:* Risk of overfitting to duplicated samples; SMOTE can generate unrealistic
  samples in complex feature spaces.

**2. Undersampling (Random or Informed)**
- *How:* Reduce the number of majority class samples. Random undersampling discards
  majority samples randomly; methods like Tomek Links remove borderline majority samples.
- *Advantage:* Faster training (smaller dataset); can improve precision for the
  majority class boundary.
- *Limitation:* Discards potentially useful data. With 99.5 % majority, random
  undersampling to balance discards 99 % of all data.

**3. Class weighting (cost-sensitive learning)**
- *How:* Assign higher loss weight to minority class samples:
  `weight_fraud = N_total / (2 * N_fraud)`.  Pass `class_weight='balanced'` in sklearn
  or `scale_pos_weight` in XGBoost.
- *Advantage:* No data manipulation; works with any differentiable loss; computationally
  cheap.
- *Limitation:* Changes the calibration of predicted probabilities. Threshold of 0.5 is
  no longer optimal; recalibration is required.

**4. Threshold adjustment (decision threshold tuning)**
- *How:* Instead of using the default 0.5 threshold, find the threshold that optimises
  a domain-specific metric (e.g., minimum cost, target recall >= 0.90) on a validation
  set.
- *Advantage:* Zero training cost; can be applied post-hoc to any trained model.
- *Limitation:* Does not change the model's learned representations; a fundamentally
  poor model remains poor at any threshold. The model must produce well-separated
  probability scores for threshold tuning to be effective.

---

### Part C Solution

**(a) How SMOTE generates synthetic samples:**

For each minority (fraud) sample x_i, SMOTE:
1. Finds the k nearest neighbours among the minority class (typically k=5) using
   Euclidean distance in feature space.
2. Randomly selects one neighbour x_j from the k-nearest.
3. Generates a synthetic sample by interpolating between x_i and x_j:
   ```
   x_synthetic = x_i + lambda * (x_j - x_i),  lambda ~ Uniform(0, 1)
   ```

This creates new samples along the line segments connecting pairs of existing minority
samples. The result is a denser, more evenly covered minority class region.

**(b) Why SMOTE must be applied only to the training fold:**

SMOTE must be fitted and applied only to the training fold within each CV iteration.
If applied to the full dataset before splitting:

1. Synthetic fraud samples will be placed in the validation fold.
2. These synthetic samples are derived from the training fold's fraud examples and will
   be very similar (sometimes nearly identical) to training samples.
3. The model effectively "sees" information from the training fold when evaluated on the
   validation fold -- this is **data leakage**.
4. The CV score will be optimistically biased, and the deployed model will perform worse
   than the CV score predicts.

The correct approach with sklearn:
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf',   GradientBoostingClassifier()),
])
# SMOTE is refit on each training fold only; never applied to validation fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
```

**(c) Failure mode on high-dimensional or sparse features:**

SMOTE uses Euclidean distance to find nearest neighbours. In high-dimensional spaces,
Euclidean distance becomes unreliable (the "curse of dimensionality" -- all points tend
to be equidistant). Additionally, if features include one-hot encoded categorical
variables or sparse bag-of-words vectors:

- Interpolating between two one-hot vectors produces fractional (non-integer) values that
  are not valid categorical encodings.
- Synthetic samples fall outside the valid feature manifold (e.g., a feature representing
  "merchant category code" takes integer values 1-100; SMOTE may generate value 47.3).

Alternatives for these cases: **SMOTENC** (handles mixed numerical/categorical features)
or **ADASYN** (adaptive synthetic sampling with density weighting).

---

### Part D Solution

**(a) Focal loss formula:**

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where:
  p_t = p      if y = 1 (true positive class)
  p_t = 1 - p  if y = 0 (true negative class)

  alpha_t = alpha      if y = 1  (minority class weight)
  alpha_t = 1 - alpha  if y = 0  (majority class weight)

  gamma >= 0: focusing parameter (typically 1-5)
  alpha in (0,1): balancing weight for the positive class
```

- **`log(p_t)`:** Standard cross-entropy. High loss when the model is wrong.
- **`(1 - p_t)^gamma`:** The **modulating factor**. When the model is confident and
  correct (p_t near 1), `(1-p_t)^gamma` is near 0, down-weighting the loss for "easy"
  well-classified examples. When the model is uncertain (p_t near 0.5) or wrong (p_t
  near 0), `(1-p_t)^gamma` is near 1, keeping the loss large.
- **`alpha_t`:** Balances the contribution of positive vs negative examples, similar to
  class weighting.

**(b) Focal loss vs cross-entropy with class weights:**

**Weighted cross-entropy** applies a fixed multiplicative weight to all positive
(minority) class samples regardless of how easy or hard they are. Easy positives (which
the model already classifies correctly with high confidence) receive the same up-weighting
as hard positives (which the model is confused about).

**Focal loss** dynamically adjusts per-sample: easy, well-classified examples (whether
positive or negative) are down-weighted. The model focuses its attention and gradient
signal on the hard examples -- the samples it is uncertain about or wrong about. This
is particularly effective for imbalanced datasets where the majority of easy examples are
negatives that would otherwise dominate the loss.

Intuitively: a highly imbalanced dataset has 99 % easy negative examples. With standard
cross-entropy, even though each negative is correctly classified with high confidence
(p_t near 1), their sheer volume swamps the gradient from the minority class. Focal loss
suppresses these easy negatives so the minority class's harder samples drive training.

**(c) Compute focal loss with gamma=2, alpha=0.25, p=0.95 (correct positive class):**

```
y = 1 (fraud), p = 0.95 (correct prediction)

p_t    = p = 0.95
alpha_t = alpha = 0.25  (since y = 1)

Modulating factor: (1 - p_t)^gamma = (1 - 0.95)^2 = (0.05)^2 = 0.0025

log(p_t) = log(0.95) = -0.05129  (natural log)
         or log2(0.95) = -0.07400  (base-2)

Using natural log (standard in PyTorch cross-entropy):
FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
   = -0.25 * 0.0025 * (-0.05129)
   = 0.25 * 0.0025 * 0.05129
   = 3.206e-5
```

For comparison, standard cross-entropy on the same sample:
```
CE = -log(0.95) = 0.05129
```

Focal loss (3.2e-5) is approximately **1600x smaller** than standard cross-entropy
(0.051) for this easy, correctly classified sample. The model barely updates its weights
for samples it already classifies with 95 % confidence. This is the desired behaviour:
training effort concentrates on ambiguous or misclassified cases.

---

### Part E Solution

**Problem identification:**

The model was trained on a 50/50 oversampled training set but deployed on a real dataset
where fraud prevalence is 0.5 %. The model has been trained to be "optimistic" about the
fraud class -- its probability scores are not calibrated to the true deployment prior.

Specifically, the model's internal decision boundary was learned for a 50 % fraud prior.
At the default threshold of 0.5, the model predicts fraud whenever its internal score
exceeds 0.5, but a score of 0.5 in a 50/50 world corresponds to a much lower probability
in a 0.5 % world. The model is flagging 12 % of transactions because its threshold is
far too low for the deployment prevalence.

**Threshold calibration strategy:**

**Step 1: Recalibrate probabilities using Platt scaling or isotonic regression** on a
validation set with the true class distribution (not the oversampled version):

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_classifier, method='isotonic', cv='prefit')
calibrated.fit(X_val, y_val)  # X_val, y_val have the real 0.5% fraud prevalence
```

**Step 2: Choose the operating threshold** by computing the precision-recall curve on the
calibration validation set and selecting the threshold that:
- Meets the analyst capacity constraint (flag at most 0.5 % of transactions = 5000
  per million), OR
- Minimises the cost function (cost of missed fraud vs. cost of false alarm).

```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_val, calibrated.predict_proba(X_val)[:,1])

# Find threshold that flags ~0.5% of transactions
for t in thresholds:
    flag_rate = (calibrated.predict_proba(X_val)[:,1] >= t).mean()
    if flag_rate <= 0.005:
        operating_threshold = t
        break
```

**Step 3: Report precision and recall at the chosen threshold** to stakeholders rather
than F1 at the default threshold.

---

### Part F Solution

**Complete experimental protocol:**

**Objective:** Select the best imbalance-handling strategy for deployment under the
constraint that at most 0.5 % of transactions can be flagged (analyst capacity).

**Data splits:**
```
Full labelled dataset: 1,000,000 transactions (5,000 fraud)
  Train/validation: 800,000  (4,000 fraud)
  Test (held out, never touched until final evaluation): 200,000  (1,000 fraud)
```

**Cross-validation:** Stratified 5-fold on the train/validation set. For each fold:
- Apply the imbalance strategy to the training fold only (never the validation fold).
- Train the model (XGBoost, same hyperparameters across all strategies to isolate the
  imbalance treatment effect).
- Evaluate on the validation fold using the true class distribution (no oversampling).

**Imbalance strategies evaluated:**
1. Random oversampling (duplicate fraud to 50/50 balance).
2. SMOTE (synthetic oversampling, k=5 neighbours).
3. Focal loss (gamma=2, alpha=0.25; standard cross-entropy with XGBoost replaced by
   custom objective; or use a neural network baseline for fair comparison).

**Metrics to report at the operating point (top 0.5 % of scores flagged):**

| Metric         | Why                                                        |
|----------------|-------------------------------------------------------------|
| Precision at k | Fraction of flagged transactions that are truly fraud       |
| Recall at k    | Fraction of all fraud caught within the flagged set         |
| AUC-PR         | Threshold-independent measure of ranking quality            |
| AUC-ROC        | For completeness; less meaningful on this imbalanced set    |
| Calibration    | Do predicted probabilities match empirical fraud rates?     |

**What to report to stakeholders:**

1. **AUC-PR** (primary model selection criterion, threshold-independent).
2. **Precision and recall at the 0.5 % flag rate** (operational constraint).
3. **Expected fraud caught per day** = Recall * daily fraud volume.
4. **False alarm rate for customers** = FP / total legitimate transactions flagged.
5. **Cross-validated mean ± std** of each metric to assess stability.

Final model selection: choose the strategy with the best AUC-PR. Confirm on the held-out
test set (evaluate once, results are final). Report test set metrics alongside CV metrics
to stakeholders. If test performance is substantially worse than CV performance, revisit
the CV design for potential data leakage.
