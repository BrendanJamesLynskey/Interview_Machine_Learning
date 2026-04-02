# Metrics: Precision, Recall and F1

## Prerequisites
- Binary and multi-class classification
- Probability and basic statistics
- Understanding of model outputs (logits, softmax probabilities)

---

## Concept Reference

### The Confusion Matrix

Every classification metric derives from four counts obtained by comparing predicted labels
to ground-truth labels on a held-out evaluation set.

```
                  Predicted Positive    Predicted Negative
Actual Positive        TP                      FN
Actual Negative        FP                      TN
```

- **TP (True Positive):** Model correctly predicts the positive class.
- **TN (True Negative):** Model correctly predicts the negative class.
- **FP (False Positive):** Model predicts positive, but the true label is negative. Type I error.
- **FN (False Negative):** Model predicts negative, but the true label is positive. Type II error.

All four cells are needed. Reporting only accuracy discards information about which errors
the model makes, which is critical for imbalanced datasets and asymmetric cost scenarios.

---

### Core Metrics Derived from the Confusion Matrix

#### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

What fraction of all predictions are correct.

**When it misleads:** On a dataset that is 99 % negative, a classifier that always predicts
negative achieves 99 % accuracy while being completely useless.

#### Precision

```
Precision = TP / (TP + FP)
```

Of all instances the model labelled positive, what fraction truly are positive.

Precision answers: "When the model raises an alarm, how trustworthy is that alarm?"

High precision is critical when false positives are costly (e.g., spam detection, where a
false alarm blocks legitimate email).

#### Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

Of all instances that are truly positive, what fraction did the model correctly identify.

Recall answers: "How many of the actual positives did we catch?"

High recall is critical when false negatives are costly (e.g., cancer screening, where
missing a true positive has severe consequences).

#### Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

Of all true negatives, what fraction did the model correctly identify.

#### F1 Score

The harmonic mean of precision and recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2*TP / (2*TP + FP + FN)
```

The harmonic mean penalises extreme imbalance between precision and recall more than the
arithmetic mean would. A model with precision=1.0 and recall=0.01 has F1=0.02, not 0.505.

F1 is the default single-number summary when both false positives and false negatives matter
and the dataset is imbalanced.

#### F-beta Score

A generalisation that lets you weight recall beta times as important as precision:

```
F_beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
```

- beta = 2: recall weighted twice as heavily (medical diagnosis, fraud detection).
- beta = 0.5: precision weighted twice as heavily (search engine ranking).

---

### The Precision-Recall Trade-off

Most classifiers produce a real-valued score (probability, logit) rather than a hard label.
The decision threshold converts scores to labels. Varying this threshold traces out a
**Precision-Recall curve**.

```
High threshold -> fewer positive predictions -> higher precision, lower recall
Low threshold  -> more positive predictions  -> lower precision, higher recall
```

**Area Under the Precision-Recall Curve (AUC-PR):**
Summarises the curve as a single number. Ranges from 0 to 1. A random classifier has
AUC-PR equal to the positive class prevalence p. A perfect classifier has AUC-PR = 1.

AUC-PR is preferable to AUC-ROC when the dataset is heavily imbalanced, because it focuses
on the positive class and is not inflated by the large number of true negatives.

---

### ROC Curve and AUC-ROC

The **Receiver Operating Characteristic (ROC) curve** plots the True Positive Rate (recall)
against the False Positive Rate (1 - specificity) at every threshold.

```
True Positive Rate  (TPR) = TP / (TP + FN)   = Recall
False Positive Rate (FPR) = FP / (FP + TN)   = 1 - Specificity
```

**Area Under the ROC Curve (AUC-ROC):**
- Perfect classifier: AUC = 1.0 (curve hugs the top-left corner).
- Random classifier: AUC = 0.5 (diagonal line).
- AUC has a probabilistic interpretation: the probability that a randomly chosen positive
  instance is ranked higher than a randomly chosen negative instance by the model.

**When to prefer AUC-PR over AUC-ROC:**
On highly imbalanced datasets (e.g., 1 positive per 10,000 negatives), the ROC curve can
look optimistic because the large TN count suppresses FPR. AUC-PR, which does not involve
TN, better reflects real-world performance.

---

### Multi-Class Extension

For K-class classification, extend binary metrics via averaging strategies:

| Strategy      | How                                                         | When to use                      |
|---------------|-------------------------------------------------------------|----------------------------------|
| Macro average | Compute metric per class, take unweighted mean              | Each class equally important     |
| Weighted avg  | Compute metric per class, weight by class support           | Class imbalance present          |
| Micro average | Aggregate all TPs, FPs, FNs across classes, then compute    | Total instance count matters     |

For macro F1, rare classes receive the same weight as common ones. For micro F1, the
dominant class drives the result. Weighted F1 is a pragmatic compromise.

---

## Interview Questions by Difficulty

### Fundamentals

**Q1.** A binary classifier is evaluated on 1000 test samples (100 positive, 900 negative).
It predicts 80 samples as positive, of which 70 are truly positive. Fill in the confusion
matrix and compute precision, recall, and F1.

**Answer:**

Confusion matrix:
```
TP = 70   (truly positive, predicted positive)
FP = 10   (truly negative, predicted positive)   [80 predicted positive - 70 TP]
FN = 30   (truly positive, predicted negative)   [100 positive - 70 TP]
TN = 890  (truly negative, predicted negative)   [900 negative - 10 FP]
```

Metrics:
```
Accuracy  = (70 + 890) / 1000           = 0.960
Precision = 70 / (70 + 10)             = 0.875
Recall    = 70 / (70 + 30)             = 0.700
F1        = 2 * (0.875 * 0.700) / (0.875 + 0.700)
          = 1.225 / 1.575
          = 0.778
```

Note: accuracy of 0.96 looks impressive but recall of 0.70 means we missed 30 % of all
positive cases. On a cancer screening task this would be unacceptable.

---

**Q2.** What does AUC-ROC = 0.5 mean, and when would you see it?

**Answer:**

AUC-ROC = 0.5 means the model has no discriminative power -- it performs no better than
randomly assigning scores to instances. Equivalently, the probability that the model ranks
a random positive above a random negative is exactly 0.5 (coin flip). This arises when
the model's scores are independent of the true label, or when the model always predicts
the same value for all instances.

---

**Q3.** You have a model that always predicts the negative class. What are its precision,
recall, and F1 for the positive class?

**Answer:**

If the model never predicts positive:
```
TP = 0, FP = 0, FN = all actual positives, TN = all actual negatives.

Precision = 0 / (0 + 0) = undefined (no positive predictions made).
Recall    = 0 / (0 + FN) = 0.
F1        = 0 (by convention, since recall = 0).
```

Precision is undefined because the denominator (TP + FP) is zero. Most implementations
default undefined precision to 0.0 and issue a warning. F1 = 0 correctly captures that
this model is useless for detecting positive cases.

---

### Intermediate

**Q4.** Your team has built a fraud detection model. The business can investigate at most
100 flagged transactions per day. A missed fraud costs $500; a false positive costs $20.
Should you optimise for precision, recall, or F1? How does threshold choice affect the
decision?

**Answer:**

Given the capacity constraint of 100 flags per day, you need high **precision** to ensure
investigators spend their limited capacity on real fraud. However, you must also consider
the asymmetric cost: each missed fraud costs 25x more than a false alarm ($500 vs $20).

The optimal threshold minimises expected daily cost:
```
Cost(threshold t) = FP(t) * $20 + FN(t) * $500
```

Steps:
1. Compute the precision-recall curve over the validation set.
2. For each threshold, compute the number of predicted positives. Filter to thresholds
   where predicted positives <= 100 (the daily capacity).
3. Among those, compute expected cost and choose the minimising threshold.

F1 ignores the cost asymmetry and the operational constraint, so it is not the right
objective here. Domain-specific cost-weighted metrics, or directly optimising the threshold
against the cost function on a validation set, are preferred.

---

**Q5.** Explain the difference between macro, micro, and weighted F1 for a 3-class problem
with class counts [1000, 100, 10]. Which should you report and why?

**Answer:**

Suppose per-class F1 scores are: Class A (majority) = 0.90, Class B = 0.70, Class C
(minority) = 0.50.

```
Macro F1    = (0.90 + 0.70 + 0.50) / 3
            = 0.700
            Classes weighted equally. Rare class has equal influence.

Weighted F1 = (0.90 * 1000 + 0.70 * 100 + 0.50 * 10) / (1000 + 100 + 10)
            = (900 + 70 + 5) / 1110
            = 0.878
            Dominated by the majority class.

Micro F1    = Aggregate TP, FP, FN across classes then compute.
            Also dominated by the majority class for imbalanced datasets.
```

For this dataset, **macro F1** is most informative because it reveals the model struggles
on minority classes. Reporting only weighted or micro F1 would hide poor performance on
Class C, which may be the most important class in practice (e.g., a rare but critical
disease category). Best practice: report all three plus per-class F1 whenever class
imbalance is significant.

---

### Advanced

**Q6.** Prove that the harmonic mean penalises imbalance more than the arithmetic mean.
Then explain why F1 specifically uses the harmonic mean.

**Answer:**

**Proof:** For any two non-negative values a and b:
```
Arithmetic mean  A = (a + b) / 2
Harmonic mean    H = 2ab / (a + b)

H / A = 4ab / (a + b)^2

By AM-GM inequality: (a + b)^2 >= 4ab, so H/A <= 1.
Equality holds iff a = b.
As one value approaches 0 while the other is fixed, H -> 0 but A -> constant/2.
```

Example: precision = 0.99, recall = 0.01
```
Arithmetic mean = (0.99 + 0.01) / 2 = 0.50   (misleadingly high)
Harmonic mean   = 2 * 0.99 * 0.01 / 1.00    = 0.0198  (correctly near 0)
```

**Why harmonic mean for F1?** F1 should be high only when *both* precision and recall are
high. The harmonic mean is dominated by whichever component is *smaller* -- a single very
low value drags the result toward zero. A model that flags every instance as positive
achieves recall=1 but precision=prevalence (e.g., 0.001); its F1 should be ~0.002, not
~0.5. The harmonic mean delivers this behaviour correctly; the arithmetic mean does not.

---

**Q7.** A model achieves AUC-ROC = 0.97 on a 1:1000 positive-to-negative dataset.
A colleague claims this is strong evidence of a good model. Critique this claim.

**Answer:**

AUC-ROC = 0.97 is a high absolute score, but on a 1:1000 imbalanced dataset it can be
misleading for several reasons:

1. **TN dominates the FPR calculation.** With 1000 negatives per positive, even a large
   absolute number of false positives produces a small FPR. A model can look excellent on
   the ROC curve while generating thousands of false positives in deployment.

2. **AUC-PR is the right metric.** For 1:1000 imbalance, the precision-recall curve is
   far more informative. A random classifier has AUC-PR = 0.001 (the positive prevalence).
   An AUC-PR of 0.05 represents a 50x improvement over random -- but AUC-ROC does not
   reveal this directly.

3. **Operating point matters.** At the deployment threshold, check actual precision. If
   precision = 0.01 at recall = 0.90, then 99 % of flagged instances are false positives,
   which may be operationally infeasible regardless of AUC-ROC.

4. **Test set distribution may not match deployment.** If the test set was constructed
   with a more balanced split than production data, the AUC estimate is optimistic.

A complete evaluation should include: AUC-PR, precision at the target operational recall
level, and calibration plots (reliability diagrams) to verify that predicted probabilities
are meaningful at the operating point.
