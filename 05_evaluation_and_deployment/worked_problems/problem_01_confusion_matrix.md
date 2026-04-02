# Problem 01: Confusion Matrix Analysis

**Topic:** Deriving and interpreting all classification metrics from a confusion matrix  
**Difficulty:** Fundamentals (Parts A-C), Intermediate (Parts D-E), Advanced (Part F)  
**Prerequisites:** `metrics_precision_recall_f1.md`

---

## Problem Statement

A medical imaging model has been evaluated on a held-out test set of 2500 patients.
The task is binary classification: **malignant** (positive class) vs **benign** (negative
class). The model produces a probability score and a hard label at a 0.5 threshold.

The results on the test set are:

|                        | Predicted Malignant | Predicted Benign |
|------------------------|---------------------|------------------|
| **Actually Malignant** | 180                 | 45               |
| **Actually Benign**    | 30                  | 2245             |

---

### Part A (Fundamentals)

Identify the four cells of the confusion matrix (TP, FP, FN, TN) from the table above.

---

### Part B (Fundamentals)

Compute the following metrics to 4 decimal places:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score

---

### Part C (Fundamentals)

The hospital's radiologist argues that the model's accuracy of over 97 % is excellent and
recommends deployment. A data scientist objects. Who is right and why?

---

### Part D (Intermediate)

The hospital chief decides that missing a malignant tumour (false negative) is 10 times
more harmful than a false alarm (false positive). They want a metric that reflects this
cost asymmetry. Compute the F-beta score with the appropriate beta value.

---

### Part E (Intermediate)

The model's threshold is adjusted from 0.5 to 0.3 (more aggressive positive prediction).
At the new threshold, the confusion matrix becomes:

|                        | Predicted Malignant | Predicted Benign |
|------------------------|---------------------|------------------|
| **Actually Malignant** | 210                 | 15               |
| **Actually Benign**    | 120                 | 2155             |

(a) Recompute precision, recall, and F1.
(b) Explain the trade-off made by lowering the threshold.
(c) Given the 10:1 cost asymmetry from Part D, which threshold (0.5 or 0.3) should
    the hospital prefer? Show the calculation.

---

### Part F (Advanced)

For a multi-class extension: suppose the model is extended to 3 classes: **malignant**,
**benign**, and **indeterminate**. The confusion matrix over 3000 test samples is:

```
Predicted:    Malignant   Benign   Indeterminate
Actual:
Malignant         170       20            35       (225 actual malignant)
Benign             25      930            45       (1000 actual benign)
Indeterminate      40       80          1655       (1775 actual indeterminate)
```

Compute:
(a) Per-class precision, recall, and F1 for each of the three classes.
(b) Macro F1 and weighted F1.
(c) Which class is hardest for the model? What does the confusion pattern suggest?

---

## Solutions

### Part A Solution

Reading from the table:

```
TP = 180   Predicted malignant AND actually malignant (correct positive)
FP =  30   Predicted malignant BUT actually benign    (false alarm, Type I error)
FN =  45   Predicted benign    BUT actually malignant (missed cancer, Type II error)
TN = 2245  Predicted benign    AND actually benign    (correct negative)

Total = 180 + 30 + 45 + 2245 = 2500  (matches test set size, as expected)
```

Note the class distribution:
```
Positive (malignant) prevalence = (180 + 45) / 2500 = 225 / 2500 = 9 %
Negative (benign)    prevalence = (30 + 2245) / 2500 = 2275 / 2500 = 91 %
```

This is a moderately imbalanced dataset (~10x more benign than malignant).

---

### Part B Solution

Using the standard formulas with TP=180, FP=30, FN=45, TN=2245:

```
Total = 2500

Accuracy  = (TP + TN) / Total
          = (180 + 2245) / 2500
          = 2425 / 2500
          = 0.9700   (97.00 %)

Precision = TP / (TP + FP)
          = 180 / (180 + 30)
          = 180 / 210
          = 0.8571

Recall (Sensitivity)
          = TP / (TP + FN)
          = 180 / (180 + 45)
          = 180 / 225
          = 0.8000

Specificity
          = TN / (TN + FP)
          = 2245 / (2245 + 30)
          = 2245 / 2275
          = 0.9868

F1        = 2 * (Precision * Recall) / (Precision + Recall)
          = 2 * (0.8571 * 0.8000) / (0.8571 + 0.8000)
          = 2 * 0.6857 / 1.6571
          = 1.3714 / 1.6571
          = 0.8276
```

Summary table:
```
Metric       Value
-----------  ------
Accuracy     0.9700
Precision    0.8571
Recall       0.8000
Specificity  0.9868
F1           0.8276
```

---

### Part C Solution

**The data scientist is right.** The radiologist's argument is an example of the
"accuracy paradox" on imbalanced datasets.

- 91 % of patients are benign. A naive classifier that **always predicts benign** achieves
  91 % accuracy without doing any work. The model's 97 % accuracy is only 6 percentage
  points better than this trivial baseline.

- More critically, **recall = 0.80** means the model misses 20 % of malignant tumours
  (the FN count of 45). In a clinical setting, each missed malignancy could result in
  delayed treatment and patient harm. A recall of 80 % for cancer detection is
  unacceptably low for a screening tool.

- The appropriate metrics for this problem are **recall** (sensitivity) -- which should
  be as high as possible -- and **precision** (to bound false alarm rates that cause
  unnecessary biopsies). F1 balances these but even F1 = 0.83 should be interpreted in
  context of the specific cost trade-off.

The radiologist made the common mistake of reporting only accuracy without examining
the confusion matrix or domain-appropriate metrics.

---

### Part D Solution

When missing a malignancy (FN) costs 10x more than a false alarm (FP), we want to weight
**recall** more heavily than precision. This corresponds to **F-beta with beta = sqrt(10)**
or, by convention, **beta = 2** (a common choice for "recall is more important") or the
exact beta can be derived from the cost ratio.

**Formal derivation:** F-beta weights recall beta^2 times as much as precision in the
harmonic mean. For recall to be 10x as important:

```
beta^2 = 10  =>  beta = sqrt(10) ≈ 3.162
```

However, in practice **beta = 2** is the standard choice when recall is prioritised and
beta = 0.5 when precision is prioritised. Using beta = 2 here:

```
F_2 = (1 + 4) * (Precision * Recall) / (4 * Precision + Recall)
    = 5 * (0.8571 * 0.8000) / (4 * 0.8571 + 0.8000)
    = 5 * 0.6857 / (3.4284 + 0.8000)
    = 3.4285 / 4.2284
    = 0.8107
```

Using the exact beta = sqrt(10) ≈ 3.162:

```
beta^2 = 10
F_beta = (1 + 10) * (0.8571 * 0.8000) / (10 * 0.8571 + 0.8000)
       = 11 * 0.6857 / (8.571 + 0.8000)
       = 7.5427 / 9.371
       = 0.8049
```

The F_beta score (beta=sqrt(10)) = **0.8049** weights the failure to detect malignancy
(recall shortfall) much more heavily. A model with recall = 0.95 and precision = 0.60
would score higher on this metric than the current model, even though its precision is
substantially lower.

---

### Part E Solution

**New threshold 0.3 confusion matrix:** TP=210, FP=120, FN=15, TN=2155.

**(a) Recompute metrics:**

```
Precision = 210 / (210 + 120) = 210 / 330 = 0.6364
Recall    = 210 / (210 + 15)  = 210 / 225 = 0.9333
F1        = 2 * (0.6364 * 0.9333) / (0.6364 + 0.9333)
          = 2 * 0.5940 / 1.5697
          = 0.7570

Compare with threshold=0.5:
  Precision = 0.8571, Recall = 0.8000, F1 = 0.8276
```

**(b) Trade-off explanation:**

Lowering the threshold makes the model more "trigger-happy" -- it predicts malignant more
often. This:
- **Increases recall** from 0.800 to 0.933: more true malignancies are caught (FN drops
  from 45 to 15, i.e., only 7 % of malignancies are missed instead of 20 %).
- **Decreases precision** from 0.857 to 0.636: more benign cases are flagged as malignant
  (FP increases from 30 to 120). More patients undergo unnecessary follow-up biopsies.
- **F1 decreases** from 0.828 to 0.757 because the precision drop more than offsets the
  recall gain.

**(c) Cost comparison with 10:1 FN/FP asymmetry:**

```
Cost(threshold=0.5) = FP * C_FP + FN * C_FN
                    = 30 * 1  + 45 * 10
                    = 30 + 450
                    = 480

Cost(threshold=0.3) = FP * C_FP + FN * C_FN
                    = 120 * 1  + 15 * 10
                    = 120 + 150
                    = 270
```

**The hospital should prefer threshold = 0.3** (lower cost = 270 vs 480). Even though
precision drops and more false alarms occur, the dramatic reduction in missed cancers
(from 45 to 15) dominates the calculation under the 10:1 cost asymmetry.

This illustrates a key principle: **F1 is not the right optimisation target when costs are
asymmetric**. Threshold selection should be guided by the actual cost structure of the
deployment context.

---

### Part F Solution

**Three-class confusion matrix:**

```
Predicted:    Malignant   Benign   Indeterminate   Row totals
Actual:
Malignant         170       20            35              225
Benign             25      930            45             1000
Indeterminate      40       80          1655             1775

Column totals:    235     1030          1735             3000
```

**(a) Per-class precision, recall, and F1:**

For each class c, treat it as "positive" and all others as "negative":

```
Malignant class (positive = malignant):
  TP = 170  (correctly predicted malignant)
  FP = 25 + 40 = 65  (other classes predicted as malignant)
  FN = 20 + 35 = 55  (malignant predicted as another class)

  Precision = 170 / (170 + 65)  = 170 / 235 = 0.7234
  Recall    = 170 / (170 + 55)  = 170 / 225 = 0.7556
  F1        = 2 * (0.7234 * 0.7556) / (0.7234 + 0.7556)
            = 2 * 0.5466 / 1.4790
            = 0.7391

Benign class (positive = benign):
  TP = 930
  FP = 20 + 80 = 100
  FN = 25 + 45 = 70

  Precision = 930 / (930 + 100) = 930 / 1030 = 0.9029
  Recall    = 930 / (930 + 70)  = 930 / 1000 = 0.9300
  F1        = 2 * (0.9029 * 0.9300) / (0.9029 + 0.9300)
            = 2 * 0.8397 / 1.8329
            = 0.9163

Indeterminate class (positive = indeterminate):
  TP = 1655
  FP = 35 + 45 = 80
  FN = 40 + 80 = 120

  Precision = 1655 / (1655 + 80)  = 1655 / 1735 = 0.9539
  Recall    = 1655 / (1655 + 120) = 1655 / 1775 = 0.9324
  F1        = 2 * (0.9539 * 0.9324) / (0.9539 + 0.9324)
            = 2 * 0.8894 / 1.8863
            = 0.9431
```

Summary:
```
Class             Precision   Recall   F1      Support
Malignant         0.7234      0.7556   0.7391    225
Benign            0.9029      0.9300   0.9163   1000
Indeterminate     0.9539      0.9324   0.9431   1775
```

**(b) Macro F1 and weighted F1:**

```
Macro F1 = (0.7391 + 0.9163 + 0.9431) / 3
         = 2.5985 / 3
         = 0.8662

Weighted F1 = (0.7391 * 225 + 0.9163 * 1000 + 0.9431 * 1775) / 3000
            = (166.30 + 916.30 + 1674.00) / 3000
            = 2756.60 / 3000
            = 0.9189
```

The weighted F1 (0.919) is substantially higher than the macro F1 (0.866) because the
two larger, easier classes dominate the weighted average.

**(c) Hardest class and confusion pattern:**

The **malignant** class is hardest (F1 = 0.739). Key observations:

- **Recall = 0.756:** 24.4 % of actual malignant cases are misclassified. The off-diagonal
  cells show 20 malignant cases predicted as benign and 35 as indeterminate.
- **Precision = 0.723:** 27.7 % of malignant predictions are wrong -- 25 benign and 40
  indeterminate cases are incorrectly labelled malignant.

The confusion pattern suggests:
1. **Malignant-indeterminate confusion is the dominant error:** 35 malignant cases are
   called indeterminate (borderline cases with ambiguous features misclassified as
   uncertain), and 40 indeterminate cases are called malignant. This suggests the model
   struggles to distinguish between truly malignant and ambiguous borderline cases.
2. **Malignant-benign confusion is secondary:** 20 malignant cases called benign is the
   most clinically dangerous error. These are cancers the model completely misses.

**Remediation:**
- Collect more labelled malignant training examples (data imbalance: malignant = 7.5 %
  of the dataset).
- Investigate whether indeterminate labels in the training set are noisy (inter-annotator
  disagreement between radiologists may create label noise on this class).
- Adjust the decision threshold for the malignant class to prioritise recall, accepting
  more indeterminate predictions being reclassified as malignant.
