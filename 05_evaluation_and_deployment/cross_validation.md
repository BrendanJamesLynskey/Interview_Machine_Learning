# Cross Validation

## Prerequisites
- Train/validation/test split concepts
- Bias-variance trade-off
- Overfitting and generalisation

---

## Concept Reference

### Why Cross Validation Exists

A single train/validation split wastes data and produces a high-variance estimate of
generalisation performance. If the split is unlucky, the validation set may not be
representative of the data distribution, causing you to over-select or under-select models
based on noise.

Cross validation (CV) systematically rotates which data serves as the validation set,
producing k different estimates of held-out performance and averaging them. This:

1. **Reduces variance** of the performance estimate by averaging over k folds.
2. **Makes better use of limited data** -- every sample appears in both training and
   validation sets across folds.
3. **Enables reliable model comparison** -- differences in cross-validated scores are more
   likely to reflect genuine differences in model quality than single-split scores.

---

### K-Fold Cross Validation

The standard algorithm:

```
1. Shuffle the dataset (if order is not meaningful).
2. Partition it into k equal-sized folds.
3. For i = 1 to k:
     - Hold out fold i as the validation set.
     - Train on the remaining k-1 folds.
     - Evaluate on fold i, recording the metric.
4. Report mean and standard deviation of the k metric values.
```

```
Dataset: [F1][F2][F3][F4][F5]   (k=5 folds)

Iteration 1:  Train=[F2,F3,F4,F5]  Val=[F1]
Iteration 2:  Train=[F1,F3,F4,F5]  Val=[F2]
Iteration 3:  Train=[F1,F2,F4,F5]  Val=[F3]
Iteration 4:  Train=[F1,F2,F3,F5]  Val=[F4]
Iteration 5:  Train=[F1,F2,F3,F4]  Val=[F5]
```

**Choosing k:**
- k = 5 or k = 10 are standard. k = 10 gives lower bias (training sets are 90 % of data)
  but higher computational cost.
- Larger k -> lower bias, higher variance of the CV estimate, higher computation.
- k = N (leave-one-out) is the extreme case (see below).

---

### Stratified K-Fold

Standard k-fold assigns samples to folds by position (or random shuffle), which can
produce folds with very different class distributions, especially when the dataset is
imbalanced or small.

**Stratified k-fold** preserves the class distribution in each fold: if the overall dataset
has 10 % positives, each fold also has approximately 10 % positives.

```
Original:  [+, +, -, -, -, -, -, -, -, -]   (10 % positive)
Stratified: each of k folds also contains ~10 % positive samples.
```

Stratified k-fold is the default choice for classification tasks. It is especially
important when:
- The dataset is small (a random fold might contain zero positive samples).
- The dataset is highly imbalanced (class ratios deviate badly across folds with standard
  k-fold).
- Evaluation metrics are sensitive to class distribution (accuracy, AUC-PR).

---

### Leave-One-Out Cross Validation (LOO-CV)

LOO-CV is k-fold where k = N (the dataset size). Each fold holds out a single sample,
trains on the remaining N-1, and evaluates on that one sample.

**Properties:**
- Produces a nearly unbiased estimate of the true generalisation error (because each
  training set differs from the full dataset by only one sample).
- High variance: the N estimates are highly correlated because N-1 of the N training
  samples are shared between any two folds.
- Computationally expensive: O(N) model fits. Impractical for large N or expensive models.
- Analytically tractable for some linear models (e.g., ridge regression has a closed-form
  LOO-CV score via the hat matrix), making it efficient in those cases.

**When to use LOO-CV:**
- Very small datasets (N < 50) where no data can be afforded to a held-out test set.
- Linear models where the closed-form LOO trick applies.
- Otherwise, 5-fold or 10-fold CV is preferred.

---

### Repeated K-Fold

Run k-fold CV multiple times with different random splits, then average all scores.

```
Repeated 5-fold, 10 repeats: 50 model fits, 50 validation scores.
```

Reduces variance of the CV estimate further. Useful when the dataset is small and a
single k-fold CV estimate would have high variance. The computational cost scales linearly
with the number of repeats.

---

### Time Series Cross Validation (Walk-Forward Validation)

Standard k-fold violates the temporal ordering of time series data: training on future
data to predict the past causes **data leakage**, producing optimistic estimates that
do not hold in deployment.

**Walk-forward (expanding window) validation:**

```
Fold 1:  Train=[t1..t4]       Val=[t5]
Fold 2:  Train=[t1..t5]       Val=[t6]
Fold 3:  Train=[t1..t6]       Val=[t7]
...
```

Each validation fold uses only data that would have been available at that point in time.
The training window expands as more data accumulates.

**Sliding window validation** uses a fixed-size training window:

```
Fold 1:  Train=[t1..t4]       Val=[t5]
Fold 2:  Train=[t2..t5]       Val=[t6]
Fold 3:  Train=[t3..t6]       Val=[t7]
...
```

Useful when the data distribution is non-stationary and older data is less relevant.

**Gap between train and validation:** In practice, add a gap period between the last
training sample and the first validation sample to account for the prediction horizon
(e.g., if your model predicts 30 days ahead, leave a 30-day gap).

```
Fold 1:  Train=[t1..t4]  Gap=[t5,t6]  Val=[t7]
```

---

### Nested Cross Validation

Used when both hyperparameter tuning **and** model selection/evaluation must be done
without a separate test set.

```
Outer loop (k_outer folds): estimates the generalisation error of the full
                             "select-and-train" pipeline.
  Inner loop (k_inner folds): selects hyperparameters using CV within each
                               outer training set.
```

```
Outer fold 1:
  Outer train = folds [2..k_outer]
    Inner CV on outer train -> best hyperparameters
    Retrain on full outer train with best hyperparameters
  Outer val = fold 1 -> record outer validation score

Repeat for all outer folds.
Final estimate: mean of outer validation scores.
```

Nested CV is the correct way to report an unbiased estimate of generalisation performance
when hyperparameter tuning is part of the modelling workflow. Without nesting, using the
same data for tuning and evaluation causes optimistic bias ("selection bias").

---

### Common Mistakes

**1. Leaking the test set into cross validation.**
Fit preprocessing (e.g., StandardScaler, PCA) on the entire dataset including the
validation fold, then apply it to train and val. This leaks val statistics into training.
**Fix:** Always fit preprocessing inside the cross-validation loop, only on the training
fold. Use sklearn `Pipeline` objects to automate this.

**2. Using standard k-fold for time series data.**
Randomly shuffling and splitting time series allows training on future data, producing
artificially optimistic CV scores. **Fix:** Use `TimeSeriesSplit` or walk-forward CV.

**3. Treating the CV mean as an exact performance guarantee.**
The CV mean is an estimate with uncertainty. Report the standard deviation across folds
alongside the mean. A difference of 0.001 F1 between two models is not meaningful if the
standard deviation is 0.01.

**4. Running hyperparameter search and using the same k-fold for evaluation.**
This is "information leakage through the outer loop." **Fix:** Use nested CV.

---

## Interview Questions by Difficulty

### Fundamentals

**Q1.** Why is 10-fold CV generally preferred over a single 80/20 train-validation split?

**Answer:**

A single 80/20 split uses only 20 % of the data for validation. If this split is unlucky,
the 20 % may not be representative and the performance estimate will be noisy. 10-fold CV
trains 10 models, each evaluated on a different 10 % of the data. The mean of the 10
scores has much lower variance than any single score. Additionally, each sample participates
in training in 9 of the 10 folds, making better use of limited data. The cost is 10x more
model fits, which is acceptable for all but the most expensive models.

---

**Q2.** When should you use stratified k-fold instead of standard k-fold?

**Answer:**

Always use stratified k-fold for classification tasks, particularly when:
- The dataset is imbalanced (any class comprises less than 20-30 % of the data).
- The dataset is small (random folds may produce zero samples of a rare class).
- Evaluation metrics are sensitive to class proportions (F1, AUC-PR, accuracy on imbalanced
  data).

Standard k-fold randomly assigns samples to folds. For a dataset with 5 % positives,
random assignment can produce a fold with 0-2 % positives, making the fold-level metric
unstable and the overall CV estimate noisy. Stratification prevents this by design.

---

### Intermediate

**Q3.** You are building a churn prediction model using three years of monthly customer
data. How should you design the cross-validation strategy?

**Answer:**

Temporal ordering must be respected to avoid data leakage. Use walk-forward (expanding
window) CV:

1. Define a minimum training window (e.g., 12 months) to ensure sufficient history.
2. Iterate month by month, training on all data up to month t and predicting month t+1.
3. Include a gap equal to the prediction horizon (e.g., if predicting 30-day churn, gap
   of 1 month between last training sample and first validation sample).

```
Fold 1:  Train=[months 1-12]   Val=[month 13]
Fold 2:  Train=[months 1-13]   Val=[month 14]
...
Fold 24: Train=[months 1-35]   Val=[month 36]
```

Report mean AUC-ROC or F1 across all folds, weighted by the number of customers in each
validation month if desired. This setup matches the deployment scenario exactly: you will
always train on past data and predict future churn.

---

**Q4.** What is the bias-variance trade-off in LOO-CV vs 5-fold CV?

**Answer:**

**LOO-CV:**
- **Low bias:** Each model trains on N-1 samples, very close to the full dataset size.
  The performance estimate is nearly unbiased for the "train on N samples" scenario.
- **High variance:** The N CV estimates are highly correlated (N-1 shared training samples
  between any two folds). A few difficult samples have an outsized effect. The standard
  deviation of LOO-CV scores is typically larger than for k-fold.

**5-fold CV:**
- **Moderate bias:** Each model trains on 80 % of the data, slightly fewer samples than
  the full dataset. The estimate is slightly pessimistic (the model trained on 100 % of
  data would likely be marginally better).
- **Lower variance:** The 5 folds have more diverse training sets, reducing correlation
  between estimates. The standard deviation of 5-fold scores is lower than LOO-CV.

In practice, for most datasets of moderate size (N > 100), 5-fold or 10-fold CV provides
a better bias-variance trade-off than LOO-CV. LOO-CV is preferable only for very small
datasets (N < 30) or linear models with closed-form LOO scores.

---

### Advanced

**Q5.** A colleague proposes the following workflow for a medical diagnosis task:
(a) standardise all features on the full dataset, (b) run 5-fold CV to compare models,
(c) report the best model's CV score as the expected test accuracy. Identify all flaws
in this workflow and explain how to fix each.

**Answer:**

**Flaw 1: Preprocessing on the full dataset before CV.**
Standardising (computing mean and variance) on the full dataset includes the validation
folds. The standardisation statistics "see" validation data during training, which is a
form of data leakage. On large datasets the effect is small but on small datasets it can
produce optimistic estimates.

*Fix:* Fit the StandardScaler only on the training fold inside each CV iteration. Use a
`Pipeline(steps=[('scaler', StandardScaler()), ('model', model)])` passed to
`cross_val_score`, which ensures the scaler is refit on each training fold.

**Flaw 2: Reporting the best model's CV score as expected test accuracy.**
If multiple models are compared and the best one is selected based on CV score, the
reported CV score is optimistically biased due to selection bias. The score corresponds
to the best of multiple evaluations, not an unbiased single evaluation.

*Fix:* Use nested CV. The outer loop provides an unbiased estimate of the
"select-and-train" pipeline. Alternatively, reserve a truly held-out test set that is not
used for model selection at all, and evaluate only once on it after the final model is
chosen.

**Flaw 3: No discussion of CV variance.**
Reporting only the mean CV score without standard deviation across folds can make
differences between models look more significant than they are.

*Fix:* Report mean ± standard deviation of CV scores. Use paired statistical tests (e.g.,
corrected resampled t-test for k-fold CV) before claiming one model is superior.
