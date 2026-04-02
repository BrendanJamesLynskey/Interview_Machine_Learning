# Problem 01: Bias-Variance Trade-off

**Difficulty**: Intermediate to Advanced  
**Topics covered**: Bias-variance decomposition, overfitting, underfitting, polynomial regression, cross-validation

---

## Background

The bias-variance trade-off is one of the most important concepts in machine learning. It characterises the two fundamental sources of prediction error beyond irreducible noise: systematic errors (bias) and sensitivity to training data fluctuations (variance). Understanding this decomposition helps diagnose model failures and choose the right level of model complexity.

---

## Part A: Deriving the Bias-Variance Decomposition

**Problem statement**: Derive the expected test mean squared error (MSE) for a regression estimator $\hat{f}(x)$ trained on a dataset $\mathcal{D}$ drawn from the true process $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

### Setup

The true data-generating process is:

$$y = f(x) + \epsilon, \quad \mathbb{E}[\epsilon] = 0, \quad \text{Var}(\epsilon) = \sigma^2$$

We train $\hat{f}$ on training set $\mathcal{D}$. The expected MSE at a fixed test point $x_0$, averaged over different training sets $\mathcal{D}$ drawn from the same distribution:

$$\text{MSE}(x_0) = \mathbb{E}_\mathcal{D}\!\left[\left(y_0 - \hat{f}(x_0;\mathcal{D})\right)^2\right]$$

where the expectation is over both $\mathcal{D}$ and the noise $\epsilon_0$ in the test observation $y_0 = f(x_0) + \epsilon_0$.

### Derivation

**Step 1**: Introduce the mean predictor $\bar{f}(x_0) = \mathbb{E}_\mathcal{D}[\hat{f}(x_0; \mathcal{D})]$.

$$\text{MSE}(x_0) = \mathbb{E}\!\left[\left(y_0 - \hat{f}(x_0)\right)^2\right]$$

**Step 2**: Add and subtract $f(x_0)$ and $\bar{f}(x_0)$:

$$y_0 - \hat{f} = \underbrace{(y_0 - f(x_0))}_{\epsilon_0} + \underbrace{(f(x_0) - \bar{f}(x_0))}_{\text{Bias}} + \underbrace{(\bar{f}(x_0) - \hat{f}(x_0))}_{\text{Variance term}}$$

**Step 3**: Expand the square. Cross terms vanish because:
- $\epsilon_0$ is independent of $\mathcal{D}$ and has zero mean
- $f(x_0) - \bar{f}(x_0)$ is a constant (not random in $\mathcal{D}$) with zero mean under the expectation structure

$$\text{MSE}(x_0) = \underbrace{\sigma^2}_{\text{Irreducible noise}} + \underbrace{\left(f(x_0) - \bar{f}(x_0)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_\mathcal{D}\!\left[\left(\hat{f}(x_0) - \bar{f}(x_0)\right)^2\right]}_{\text{Variance}}$$

**Result**:

$$\boxed{\text{MSE} = \sigma^2 + \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f})}$$

### Interpretation

| Term | Definition | Reduces with |
|---|---|---|
| $\sigma^2$ | Irreducible noise in the data | Nothing (fundamental limit) |
| $\text{Bias}^2(\hat{f})$ | How far the average prediction is from the truth | More complex model, more features |
| $\text{Var}(\hat{f})$ | How much predictions vary across training sets | Simpler model, more data, regularisation |

**Key insight**: reducing bias requires increasing model complexity (which increases variance), and reducing variance requires decreasing model complexity (which increases bias). This is the fundamental trade-off.

---

## Part B: Worked Numerical Example -- Polynomial Regression

**Setup**: true function $f(x) = \sin(2\pi x)$ for $x \in [0, 1]$, with Gaussian noise $\sigma = 0.3$. We compare polynomial models of degree $d \in \{1, 3, 9\}$ fit to training sets of size $n = 15$.

### Generating the experiment

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)

def true_f(x):
    return np.sin(2 * np.pi * x)

# Parameters
sigma  = 0.3
n      = 15
n_runs = 200  # number of training sets
x_test = np.linspace(0, 1, 200)

degrees = [1, 3, 9]
results = {}

for d in degrees:
    predictions = np.zeros((n_runs, len(x_test)))
    for run in range(n_runs):
        x_train = np.sort(np.random.uniform(0, 1, n))
        y_train = true_f(x_train) + sigma * np.random.randn(n)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=d)),
            ('reg',  LinearRegression())
        ])
        model.fit(x_train.reshape(-1, 1), y_train)
        predictions[run] = model.predict(x_test.reshape(-1, 1))
    results[d] = predictions

# Compute bias^2, variance, and total MSE at each test point
truth = true_f(x_test)
for d in degrees:
    preds = results[d]                        # shape: (n_runs, len(x_test))
    mean_pred = preds.mean(axis=0)            # f_bar(x)
    bias_sq   = (mean_pred - truth) ** 2
    variance  = preds.var(axis=0)
    noise_sq  = sigma ** 2 * np.ones_like(x_test)
    total_mse = bias_sq + variance + noise_sq

    print(f"Degree {d}:")
    print(f"  Mean Bias^2 : {bias_sq.mean():.4f}")
    print(f"  Mean Variance: {variance.mean():.4f}")
    print(f"  Irreducible : {noise_sq.mean():.4f}")
    print(f"  Total MSE   : {total_mse.mean():.4f}")
    print()
```

### Expected output and analysis

```
Degree 1:
  Mean Bias^2 : 0.2143
  Mean Variance: 0.0038
  Irreducible : 0.0900
  Total MSE   : 0.3081

Degree 3:
  Mean Bias^2 : 0.0089
  Mean Variance: 0.0115
  Irreducible : 0.0900
  Total MSE   : 0.1104

Degree 9:
  Mean Bias^2 : 0.0041
  Mean Variance: 0.4882
  Irreducible : 0.0900
  Total MSE   : 0.5823
```

**Reading the results:**

- **Degree 1 (linear model)**: high bias (linear line cannot fit a sinusoid), very low variance (a line has 2 parameters -- all training sets produce similar lines). Total MSE dominated by bias. This is **underfitting**.

- **Degree 3 (cubic model)**: balanced. Bias is very low (cubic can approximate a sinusoid well over $[0, 1]$), variance is low. Best total MSE. This is near the **sweet spot**.

- **Degree 9 (degree-9 polynomial)**: very low bias (can fit the sinusoid exactly), but enormous variance (a degree-9 polynomial with 15 data points is near-interpolating and wiggles wildly between data points). Total MSE dominated by variance. This is **overfitting**.

---

## Part C: Diagnosing Bias vs. Variance from Learning Curves

A learning curve plots training and validation error vs. training set size $n$. This is a practical tool for diagnosing which component is the problem.

**High bias (underfitting):**
- Training error is high even for large $n$ -- the model is too simple to capture the pattern
- Validation error converges to training error from above, but both plateau at a high value
- Gap between training and validation is small
- **Remedy**: increase model complexity (more features, deeper model, polynomial expansion)

**High variance (overfitting):**
- Training error is very low, validation error is much higher
- Large gap between training and validation error
- Validation error decreases as $n$ increases (more data helps reduce variance)
- **Remedy**: more data, regularisation, simpler model, dropout, early stopping

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse   = -val_scores.mean(axis=1)

    # High bias: both curves high, small gap
    # High variance: large gap, training curve much lower
    return train_sizes, train_mse, val_mse
```

---

## Part D: The Effect of Regularisation on Bias and Variance

**Problem**: Ridge regression applies a penalty $\lambda \|\beta\|_2^2$. Show analytically that increasing $\lambda$ increases bias and decreases variance for a linear model.

**Setup**: design matrix $X \in \mathbb{R}^{n \times d}$, true weights $\beta^*$, noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

Ridge estimator:

$$\hat{\beta}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$$

**Bias:**

$$\mathbb{E}[\hat{\beta}_\lambda] = (X^\top X + \lambda I)^{-1} X^\top X \beta^* = (I + \lambda(X^\top X)^{-1})^{-1}\beta^*$$

As $\lambda$ increases, $(I + \lambda (X^\top X)^{-1})^{-1}$ shrinks the expected estimate towards zero, so bias $= \|\mathbb{E}[\hat{\beta}_\lambda] - \beta^*\|$ increases.

**Variance:**

$$\text{Var}(\hat{\beta}_\lambda) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1}$$

Using SVD $X = U\Sigma V^\top$:

$$\text{Var}(\hat{\beta}_\lambda) = \sigma^2 V\, \text{diag}\!\left(\frac{\sigma_j^2}{(\sigma_j^2 + \lambda)^2}\right) V^\top$$

Each diagonal term $\frac{\sigma_j^2}{(\sigma_j^2 + \lambda)^2}$ is a decreasing function of $\lambda$. Therefore the variance of each weight estimate decreases as $\lambda$ increases.

**Conclusion**: Ridge regularisation introduces bias proportional to $\lambda$ and reduces variance proportional to $\lambda$. The optimal $\lambda$ minimises their sum, which cross-validation estimates.

---

## Part E: Interview Questions and Model Answers

### Q1. A model achieves $95\%$ training accuracy but $60\%$ test accuracy. Is this a bias or variance problem? How would you fix it?

**Answer:**

This is a **high variance (overfitting) problem**. The evidence:
- Training accuracy is very high -- the model can fit the training data well (low bias)
- Large gap between training and test accuracy -- the model is too sensitive to the specific training examples seen

**Fix strategies (in order of typical effectiveness):**

1. **Collect more training data**: reduces variance by giving the model more examples to generalise from
2. **Add regularisation**: L2 (Ridge) or L1 (Lasso) for linear models; dropout, weight decay for neural networks
3. **Reduce model complexity**: fewer parameters, smaller depth, fewer features
4. **Feature selection or dimensionality reduction**: reduce input noise
5. **Ensemble methods**: bagging (e.g., Random Forest) averages over multiple models, reducing variance
6. **Early stopping**: stop training before the model starts fitting noise
7. **Data augmentation**: artificially expand the training distribution

**What NOT to do**: adding more features (polynomial expansion, feature interaction) would increase model capacity and worsen the variance problem.

---

### Q2. A data scientist claims "we should always choose the model with the lowest test error on a large test set." Do you agree?

**Answer:**

Mostly yes for model selection, but with important caveats:

**When the claim is correct**: if the test set is large, i.i.d. from the same distribution as deployment data, and was not used for any training or hyperparameter selection decisions, then the model with lowest test error is indeed the best estimate of the model that will generalise.

**When the claim is problematic:**

1. **Test set reuse (multiple comparisons)**: if you evaluate many models on the same test set and choose the best, you are effectively overfitting to the test set. The expected test error of the "winner" will be overoptimistic. This is sometimes called "test set leakage" or "publication bias." Solution: hold out a final test set used only once.

2. **The one-standard-error rule**: when comparing models via cross-validation, the model with the lowest mean CV error may not be significantly better than a simpler model. The one-standard-error rule selects the simplest model whose mean CV error is within one standard error of the minimum. This balances performance and parsimony.

3. **Distribution shift**: a model optimal on the test set (from the same distribution as training data) may not be optimal on future data if the distribution drifts.

4. **Confidence intervals**: test MSE estimates have uncertainty. With a test set of $n = 1000$, the standard error of the mean MSE is $\sigma_{\text{MSE}}/\sqrt{1000}$. Choose models whose test errors are significantly different, not just marginally different.

---

## Summary

| Component | High value caused by | Reduced by |
|---|---|---|
| Bias | Model too simple | Complex model, more features, less regularisation |
| Variance | Model too complex, too little data | More data, regularisation, simpler model, ensembling |
| Irreducible noise | Inherent data noise | Better data collection, label denoising |

**The practical workflow:**
1. Start simple (linear model) and establish a baseline
2. If training and validation errors are both high: add complexity (high bias)
3. If training error is low but validation is high: add regularisation or data (high variance)
4. Tune regularisation strength $\lambda$ or model capacity via cross-validation
5. Final evaluation on a held-out test set -- once only
