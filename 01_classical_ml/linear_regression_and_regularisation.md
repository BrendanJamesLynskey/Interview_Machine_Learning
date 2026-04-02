# Linear Regression and Regularisation

## Prerequisites
- Linear algebra: matrix transpose, inverse, positive semi-definiteness
- Calculus: partial derivatives, gradient, setting gradient to zero
- Basic probability: Gaussian distribution, maximum likelihood estimation
- Python and NumPy for the implementation sections

---

## Concept Reference

### Ordinary Least Squares (OLS)

Given $n$ training examples $\{(x_i, y_i)\}$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, the linear model predicts:

$$\hat{y} = X\beta$$

where $X \in \mathbb{R}^{n \times d}$ is the design matrix (rows are examples, columns are features), and $\beta \in \mathbb{R}^d$ is the weight vector. A bias term is included by prepending a column of ones to $X$.

The OLS objective minimises the residual sum of squares:

$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 = (y - X\beta)^\top (y - X\beta)$$

**Normal equations derivation:**

Expand the loss:

$$\mathcal{L}(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta$$

Take the gradient with respect to $\beta$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial \beta} = -2X^\top y + 2X^\top X \beta = 0$$

$$\Rightarrow \hat{\beta}_{\text{OLS}} = (X^\top X)^{-1} X^\top y$$

This is the **normal equation**. The matrix $X^\top X$ must be invertible (full column rank). If features are linearly dependent, $X^\top X$ is singular and the solution is not unique.

**Geometric interpretation**: $\hat{y} = X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$. The residual $y - \hat{y}$ is perpendicular to every column of $X$.

**Probabilistic interpretation**: OLS is the maximum likelihood estimator under the Gaussian noise model $y = X\beta + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

### Assumptions of Linear Regression

1. **Linearity**: $\mathbb{E}[y \mid x] = x^\top \beta$ (the mean response is linear in features)
2. **Independence**: residuals $\epsilon_i$ are independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ is constant across all $i$
4. **No perfect multicollinearity**: $X$ has full column rank
5. **No endogeneity**: $\mathbb{E}[\epsilon \mid X] = 0$ (errors uncorrelated with features)

Under assumptions 1--5, the **Gauss-Markov theorem** states that OLS is the **BLUE** (Best Linear Unbiased Estimator) -- it has minimum variance among all linear unbiased estimators.

### The Problem with OLS: Variance and Multicollinearity

OLS is unbiased but can have high variance when:
- $d$ is large relative to $n$ (many features, few examples)
- Features are highly correlated (near-multicollinearity makes $X^\top X$ ill-conditioned)
- The model is overparameterised

The variance of OLS estimates is:

$$\text{Var}(\hat{\beta}_{\text{OLS}}) = \sigma^2 (X^\top X)^{-1}$$

When $X^\top X$ has small eigenvalues (near-singular), the inverse has large eigenvalues, inflating variance. High variance means weights can take arbitrarily large values and the model overfits.

### L2 Regularisation -- Ridge Regression

Ridge regression adds a squared $\ell_2$ penalty on the weights:

$$\mathcal{L}_{\text{Ridge}}(\beta) = \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

Taking the gradient and setting to zero:

$$-2X^\top y + 2X^\top X \beta + 2\lambda \beta = 0$$

$$\hat{\beta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

**Key properties:**
- Adding $\lambda I$ to $X^\top X$ ensures the matrix is always invertible (positive definite), regardless of multicollinearity
- All weights are shrunk towards zero proportionally; no weight is set exactly to zero
- The solution is unique even when $d > n$
- Equivalent to placing a Gaussian prior $\beta_j \sim \mathcal{N}(0, 1/\lambda)$ and computing the MAP estimate

**Bias-variance trade-off**: Ridge introduces bias $\mathbb{E}[\hat{\beta}_{\text{Ridge}}] \neq \beta^*$ but reduces variance. The net effect on test MSE can be positive when the variance reduction exceeds the bias increase.

### L1 Regularisation -- Lasso

Lasso (Least Absolute Shrinkage and Selection Operator) adds an $\ell_1$ penalty:

$$\mathcal{L}_{\text{Lasso}}(\beta) = \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

There is no closed-form solution because $|\beta_j|$ is not differentiable at $\beta_j = 0$. Lasso is solved with coordinate descent or subgradient methods.

**Key property -- sparsity**: Lasso produces sparse solutions. Many weights are set exactly to zero, performing implicit feature selection. This is the critical distinction from Ridge.

**Why L1 induces sparsity (geometric intuition):**

The constraint region for $\ell_1$ is a diamond (in 2D) with corners on the axes. The unconstrained OLS solution is an ellipsoid. The Lasso solution is where the ellipsoid first touches the diamond. The corners of the diamond lie on coordinate axes, so the solution frequently has some $\beta_j = 0$.

The constraint region for $\ell_2$ is a circle (in 2D). The ellipsoid typically touches the circle at a non-axis point, so Ridge rarely produces exact zeros.

**Bayesian interpretation**: Lasso corresponds to a Laplace prior $\beta_j \sim \text{Laplace}(0, 1/\lambda)$ on the weights. The Laplace distribution has heavier tails and a sharper peak at zero than the Gaussian, which favours sparse solutions.

### Elastic Net

Elastic net combines L1 and L2 penalties:

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

Or equivalently with a mixing parameter $\alpha \in [0, 1]$:

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right]$$

**When to use elastic net over Lasso:**
- When features are strongly correlated: Lasso tends to pick one feature from a correlated group arbitrarily; elastic net selects the whole group together (grouped selection)
- When $d \gg n$: Lasso can select at most $n$ non-zero features; elastic net has no such constraint
- When you need both sparsity and stability

### Comparison Summary

| Property | OLS | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|---|
| Closed form | Yes | Yes | No | No |
| Biased | No | Yes | Yes | Yes |
| Sparse solution | No | No | Yes | Yes |
| Handles multicollinearity | Poorly | Well | Partially | Well |
| Feature selection | No | No | Yes | Yes |
| Handles $d > n$ | No | Yes | Yes (limited) | Yes |
| Grouped selection | No | No | No | Yes |

---

## Tier 1 -- Fundamentals

### Q1. Derive the normal equations for ordinary least squares.

**Answer:**

The objective is to minimise:

$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2$$

Expand using the identity $\|v\|^2 = v^\top v$:

$$\mathcal{L}(\beta) = (y - X\beta)^\top (y - X\beta)$$
$$= y^\top y - y^\top X\beta - \beta^\top X^\top y + \beta^\top X^\top X \beta$$
$$= y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta$$

(Note: $y^\top X\beta = \beta^\top X^\top y$ since both are scalars.)

Differentiate with respect to $\beta$ using matrix calculus rules:
- $\frac{\partial}{\partial \beta}(\beta^\top a) = a$ for constant $a$
- $\frac{\partial}{\partial \beta}(\beta^\top A \beta) = 2A\beta$ for symmetric $A$

$$\frac{\partial \mathcal{L}}{\partial \beta} = -2X^\top y + 2X^\top X \beta$$

Setting to zero:

$$X^\top X \beta = X^\top y$$

$$\hat{\beta} = (X^\top X)^{-1} X^\top y \quad \text{(assuming } X^\top X \text{ is invertible)}$$

**Computational note**: In practice, computing the explicit inverse $(X^\top X)^{-1}$ is numerically unstable. Implementations solve the system $X^\top X \hat{\beta} = X^\top y$ using Cholesky factorisation, or use the QR decomposition of $X$ directly.

---

### Q2. What is multicollinearity and how does Ridge regression address it?

**Answer:**

**Multicollinearity** occurs when two or more features are highly correlated -- one feature can be approximately expressed as a linear combination of others. In the extreme case, $X^\top X$ becomes singular (non-invertible) and OLS has no unique solution.

Even when not perfectly collinear, near-multicollinearity causes $X^\top X$ to have very small eigenvalues. The inverse $(X^\top X)^{-1}$ then has very large eigenvalues, making the OLS estimates extremely sensitive to small changes in the data -- high variance.

**Ridge regression** adds $\lambda I$ to $X^\top X$:

$$\hat{\beta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

Since $\lambda I$ is positive definite, $X^\top X + \lambda I$ is always invertible. The minimum eigenvalue of $X^\top X + \lambda I$ is at least $\lambda$, controlling the condition number of the system. This "shrinks" the effective influence of near-collinear features, reducing variance at the cost of introducing bias.

---

### Q3. Explain the difference between L1 and L2 regularisation. Why does L1 produce sparse weights?

**Answer:**

**L2 (Ridge)**: penalises $\|\beta\|_2^2 = \sum_j \beta_j^2$. The gradient of the penalty is $2\lambda \beta_j$, which pulls each weight proportionally towards zero but never reaches exactly zero (the gradient becomes smaller as $\beta_j$ approaches zero, slowing convergence but not stopping at exactly zero).

**L1 (Lasso)**: penalises $\|\beta\|_1 = \sum_j |\beta_j|$. The subgradient of $|\beta_j|$ is $\text{sign}(\beta_j)$ -- a constant magnitude regardless of the size of $\beta_j$. This constant pull towards zero means weights can be pushed all the way to zero and held there.

**Geometric explanation:**

The Lasso constraint $\|\beta\|_1 \leq t$ defines a diamond (in 2D) with vertices at $(\pm t, 0)$ and $(0, \pm t)$. The OLS solution is the centre of a family of elliptical level sets. The Lasso solution is the first point where these ellipses touch the diamond. The corners of the diamond are at the coordinate axes, and the typical touching point is at a corner, producing $\beta_j = 0$ for one or more $j$.

**Practical consequence**: Lasso performs **automatic feature selection** -- it identifies and zeros out irrelevant features. Ridge keeps all features but shrinks their coefficients. When many features are truly irrelevant (sparse signal), Lasso is preferable. When all features have some effect, Ridge is typically better.

---

## Tier 2 -- Intermediate

### Q4. Prove that Ridge regression is equivalent to MAP estimation with a Gaussian prior on the weights.

**Answer:**

Assume the likelihood model:

$$y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

So the likelihood is:

$$P(y \mid X, \beta) = \mathcal{N}(y \mid X\beta, \sigma^2 I) \propto \exp\!\left(-\frac{1}{2\sigma^2}\|y - X\beta\|_2^2\right)$$

Assume a Gaussian prior on the weights:

$$P(\beta) = \mathcal{N}(0, \tau^2 I) \propto \exp\!\left(-\frac{1}{2\tau^2}\|\beta\|_2^2\right)$$

By Bayes' rule, the posterior is:

$$P(\beta \mid y, X) \propto P(y \mid X, \beta)\, P(\beta)$$

$$\propto \exp\!\left(-\frac{1}{2\sigma^2}\|y - X\beta\|_2^2 - \frac{1}{2\tau^2}\|\beta\|_2^2\right)$$

The MAP (maximum a posteriori) estimate maximises this, equivalently minimising the negative log posterior:

$$\hat{\beta}_{\text{MAP}} = \arg\min_\beta \left[\|y - X\beta\|_2^2 + \frac{\sigma^2}{\tau^2}\|\beta\|_2^2\right]$$

This is exactly the Ridge objective with $\lambda = \sigma^2 / \tau^2$. A smaller prior variance $\tau^2$ (stronger belief weights are near zero) corresponds to larger $\lambda$ (stronger regularisation).

---

### Q5. How does the choice of regularisation strength $\lambda$ affect the bias-variance trade-off?

**Answer:**

Define the test MSE decomposition for a fixed test point $x_0$:

$$\text{MSE}(\hat{y}_0) = \underbrace{\text{Bias}^2(\hat{y}_0)}_{\text{systematic error}} + \underbrace{\text{Var}(\hat{y}_0)}_{\text{estimation noise}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

For Ridge regression:

**As $\lambda \to 0$**: Ridge approaches OLS. Bias approaches zero (OLS is unbiased), but variance is large when features are correlated or $n$ is small.

**As $\lambda \to \infty$**: $\hat{\beta}_{\text{Ridge}} \to 0$. The model predicts the training mean for every input. Bias is $\|X\beta^*\|_2^2 / n$ (large), but variance approaches zero.

**Optimal $\lambda$**: There exists a $\lambda^*$ that minimises test MSE by trading off bias and variance. This is found empirically via cross-validation.

**Why the trade-off can favour Ridge over OLS**: Even though OLS is unbiased, if its variance is large (causing the squared error to vary wildly across samples), the expected MSE may exceed that of a biased-but-low-variance Ridge estimator.

**Practical guidance:**
- Start with a logarithmic grid: $\lambda \in \{10^{-4}, 10^{-3}, \ldots, 10^3\}$
- Use $k$-fold cross-validation to estimate validation MSE at each $\lambda$
- Select the $\lambda$ with minimum validation MSE (or the largest $\lambda$ within one standard error of the minimum, following the "one-standard-error rule" for parsimony)

---

### Q6. What are the limitations of linear regression, and how can you extend it to handle non-linear relationships?

**Answer:**

**Limitations:**
1. Linear assumption: $\mathbb{E}[y \mid x]$ must be linear in $x$. Fails for quadratic, exponential, or interaction effects.
2. Homoscedasticity: assumes constant error variance; fails for many real datasets where variance scales with the mean.
3. Gaussian errors: OLS is optimal for Gaussian errors; for binary outcomes or count data, it is the wrong model.
4. Sensitivity to outliers: squared loss weights large residuals heavily; a single extreme observation can dominate the solution.

**Extensions:**

| Technique | Mechanism | When to use |
|---|---|---|
| Polynomial features | Augment $x$ with $x^2, x^3, x_i x_j$ | Known low-degree polynomial relationship |
| Basis function expansion | Map $x$ to $\phi(x)$ using RBF, Fourier, or spline bases | Smooth non-linearities |
| Generalised Linear Models (GLMs) | Change the link function and noise distribution | Binary outcomes (logit), counts (Poisson) |
| Robust regression (Huber loss) | Replace squared loss with Huber loss | Outliers are present |
| Kernel ridge regression | Implicitly work in infinite-dimensional feature space | Non-linear regression with kernel trick |

**Polynomial feature caution**: Adding polynomial features of degree $p$ increases the feature count from $d$ to $O(d^p)$, causing an explosion in the number of parameters and the risk of overfitting. Regularisation (Ridge, Lasso) becomes critical.

---

## Tier 3 -- Advanced

### Q7. Derive the Ridge regression solution using singular value decomposition and explain what happens to each singular value component.

**Answer:**

Let the SVD of the design matrix be $X = U \Sigma V^\top$, where:
- $U \in \mathbb{R}^{n \times d}$, orthonormal columns (left singular vectors)
- $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_d)$, singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_d \geq 0$
- $V \in \mathbb{R}^{d \times d}$, orthogonal (right singular vectors)

**OLS fitted values** using the pseudo-inverse:

$$\hat{y}_{\text{OLS}} = X(X^\top X)^{-1}X^\top y = U U^\top y = \sum_{j=1}^{d} u_j u_j^\top y$$

Each component projects $y$ onto $u_j$ without shrinkage.

**Ridge fitted values:**

$$X^\top X + \lambda I = V(\Sigma^2 + \lambda I)V^\top$$

$$(X^\top X + \lambda I)^{-1} = V \text{diag}\!\left(\frac{1}{\sigma_j^2 + \lambda}\right) V^\top$$

$$\hat{y}_{\text{Ridge}} = X(X^\top X + \lambda I)^{-1}X^\top y = U \text{diag}\!\left(\frac{\sigma_j^2}{\sigma_j^2 + \lambda}\right) U^\top y = \sum_{j=1}^{d} \frac{\sigma_j^2}{\sigma_j^2 + \lambda}\, u_j u_j^\top y$$

**Interpretation of shrinkage factors** $d_j = \frac{\sigma_j^2}{\sigma_j^2 + \lambda}$:

- When $\sigma_j \gg \sqrt{\lambda}$: $d_j \approx 1$ -- principal component $j$ is kept nearly unchanged
- When $\sigma_j \ll \sqrt{\lambda}$: $d_j \approx 0$ -- principal component $j$ is heavily shrunk (nearly discarded)
- Ridge shrinks directions of small variance (small $\sigma_j$) more aggressively

This is precisely the bias-variance trade-off: directions of low data variance carry mostly noise, so shrinking them reduces variance without much bias cost. Directions of high data variance carry signal, so they are preserved.

**Effective degrees of freedom** of Ridge regression:

$$\text{df}(\lambda) = \sum_{j=1}^{d} \frac{\sigma_j^2}{\sigma_j^2 + \lambda}$$

This interpolates between $d$ (OLS, $\lambda = 0$) and $0$ (null model, $\lambda \to \infty$).

---

### Q8. Compare Lasso, Ridge, and elastic net in the regime where the number of features $d$ is much larger than the number of samples $n$ (the "$p \gg n$" problem). What are the guarantees and failure modes of each?

**Answer:**

**Ridge ($p \gg n$):**

Ridge always has a unique solution since $X^\top X + \lambda I$ is always invertible. However, Ridge cannot perform feature selection -- it keeps all $d$ features with non-zero coefficients. In genomics (e.g., $n = 100$ patients, $d = 20{,}000$ genes), Ridge produces a model that uses all 20,000 genes, which is uninterpretable and potentially overfit even though regularised.

Ridge is consistent and achieves near-optimal prediction error under appropriate conditions, but it is not **model-selection consistent** -- it does not recover the true sparse support.

**Lasso ($p \gg n$):**

Lasso can select at most $n$ non-zero features (because the solution is a vertex of a polytope in $\mathbb{R}^d$ and there are only $n$ constraints from the data). When the true model has $s \ll n$ non-zero features, Lasso can, under certain conditions, recover exactly those $s$ features.

**Restricted Isometry Property (RIP) / Incoherence condition**: Lasso achieves exact support recovery when the design matrix satisfies the restricted eigenvalue condition and the signal-to-noise ratio is sufficient. Critically, features must not be too strongly correlated.

**Failure mode**: When relevant features are highly correlated, Lasso arbitrarily picks one from a correlated group and zeros out the rest. This is unstable -- small changes in the data flip which feature is selected.

**Elastic net ($p \gg n$):**

Elastic net overcomes the Lasso limitation of selecting at most $n$ features. It can select more than $n$ features, and the $\ell_2$ component encourages grouping of correlated features. It achieves a middle ground: sparsity without the instability of Lasso under correlation.

**Practical recommendation for $p \gg n$:**
- Strong true sparsity, uncorrelated features: Lasso
- Strong true sparsity, correlated feature groups: elastic net
- Dense true signal or prediction-only task: Ridge
- Unknown structure: cross-validate all three

---

## Implementation Reference

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Always scale features before regularised regression
# Regularisation penalises raw coefficient magnitude, which is scale-dependent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# OLS via normal equations (NumPy)
beta_ols = np.linalg.lstsq(X_train_scaled, y_train, rcond=None)[0]

# Ridge: closed-form solution (sklearn uses SVD internally)
ridge = Ridge(alpha=1.0)  # alpha is lambda in sklearn notation
ridge.fit(X_train_scaled, y_train)

# Lasso: coordinate descent
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
print(f"Non-zero Lasso features: {np.sum(lasso.coef_ != 0)}")

# Elastic net: l1_ratio=1 is Lasso, l1_ratio=0 is Ridge
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train_scaled, y_train)

# Select lambda via cross-validation
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

**Critical note**: Always standardise features (zero mean, unit variance) before applying regularisation. The penalty $\lambda \|\beta\|_2^2$ treats all coefficients equally, so features on different scales would be penalised unequally. Standardisation ensures the regularisation is invariant to input scale. Do NOT standardise the target $y$ in regression unless you want to interpret coefficients in standardised units.

---

## Quick Reference Quiz

**Q: The OLS estimator is BLUE (Best Linear Unbiased Estimator). Does this mean OLS always has lower test MSE than Ridge?**

A) Yes, because BLUE means minimum error  
B) No, because BLUE only concerns bias; Ridge reduces variance, which can reduce test MSE  
C) Yes, because Ridge is biased and bias always increases MSE  
D) No, because Ridge is more computationally efficient  

**Answer: B.** BLUE means minimum variance among linear *unbiased* estimators. Ridge is biased, so it is outside the class of estimators that the Gauss-Markov theorem considers. Ridge can achieve lower test MSE by accepting bias in exchange for a larger reduction in variance.

---

**Q: When fitting Lasso with $\lambda = 100$, you find that all coefficients are exactly zero. What should you do?**

A) Increase $\lambda$ further  
B) Decrease $\lambda$  
C) Switch to Ridge  
D) The model is optimal -- zero coefficients mean no features are relevant  

**Answer: B.** All-zero coefficients means $\lambda$ is too large and you have shrunk everything to zero. Decrease $\lambda$ to allow the model to use features. Use cross-validation to find the largest $\lambda$ that still achieves good validation performance.
