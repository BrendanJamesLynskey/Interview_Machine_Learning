# Problem 02: Regularisation Derivation

**Difficulty**: Intermediate to Advanced  
**Topics covered**: L1 and L2 regularisation, proximal operators, coordinate descent, subgradients, Bayesian interpretation, effect on weight distributions

---

## Background

Regularisation constrains model weights to reduce overfitting. This problem derives rigorously why L1 and L2 penalties have fundamentally different effects on weights: L2 shrinks all weights proportionally, while L1 drives weights exactly to zero. Understanding these derivations is essential for choosing between Lasso, Ridge, and elastic net in practice.

---

## Part A: The Geometry of Regularisation Constraints

### Equivalent formulations

The regularised regression objectives can be equivalently written as **constrained optimisation** problems:

**Ridge (L2 constraint form):**

$$\min_\beta \|y - X\beta\|_2^2 \quad \text{s.t.}\quad \|\beta\|_2^2 \leq t$$

**Lasso (L1 constraint form):**

$$\min_\beta \|y - X\beta\|_2^2 \quad \text{s.t.}\quad \|\beta\|_1 \leq t$$

By the Lagrangian duality theorem, each value of the constraint budget $t$ corresponds to a unique value of the penalty $\lambda$ in the unconstrained penalised form, and vice versa.

### Geometric argument for L1 sparsity

Consider $\beta \in \mathbb{R}^2$ for illustration.

**L2 constraint region** ($\|\beta\|_2^2 \leq t$): a circle centred at the origin. Its boundary is smooth everywhere -- no corners.

**L1 constraint region** ($\|\beta\|_1 \leq t$): a diamond (square rotated 45°) with vertices at $(\pm t, 0)$ and $(0, \pm t)$. Its boundary has corners on the coordinate axes.

The OLS objective $\|y - X\beta\|_2^2$ has level sets that are ellipses (or ellipsoids in higher dimensions), centred at $\hat{\beta}_{\text{OLS}}$.

**The Lasso solution**: expand the OLS ellipsoid from the unconstrained minimum until it first touches the diamond. Because the diamond has corners aligned with the coordinate axes, the first contact point is frequently at a corner, where one or more $\beta_j = 0$.

**The Ridge solution**: expand the OLS ellipsoid until it first touches the circle. The circle is smooth -- the contact point is generically not on a coordinate axis, so both components are non-zero.

**In $d$ dimensions**: the L1 ball has $2^d$ corners and $2d$ edges (face intersections), many of which lie on coordinate hyperplanes. As $d$ increases, the fraction of the L1 ball's surface that lies on coordinate hyperplanes grows, making sparsity increasingly likely.

---

## Part B: Subgradient Derivation -- Why L1 Sets Weights to Zero

For smooth objectives, we set the gradient to zero. For non-smooth objectives (L1 contains $|\beta_j|$ which is non-differentiable at $\beta_j = 0$), we use **subgradients**.

### Subgradient of the absolute value

The subgradient of $g(\beta_j) = |\beta_j|$ is the set:

$$\partial g(\beta_j) = \begin{cases} \{-1\} & \text{if } \beta_j < 0 \\ [-1, 1] & \text{if } \beta_j = 0 \\ \{+1\} & \text{if } \beta_j > 0\end{cases}$$

A subgradient $s \in \partial g(\beta_j)$ satisfies $g(z) \geq g(\beta_j) + s(z - \beta_j)$ for all $z$ -- it is a supporting hyperplane at $\beta_j$.

### Optimality condition for Lasso

The Lasso objective for a single coordinate $j$ (holding others fixed):

$$\mathcal{L}(\beta_j) = \frac{1}{2}r_j(\beta_j)^2 + \lambda |\beta_j|$$

where $r_j = \|y - X_{-j}\beta_{-j} - X_j\beta_j\|^2$ contributes a term involving $\beta_j$.

Let $\rho_j = X_j^\top (y - X_{-j}\beta_{-j})$ be the partial correlation between feature $j$ and the current residual (holding all other weights fixed). Assuming $X$ has standardised columns ($\|X_j\|^2 = 1$), the 1D Lasso subproblem becomes:

$$\min_{\beta_j} \frac{1}{2}(\rho_j - \beta_j)^2 + \lambda|\beta_j|$$

**Optimality condition (0 must be a subgradient of the full objective at the minimum):**

$$0 \in -(\rho_j - \beta_j) + \lambda \partial |\beta_j|$$

**Case 1: $\rho_j > \lambda$ (positive partial correlation, large enough to overcome penalty)**

$$\beta_j = \rho_j - \lambda > 0$$

**Case 2: $\rho_j < -\lambda$ (negative partial correlation, large enough to overcome penalty)**

$$\beta_j = \rho_j + \lambda < 0$$

**Case 3: $|\rho_j| \leq \lambda$ (partial correlation too small to overcome penalty)**

$$0 = -\rho_j + \lambda s, \quad s \in [-1, 1] \;\Rightarrow\; \beta_j = 0 \text{ (with } s = \rho_j/\lambda \in [-1, 1])$$

This gives the **soft-thresholding operator** (proximal operator of L1):

$$\hat{\beta}_j = \mathcal{S}_\lambda(\rho_j) = \text{sign}(\rho_j)\max(|\rho_j| - \lambda, 0)$$

```
|beta_j|
    /
   /
  /
 +-------0------+        rho_j axis
        |       
        lambda           <- threshold
```

**Interpretation**: if the partial correlation $|\rho_j|$ is smaller than $\lambda$, the feature does not provide enough signal to overcome the penalty and is exactly zeroed out. If it exceeds $\lambda$, it is included but shrunk by $\lambda$ towards zero.

### Contrast with L2 (Ridge)

The Ridge subproblem:

$$\min_{\beta_j} \frac{1}{2}(\rho_j - \beta_j)^2 + \frac{\lambda}{2}\beta_j^2$$

The gradient (smooth everywhere):

$$0 = -(\rho_j - \beta_j) + \lambda \beta_j \;\Rightarrow\; \hat{\beta}_j = \frac{\rho_j}{1 + \lambda}$$

This is proportional shrinkage: the weight is always $\rho_j/(1 + \lambda)$. As $\lambda \to \infty$, $\hat{\beta}_j \to 0$ but never exactly zero for any finite $\lambda$.

**Visualisation of shrinkage operators:**

```python
import numpy as np
import matplotlib.pyplot as plt

rho = np.linspace(-3, 3, 300)
lam = 1.0

# Soft threshold (Lasso / L1)
lasso_shrink = np.sign(rho) * np.maximum(np.abs(rho) - lam, 0)

# Proportional shrink (Ridge / L2)
ridge_shrink = rho / (1 + lam)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rho, rho,           '--', color='gray',   label='No shrinkage (OLS)', linewidth=1)
ax.plot(rho, lasso_shrink,  '-',  color='#d62728', label=f'Lasso (lambda={lam})', linewidth=2)
ax.plot(rho, ridge_shrink,  '-',  color='#1f77b4', label=f'Ridge (lambda={lam})', linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Partial correlation rho_j')
ax.set_ylabel('Estimated weight beta_j')
ax.set_title('Shrinkage operators: Lasso (soft-threshold) vs. Ridge (proportional)')
ax.legend()
```

---

## Part C: Coordinate Descent for Lasso

The soft-thresholding operator derived above is the basis for coordinate descent optimisation of Lasso.

**Algorithm:**

```python
def lasso_coordinate_descent(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Coordinate descent for Lasso regression.
    Assumes X has been standardised (columns have unit norm).
    """
    n, d = X.shape
    beta  = np.zeros(d)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(d):
            # Partial residual (residual when beta_j is removed)
            residual = y - X @ beta + X[:, j] * beta[j]
            
            # Partial correlation
            rho_j = X[:, j] @ residual / n
            
            # Soft-thresholding update
            beta[j] = np.sign(rho_j) * max(abs(rho_j) - lam, 0)
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break
    
    return beta

# Test on synthetic data
np.random.seed(42)
n, d = 100, 20
true_beta = np.zeros(d)
true_beta[:5] = [2.0, -1.5, 3.0, -2.5, 1.0]   # only 5 non-zero features

X = np.random.randn(n, d)
X /= np.linalg.norm(X, axis=0)   # standardise columns
y = X @ true_beta + 0.1 * np.random.randn(n)

lam = 0.05
beta_est = lasso_coordinate_descent(X, y, lam)
print(f"Estimated non-zero features: {np.where(beta_est != 0)[0].tolist()}")
print(f"True non-zero features:      {np.where(true_beta != 0)[0].tolist()}")
```

**Why coordinate descent works for Lasso**: the subproblem for each coordinate has a closed-form solution (soft thresholding), so each step is $O(n)$ and the full pass over all coordinates is $O(nd)$. The coordinate descent converges because the Lasso objective is separable (if $X$ is orthonormal) and is convex.

---

## Part D: Effect on Weight Distribution -- Bayesian Interpretation

Both L1 and L2 regularisation can be derived as MAP estimation with specific priors. The prior determines the shape of the weight distribution.

### L2 $\Leftrightarrow$ Gaussian prior

Prior: $P(\beta_j) = \mathcal{N}(0, 1/\lambda)$

Log-prior: $\log P(\beta_j) = -\frac{\lambda}{2}\beta_j^2 + \text{const}$

Negative log-posterior: $\|y - X\beta\|^2 + \lambda\|\beta\|_2^2$

The Gaussian prior has a smooth, rounded peak at zero. It prefers small weights but does not strongly prefer exactly-zero weights. The posterior mode (MAP) is Ridge regression.

### L1 $\Leftrightarrow$ Laplace prior

Prior: $P(\beta_j) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)$

Log-prior: $\log P(\beta_j) = -\lambda|\beta_j| + \text{const}$

Negative log-posterior: $\|y - X\beta\|^2 + \lambda\|\beta\|_1$

The Laplace distribution has a **sharp peak** at zero and heavier tails than the Gaussian. It strongly favours weight values close to or exactly zero, while still allowing large weights for informative features.

```python
import scipy.stats as stats

beta_values = np.linspace(-4, 4, 400)
lam = 1.0

# Gaussian prior (Ridge)
gaussian_prior = stats.norm.pdf(beta_values, loc=0, scale=1/np.sqrt(lam))

# Laplace prior (Lasso)
laplace_prior = stats.laplace.pdf(beta_values, loc=0, scale=1/lam)

# Key difference: Laplace has sharper peak at 0 and heavier tails
# The sharp peak is what causes sparsity in MAP estimation
```

### Numerical comparison of weight distributions

For a dataset with 100 features but only 10 truly relevant, fit Ridge and Lasso with tuned regularisation and inspect the weight distributions:

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardise features (critical for fair comparison)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_s, y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(100), ridge.coef_, color='steelblue', alpha=0.7)
axes[0].set_title(f'Ridge: {np.sum(ridge.coef_ != 0)} non-zero weights')
axes[0].set_xlabel('Feature index')
axes[0].set_ylabel('Weight value')

axes[1].bar(range(100), lasso.coef_, color='coral', alpha=0.7)
axes[1].set_title(f'Lasso: {np.sum(lasso.coef_ != 0)} non-zero weights')
axes[1].set_xlabel('Feature index')
axes[1].set_ylabel('Weight value')
```

**Expected output**: Ridge produces 100 non-zero weights with most values small but non-zero. Lasso produces ~10 non-zero weights matching the true non-zero features, with the rest set to exactly zero.

---

## Part E: Elastic Net Derivation

The elastic net penalty is a convex combination of L1 and L2:

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda\alpha\|\beta\|_1 + \frac{\lambda(1-\alpha)}{2}\|\beta\|_2^2$$

The coordinate descent update for elastic net:

**Subproblem for coordinate $j$:**

$$\min_{\beta_j} \frac{1}{2}(\rho_j - \beta_j)^2 + \lambda\alpha|\beta_j| + \frac{\lambda(1-\alpha)}{2}\beta_j^2$$

The L2 term adds a smooth quadratic to the L1 subproblem. Differentiating the smooth part and applying the subgradient condition for the L1 part:

$$\hat{\beta}_j = \frac{\mathcal{S}_{\lambda\alpha}(\rho_j)}{1 + \lambda(1-\alpha)}$$

This is a **scaled soft-threshold**: first apply soft-thresholding with threshold $\lambda\alpha$ (L1 induces sparsity), then divide by $1 + \lambda(1-\alpha)$ (L2 shrinks the remaining non-zero weight).

The combined effect:
- L1 component: sets small weights exactly to zero (sparsity)
- L2 component: shrinks the remaining non-zero weights proportionally (stabilises grouped features)

This is why elastic net combines the benefits of both: it performs feature selection while keeping correlated features grouped together.

---

## Part F: Interview Questions and Model Answers

### Q1. A colleague suggests doubling all regularisation parameters will always improve generalisation. Is this correct?

**Answer:**

No. Regularisation follows the bias-variance trade-off:

- Increasing $\lambda$ from a very small value: initially reduces variance more than it increases bias, improving generalisation (test error decreases)
- Continuing to increase $\lambda$ beyond the optimal: bias grows faster than variance decreases, worsening generalisation (test error increases)
- At $\lambda \to \infty$: all weights go to zero (Ridge) or to zero (Lasso), producing a trivial model with high bias

Doubling $\lambda$ will improve generalisation only if the current $\lambda$ is below the optimal value. If it is at or above the optimal, doubling it will hurt performance.

**The correct approach**: select $\lambda$ via cross-validation. The optimal $\lambda$ balances the bias-variance trade-off for the specific dataset, model class, and problem.

---

### Q2. Derive what happens to the Lasso solution when two features are perfectly correlated.

**Answer:**

Let $X_1 = X_2$ (two identical feature columns) and the true model use only $X_1$ with weight $\beta_1^* = 1$, $\beta_2^* = 0$. The OLS problem has infinitely many solutions: $\beta_1 + \beta_2 = 1$ with arbitrary split.

**Lasso under perfect correlation:**

The Lasso objective with $X_1 = X_2$:

$$\|y - X_1(\beta_1 + \beta_2)\|_2^2 + \lambda(|\beta_1| + |\beta_2|)$$

Since $y \approx X_1 \cdot 1$, we need $\beta_1 + \beta_2 = 1$. Subject to $\beta_1 + \beta_2 = 1$, the penalty $|\beta_1| + |\beta_2|$ is minimised by setting one variable to $1$ and the other to $0$ (corner of the diamond). Which one is selected is arbitrary (depends on initialisation or tie-breaking).

**Problem**: Lasso arbitrarily selects one of the two correlated features. The selection is not stable -- small perturbations in the data flip which is selected. The result is not reproducible.

**Elastic net solution**: with the L2 component, the penalty for $\beta_1 + \beta_2 = 1$ becomes $\lambda\alpha(|\beta_1| + |\beta_2|) + \frac{\lambda(1-\alpha)}{2}(\beta_1^2 + \beta_2^2)$. This is minimised at $\beta_1 = \beta_2 = 0.5$ (the L2 term prefers equal split). Elastic net selects both correlated features with equal weight -- **grouped selection** -- which is more stable and interpretable.

---

### Q3. You have 500 features and 200 training examples. You must produce a model that is both accurate and interpretable (few features). Which regularisation would you choose and why?

**Answer:**

With $d = 500$ and $n = 200$ (a $p > n$ problem), OLS is undefined (singular $X^\top X$) and Ridge produces all 500 non-zero features (not interpretable). The choice is between Lasso and elastic net.

**Lasso** would be the first choice:
- Lasso produces a sparse model with $\leq n = 200$ non-zero features
- With true sparsity (only a small fraction of the 500 features truly relevant), Lasso can recover the support under the restricted eigenvalue condition
- Interpretable: the non-zero features are the selected variables

**But use elastic net if:**
- Features are correlated (genomics data, text TF-IDF): Lasso's arbitrary selection among correlated features is problematic for interpretability ("why is gene $A$ selected but not gene $B$ when they are highly co-expressed?")
- Need stable selection: elastic net's grouped selection is reproducible across different training samples

**Practical workflow:**
1. Standardise all features
2. Run Lasso with cross-validated $\lambda$ (use `LassoCV`)
3. Check: are many selected features highly correlated? If yes, switch to elastic net with cross-validated $\lambda$ and $\alpha$
4. Verify selected features on held-out data; compute confidence intervals via bootstrap to assess selection stability

---

## Summary of Key Results

| Regulariser | Update rule | Effect on weights | Bayesian prior |
|---|---|---|---|
| Ridge (L2) | $\hat{\beta}_j = \rho_j/(1 + \lambda)$ | Proportional shrinkage, never zero | Gaussian |
| Lasso (L1) | $\hat{\beta}_j = \mathcal{S}_\lambda(\rho_j)$ | Soft-threshold, exact zeros | Laplace |
| Elastic net | $\hat{\beta}_j = \mathcal{S}_{\lambda\alpha}(\rho_j)/(1 + \lambda(1-\alpha))$ | Threshold then shrink, grouped | Gaussian $\times$ Laplace |

**Soft-threshold operator**: $\mathcal{S}_\lambda(\rho) = \text{sign}(\rho)\max(|\rho| - \lambda, 0)$

The key insight: L1 sets weights to zero because its subgradient at zero spans $[-1, 1]$, allowing the optimality condition $0 \in \partial \mathcal{L}$ to be satisfied without moving $\beta_j$ away from zero. L2's gradient at zero is exactly zero and pushes the weight to $\rho_j/(1+\lambda)$ -- always non-zero for non-zero $\rho_j$.
