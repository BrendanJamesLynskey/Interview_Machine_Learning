# SVM and Kernels

## Prerequisites
- Linear algebra: dot products, norms, hyperplanes
- Calculus: Lagrange multipliers, constrained optimisation
- Basic familiarity with duality in optimisation
- Understanding of overfitting and the bias-variance trade-off

---

## Concept Reference

### The Maximum Margin Classifier

For a binary classification problem with $y_i \in \{-1, +1\}$, a linear classifier predicts $\hat{y} = \text{sign}(w^\top x + b)$. The decision boundary is the hyperplane $w^\top x + b = 0$.

Many hyperplanes can separate linearly separable data. The SVM chooses the one that **maximises the geometric margin** -- the distance between the hyperplane and the nearest training examples.

**Geometric margin**: the signed distance from point $x_i$ to the hyperplane is:

$$\gamma_i = y_i \frac{w^\top x_i + b}{\|w\|}$$

The margin of the classifier is $\gamma = \min_i \gamma_i$.

Without loss of generality, rescale $w$ so that $\min_i y_i(w^\top x_i + b) = 1$ (this is the **functional margin normalisation**). Under this convention:
- Support vectors lie on the planes $w^\top x + b = \pm 1$
- The geometric margin is $\frac{2}{\|w\|}$

Maximising the margin is equivalent to minimising $\|w\|^2$, giving the **hard-margin SVM primal**:

$$\min_{w, b}\; \frac{1}{2}\|w\|^2 \quad \text{s.t.}\; y_i(w^\top x_i + b) \geq 1, \quad \forall i$$

### Support Vectors

Support vectors are the training examples that lie exactly on the margin boundaries $w^\top x + b = \pm 1$. These are the points for which the constraint is active (equality holds). All other examples are further from the hyperplane and have no influence on the solution -- the SVM decision boundary depends only on support vectors.

This is a key property: SVMs are controlled by a small subset of the training data, making them memory-efficient and robust to non-support-vector examples.

### The Dual Problem and Lagrange Multipliers

The Lagrangian of the primal:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i \left[y_i(w^\top x_i + b) - 1\right], \quad \alpha_i \geq 0$$

Setting partial derivatives to zero:

$$\frac{\partial \mathcal{L}}{\partial w} = 0 \;\Rightarrow\; w = \sum_i \alpha_i y_i x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0$$

Substituting back into $\mathcal{L}$ gives the **dual problem** (maximise):

$$\mathcal{D}(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^\top x_j$$

subject to $\alpha_i \geq 0$ and $\sum_i \alpha_i y_i = 0$.

**KKT conditions** (necessary and sufficient for the global optimum):
- $\alpha_i \geq 0$
- $y_i(w^\top x_i + b) - 1 \geq 0$
- $\alpha_i [y_i(w^\top x_i + b) - 1] = 0$ (complementary slackness)

The last condition implies: $\alpha_i > 0$ only for support vectors (constraints active); all non-support-vector examples have $\alpha_i = 0$.

**Critical insight**: the dual objective and the prediction function $w^\top x = \sum_i \alpha_i y_i x_i^\top x$ depend on inputs only through their **dot products** $x_i^\top x_j$. This enables the kernel trick.

### The Kernel Trick

To handle non-linearly separable data, map inputs to a higher-dimensional feature space $\phi(x)$ and find a linear separator there. The dual objective becomes:

$$\mathcal{D}(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j \phi(x_i)^\top \phi(x_j)$$

The key observation: we never need $\phi(x)$ explicitly -- only the dot product $\phi(x_i)^\top \phi(x_j)$. Define the **kernel function**:

$$K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$$

The dual becomes:

$$\mathcal{D}(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

Prediction: $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$

**Mercer's theorem**: a symmetric function $K(x, z)$ is a valid kernel if and only if the kernel matrix $K_{ij} = K(x_i, x_j)$ is positive semi-definite for all datasets. This is equivalent to the existence of a feature map $\phi$ such that $K(x, z) = \phi(x)^\top \phi(z)$.

### Common Kernels

**Linear kernel**: $K(x, z) = x^\top z$

Equivalent to the original feature space. Fast, interpretable. Use when data is linearly separable or $d$ is large and features are already informative.

**Polynomial kernel**: $K(x, z) = (\gamma x^\top z + r)^d$

Corresponds to a feature map including all monomials up to degree $d$. Controls the degree of polynomial interactions. Hyperparameters: degree $d$, $\gamma$, $r$.

**Radial Basis Function (RBF) / Gaussian kernel**: $K(x, z) = \exp\!\left(-\gamma \|x - z\|^2\right)$

The corresponding feature map $\phi$ has **infinite dimensions** (it spans the full space of Gaussian-smoothed functions). Two points close in input space have kernel value near 1; distant points have kernel value near 0. The hyperparameter $\gamma = 1/(2\sigma^2)$ controls the width:
- Large $\gamma$ (narrow kernel): each support vector influences a small region -- high complexity, prone to overfitting
- Small $\gamma$ (wide kernel): each support vector influences a large region -- smoother decision boundary

**Sigmoid kernel**: $K(x, z) = \tanh(\gamma x^\top z + r)$

Not always PSD (not a valid Mercer kernel for all parameter values). Used historically; generally outperformed by RBF.

### Soft-Margin SVM (C-SVM)

Real data is rarely linearly separable. The soft-margin SVM allows constraints to be violated using slack variables $\xi_i \geq 0$:

$$\min_{w, b, \xi}\; \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

$$\text{s.t.}\; y_i(w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i$$

The parameter $C$ controls the trade-off:
- Large $C$: allow few violations -- small margin, potentially overfits (approaches hard-margin with correct data)
- Small $C$: allow many violations -- large margin, but more training errors (underfits)

The dual of the soft-margin SVM is identical to the hard-margin dual, but the Lagrange multipliers are bounded: $0 \leq \alpha_i \leq C$.

**Hinge loss interpretation**: the soft-margin SVM objective is equivalent to minimising the regularised empirical risk with the hinge loss:

$$\mathcal{L} = \frac{1}{n}\sum_i \max(0, 1 - y_i(w^\top x_i + b)) + \frac{1}{2C}\|w\|^2$$

The hinge loss $\max(0, 1 - yf)$ is zero for correctly classified examples outside the margin and linear for margin violations. It is convex and upper bounds the 0-1 loss.

---

## Tier 1 -- Fundamentals

### Q1. What is the maximum margin hyperplane and why is maximising the margin a good objective?

**Answer:**

The maximum margin hyperplane is the linear decision boundary that maximises the distance to the nearest training examples on either side (the margin). For a linearly separable dataset, this boundary is unique.

**Why maximising margin is a good objective:**

1. **Generalisation theory (VC theory)**: the generalisation bound for margin classifiers depends on the ratio of the feature space radius to the margin. A larger margin corresponds to a smaller VC dimension and tighter generalisation bounds -- the classifier is less likely to overfit.

2. **Geometric robustness**: the maximum-margin classifier is as far as possible from both classes, so it can tolerate the largest perturbations in new examples without misclassifying them. This gives intuitive robustness to measurement noise.

3. **Uniqueness**: of all hyperplanes that correctly classify the training data (infinitely many), the maximum margin one is uniquely defined. This removes the ambiguity of which separator to choose.

4. **Practical performance**: SVMs historically achieved state-of-the-art performance on many tasks precisely because the margin principle implicitly regularises the solution -- it is related to L2 regularisation on the weight vector.

**Common mistake**: confusing the geometric margin $2/\|w\|$ with the functional margin. The geometric margin is invariant to scaling of $(w, b)$ and is the true distance measure. Maximising $2/\|w\|$ is equivalent to minimising $\|w\|^2$.

---

### Q2. What are support vectors? How many support vectors does a typical SVM have, and why?

**Answer:**

Support vectors are the training examples that satisfy the constraint with equality: $y_i(w^\top x_i + b) = 1$. They lie exactly on the margin boundaries. All other training examples satisfy $y_i(w^\top x_i + b) > 1$ (strictly outside the margin).

From the KKT complementary slackness condition, $\alpha_i[y_i(w^\top x_i + b) - 1] = 0$, so $\alpha_i > 0$ only for support vectors. The weight vector is $w = \sum_i \alpha_i y_i x_i$, which is a linear combination only of support vectors.

**Typical count**: in $d$ dimensions, the hard-margin SVM always has at least 1 and at most $d + 1$ support vectors for a linearly separable problem (by the geometry of touching hyperplanes). In practice:
- Soft-margin SVM with RBF kernel: typically dozens to hundreds of support vectors, depending on $C$ and data complexity
- Small $C$ (large margin, many violations): more support vectors
- Large $C$ (small margin, few violations): fewer support vectors

**Computational implication**: inference time scales with the number of support vectors $|SV|$, since prediction requires computing $\sum_i \alpha_i y_i K(x_i, x)$ over all support vectors. For large datasets, this can be slow -- this is one reason kernel SVMs are less commonly used at scale compared to neural networks or gradient boosting.

---

### Q3. What does the regularisation parameter $C$ control in a soft-margin SVM, and how do you choose it?

**Answer:**

The soft-margin SVM primal minimises:

$$\frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$

$C$ controls the trade-off between maximising the margin (first term) and minimising constraint violations (second term).

**Effect of $C$:**

| $C$ value | Margin | Constraint violations | Complexity |
|---|---|---|---|
| Large $C$ (e.g., 100) | Small | Few allowed | High (can overfit) |
| Small $C$ (e.g., 0.01) | Large | Many allowed | Low (can underfit) |

**Choosing $C$:** use cross-validation over a logarithmic grid. A common starting range is $\{10^{-3}, 10^{-2}, \ldots, 10^3\}$. For RBF kernels, jointly tune $C$ and $\gamma$ using a 2D grid search or random search.

**Intuition**: if you believe the data is nearly linearly separable with a few outliers, use large $C$. If you expect significant overlap between classes, use small $C$ to get a more robust, wider-margin separator.

---

## Tier 2 -- Intermediate

### Q4. Explain the kernel trick. Specifically, show how the RBF kernel corresponds to an infinite-dimensional feature map.

**Answer:**

**The kernel trick** allows an SVM to implicitly operate in a very high-dimensional feature space $\phi(x)$ without ever computing $\phi(x)$ explicitly. The dual objective and prediction depend on inputs only through $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$.

**RBF kernel and infinite-dimensional feature map:**

$$K(x, z) = \exp\!\left(-\gamma \|x - z\|^2\right)$$

Expanding using the Taylor series of $e^{-\gamma\|x-z\|^2}$:

$$\|x - z\|^2 = \|x\|^2 - 2x^\top z + \|z\|^2$$

$$K(x, z) = e^{-\gamma\|x\|^2} e^{2\gamma x^\top z} e^{-\gamma\|z\|^2}$$

Expanding $e^{2\gamma x^\top z} = \sum_{n=0}^{\infty} \frac{(2\gamma x^\top z)^n}{n!}$ and using the multinomial theorem to expand $(x^\top z)^n$, one can show that the kernel equals the dot product of infinite-dimensional vectors $\phi(x)$ whose components include all monomials of $x$ weighted by Gaussian factors.

**Concrete example** (1D, $\gamma = 1$):

$$K(x, z) = e^{-x^2}e^{-z^2} \sum_{n=0}^{\infty} \frac{(2xz)^n}{n!} = \sum_{n=0}^{\infty} e^{-x^2}e^{-z^2} \frac{2^n x^n z^n}{n!}$$

$$= \sum_{n=0}^{\infty} \underbrace{\left(\frac{e^{-x^2} (\sqrt{2}x)^n}{\sqrt{n!}}\right)}_{\phi_n(x)} \cdot \underbrace{\left(\frac{e^{-z^2} (\sqrt{2}z)^n}{\sqrt{n!}}\right)}_{\phi_n(z)}$$

So $K(x, z) = \phi(x)^\top \phi(z)$ where $\phi = [\phi_0, \phi_1, \phi_2, \ldots]$ is an infinite-dimensional vector.

**Practical consequence**: the RBF SVM can represent arbitrarily complex decision boundaries while training in the original (low-dimensional) input space, only ever computing $n(n-1)/2$ kernel evaluations between training points. The implicit feature map of infinite dimension would be computationally infeasible to use explicitly.

---

### Q5. Derive the dual of the hard-margin SVM and explain the role of each Lagrange multiplier.

**Answer:**

**Primal:**

$$\min_{w, b}\; \frac{1}{2}\|w\|^2 \quad \text{s.t.}\; y_i(w^\top x_i + b) \geq 1, \quad i = 1, \ldots, n$$

**Lagrangian** (Lagrange multipliers $\alpha_i \geq 0$):

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i \left[y_i(w^\top x_i + b) - 1\right]$$

**Stationarity conditions** (set partial derivatives to zero):

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \;\Rightarrow\; w^* = \sum_i \alpha_i y_i x_i \quad (*)$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0 \quad (**)$$

**Substituting (*) into the Lagrangian:**

$$\mathcal{L} = \frac{1}{2}\left(\sum_i \alpha_i y_i x_i\right)^\top\!\!\left(\sum_j \alpha_j y_j x_j\right) - \sum_i \alpha_i y_i x_i^\top \sum_j \alpha_j y_j x_j - b\underbrace{\sum_i \alpha_i y_i}_{=0} + \sum_i \alpha_i$$

$$= \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j x_i^\top x_j$$

**Dual problem** (maximise with respect to $\alpha$):

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{s.t.}\; \alpha_i \geq 0,\; \sum_i \alpha_i y_i = 0$$

**Role of each $\alpha_i$:**

- $\alpha_i = 0$: example $i$ is not a support vector; it lies strictly outside the margin and has no influence on the decision boundary
- $\alpha_i > 0$: example $i$ is a support vector lying on the margin boundary $y_i(w^\top x_i + b) = 1$; it directly contributes to defining $w^*$
- The constraint $\sum_i \alpha_i y_i = 0$ ensures the decision boundary is balanced between the two classes

**Strong duality** holds by Slater's condition (the feasible set has an interior when the data is separable), so the dual solution equals the primal solution -- solving the dual gives the exact primal solution.

---

### Q6. Compare SVM to logistic regression. When does each approach dominate?

**Answer:**

Both are linear classifiers that minimise a regularised empirical risk. The key difference is the loss function:

| Property | SVM (hinge loss) | Logistic Regression (log loss) |
|---|---|---|
| Loss for correct predictions outside margin | 0 (sparse) | Small positive (dense gradient) |
| Loss for misclassifications | Linear growth | Logarithmic growth (infinite for confident errors) |
| Decision function | Margin-maximising hyperplane | Conditional probability $P(y \mid x)$ |
| Outputs calibrated probabilities | Not directly | Yes (well-calibrated) |
| Sparsity in dual | Sparse (only support vectors) | Dense ($\alpha_i > 0$ for all examples) |
| Kernel extension | Natural and efficient | Less natural (requires kernel logistic regression) |
| Computation at scale | Quadratic in $n$ (naive) | Linear in $n$ (stochastic gradient descent) |

**When SVM dominates:**
1. Small datasets with clear margin (e.g., text classification with TF-IDF features, $n < 100{,}000$)
2. High-dimensional feature spaces where kernel choice captures known structure (string kernels for text, graph kernels for molecules)
3. When outliers in the label space are a concern: hinge loss is robust to points outside the margin (zero gradient), while log loss continues to be affected by all examples

**When logistic regression dominates:**
1. When calibrated probabilities are needed downstream (decision theory, risk scoring)
2. Large-scale problems: logistic regression scales to millions of examples with SGD; kernel SVM is $O(n^2)$ in memory and $O(n^3)$ in training
3. When features are linearly informative and the interpretability of log-odds coefficients is valuable
4. In neural networks: logistic regression is the output layer for binary classification -- its clean gradient $(p-y)$ facilitates training

---

## Tier 3 -- Advanced

### Q7. What is the VC dimension of an SVM with an RBF kernel, and what does this mean for the bias-variance trade-off?

**Answer:**

**VC dimension background**: the VC (Vapnik-Chervonenkis) dimension of a classifier is the maximum number of points that can be **shattered** (correctly labelled in all $2^m$ ways) by the classifier class. A higher VC dimension means higher capacity (can fit more complex patterns) but also requires more data to generalise.

**Linear SVM in $d$ dimensions**: VC dimension is $d + 1$. The margin constraints effectively reduce the functional VC dimension below $d + 1$; for large margins, the effective capacity is controlled by $R^2/\gamma^2$ where $R$ is the radius of the data and $\gamma$ is the margin.

**SVM with RBF kernel**: the VC dimension is **infinite**. The RBF kernel corresponds to an infinite-dimensional feature space, and in principle an RBF SVM can shatter any finite set of points (for small enough $\gamma$).

**What this means in practice:**

The generalisation bound for margin classifiers (Vapnik, 1998) is controlled not just by VC dimension but also by the **fat-shattering dimension** tied to the margin:

$$P(\text{test error}) \leq P(\text{train margin violations}) + O\!\left(\sqrt{\frac{R^2 \|w\|^2}{n\, \gamma^2}}\right)$$

For a fixed training set, a large-$C$ RBF SVM has small $\|w\|$ relative to the data spread (small $\|w\|$ means large margin), bounding the effective capacity. For small $C$ (large $\gamma$ in the kernel), each support vector has narrow influence, allowing the classifier to shatter training points -- but the corresponding generalisation bound deteriorates.

**Practical implication:**

The RBF SVM with properly tuned $(C, \gamma)$ achieves the bias-variance sweet spot:
- Too large $\gamma$: narrow kernels, complex boundary, overfits (high variance)
- Too small $\gamma$: wide kernels, smooth boundary, underfits (high bias)
- Too large $C$: hard margin, few support vectors, overfits
- Too small $C$: wide margin, many violations, underfits

Cross-validation on a 2D grid of $(C, \gamma)$ is the standard approach. A useful heuristic is to set $\gamma = 1/(d\, \sigma_X^2)$ as an initial estimate and then tune $C$.

---

### Q8. Describe three scenarios where SVMs with non-trivial kernels significantly outperform neural networks, and explain why.

**Answer:**

**Scenario 1: Structured data with known similarity measures -- computational biology**

In protein remote homology detection, the input is an amino acid sequence and the task is to determine whether two proteins share a common evolutionary ancestor. A **string kernel** (e.g., the spectrum kernel, which counts $k$-mer frequencies) captures the biological notion of sequence similarity without needing to define a fixed-length feature vector.

Why SVMs win: the kernel encodes domain knowledge about what constitutes "similarity" for sequences. Designing a neural architecture to learn this from scratch requires much more data. SVMs with the spectrum kernel achieve state-of-the-art results with hundreds of training examples; a competitive neural network would require orders of magnitude more data.

**Scenario 2: Very small datasets -- materials science or drug discovery**

Predicting molecular properties from chemical descriptors. With $n = 50$--$200$ labelled molecules and $d = 100$--$1{,}000$ descriptor features:
- A neural network with enough capacity to model non-linearities has far too many parameters to avoid overfitting
- An RBF SVM has an implicit infinite-dimensional feature space but is regularised by the margin constraint, achieving good generalisation with few examples

The SVM's margin principle provides strong regularisation that is geometrically motivated and requires fewer hyperparameter choices than neural network regularisation (depth, width, dropout, batch norm, learning rate, etc.).

**Scenario 3: Provable worst-case guarantees -- adversarial robustness**

Standard neural networks are fragile to adversarial perturbations -- tiny changes in inputs cause large changes in output. The SVM margin guarantee provides a certificate: any input within distance $1/(2\|w\|)$ of the decision boundary (in feature space) will not cross it.

For problems where certified robustness to bounded perturbations is required (medical diagnosis, safety-critical systems), a max-margin classifier provides interpretable, provable safety guarantees. Neural networks require specialised architectures and training methods (certified defences) to achieve comparable guarantees, and these typically underperform SVMs on small datasets.

---

## Implementation Reference

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Always standardise: SVM is sensitive to feature scale
# The RBF kernel computes ||x - z||^2 directly
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Linear SVM (best for high-dimensional sparse data like TF-IDF)
from sklearn.svm import LinearSVC
linear_svm = LinearSVC(C=1.0, max_iter=10000)
linear_svm.fit(X_train_s, y_train)

# RBF kernel SVM -- tune C and gamma
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
# gamma='scale' uses 1 / (n_features * X.var())

# Grid search over (C, gamma)
param_grid = {
    'C':     [0.01, 0.1, 1, 10, 100],
    'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
}
grid_search = GridSearchCV(
    SVC(kernel='rbf', probability=True),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_s, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_

# Inspect support vectors
print(f"Number of support vectors: {best_svm.n_support_}")
print(f"Support vector indices: {best_svm.support_}")

# Custom kernel: precomputed kernel matrix
# Useful when domain-specific similarity is available
K_train = compute_custom_kernel(X_train, X_train)   # n x n matrix
K_test  = compute_custom_kernel(X_test,  X_train)   # m x n matrix

svm_custom = SVC(kernel='precomputed', C=1.0)
svm_custom.fit(K_train, y_train)
preds = svm_custom.predict(K_test)
```

---

## Quick Reference Quiz

**Q: An SVM is trained with a very large value of $C$. Which of the following is most likely?**

A) The margin will be wide and many training points will be misclassified  
B) The margin will be narrow and training accuracy will be near 100%  
C) The model will be identical to logistic regression  
D) The number of support vectors will increase  

**Answer: B.** Large $C$ penalises constraint violations heavily, so the optimiser minimises violations at the expense of a narrow margin. This tends to produce high training accuracy (near-zero slack) but can overfit, especially with noisy data. The number of support vectors typically decreases with large $C$ because fewer points need to be on or inside the margin.

---

**Q: The RBF kernel $K(x, z) = \exp(-\gamma\|x-z\|^2)$ satisfies which property?**

A) It always produces a kernel matrix with exactly $d$ non-zero eigenvalues  
B) The corresponding feature map $\phi$ is finite-dimensional  
C) The kernel matrix is positive semi-definite for any finite dataset  
D) It is equivalent to the polynomial kernel of degree 2  

**Answer: C.** The RBF kernel satisfies Mercer's condition -- the Gram matrix $K_{ij} = K(x_i, x_j)$ is always PSD for any finite set of points. This means it corresponds to a valid inner product in some (infinite-dimensional) feature space and can be used in any kernel method. The feature map is infinite-dimensional (not finite), and it is not equivalent to any polynomial kernel.
