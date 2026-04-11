# Logistic Regression

## Prerequisites
- Linear regression and the idea of a linear decision boundary
- Probability fundamentals: Bernoulli distribution, likelihood
- Calculus: chain rule, partial derivatives
- Understanding of gradient descent

---

## Concept Reference

### Motivation: Why Not Use Linear Regression for Classification?

For a binary classification problem $y \in \{0, 1\}$, linear regression predicts $\hat{y} = x^\top \beta$, which can produce values outside $[0, 1]$ -- meaningless as a probability. Moreover, the squared loss is poorly matched to the structure of the problem: it penalises confident correct predictions unnecessarily.

The fix is to transform the linear output into a valid probability using the **sigmoid function**.

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Output range: $(0, 1)$ -- a valid probability
- $\sigma(0) = 0.5$
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (elegant and efficient to compute)
- Monotonically increasing: larger $z$ means higher predicted probability

The logistic regression model:

$$P(y = 1 \mid x) = \sigma(x^\top \beta) = \frac{1}{1 + e^{-x^\top \beta}}$$

**Log-odds (logit) interpretation:**

$$\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = x^\top \beta$$

The log-odds ratio is a linear function of $x$. A unit increase in feature $x_j$ changes the log-odds by $\beta_j$, which is a multiplicative change of $e^{\beta_j}$ in the odds. This is the standard coefficient interpretation.

### The Decision Boundary

The decision boundary is the set of inputs where the model is equally uncertain, i.e., $P(y=1 \mid x) = 0.5$:

$$\sigma(x^\top \beta) = 0.5 \Leftrightarrow x^\top \beta = 0$$

This is a **hyperplane** in input space. Logistic regression always produces a linear decision boundary. To model non-linear boundaries, one must either engineer non-linear features (e.g., polynomial) or use a more expressive model.

### Cross-Entropy Loss Derivation

**Maximum likelihood setup:** Given $n$ i.i.d. examples, the likelihood under the Bernoulli model is:

$$\mathcal{L}(\beta) = \prod_{i=1}^{n} P(y_i \mid x_i; \beta) = \prod_{i=1}^{n} \hat{p}_i^{y_i}(1 - \hat{p}_i)^{1 - y_i}$$

where $\hat{p}_i = \sigma(x_i^\top \beta)$.

**Negative log-likelihood** (which we minimise):

$$-\log \mathcal{L}(\beta) = -\sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

This is the **binary cross-entropy loss**. For a single example:

$$\mathcal{L}(y, \hat{p}) = -y \log \hat{p} - (1-y)\log(1-\hat{p})$$

**Behaviour:**
- When $y = 1$: loss is $-\log \hat{p}$. If $\hat{p} \to 1$ (correct and confident), loss $\to 0$. If $\hat{p} \to 0$ (wrong and confident), loss $\to \infty$.
- When $y = 0$: loss is $-\log(1-\hat{p})$. Symmetric behaviour.

Cross-entropy penalises confidently wrong predictions with infinite loss, which strongly pushes the model away from such mistakes.

### Gradient of the Loss

$$\frac{\partial \mathcal{L}}{\partial \beta} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i) x_i = \frac{1}{n} X^\top (\hat{p} - y)$$

Derivation using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \beta_j} = \frac{\partial \mathcal{L}}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial z} \cdot \frac{\partial z}{\partial \beta_j}$$

where $z = x^\top \beta$.

$$\frac{\partial \mathcal{L}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}$$

$$\frac{\partial \hat{p}}{\partial z} = \sigma'(z) = \hat{p}(1-\hat{p})$$

$$\frac{\partial z}{\partial \beta_j} = x_j$$

Multiplying through: $\frac{\partial \mathcal{L}}{\partial \beta_j} = (\hat{p} - y) x_j$

The sigmoid derivative and the loss derivative cancel beautifully, giving a remarkably clean gradient. There is **no closed-form solution** (unlike OLS), but the loss is convex in $\beta$, so gradient descent converges to the global minimum.

### Multi-Class Classification -- Softmax

For $K > 2$ classes, generalise with the **softmax function**:

$$P(y = k \mid x) = \frac{e^{x^\top \beta_k}}{\sum_{j=1}^{K} e^{x^\top \beta_j}}$$

Properties:
- Outputs sum to 1: $\sum_k P(y=k \mid x) = 1$
- Each output is in $(0, 1)$
- The argmax of logits equals the argmax of probabilities

**Cross-entropy loss for multi-class (categorical cross-entropy):**

$$\mathcal{L} = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y = k \mid x_i)$$

Or equivalently using one-hot vectors $y_i$:

$$\mathcal{L} = -\sum_{i=1}^{n} y_i^\top \log \hat{p}_i$$

**Note on parameter redundancy**: Softmax has one redundant parameter set ($K$ weight vectors when $K - 1$ would suffice). This is usually handled implicitly; the model is still identifiable because the loss surface has a unique minimum up to the softmax invariance.

**One-vs-rest vs. multinomial logistic regression:**
- One-vs-rest: train $K$ independent binary classifiers; no guarantee that class probabilities sum to 1
- Multinomial (softmax): joint training; probabilities are coherent; preferred for most applications

### Numerical Stability: Log-Sum-Exp Trick

Computing $\log \sum_k e^{z_k}$ directly can overflow if any $z_k$ is large. The stable form is:

$$\log \sum_{k} e^{z_k} = c + \log \sum_{k} e^{z_k - c}, \quad c = \max_k z_k$$

Subtracting the maximum before exponentiation keeps values in a safe numerical range.

---

## Tier 1 -- Fundamentals

### Q1. Why is the sigmoid function chosen for logistic regression rather than a step function or linear activation?

**Answer:**

Three reasons motivate the sigmoid over alternatives:

1. **Outputs a probability**: The sigmoid maps any real number to $(0, 1)$, yielding a valid probability. A step function outputs $\{0, 1\}$ (not a probability), and a linear activation outputs $(-\infty, \infty)$ (not bounded).

2. **Smooth and differentiable**: Gradient descent requires a differentiable loss. The step function has zero gradient almost everywhere and is undefined at the threshold -- gradient descent cannot work with it. The sigmoid has a well-defined, smooth gradient everywhere.

3. **Natural connection to log-odds**: The inverse of the sigmoid is the logit (log-odds) function: $\sigma^{-1}(p) = \log(p/(1-p))$. Logistic regression assumes the log-odds of the outcome is linear in the features, which is a natural and interpretable model for many real-world problems.

**Common mistake**: Confusing the softmax (multi-class) with the sigmoid (binary). For two-class softmax, the outputs are equivalent to sigmoid, but for $K > 2$ classes you must use softmax -- not sigmoid applied independently to each class (which would not produce probabilities summing to 1).

---

### Q2. A logistic regression model predicts $P(y=1 \mid x) = 0.9$ for a test example but the true label is $y = 0$. What is the cross-entropy loss for this single example, and why is it much larger than if the model had predicted $0.6$?

**Answer:**

For $y = 0$, the cross-entropy loss is:

$$\mathcal{L} = -\log(1 - \hat{p})$$

For $\hat{p} = 0.9$:
$$\mathcal{L} = -\log(0.1) = \log(10) \approx 2.303$$

For $\hat{p} = 0.6$:
$$\mathcal{L} = -\log(0.4) \approx 0.916$$

The model predicting $0.9$ is confidently wrong. Cross-entropy loss penalises this with a loss of $2.303$ vs $0.916$ -- nearly $2.5\times$ higher. As $\hat{p} \to 1$ for a $y = 0$ example, the loss $-\log(1-\hat{p}) \to \infty$, providing an infinitely large gradient signal to correct the overconfident wrong prediction.

This is the key advantage of cross-entropy over squared loss for classification: it imposes a much stronger penalty on confident mistakes, which is appropriate because confidently wrong predictions are more damaging in practice.

---

### Q3. What is the decision boundary of a logistic regression model, and can it represent a circle?

**Answer:**

The decision boundary is where $P(y=1 \mid x) = 0.5$, i.e., where $x^\top \beta = 0$. This is a **hyperplane** -- a line in 2D, a plane in 3D.

A logistic regression with raw features $x_1, x_2$ **cannot** represent a circular decision boundary because $\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0$ is always a line.

To represent a circle, add quadratic features: let $\phi(x) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]$. Then $\phi(x)^\top \beta = 0$ can be a circle or ellipse. For example, $x_1^2 + x_2^2 = r^2$ is recovered with $\beta_3 = \beta_4 = 1$, $\beta_5 = 0$, $\beta_0 = -r^2$, $\beta_1 = \beta_2 = 0$.

**Key insight**: Logistic regression is linear in the feature space but can be non-linear in the original input space if non-linear features are added. This is the basis of the kernel trick for logistic regression.

---

## Tier 2 -- Intermediate

### Q4. Derive the gradient of the binary cross-entropy loss with respect to the logistic regression weights, and explain why the result has such a clean form.

**Answer:**

For a single example $(x, y)$ with prediction $\hat{p} = \sigma(z)$, $z = x^\top \beta$:

**Loss:**
$$\mathcal{L} = -y \log \hat{p} - (1-y)\log(1-\hat{p})$$

**Step 1 -- Gradient with respect to $\hat{p}$:**
$$\frac{\partial \mathcal{L}}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}} = \frac{\hat{p}(1-y) - y(1-\hat{p})}{\hat{p}(1-\hat{p})} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}$$

**Step 2 -- Gradient of sigmoid with respect to $z$:**
$$\frac{\partial \hat{p}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{p}(1-\hat{p})$$

**Step 3 -- Chain rule:**
$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial z} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})} \cdot \hat{p}(1-\hat{p}) = \hat{p} - y$$

**Step 4 -- Gradient with respect to $\beta$:**
$$\frac{\partial z}{\partial \beta} = x$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = (\hat{p} - y) x$$

**Why the clean form**: The Bernoulli log-likelihood and the sigmoid are a **natural conjugate pair** -- the sigmoid is the canonical link function for the Bernoulli family in the exponential family framework. The gradient of the log-likelihood with respect to the natural parameter always takes the form (predicted - observed) times the sufficient statistic. The sigmoid derivative cancels exactly with the denominator of the loss gradient, producing this elegant result.

This also appears in neural networks: the gradient flowing back from a cross-entropy loss through a sigmoid activation is simply $\hat{p} - y$ per example -- no chain-rule multiplication needed beyond that point.

---

### Q5. How does L2 regularisation affect logistic regression, and why might you prefer L1 regularisation for high-dimensional text classification?

**Answer:**

**L2 regularisation (Ridge logistic regression):**

Adds $\frac{\lambda}{2}\|\beta\|_2^2$ to the loss:

$$\mathcal{L}_{\text{reg}} = -\frac{1}{n}\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)] + \frac{\lambda}{2}\|\beta\|_2^2$$

Gradient: $\nabla_\beta \mathcal{L}_{\text{reg}} = \frac{1}{n}X^\top(\hat{p} - y) + \lambda \beta$

All weights are shrunk towards zero but none are set exactly to zero. This is standard practice and usually improves generalisation. In scikit-learn, the parameter `C = 1/λ` (inverse regularisation strength) controls this.

**L1 regularisation for text classification:**

Text feature vectors are typically sparse bag-of-words or TF-IDF representations with $d = 100{,}000$+ features (vocabulary size). Most words are irrelevant for a given classification task. L1 regularisation:

1. Produces exact zeros, effectively selecting the relevant vocabulary subset
2. Creates a sparse model that requires less memory and is faster at inference
3. Is more interpretable: the non-zero features are the discriminative words
4. Can be more stable than L2 when features are genuinely sparse signals

**Practical example**: For binary sentiment classification on movie reviews with 50,000 vocabulary features, L1 logistic regression might select only 500 discriminative words (e.g., "terrible", "brilliant", "boring"), while L2 keeps all 50,000 features with varying small weights. The L1 model is more interpretable and often equally accurate.

**Note**: scikit-learn's `LogisticRegression` uses `solver='liblinear'` or `'saga'` for L1, as standard solvers require smooth gradients.

---

### Q6. Explain the connection between logistic regression and the exponential family. What does this reveal about the choice of cross-entropy loss?

**Answer:**

The **exponential family** is a class of probability distributions of the form:

$$P(y \mid \eta) = h(y) \exp\!\left(\eta^\top T(y) - A(\eta)\right)$$

where $\eta$ is the natural parameter, $T(y)$ the sufficient statistic, $A(\eta)$ the log-partition function, and $h(y)$ the base measure.

**Bernoulli distribution in exponential form:**

$$P(y \mid p) = p^y(1-p)^{1-y} = \exp\!\left(y \log\frac{p}{1-p} + \log(1-p)\right)$$

So the natural parameter is $\eta = \log\frac{p}{1-p}$ (the log-odds), $T(y) = y$, and $A(\eta) = \log(1 + e^\eta)$.

**Connection to logistic regression**: Setting $\eta = x^\top \beta$ (linear in features), and inverting the natural parameter mapping gives $p = \sigma(x^\top \beta)$. The sigmoid is the **canonical link function** of the Bernoulli distribution.

**What this reveals:**

1. **Optimality of cross-entropy loss**: The maximum likelihood estimator for exponential family models is always consistent and asymptotically efficient. Cross-entropy is the correct loss because it equals the negative log-likelihood of the Bernoulli family, not just an ad-hoc choice.

2. **Log-partition function and calibration**: The log-partition function $A(\eta) = \log(1 + e^\eta)$ ensures the distribution is normalised. Because the model is well-calibrated by construction under the Bernoulli assumption, logistic regression typically produces better-calibrated probabilities than discriminative approaches that don't explicitly model the likelihood.

3. **Generalisations**: This framework explains why:
   - For Poisson-distributed counts, the canonical link is log (Poisson regression)
   - For Gaussian targets, the canonical link is identity (linear regression)
   - For multinomial outcomes, the canonical link is log-softmax

---

## Tier 3 -- Advanced

### Q7. What is the problem of class imbalance in logistic regression, and what are four distinct techniques to address it?

**Answer:**

When one class is rare (e.g., 1% positive examples in fraud detection), the maximum likelihood logistic regression estimator is biased towards predicting the majority class. The cross-entropy loss gradient is dominated by majority-class examples, causing the model to assign low probability to positive cases even when a strong signal exists.

**Technique 1: Class weighting**

Weight each example's loss contribution inversely proportional to its class frequency:

$$\mathcal{L} = -\frac{1}{n}\sum_i w_{y_i}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

where $w_1 = n/(2 n_1)$, $w_0 = n/(2 n_0)$ for a balanced weighting. In sklearn: `class_weight='balanced'`. This is equivalent to oversampling the minority class.

**Technique 2: Resampling -- SMOTE (Synthetic Minority Oversampling Technique)**

Generate synthetic minority-class examples by interpolating between existing minority-class examples in feature space. For each minority example, select $k$ nearest minority neighbours and create synthetic points along the line segments between them. This increases training set diversity without simple duplication.

**Technique 3: Threshold tuning**

The default classification threshold is $0.5$. For imbalanced problems, lower thresholds (e.g., $0.1\text{--}0.3$) increase recall at the cost of precision. Choose the threshold on the validation set by optimising the metric that matters for your application (F1-score, precision at fixed recall, etc.). The ROC curve and precision-recall curve are invaluable tools here.

**Technique 4: Alternative loss functions**

- **Focal loss** (from object detection): $\mathcal{L}_{\text{focal}} = -\alpha(1-\hat{p})^\gamma \log \hat{p}$ for positive examples. The factor $(1-\hat{p})^\gamma$ downweights easy examples where the model is already confident, focusing training capacity on hard examples near the decision boundary.
- **Asymmetric loss**: directly penalise false negatives more than false positives.

**Important**: Evaluation on imbalanced datasets should use precision-recall AUC or F1-score, not accuracy. A classifier predicting "always negative" achieves 99% accuracy on a 1% positive-rate problem but is entirely useless.

---

### Q8. Logistic regression is a convex optimisation problem. Prove convexity of the binary cross-entropy loss and explain what this guarantees algorithmically.

**Answer:**

**Claim**: The binary cross-entropy loss $\mathcal{L}(\beta) = -\frac{1}{n}\sum_i [y_i \log \sigma(x_i^\top \beta) + (1-y_i)\log(1-\sigma(x_i^\top \beta))]$ is convex in $\beta$.

**Proof via Hessian positive semi-definiteness:**

For a single example, the loss is:

$$\ell(z) = -y \log \sigma(z) - (1-y)\log(1-\sigma(z)), \quad z = x^\top \beta$$

We showed $\frac{\partial \ell}{\partial z} = \hat{p} - y$. Taking the second derivative:

$$\frac{\partial^2 \ell}{\partial z^2} = \frac{\partial \hat{p}}{\partial z} = \hat{p}(1-\hat{p}) \geq 0 \quad \forall z$$

This means $\ell$ is convex as a function of $z$. Since $z = x^\top \beta$ is linear (hence convex) in $\beta$, the composition $\ell(x^\top \beta)$ is convex in $\beta$ (composition of convex function with affine map is convex).

The sum of convex functions is convex, so the full loss $\mathcal{L}(\beta)$ is convex.

The Hessian of $\mathcal{L}$ with respect to $\beta$:

$$H = \frac{1}{n} X^\top \text{diag}(\hat{p}_i(1-\hat{p}_i)) X = \frac{1}{n} X^\top W X$$

where $W = \text{diag}(\hat{p}_i(1-\hat{p}_i))$. Since $0 < \hat{p}_i < 1$, $W$ is positive definite. $X^\top W X$ is positive semi-definite (PSD) for any $X$. Thus $H \succeq 0$, confirming convexity.

**Algorithmic guarantees of convexity:**

1. **Any local minimum is a global minimum**: gradient descent converges to the optimal solution regardless of initialisation.
2. **No saddle point traps**: with a convex function, all saddle points are global minima (trivially true since there are none above the global min).
3. **Convergence rate guarantees**: for strongly convex losses (with L2 regularisation, which makes $H \succ 0$), gradient descent converges geometrically (linear convergence rate).
4. **Warm starting**: the solution changes smoothly with hyperparameters (e.g., $\lambda$), so solution paths can be traced efficiently.

**Note**: Without L2 regularisation, logistic regression is convex but not strictly convex when features are linearly separable. In this case, the maximum likelihood estimate diverges to infinity (weights grow without bound, pushing $\hat{p} \to 1$ for positive examples and $\hat{p} \to 0$ for negative). L2 regularisation restores strict convexity and ensures a finite, unique solution.

---

## Implementation Reference

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Standardise features (important for regularised logistic regression)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Binary logistic regression with L2 regularisation (default)
# C = 1/lambda: smaller C = stronger regularisation
clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
clf.fit(X_train_s, y_train)

# Probabilities and predictions
probs = clf.predict_proba(X_test_s)[:, 1]   # P(y=1)
preds = clf.predict(X_test_s)               # threshold at 0.5 by default

print(classification_report(y_test, preds))
print(f"AUC-ROC: {roc_auc_score(y_test, probs):.4f}")

# Custom threshold (for imbalanced problems)
threshold = 0.3
preds_custom = (probs >= threshold).astype(int)

# Multi-class with softmax (multinomial)
clf_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0)
clf_multi.fit(X_train_s, y_train_multi)

# Logistic regression from scratch (for understanding)
def sigmoid(z):
    # Numerically stable sigmoid
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def cross_entropy_loss(y, p):
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def logistic_gradient_descent(X, y, lr=0.01, n_iter=1000, lam=0.01):
    n, d = X.shape
    beta = np.zeros(d)
    for _ in range(n_iter):
        p = sigmoid(X @ beta)
        grad = (X.T @ (p - y)) / n + lam * beta   # L2 gradient
        beta -= lr * grad
    return beta
```

---

## Quick Reference Quiz

**Q: Logistic regression is trained on linearly separable data with no regularisation. What happens to the weights?**

A) They converge to the maximum margin hyperplane (same as SVM)  
B) They diverge to infinity along the separating direction  
C) They converge to a finite solution that misclassifies a few points  
D) Gradient descent fails to converge at all  

**Answer: B.** With linearly separable data and no regularisation, the cross-entropy loss can always be reduced further by scaling the weights by a constant $c > 1$, pushing predictions towards $0$ and $1$ with higher confidence. The loss approaches $0$ only as $\|\beta\| \to \infty$. This is a practical problem: add L2 regularisation to ensure a finite, well-behaved solution.

---

**Q: For a 3-class softmax logistic regression with $d = 10$ features, how many learnable parameters are there (including bias)?**

A) 11  
B) 30  
C) 33  
D) 32  

**Answer: C.** Each of the 3 classes has a weight vector of length $d = 10$ plus one bias term, giving $3 \times (10 + 1) = 33$ parameters. Although one set is redundant (can be set to zero), all 33 are typically learned in practice (with regularisation preventing instability).
