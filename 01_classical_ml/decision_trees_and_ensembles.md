# Decision Trees and Ensembles

## Prerequisites
- Basic probability: entropy, expectation
- Understanding of overfitting and bias-variance trade-off
- Gradient descent at a conceptual level (for gradient boosting)

---

## Concept Reference

### Decision Trees

A decision tree recursively partitions the feature space by axis-aligned splits. At each internal node, a feature $j$ and threshold $t$ are chosen to split examples into two subsets. Leaf nodes output a prediction (class label or real value).

**Inductive bias**: decision trees assume that good decision boundaries are axis-aligned and hierarchical. They make no assumptions about the data distribution (non-parametric).

### Splitting Criteria

The goal at each node is to choose the split that maximises the "purity" of the resulting child nodes. Two common criteria:

**1. Information Gain (uses entropy)**

For a node containing examples from $K$ classes with proportions $p_1, \ldots, p_K$, the **entropy** is:

$$H(p_1, \ldots, p_K) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Entropy is maximised at $\log_2 K$ (uniform distribution) and minimised at $0$ (all one class).

The **information gain** of a split that divides node $S$ into children $S_L$ and $S_R$:

$$IG = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$$

We choose the split $(j, t)$ that maximises $IG$. This is the criterion used in **ID3** and **C4.5** algorithms.

**2. Gini Impurity**

$$G(p_1, \ldots, p_K) = 1 - \sum_{k=1}^{K} p_k^2$$

Interpretation: the probability that two randomly drawn examples from the node belong to different classes. $G = 0$ for a pure node; $G = 1 - 1/K$ for a completely mixed node.

Gini impurity gain for a split:

$$\Delta G = G(S) - \frac{|S_L|}{|S|} G(S_L) - \frac{|S_R|}{|S|} G(S_R)$$

CART (Classification and Regression Trees) uses Gini for classification and variance reduction for regression.

**Practical difference**: Entropy and Gini produce nearly identical trees in practice. Gini is slightly cheaper to compute (no logarithm). Entropy is slightly more sensitive to equal-probability splits.

**Regression trees**: use **variance (MSE) reduction** as the split criterion:

$$\Delta \text{Var} = \text{Var}(S) - \frac{|S_L|}{|S|}\text{Var}(S_L) - \frac{|S_R|}{|S|}\text{Var}(S_R)$$

Leaf prediction: mean of target values in the leaf.

### Tree Depth and Overfitting

An unconstrained decision tree will grow until every leaf contains a single training example -- zero training error but severe overfitting. Control strategies:

| Hyperparameter | Effect |
|---|---|
| `max_depth` | Limits tree height directly |
| `min_samples_split` | Node must have at least $k$ examples to split |
| `min_samples_leaf` | Each leaf must contain at least $k$ examples |
| `max_features` | Consider only a random subset of features at each split |
| `min_impurity_decrease` | Only split if impurity decrease exceeds threshold |

### Random Forest

Random Forest builds an ensemble of decorrelated decision trees using two sources of randomness:

1. **Bootstrap aggregating (bagging)**: each tree trains on a bootstrap sample (random sample with replacement) of the training data, size $n$. Approximately $63.2\%$ of examples appear in each bootstrap sample; the rest (out-of-bag examples) can be used for free validation.

2. **Feature subsampling**: at each split, only a random subset of $m$ features is considered, typically $m = \sqrt{d}$ for classification and $m = d/3$ for regression.

**Prediction**: average of all tree predictions (regression) or majority vote (classification). Optionally, soft votes using predicted probabilities are averaged.

**Why decorrelation matters**: If all trees were identical, averaging them would not reduce variance. The variance of the average of $B$ identically distributed variables with pairwise correlation $\rho$ and variance $\sigma^2$ is:

$$\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$, the second term vanishes, leaving $\rho \sigma^2$. The correlation $\rho$ is the limiting factor; feature subsampling reduces $\rho$ by preventing all trees from making the same splits on dominant features.

**Out-of-bag (OOB) error**: each training example is out-of-bag for approximately $37\%$ of trees. Averaging predictions from only those trees gives an approximately unbiased estimate of generalisation error -- a free cross-validation estimate that requires no explicit held-out set.

### Gradient Boosting

Gradient boosting builds an additive ensemble **sequentially**, where each new tree corrects the residuals of the current ensemble.

**Algorithm:**

1. Initialise $F_0(x) = \text{const}$ (e.g., mean of targets for regression)
2. For $m = 1, \ldots, M$:
   a. Compute **pseudo-residuals** (negative gradient of the loss):

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

   b. Fit a regression tree $h_m(x)$ to the pseudo-residuals $\{(x_i, r_{im})\}$
   c. Compute optimal step size $\gamma_m$:

$$\gamma_m = \arg\min_\gamma \sum_{i=1}^n \mathcal{L}(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

   d. Update: $F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)$, where $\nu$ is the learning rate (shrinkage)

**Pseudo-residuals for common losses:**
- MSE loss $\mathcal{L} = \frac{1}{2}(y - F)^2$: residuals are $r_i = y_i - F(x_i)$
- Log-loss (classification): residuals are $r_i = y_i - \sigma(F(x_i))$ -- same clean form as logistic regression gradient
- Huber loss: residuals are clipped, making the method robust to outliers

**Key hyperparameters:**
- Learning rate $\nu \in (0, 1]$: smaller values require more trees but generalise better
- Number of trees $M$: chosen via early stopping on a validation set
- Max tree depth (usually shallow: depth 3--6): controls the order of interactions modelled

### XGBoost

XGBoost (eXtreme Gradient Boosting) extends gradient boosting with several practical improvements:

**1. Second-order Taylor expansion of the loss:**

$$\mathcal{L} \approx \sum_{i} \left[g_i f_m(x_i) + \frac{1}{2}h_i f_m^2(x_i)\right] + \Omega(f_m)$$

where $g_i = \partial_{F} \mathcal{L}(y_i, F)$ (first-order gradient) and $h_i = \partial^2_F \mathcal{L}(y_i, F)$ (second-order Hessian). Using curvature information leads to better step sizes and faster convergence.

**2. Regularisation term on tree structure:**

$$\Omega(f) = \gamma T + \frac{\lambda}{2}\sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

where $T$ is the number of leaves, $w_j$ the leaf weights, and $\gamma$, $\lambda$, $\alpha$ are regularisation hyperparameters. This penalises both tree complexity and large leaf values.

**3. Approximate split finding**: an efficient histogram-based algorithm avoids scanning all possible split points, enabling training on datasets that do not fit in memory.

**4. Handling missing values**: learns an optimal default direction for each split when values are missing.

**5. Column subsampling**: like Random Forest, randomly samples features at each level or tree.

**LightGBM** (Microsoft) further improves on XGBoost with gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB), achieving much faster training on large datasets.

### Bagging vs. Boosting

| Property | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|---|---|---|
| Tree construction | Parallel | Sequential |
| Focus | Variance reduction | Bias reduction |
| Base learner | Deep trees (low bias, high variance) | Shallow trees (high bias, low variance) |
| Sensitive to outliers | Less | More (fits residuals) |
| Overfitting | Hard to overfit with more trees | Can overfit; needs early stopping |
| Interpretability | Lower (many deep trees) | Slightly higher (shallow trees) |
| Training speed | Embarrassingly parallelisable | Sequential, harder to parallelise |

---

## Tier 1 -- Fundamentals

### Q1. Explain how a decision tree chooses a split using information gain, and work through a small example.

**Answer:**

At each node, we evaluate every possible (feature, threshold) pair and choose the one that maximises information gain.

**Example**: 10 examples: 5 class A, 5 class B. Feature $x_1$: values $\{1,1,1,2,2,2,2,3,3,3\}$ with labels $\{A,A,A,A,A,B,B,B,B,B\}$.

**Parent entropy:**
$$H(S) = -\frac{5}{10}\log_2\frac{5}{10} - \frac{5}{10}\log_2\frac{5}{10} = 1.0 \text{ bit}$$

**Split at $x_1 \leq 1.5$**: Left = $\{1,1,1\}$ (3 examples, all A); Right = $\{2,2,2,2,3,3,3\}$ (7 examples: 2A, 5B)

$$H(S_L) = 0 \text{ (pure)}$$
$$H(S_R) = -\frac{2}{7}\log_2\frac{2}{7} - \frac{5}{7}\log_2\frac{5}{7} \approx 0.863 \text{ bits}$$
$$IG = 1.0 - \frac{3}{10}(0) - \frac{7}{10}(0.863) \approx 0.396 \text{ bits}$$

**Split at $x_1 \leq 2.5$**: Left = $\{1,1,1,2,2,2,2\}$ (7 examples: 5A, 2B); Right = $\{3,3,3\}$ (3 examples, all B)

$$H(S_L) \approx 0.863, \quad H(S_R) = 0$$
$$IG = 1.0 - \frac{7}{10}(0.863) - \frac{3}{10}(0) \approx 0.396 \text{ bits}$$

Both splits give identical information gain by symmetry. The tree picks either; further splits would refine the remaining impure nodes.

---

### Q2. What is the difference between bagging and boosting?

**Answer:**

Both build ensembles of weak learners but differ in strategy and error reduction:

**Bagging (Bootstrap Aggregating):**
- Trains each learner **independently and in parallel** on a bootstrap sample of the data
- Each learner has the same expected bias as a single learner trained on the full dataset
- Averaging reduces variance: $\text{Var}(\bar{T}) \approx \rho \sigma^2$ in the limit
- Best applied to high-variance, low-bias learners (deep decision trees)
- Random Forest is the canonical bagging algorithm for trees

**Boosting:**
- Trains learners **sequentially**, each correcting the errors of the current ensemble
- Each new learner focuses on examples the ensemble gets wrong (either by reweighting examples in AdaBoost, or by fitting residuals in gradient boosting)
- Primarily reduces bias: the ensemble progressively models more complex patterns
- Best applied to high-bias, low-variance learners (shallow trees -- stumps or depth-3 trees)
- Risk: can overfit training data if too many trees are added; requires early stopping

**Rule of thumb**: if training error is high (underfitting), boosting helps. If training error is low but test error is high (overfitting), bagging helps.

---

### Q3. Why does Random Forest use feature subsampling at each split? What problem would occur without it?

**Answer:**

Without feature subsampling, all trees in the forest would see the same features at every split. If one feature is highly predictive (e.g., the strongest predictor of the target), virtually every tree would choose that feature for its root split, making all trees similar (highly correlated). Averaging correlated trees does not reduce variance significantly because correlated errors cancel only slightly.

The variance of the mean of $B$ trees with pairwise correlation $\rho$:

$$\text{Var}\!\left(\frac{1}{B}\sum T_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

With $\rho$ close to 1 and $B$ large: $\text{Var} \approx \rho \sigma^2 \approx \sigma^2$ (no benefit from averaging).

With $\rho$ small: $\text{Var} \approx \sigma^2 / B$ (full variance reduction).

Feature subsampling forces different trees to use different features for early splits, decorrelating them. The typical choice of $m = \sqrt{d}$ features balances the trade-off between decorrelation (want small $m$) and individual tree quality (want large $m$).

---

## Tier 2 -- Intermediate

### Q4. Derive the update rule for gradient boosting and explain why it is called "gradient" boosting.

**Answer:**

Let $F_{m-1}(x)$ be the current ensemble. We want to add a new tree $h_m(x)$ to reduce the total loss:

$$\mathcal{L}(F_{m-1} + h_m) = \sum_i \mathcal{L}(y_i, F_{m-1}(x_i) + h_m(x_i))$$

The optimal additive correction would be $h_m^*(x_i) = -\frac{\partial \mathcal{L}}{\partial F}$ evaluated at $F = F_{m-1}$, i.e., the negative gradient of the loss with respect to the current predictions.

We cannot compute a function exactly equal to the negative gradient for unseen $x$, so we fit a regression tree to approximate it. The pseudo-residuals are:

$$r_{im} = -\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F = F_{m-1}}$$

The tree $h_m$ is fit to $\{(x_i, r_{im})\}$ using squared error, making $h_m(x) \approx r_m(x)$.

The update:

$$F_m(x) = F_{m-1}(x) + \nu h_m(x)$$

**Why "gradient"**: this is steepest descent in the functional space of predictions. Instead of moving in the direction of the negative gradient in parameter space (as in standard gradient descent), we move in the direction of the negative gradient in **function space** -- represented by a tree. The algorithm is gradient descent where the step is a regression tree rather than a parameter vector update.

**The learning rate** $\nu$ acts like a step size in gradient descent -- small values take small steps along the gradient direction, requiring more trees but generalising better (less likely to overshoot the minimum).

---

### Q5. Compare the behaviour of a decision tree, Random Forest, and Gradient Boosting when you add more data (increase $n$). Which benefits most and why?

**Answer:**

**Decision tree:**
- A single tree has high variance. With more data, the variance of individual splits decreases and the tree can learn finer-grained splits without overfitting. However, a fully grown tree will still overfit on the training data.
- More data primarily helps by stabilising the split choices and reducing the risk of spurious splits based on small samples.

**Random Forest:**
- Benefits substantially from more data because the bootstrap samples become more representative of the true distribution.
- With more data, both bias and variance of individual trees decrease. The averaging further reduces variance.
- Random Forest has a hard floor on performance set by the correlation $\rho$ between trees -- adding more data helps but the forest still averages correlated errors.

**Gradient Boosting:**
- Benefits greatly from more data. Gradient boosting has low bias by design (it fits residuals aggressively), but can overfit with limited data if too many trees are used.
- With more data, the variance of each tree fit decreases, allowing more boosting iterations without overfitting. This lets the model learn more complex patterns.
- Empirically, XGBoost and LightGBM are among the strongest models on tabular data at large scales ($n > 10^5$), often outperforming neural networks.

**Summary**: all three benefit from more data. Gradient boosting typically benefits most on tabular data because it has the capacity to reduce both bias and variance with sufficient data, while Random Forest's variance floor ($\rho\sigma^2$) limits its improvement beyond a certain scale.

---

### Q6. What is the out-of-bag (OOB) error in Random Forest, and how does it compare to k-fold cross-validation?

**Answer:**

For each tree in the forest, approximately $36.8\%$ ($e^{-1}$) of the training examples are not sampled (out-of-bag). These OOB examples can be used to estimate the tree's prediction without information leakage -- the tree never saw them during training.

**OOB error computation**: for each training example $x_i$, average predictions only over trees for which $x_i$ was out-of-bag. The resulting OOB prediction for each example provides an unbiased estimate of the generalisation error.

**Comparison with $k$-fold cross-validation:**

| Property | OOB Error | k-fold CV |
|---|---|---|
| Additional training runs | None (computed from training run) | Requires $k$ separate training runs |
| Computational cost | Free | $k \times$ training cost |
| Effective training set fraction | $\approx 63.2\%$ per tree | $(k-1)/k$ per fold |
| Bias | Slightly pessimistic (smaller effective training set) | Depends on $k$; $k=10$ is nearly unbiased |
| Applicable models | Random Forest only | Any model |
| Variance | Moderate | Lower with higher $k$ |

**When to use OOB**: when training time is limited and you need a quick, approximately unbiased error estimate without running full cross-validation. For model selection between fundamentally different models, use k-fold CV.

---

## Tier 3 -- Advanced

### Q7. Explain how XGBoost uses second-order Taylor expansion to derive optimal leaf weights. What advantage does this give over standard gradient boosting?

**Answer:**

For each boosting step, XGBoost approximates the loss at tree $m$ using a second-order Taylor expansion around the current predictions $F_{m-1}(x_i)$:

$$\mathcal{L} \approx \sum_i \left[\mathcal{L}(y_i, F_{m-1}(x_i)) + g_i f_m(x_i) + \frac{1}{2}h_i f_m(x_i)^2 \right] + \Omega(f_m)$$

where $g_i = \partial_{F}\mathcal{L}(y_i, F_{m-1})|_{F_{m-1}}$ and $h_i = \partial^2_F \mathcal{L}(y_i, F_{m-1})|_{F_{m-1}}$.

For a tree with leaf set $\mathcal{J}$ mapping each example to leaf $q(x_i) \in \{1, \ldots, T\}$ with leaf weight $w_j$, and regularisation $\Omega = \gamma T + \frac{\lambda}{2}\sum_j w_j^2$:

$$\mathcal{L} \approx \sum_{j=1}^{T} \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T + \text{const}$$

where $I_j = \{i : q(x_i) = j\}$ is the set of examples in leaf $j$.

This is a quadratic in $w_j$ for each leaf independently. Taking the derivative and setting to zero:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} = -\frac{G_j}{H_j + \lambda}$$

Substituting back gives the **optimal leaf score** (gain) for a candidate split:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

**Advantages over standard gradient boosting:**

1. **Adaptive step sizes per leaf**: the optimal leaf weight $w_j^*$ uses the local curvature $H_j$ as a natural step size denominator. Flatter regions (low $H_j$) get larger steps; steep regions (high $H_j$) get smaller, more conservative steps. This is analogous to Newton's method vs. gradient descent.

2. **Principled regularisation**: the $\lambda$ term smoothly regularises the leaf values within the same optimisation, not as a separate post-processing step.

3. **Support for arbitrary losses**: as long as the loss is twice differentiable, the same algorithm applies -- only $g_i$ and $h_i$ change. This generalises to ranking, survival analysis, and custom loss functions.

4. **Faster convergence**: second-order methods typically require far fewer iterations than first-order methods to reach the same loss value, saving computation especially when each tree is expensive to fit.

---

### Q8. A Random Forest with 1000 trees achieves $92\%$ test accuracy. A single decision tree achieves $78\%$. Explain mathematically why the ensemble is better, and describe three failure modes where Random Forest underperforms.

**Answer:**

**Why the ensemble is better:**

For the ensemble prediction to be wrong (majority vote), more than half the trees must be wrong. If each tree has independent error probability $\epsilon = 0.22$, the ensemble error is:

$$P(\text{ensemble error}) = \sum_{k = \lceil B/2 \rceil}^{B} \binom{B}{k} \epsilon^k (1-\epsilon)^{B-k}$$

For $\epsilon = 0.22$, $B = 1000$: this is astronomically small compared to $0.22$.

In practice, trees are correlated (not independent), so the improvement is smaller but still substantial. The decomposition is:

$$\text{MSE}(\bar{T}) = \underbrace{\text{Bias}^2(\bar{T})}_{= \text{Bias}^2(T)} + \underbrace{\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2}_{\text{Variance}(\bar{T})}$$

Since $\rho < 1$ (decorrelation from feature subsampling), the ensemble variance is strictly less than a single tree's variance $\sigma^2$.

**Three failure modes of Random Forest:**

1. **High-cardinality categorical features**: a feature with many unique values (e.g., zip code with 30,000 values) will dominate split selection because there are many possible split thresholds. The forest learns spurious associations with rare levels. Gradient boosting handles this better via regularisation, and target encoding can mitigate it for Random Forest.

2. **Extrapolation (out-of-distribution inputs)**: decision trees predict by averaging target values in leaf nodes. For regression, if a test input has feature values outside the range seen during training, the forest returns the prediction of the leaf most closely matching those values -- effectively extrapolating by returning training-data-range values. It cannot extrapolate trends. Neural networks and linear models extrapolate differently, sometimes more accurately for structured trend data.

3. **Imbalanced classes with a dominant feature**: when one feature is highly predictive of the majority class, feature subsampling may frequently exclude this feature, causing each tree to build poor splits and learn high-variance patterns. Class weighting and stratified sampling mitigate this. Gradient boosting, which focuses on residuals, often handles this case better naturally.

---

## Implementation Reference

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import numpy as np

# Decision tree with controlled depth
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
tree.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,      # number of trees
    max_features='sqrt',   # features per split
    max_depth=None,        # grow full trees
    min_samples_leaf=5,
    oob_score=True,        # free OOB error estimate
    n_jobs=-1,             # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB accuracy: {rf.oob_score_:.4f}")

# Feature importances (mean decrease in impurity)
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# Gradient Boosting (sklearn)
gb = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,    # small learning rate: more trees needed
    max_depth=4,
    subsample=0.8,         # stochastic gradient boosting
    min_samples_leaf=20,
    validation_fraction=0.1,
    n_iter_no_change=20,   # early stopping
    random_state=42
)
gb.fit(X_train, y_train)

# XGBoost with early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,   # equivalent to min_samples_leaf
    'lambda':           1.0,  # L2 regularisation on leaf weights
    'alpha':            0.0,  # L1 regularisation on leaf weights
    'gamma':            0.1,  # min split gain (complexity penalty)
}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)
```

---

## Quick Reference Quiz

**Q: A node contains 10 examples: 6 class A, 4 class B. A split produces Left = {4A, 1B}, Right = {2A, 3B}. What is the Gini impurity of each node?**

Parent: $G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$

Left (5 examples, 4A 1B): $G_L = 1 - ((4/5)^2 + (1/5)^2) = 1 - (0.64 + 0.04) = 0.32$

Right (5 examples, 2A 3B): $G_R = 1 - ((2/5)^2 + (3/5)^2) = 1 - (0.16 + 0.36) = 0.48$

Weighted impurity: $\frac{5}{10}(0.32) + \frac{5}{10}(0.48) = 0.40$

Gini gain: $0.48 - 0.40 = 0.08$

---

**Q: Which of the following is FALSE about Random Forest?**

A) It uses bootstrap sampling of training examples  
B) It can estimate out-of-sample error without a separate validation set  
C) Adding more trees will eventually cause overfitting  
D) It randomly selects a subset of features at each split  

**Answer: C.** Unlike gradient boosting, adding more trees to a Random Forest does not increase overfitting. The averaging converges to a stable ensemble. This is one practical advantage of Random Forest: training for longer (more trees) always helps or at worst plateaus, never hurts.
