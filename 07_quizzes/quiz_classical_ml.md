# Quiz: Classical ML

## Instructions

15 multiple-choice questions covering supervised learning algorithms, regularisation,
ensemble methods, SVMs, and clustering. Each question has exactly one correct answer.
Work through all questions before checking the answer key at the end.

Difficulty distribution: Questions 1-5 Fundamentals, Questions 6-11 Intermediate,
Questions 12-15 Advanced.

---

## Questions

### Q1 (Fundamentals)

Which of the following correctly describes the bias-variance decomposition of
expected test error?

- A) Error = Bias + Variance + Irreducible Noise
- B) Error = Bias^2 + Variance + Irreducible Noise
- C) Error = Bias^2 * Variance + Irreducible Noise
- D) Error = Bias + Variance^2 + Irreducible Noise

---

### Q2 (Fundamentals)

In logistic regression, the model outputs:

- A) A real-valued score that can take any value in (-inf, +inf).
- B) A probability estimate constrained to [0, 1] via the sigmoid function.
- C) A hard binary label (0 or 1) directly without a probability.
- D) A probability estimate constrained to [0, 1] via the softmax function over two classes.

---

### Q3 (Fundamentals)

What is the purpose of the regularisation parameter lambda (or C in sklearn's SVM) in a
machine learning model?

- A) It controls the learning rate of gradient descent.
- B) It controls the trade-off between fitting the training data and keeping model
     weights small (penalising complexity).
- C) It determines the number of trees in an ensemble.
- D) It sets the minimum number of samples required at a leaf node in a decision tree.

---

### Q4 (Fundamentals)

A k-means clustering algorithm has been run on a dataset and assigned cluster labels.
The algorithm is considered converged when:

- A) The within-cluster sum of squares (inertia) reaches zero.
- B) No data point changes its cluster assignment between two consecutive iterations.
- C) The number of clusters k matches the number of distinct classes in the dataset.
- D) The cluster centroids are positioned at the mean of all data points.

---

### Q5 (Fundamentals)

Which of the following is a key difference between bagging and boosting ensemble methods?

- A) Bagging trains trees sequentially, each correcting the errors of the previous;
     boosting trains trees in parallel on bootstrap samples.
- B) Boosting trains models sequentially on re-weighted data; bagging trains models
     in parallel on bootstrap samples.
- C) Bagging uses only decision trees; boosting can use any model type.
- D) Boosting always outperforms bagging on all datasets.

---

### Q6 (Intermediate)

In a Support Vector Machine (SVM), the "support vectors" are:

- A) The feature vectors with the highest magnitude in the training set.
- B) The training examples that lie exactly on or inside the margin boundaries.
- C) All training examples used to compute the decision boundary.
- D) The eigenvectors of the kernel matrix used for dimensionality reduction.

---

### Q7 (Intermediate)

You train a decision tree to maximum depth on a training set and observe training
accuracy = 100 % and test accuracy = 60 %. Which action is MOST likely to improve
test accuracy?

- A) Increase tree depth further to memorise more patterns.
- B) Apply post-pruning or limit the maximum depth to prevent overfitting.
- C) Remove all regularisation (min_samples_leaf, min_samples_split).
- D) Switch to a linear model, which always generalises better than decision trees.

---

### Q8 (Intermediate)

The RBF (Radial Basis Function) kernel in SVM is defined as:

K(x, x') = exp(-gamma * ||x - x'||^2)

What happens to the decision boundary as gamma increases?

- A) The decision boundary becomes smoother and more linear.
- B) The influence of each support vector extends over a wider region.
- C) The decision boundary becomes more complex, fitting closer to individual training
     points and potentially overfitting.
- D) The kernel degenerates to a linear kernel.

---

### Q9 (Intermediate)

In a Random Forest, why does using a random subset of features at each split (the `max_features`
parameter) improve generalisation over a standard bagged forest that considers all features?

- A) It reduces variance by ensuring each tree is a weak learner, increasing diversity
     between trees so their errors are less correlated.
- B) It reduces bias because fewer features means the model is simpler.
- C) It speeds up training but has no effect on generalisation.
- D) It ensures that the most predictive feature is never used, forcing the model
     to learn more robust patterns.

---

### Q10 (Intermediate)

You are training a gradient boosting model (e.g., XGBoost). Increasing the `max_depth`
hyperparameter from 3 to 8 produces lower training loss but higher test loss. Which
of the following BEST explains this?

- A) Deeper trees have lower variance and higher bias, causing underfitting.
- B) Deeper trees memorise more training-set-specific interactions, increasing variance
     and causing overfitting.
- C) XGBoost always overfits at max_depth=8 regardless of the dataset size.
- D) The gradient boosting algorithm requires max_depth=3 to converge properly.

---

### Q11 (Intermediate)

The DBSCAN clustering algorithm requires two parameters: eps and min_samples. A point
is classified as a core point, border point, or noise. Which of the following correctly
defines a core point?

- A) A point that has at least min_samples other points within distance eps in its
     neighbourhood (including itself).
- B) A point that is within distance eps of exactly one cluster centroid.
- C) A point that is equidistant from all other points in the dataset.
- D) The centroid of the densest cluster in the dataset.

---

### Q12 (Advanced)

Ridge regression (L2) and Lasso regression (L1) both add a penalty to the OLS objective.
Lasso is known to produce sparse solutions (many weights exactly zero), while Ridge
shrinks all weights toward zero but rarely to exactly zero. What geometric property
of the L1 constraint set causes this sparsity?

- A) The L1 constraint set (a hypercube) has corners on the coordinate axes, and the
     OLS loss ellipsoid tends to first touch the constraint set at these corners where
     one or more coordinates are zero.
- B) The L2 constraint set (a sphere) has no corners; the L1 ball (a diamond) has sharp
     corners on the coordinate axes, and the OLS loss ellipsoid tends to first touch
     these corners, setting some coordinates to exactly zero.
- C) Lasso applies a non-linear transformation that forces weights to zero during
     optimisation.
- D) Lasso uses a different gradient descent algorithm that converges to a sparser solution.

---

### Q13 (Advanced)

Kernel PCA with an RBF kernel implicitly maps data to an infinite-dimensional feature
space. Which of the following correctly describes the relationship between the kernel
function and this feature map?

- A) K(x, x') = phi(x)^T phi(x') where phi maps to a finite-dimensional space whose
     dimension equals the number of training samples.
- B) K(x, x') = phi(x)^T phi(x') where phi maps to a potentially infinite-dimensional
     feature space, and the kernel trick avoids explicitly computing phi.
- C) The RBF kernel computes the Euclidean distance between x and x' and does not
     correspond to any inner product in a feature space.
- D) The implicit feature space of the RBF kernel has dimension equal to the number
     of input features.

---

### Q14 (Advanced)

In gradient boosting, at each stage m, a regression tree is fit to the negative gradient
of the loss function with respect to the current model's predictions. For mean squared
error loss L = (1/2)(y - F(x))^2, the negative gradient is:

- A) -(y - F(x))
- B) y - F(x)
- C) (y - F(x))^2
- D) F(x) - y

---

### Q15 (Advanced)

You have a dataset with 1000 features and 200 training samples (p >> n regime). Which
statement about applying OLS linear regression in this setting is CORRECT?

- A) OLS will find a unique solution with zero training error, but the solution will
     not generalise because the system is underdetermined.
- B) OLS is undefined because the feature matrix X is singular and (X^T X)^{-1} does not
     exist. The model will not train.
- C) Adding L2 regularisation (Ridge) makes the problem well-posed by ensuring
     (X^T X + lambda*I) is invertible for any lambda > 0.
- D) The model will have high bias because too many features confuse the linear model.

---

## Answer Key

| Q  | Answer | Difficulty    |
|----|--------|---------------|
| 1  | B      | Fundamentals  |
| 2  | B      | Fundamentals  |
| 3  | B      | Fundamentals  |
| 4  | B      | Fundamentals  |
| 5  | B      | Fundamentals  |
| 6  | B      | Intermediate  |
| 7  | B      | Intermediate  |
| 8  | C      | Intermediate  |
| 9  | A      | Intermediate  |
| 10 | B      | Intermediate  |
| 11 | A      | Intermediate  |
| 12 | B      | Advanced      |
| 13 | B      | Advanced      |
| 14 | B      | Advanced      |
| 15 | C      | Advanced      |

---

## Detailed Explanations

### Q1 - Answer: B

The bias-variance decomposition for squared error is:
```
E[(y - f_hat(x))^2] = Bias(f_hat(x))^2 + Var(f_hat(x)) + sigma^2
```
The **squared bias** appears because bias is a signed quantity (the error can be positive
or negative), and the expected squared error includes it squared. The irreducible noise
sigma^2 is the variance of the data-generating process itself.

- **A** is wrong: bias enters the decomposition squared, not linearly.
- **C** is wrong: the terms are added, not multiplied.
- **D** is wrong: variance is not squared.

---

### Q2 - Answer: B

Logistic regression applies the sigmoid function to a linear combination of features:
```
p(y=1|x) = sigmoid(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
```
The sigmoid maps any real value to (0, 1), producing a calibrated probability estimate.

- **A** describes the pre-sigmoid linear score (logit), not the model's output.
- **C** is wrong: logistic regression outputs a probability; a threshold converts it to a
  hard label, but that is a post-processing step.
- **D** is wrong: the softmax over 2 classes is mathematically equivalent to the sigmoid,
  but logistic regression specifically uses the sigmoid function. Softmax is used for K>2.

---

### Q3 - Answer: B

Lambda (or its inverse, C in sklearn's SVM) controls regularisation strength. A large
lambda heavily penalises large weights, producing a simpler model that underfits (high
bias, low variance). A small lambda allows large weights, producing a more complex model
that can overfit (low bias, high variance).

- **A** is wrong: lambda is not a learning rate. It penalises weight magnitude.
- **C** is wrong: that is the `n_estimators` parameter in ensemble methods.
- **D** is wrong: that is `min_samples_leaf` in decision trees.

---

### Q4 - Answer: B

K-means is an iterative algorithm alternating between assignment (assign each point to the
nearest centroid) and update (recompute centroids as the mean of assigned points). It
converges when the assignment step produces no changes -- all points stay in their current
cluster.

- **A** is wrong: inertia reaches zero only if k equals the number of data points, which
  is trivially useless. Normal convergence occurs when inertia stops changing, not when
  it reaches zero.
- **C** is wrong: k is a hyperparameter chosen by the user, not determined by class count.
- **D** is wrong: centroids at the global mean describes k=1 (one cluster for all data).

---

### Q5 - Answer: B

Bagging (Bootstrap Aggregating): trains multiple models in **parallel**, each on a
bootstrap sample (random sample with replacement). Predictions are averaged (regression)
or voted (classification). Reduces variance.

Boosting: trains models **sequentially**. Each model focuses on examples the previous
models got wrong (by re-weighting or fitting residuals). Reduces bias. AdaBoost re-weights
misclassified samples; Gradient Boosting fits residuals.

- **A** has the descriptions swapped.
- **C** is wrong: bagging works with any base learner, not just trees.
- **D** is wrong: boosting can overfit on noisy data; bagging can outperform boosting
  in some settings.

---

### Q6 - Answer: B

Support vectors are the training examples that lie ON or INSIDE the margin boundaries
(i.e., examples within the margin or misclassified examples for the soft-margin SVM).
These are the examples that determine the position of the decision boundary. All other
training examples (outside the margin) have no influence on the boundary. The name comes
from the fact that the optimal hyperplane can be expressed entirely as a function of
these "supporting" examples.

- **A** is wrong: magnitude of feature vectors is not the selection criterion.
- **C** is wrong: only the support vectors (a subset of training examples) define the
  boundary, not all training examples.
- **D** conflates SVM support vectors with PCA eigenvectors.

---

### Q7 - Answer: B

Training accuracy = 100 % with test accuracy = 60 % is a classic overfitting signature
(high variance). The tree has memorised the training set. The most direct fix is to
constrain tree complexity by limiting maximum depth or applying post-pruning.

- **A** would increase overfitting further.
- **C** removing regularisation makes the problem worse.
- **D** is wrong: a linear model would not necessarily outperform a properly regularised
  tree, especially if the decision boundary is non-linear.

---

### Q8 - Answer: C

In the RBF kernel, gamma controls the "bandwidth": how quickly the influence of a
training point drops off with distance. High gamma means only very close neighbours
have significant influence -- the decision boundary becomes highly non-linear and
wraps tightly around individual training points (low bias, high variance). Low gamma
produces a smoother, wider-influence decision boundary.

- **A** is wrong: high gamma produces a more complex boundary, not a smoother one.
- **B** is wrong: high gamma NARROWS the influence radius, not widens it.
- **D** is wrong: the RBF kernel never degenerates to linear.

---

### Q9 - Answer: A

In a standard bagged forest (all features at each split), the best feature tends to be
used at the top of every tree. The trees become correlated -- their errors are similar
because they are all dominated by the same strong feature. Averaging correlated trees
provides limited variance reduction.

By randomly subsampling features at each split, different trees are forced to explore
different subsets of the feature space. Their errors become less correlated. When
uncorrelated weak learners are averaged, variance reduction is maximised:
`Var(avg) = sigma^2/n` for n independent trees.

- **B** is wrong: fewer features at each split increases, not decreases, bias of
  individual trees. The ensemble may have slightly higher bias but much lower variance.
- **C** is wrong: it does substantially improve generalisation.
- **D** is wrong: the most predictive feature may still be selected but must compete
  with other randomly chosen features at each node.

---

### Q10 - Answer: B

Deeper trees in gradient boosting capture more complex feature interactions and memorise
training-specific patterns. This is the classic overfitting (high variance) symptom:
training loss falls but test loss rises. The model fits noise in the training set.

- **A** has the relationship backwards: deeper trees have higher variance and lower bias.
- **C** is wrong: overfitting depends on the dataset size and signal-to-noise ratio, not
  just max_depth. With enough data, max_depth=8 can generalise well.
- **D** is wrong: convergence of the boosting algorithm depends on learning rate and
  number of trees, not specifically on max_depth=3.

---

### Q11 - Answer: A

DBSCAN defines a core point as one with at least `min_samples` points within distance
`eps`, counting the point itself. A border point is within eps of a core point but is not
itself a core point. A noise point (outlier) is not within eps of any core point.

- **B** describes a point close to a centroid, which is a k-means concept, not DBSCAN.
- **C** is not a meaningful definition in DBSCAN.
- **D** describes a centroid, which DBSCAN does not compute.

---

### Q12 - Answer: B

The geometric explanation: the Lasso constraint set in 2D is a diamond (L1 ball), which
has sharp corners at (+/- c, 0) and (0, +/- c). The Ridge constraint set is a circle
(L2 ball) with no corners. The unconstrained OLS solution corresponds to the centre of
the loss ellipsoids. As we shrink the constraint set, the growing ellipsoids first
intersect the constraint boundary.

For the diamond, the corners are the "most prominent" parts of the boundary extending
in each coordinate direction. The loss ellipsoid typically hits a corner first, where
one coordinate is zero. For the circle, the first intersection can be anywhere on the
smooth boundary, which generically occurs at a point with no zero coordinates.

- **A** confuses L1 with a hypercube (L-infinity ball). L1 ball in 2D is a diamond.
- **C** is wrong: Lasso uses a standard subgradient or coordinate descent; there is no
  special non-linear transformation.
- **D** is wrong: the algorithm is not what causes sparsity; the geometry of the L1
  constraint is.

---

### Q13 - Answer: B

The kernel trick is the insight that many algorithms only need inner products between
examples, not the examples themselves. The kernel function K(x, x') computes this inner
product in an implicit (possibly infinite-dimensional) feature space:
```
K(x, x') = <phi(x), phi(x')>_H
```
where H is a Hilbert space (possibly infinite-dimensional, as with the RBF kernel).
The feature map phi is never computed explicitly -- only K is evaluated.

- **A** is wrong: the implicit feature space of the RBF kernel is infinite-dimensional.
- **C** is wrong: the RBF kernel is a valid Mercer kernel and corresponds to an inner
  product in a reproducing kernel Hilbert space (RKHS).
- **D** is wrong: the implicit feature space dimension does not equal the input dimension.

---

### Q14 - Answer: B

For MSE loss L = (1/2)(y - F(x))^2, the negative gradient with respect to F(x) is:
```
-dL/dF(x) = -((F(x) - y)) = y - F(x)
```
This is the residual: the difference between the true target and the current model
prediction. Fitting a tree to the residuals at each stage is why gradient boosting on
MSE is equivalent to iterative residual fitting.

- **A** is the positive gradient (wrong sign).
- **C** is the squared residual (wrong quantity).
- **D** is the negative residual (negative of the correct answer).

---

### Q15 - Answer: C

When n < p, the feature matrix X has rank at most n < p. The matrix X^T X (p x p) is
therefore singular (rank deficient) and has no unique inverse. OLS has infinitely many
solutions (any solution in the null space of X gives zero training error).

Adding L2 regularisation produces the Ridge problem:
```
w* = (X^T X + lambda * I)^{-1} X^T y
```
For any lambda > 0, (X^T X + lambda * I) is strictly positive definite and therefore
invertible. Ridge selects the minimum-norm solution among all zero-training-error
solutions, which typically generalises better than an arbitrary solution.

- **A** is partially correct (OLS will achieve zero training error in the underdetermined
  case because the system is consistent) but the full statement is misleading -- there
  are infinitely many solutions and OLS is not well-defined without additional criteria.
- **B** is wrong: OLS can be solved via pseudo-inverse (minimum-norm solution) even when
  X^T X is singular. The model technically trains; it just is not unique.
- **D** is wrong: p >> n causes high variance (overfitting), not high bias.
