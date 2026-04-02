# Model Selection

## Prerequisites
- Cross validation (`cross_validation.md`)
- Bias-variance trade-off
- Gradient-based optimisation basics

---

## Concept Reference

### What Model Selection Involves

Model selection covers two related problems:

1. **Architecture selection:** Choosing between fundamentally different model families
   (logistic regression vs random forest vs neural network).
2. **Hyperparameter tuning:** Choosing configuration values within a chosen architecture
   that are not learned by the training algorithm (learning rate, tree depth, regularisation
   strength, number of layers).

Both require evaluating candidate configurations using a held-out dataset or cross
validation. The key principle: **the data used to choose hyperparameters cannot also be
used to report the final generalisation estimate** (see nested CV in `cross_validation.md`).

---

### The Hyperparameter Landscape

Hyperparameters fall into two categories:

**Continuous:** Learning rate, regularisation lambda, dropout rate, momentum, weight decay.
These are best searched over log-scale ranges (e.g., 1e-5 to 1e-1 for learning rate)
because their effect on loss is multiplicative, not additive.

**Discrete/categorical:** Number of layers, number of units per layer, activation function,
optimiser type, kernel type. These define a combinatorial search space.

The joint search space over all hyperparameters is high-dimensional and non-convex. The
objective function (validation loss or CV metric) is:
- **Expensive to evaluate:** requires training a full model.
- **Noisy:** CV scores have variance due to random initialisation and data shuffling.
- **Black-box:** no gradient is available with respect to hyperparameters (in general).

This rules out standard gradient descent and motivates the search strategies below.

---

### Grid Search

Exhaustively evaluate every combination of a discrete set of candidate values for each
hyperparameter.

```
learning_rate: [0.001, 0.01, 0.1]
weight_decay:  [1e-4, 1e-3, 1e-2]
num_layers:    [2, 3, 4]

Grid size = 3 * 3 * 3 = 27 configurations
```

**Advantages:**
- Reproducible and easy to parallelise.
- Guaranteed to evaluate all specified combinations.

**Disadvantages:**
- Scales exponentially with the number of hyperparameters (the "curse of dimensionality").
- All evaluations of a given hyperparameter use only the discrete values specified -- you
  miss the continuous landscape between points.
- Wastes budget: if `num_layers` has little effect, you still evaluate it at every value
  for every combination of the other parameters.

Grid search is practical when you have at most 3-4 hyperparameters and small grids.

---

### Random Search

Sample hyperparameter configurations uniformly at random from a specified range (or
distribution) for each hyperparameter, independently. Evaluate a fixed budget of
configurations.

```python
# Random search: sample from continuous distributions
learning_rate = 10 ** np.random.uniform(-5, -1)  # Log-uniform in [1e-5, 0.1]
weight_decay  = 10 ** np.random.uniform(-5, -2)  # Log-uniform in [1e-5, 1e-2]
num_layers    = np.random.randint(1, 6)           # Uniform integer in [1, 5]
```

**Key result (Bergstra & Bengio, 2012):** For a fixed budget of n evaluations, random
search finds better configurations than grid search when some hyperparameters have much
more influence on performance than others (which is almost always true in practice). This
is because random search gives each important hyperparameter n distinct values, whereas
grid search gives it only the number of points in its grid dimension.

**Advantages:**
- Simple to implement.
- Naturally handles continuous ranges.
- No need to pre-specify a discrete grid.
- Embarrassingly parallel.

**Disadvantages:**
- Does not use information from past evaluations to guide future ones.
- With a limited budget, may not find the optimum if the search space is large.

Random search is the default baseline for hyperparameter tuning and almost always
outperforms grid search for the same compute budget.

---

### Bayesian Optimisation

Uses the history of (configuration, score) pairs to build a **surrogate model** of the
objective function, then uses an **acquisition function** to choose which configuration
to evaluate next, trading off exploration (high uncertainty regions) and exploitation
(regions the surrogate predicts to be good).

```
Repeat for budget B:
  1. Fit surrogate model on {(x_i, y_i)} history.
  2. Maximise acquisition function -> select next config x_next.
  3. Evaluate objective: y_next = f(x_next).  [expensive: train a model]
  4. Add (x_next, y_next) to history.
```

**Common surrogate models:**
- **Gaussian Process (GP):** Models the objective as a sample from a Gaussian process.
  Provides calibrated uncertainty estimates. Scales as O(n^3) in the number of
  observations; impractical beyond ~200 evaluations.
- **Random Forest / Tree Parzen Estimator (TPE):** Handles categorical and conditional
  hyperparameters well. Used by Hyperopt and Optuna. Scales to larger budgets.

**Common acquisition functions:**
- **Expected Improvement (EI):** Expected amount by which the next point will improve
  over the current best.
- **Upper Confidence Bound (UCB):** Balances mean prediction and uncertainty with a
  tunable exploration weight.
- **Probability of Improvement (PI):** Probability that the next point exceeds the
  current best.

**Advantages over random search:**
- Makes sequential decisions informed by all previous evaluations.
- Significantly more sample-efficient for expensive black-box objectives.
- Can find better solutions in fewer total evaluations.

**Disadvantages:**
- Sequential by design -- parallelisation requires asynchronous or batch acquisition
  variants (e.g., batch EI, qNEI).
- Surrogate model assumptions may not hold; performance depends on the surrogate quality.
- More complex to implement and debug.

**When to use Bayesian optimisation:**
When each evaluation takes hours (training a large model) and your total compute budget
allows only 20-100 evaluations. For cheap models evaluated in seconds, random search with
a larger budget typically matches or beats Bayesian optimisation.

---

### Successive Halving and Hyperband

These methods allocate resources (training epochs, data size) adaptively across
configurations:

**Successive Halving:**
1. Start with n configurations, each given budget b (e.g., 10 epochs).
2. Evaluate all n. Keep the top 1/eta fraction (e.g., eta=3 keeps top third).
3. Give survivors budget eta*b (e.g., 30 epochs). Repeat until one survives.

**Hyperband:** Runs multiple brackets of successive halving with different initial budgets
and n values, providing a principled trade-off between exploration and resource allocation.

**Advantage:** Dramatically reduces total compute vs training every configuration fully.
A bad configuration is eliminated early; only promising ones receive full training.
**Used by:** Ray Tune, Optuna (via ASHA scheduler), SageMaker.

---

### Learning Curves and Validation Curves

Before committing to expensive hyperparameter search, use diagnostic curves:

**Learning curve:** Plot training and validation metric vs training set size.
- Both metrics converging at similar values: likely high bias (underfitting). More data
  will not help; add model capacity.
- Large gap between train and val: high variance (overfitting). Add regularisation or
  more data.

**Validation curve:** Plot training and validation metric vs a single hyperparameter
(e.g., regularisation strength or tree depth).
- Identifies the rough region of a good hyperparameter value.
- Provides a 1D cross-section of the search space at low cost.
- Use this before grid/random search to narrow the search range.

---

## Interview Questions by Difficulty

### Fundamentals

**Q1.** What is the difference between a hyperparameter and a model parameter?

**Answer:**

A **model parameter** (e.g., neural network weights, linear regression coefficients) is
learned from training data by minimising a loss function. It changes during training and
is specific to a trained model instance.

A **hyperparameter** (e.g., learning rate, number of layers, regularisation lambda) is
a configuration value set *before* training begins. It controls how the training algorithm
or model architecture behaves but is not directly optimised by the training loss. Choosing
hyperparameters requires a separate evaluation procedure (CV or a held-out validation set)
because the training loss cannot be used -- a model with very high capacity will always
achieve lower training loss regardless of whether it generalises.

---

**Q2.** You have 4 hyperparameters, each with 5 candidate values. How many evaluations
does grid search require? How many does random search require for the same coverage?

**Answer:**

Grid search: 5^4 = 625 evaluations.

Random search: Bergstra & Bengio (2012) show that n random samples cover each
hyperparameter's range as well as a grid of n points for any number of hyperparameters
(because each sample is independent in each dimension). For 4 hyperparameters with 5
values each, random search needs only ~5 evaluations to cover each individual dimension
as well as the grid does. In practice, 60 random evaluations give >95 % probability of
finding a configuration within the top 5 % for any single hyperparameter dimension.
For the same budget (625), random search explores a much denser coverage of the
important hyperparameters.

---

### Intermediate

**Q3.** Describe Bayesian optimisation with a Gaussian Process surrogate. What is the
role of the acquisition function? What is its main limitation at scale?

**Answer:**

**Surrogate model:** A Gaussian Process (GP) is fitted to the observed (config, score)
pairs. A GP provides a mean prediction and an uncertainty estimate (variance) at every
unobserved point in the hyperparameter space. Points near observed evaluations have low
uncertainty; unexplored regions have high uncertainty.

**Acquisition function:** At each step, instead of evaluating the expensive true objective
(training a model), we maximise the cheap acquisition function over the GP. Expected
Improvement (EI) is a common choice:

```
EI(x) = E[max(f(x) - f_best, 0)]
```

This quantifies how much better than the current best we expect x to be. Points with
high predicted mean (exploitation) or high uncertainty (exploration) both score well.

**Main limitation at scale:** GP inference requires solving an N x N linear system
(Cholesky factorisation), which scales as O(N^3) in time and O(N^2) in memory, where N
is the number of past evaluations. Beyond ~200-500 evaluations the GP becomes
computationally prohibitive. Alternative surrogates (TPE, random forests) are used for
larger budgets.

---

**Q4.** Your validation loss curve shows that training loss decreases to 0.05 while
validation loss plateaus at 0.35 after epoch 20. What does this indicate and what are
three actions you can take?

**Answer:**

The large gap between training loss (0.05) and validation loss (0.35) indicates severe
**overfitting** (high variance). The model has memorised the training data and fails to
generalise.

Three actions:
1. **Add regularisation:** Increase weight decay (L2), add dropout layers, or use
   early stopping (halt training at the epoch of minimum validation loss).
2. **Reduce model capacity:** Decrease the number of layers or units per layer. The model
   has more parameters than the data can reliably constrain.
3. **Increase training data:** Collect more labelled examples, or apply data augmentation.
   More data makes it harder for the model to memorise specific training instances.

Additionally, check whether the train/val split is representative (stratified split) and
whether there is data leakage from preprocessing applied to the full dataset before
splitting.

---

### Advanced

**Q5.** Compare successive halving with Bayesian optimisation. When would you choose
each approach for tuning a large language model fine-tuning job?

**Answer:**

**Successive halving (Hyperband/ASHA):**
- Allocates compute adaptively: eliminates bad configs early, concentrates budget on
  promising ones.
- Assumes performance at low resource (few epochs) is correlated with final performance
  -- the "early elimination" assumption. This holds for smooth training curves but can
  fail when some configurations require longer to warm up.
- Naturally parallel: all configurations in a round can train simultaneously.
- Does not model the objective function; each evaluation is independent.

**Bayesian optimisation:**
- Builds a model of the objective and directs search intelligently.
- More sample-efficient when each full evaluation is expensive and there are a small number
  of evaluations possible (<100).
- Inherently sequential (each step uses all prior observations). Parallel variants exist
  but add complexity.
- Does not reduce individual evaluation cost; you still train for the full budget.

**For large language model fine-tuning:**
Large models take hours or days per run. The total budget may be 20-50 full fine-tuning
runs. Bayesian optimisation (with TPE or GP) is appropriate because:
- The evaluation budget is small enough for Bayesian methods to be sample-efficient.
- Each run is too expensive to waste on uninformed random exploration.
- A hybrid approach (Bayesian BO + ASHA scheduling) using tools like Optuna or Ray Tune
  with an ASHA pruner can combine both advantages: BO proposes configurations and ASHA
  prunes underperforming ones early in training.

For small models with cheap evaluations (seconds per run), random search or grid search
with a large budget is sufficient and much simpler to implement and debug.
