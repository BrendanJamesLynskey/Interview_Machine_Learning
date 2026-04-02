# Supervised vs Unsupervised Learning

## Prerequisites
- Basic probability: joint, conditional, marginal distributions
- Familiarity with common ML datasets (MNIST, ImageNet, UCI repository)
- Understanding of the train/validation/test split paradigm

---

## Concept Reference

### The Core Distinction

The fundamental difference between learning paradigms is whether a **label signal** exists during training.

| Paradigm | Training data | Goal | Example tasks |
|---|---|---|---|
| Supervised | $(x_i, y_i)$ pairs | Learn $f: X \to Y$ | Classification, regression |
| Unsupervised | $x_i$ only | Discover structure in $P(X)$ | Clustering, density estimation, dimensionality reduction |
| Self-supervised | $x_i$ only (labels derived from data) | Learn rich representations | Masked language modelling, contrastive image learning |
| Semi-supervised | Few $(x_i, y_i)$, many $x_i$ | Exploit unlabelled data | Label propagation, pseudo-labelling |
| Reinforcement learning | State-action-reward tuples | Learn a policy | Game playing, robotics |

### Supervised Learning

The model observes $n$ training examples $\{(x_i, y_i)\}_{i=1}^{n}$ and minimises an empirical risk:

$$\hat{\theta} = \arg\min_\theta \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f_\theta(x_i),\, y_i) + \lambda\, \Omega(\theta)$$

where $\mathcal{L}$ is a task-specific loss and $\Omega$ is a regularisation term.

**Subcategories:**
- **Classification**: $y \in \{1, \dots, K\}$ -- binary or multi-class
- **Regression**: $y \in \mathbb{R}$ (or $\mathbb{R}^d$)
- **Structured prediction**: $y$ is a sequence, tree, or graph (e.g., parsing, segmentation)

**Core assumption -- i.i.d.**: Training and test examples are drawn independently from the same distribution $P(X, Y)$. Violation of this causes *distribution shift*.

### Unsupervised Learning

No labels are provided. The model must find structure in $P(X)$ alone.

**Major categories:**

1. **Clustering** -- partition data into groups (K-Means, DBSCAN, GMM)
2. **Dimensionality reduction** -- find a lower-dimensional manifold (PCA, t-SNE, UMAP, autoencoders)
3. **Density estimation** -- model $P(X)$ explicitly (KDE, normalising flows, VAEs)
4. **Generative modelling** -- learn to sample from $P(X)$ (GANs, diffusion models)
5. **Association rule mining** -- find frequent co-occurrences (Apriori algorithm)

Evaluation is harder than in supervised learning because there is no ground-truth label to compute accuracy against. Metrics like silhouette score or reconstruction error serve as proxies.

### Self-Supervised Learning

Self-supervised learning is a form of unsupervised learning where labels are **constructed automatically from the data** by solving a proxy task (a *pretext task*). The learned representations are then transferred to downstream supervised tasks.

**Examples of pretext tasks:**
- **Masked Language Modelling (BERT)**: predict masked tokens from context
- **Next sentence prediction**: predict whether two sentences are consecutive
- **Contrastive learning (SimCLR, MoCo)**: learn that two augmented views of the same image are similar, and views of different images are dissimilar
- **Rotation prediction**: predict which rotation (0°, 90°, 180°, 270°) was applied to an image
- **Jigsaw puzzle solving**: predict the permutation of image patches

**Why it matters**: Self-supervised pre-training enables learning from internet-scale unlabelled data, producing representations that generalise across many downstream tasks with little labelled data. GPT, BERT, CLIP, and DINO all rely on this paradigm.

### Semi-Supervised Learning

Only a small fraction of data has labels. The objective combines supervised loss on labelled data and unsupervised loss on all data:

$$\mathcal{L} = \mathcal{L}_{\text{sup}}(\mathcal{D}_L) + \lambda\, \mathcal{L}_{\text{unsup}}(\mathcal{D}_L \cup \mathcal{D}_U)$$

**Common techniques:**
- **Pseudo-labelling**: train on labelled data, predict labels for unlabelled data, retrain on both
- **Label propagation**: spread labels through a graph built on feature similarity
- **Consistency regularisation**: enforce that the model outputs the same prediction for perturbed versions of the same input (MixMatch, FixMatch)
- **Self-training**: iterative pseudo-labelling with confidence thresholding

---

## Tier 1 -- Fundamentals

### Q1. What is the difference between supervised and unsupervised learning?

**Answer:**

In supervised learning, each training example has an associated label $y_i$. The model learns a mapping $f: X \to Y$ by minimising a loss between predictions and true labels. A separate held-out test set with known labels measures generalisation performance.

In unsupervised learning, no labels are provided. The model explores the structure of the input distribution $P(X)$ -- finding clusters, low-dimensional representations, or ways to generate new samples.

The practical implication is labelling cost. Labels often require human annotation (expensive, slow, sometimes inconsistent). Unsupervised methods can exploit vast quantities of raw data that would be infeasible to label.

**Common mistake**: Students sometimes say "unsupervised learning has no feedback." This is imprecise -- unsupervised methods do use feedback signals (e.g., reconstruction loss in autoencoders, intra-cluster cohesion in K-Means), but that signal comes from the data itself rather than human-assigned labels.

---

### Q2. Give three concrete examples of supervised learning tasks and three of unsupervised learning tasks, explaining what the inputs and outputs are in each case.

**Answer:**

**Supervised:**

| Task | Input $x$ | Label $y$ | Model output |
|---|---|---|---|
| Email spam detection | TF-IDF vector of email text | `{spam, not_spam}` | Probability of spam |
| House price prediction | Size, location, rooms, age | Sale price in $ | Real-valued price estimate |
| Medical image diagnosis | Chest X-ray pixel array | `{normal, pneumonia}` | Diagnosis probability |

**Unsupervised:**

| Task | Input $x$ | No labels | Goal |
|---|---|---|---|
| Customer segmentation | Purchase history, demographics | -- | Groups of similar customers |
| Topic modelling (LDA) | Bag-of-words document vectors | -- | Latent topics per document |
| Anomaly detection | Network packet features | -- | Identify outliers from normal distribution |

---

### Q3. What is a label and what makes labelled data expensive to obtain?

**Answer:**

A label is a human-assigned annotation that specifies the ground truth for a training example. Labelled data is expensive because:

1. **Expert time**: Medical diagnoses, legal document classification, and scientific data annotation require domain experts who are scarce and costly.
2. **Volume**: Even at $\$0.01$ per label, labelling 1 million examples costs $\$10,000$.
3. **Inter-annotator disagreement**: Ambiguous examples require multiple annotators and adjudication protocols.
4. **Consistency**: Label definitions drift over time or between annotators without rigorous guidelines.
5. **Privacy and ethics**: Some data (medical, financial) cannot be shared with external annotation services.

This cost motivates semi-supervised learning, data augmentation, transfer learning, and active learning (query only the most informative unlabelled examples).

---

### Q4. What does "i.i.d." mean, and why does violating it cause problems?

**Answer:**

i.i.d. stands for **independently and identically distributed**. Training and test examples are assumed to be:
- **Independent**: the label or features of one example provide no information about another
- **Identically distributed**: all examples are drawn from the same joint distribution $P(X, Y)$

**Violations and consequences:**

| Violation | Name | Example | Consequence |
|---|---|---|---|
| $P_{\text{train}}(X) \neq P_{\text{test}}(X)$ | Covariate shift | Train on daytime photos, test on night photos | Feature representations no longer reliable |
| $P_{\text{train}}(Y \mid X) \neq P_{\text{test}}(Y \mid X)$ | Concept drift | Spam tactics evolve over time | Decision boundary becomes stale |
| $P_{\text{train}}(Y) \neq P_{\text{test}}(Y)$ | Label shift | Clinical trial vs. general population | Calibration fails |
| Examples correlated | Non-independence | Sequential time-series, patient with multiple records | Test leakage, inflated performance metrics |

---

## Tier 2 -- Intermediate

### Q5. Explain self-supervised learning and describe how contrastive learning works. Why has this paradigm become so important?

**Answer:**

Self-supervised learning creates supervisory signals automatically from unlabelled data by defining a pretext task. The model learns representations that are useful for downstream tasks without any human labels.

**Contrastive learning (SimCLR framework):**

Given an image $x$, two independent augmentations are sampled: $\tilde{x}_i$ and $\tilde{x}_j$ (positive pair). A batch of $N$ images produces $2N$ augmented views. For each view, a representation $z = g(f(x))$ is computed. The **NT-Xent loss** (normalised temperature-scaled cross-entropy) encourages positive pairs to be similar and negative pairs to be dissimilar:

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(u, v) = \frac{u^\top v}{\|u\|\|v\|}$ is cosine similarity and $\tau$ is a temperature hyperparameter.

**Why it matters:**
1. Internet-scale data is unlabelled -- self-supervised models can train on billions of images or tokens
2. Representations learned from broad pretraining generalise to many downstream tasks with few labelled examples
3. State-of-the-art vision (DINO, DINOv2), NLP (BERT, GPT), and multimodal (CLIP) models all rely on this paradigm

**Key insight**: The model cannot trivially solve contrastive tasks by memorising colour or texture statistics -- it must learn semantic invariances to correctly identify that two augmented views come from the same image.

---

### Q6. Compare semi-supervised learning with transfer learning. When would you choose each?

**Answer:**

**Semi-supervised learning:**
- Leverages **unlabelled data from the same domain and task** as the labelled data
- The unlabelled data directly influences the learned hypothesis
- Typical setting: 100 labelled examples, 100,000 unlabelled examples, same domain
- Methods: pseudo-labelling, label propagation, consistency regularisation

**Transfer learning:**
- Leverages a model pre-trained on a **different (usually larger) dataset**
- The pre-trained model provides an initialisation; fine-tuning adapts it to the target task
- Typical setting: pre-train on ImageNet, fine-tune on a medical imaging dataset with 500 labelled examples
- Methods: feature extraction (freeze backbone), full fine-tuning, adapter layers

**Decision guidance:**

| Condition | Prefer |
|---|---|
| Large unlabelled corpus in target domain, small labelled set | Semi-supervised |
| A strong pre-trained model exists for a related domain | Transfer learning |
| Target domain is very different from any available pre-trained domain | Semi-supervised (transfer unhelpful) |
| Compute budget is very tight | Transfer + freeze backbone |
| Both unlabelled target data and pre-trained model available | Combine both (e.g., fine-tune with pseudo-labels) |

---

### Q7. What is the manifold hypothesis and why is it relevant to unsupervised learning?

**Answer:**

The **manifold hypothesis** states that real-world high-dimensional data (images, audio, text) does not uniformly occupy its nominal high-dimensional space. Instead, data points lie on or near a low-dimensional manifold embedded in the high-dimensional space.

For example, the space of all $28 \times 28$ pixel images has $784$ dimensions, but the set of images that look like handwritten digits occupies a much smaller manifold within this space -- parameterised by stroke width, digit identity, slant, position, and a handful of other factors.

**Implications for unsupervised learning:**

1. **Dimensionality reduction is meaningful**: PCA, t-SNE, and UMAP can find low-dimensional coordinates that capture the true degrees of variation without losing semantic information.
2. **Generalisation is possible**: Nearby points on the manifold represent semantically similar examples. A model that learns the manifold structure can interpolate and extrapolate sensibly.
3. **Density estimation is tractable**: The intrinsic dimensionality of data is far lower than the nominal dimensionality, making density estimation feasible with models like VAEs or normalising flows.
4. **Clustering assumptions**: If clusters correspond to distinct sub-manifolds, geometric clustering methods (DBSCAN) will find them more reliably than methods that assume spherical clusters (K-Means).

**Failure mode**: The manifold hypothesis is an assumption. For genuinely high-dimensional, noisy data (e.g., genomics data with independent features), it may not hold, and methods that rely on it will give misleading results.

---

## Tier 3 -- Advanced

### Q8. A company has 10 million user clickstream events but only 50,000 have expert-assigned purchase intent labels. Describe a complete learning strategy, explaining the role of each component.

**Answer:**

This is a classic semi-supervised learning problem with the option to use self-supervised pre-training.

**Step 1 -- Self-supervised pre-training on all 10M events**

Design a pretext task on the raw clickstream data. For sequential data, masked event prediction works well: randomly mask 15% of events in a session and train a transformer to predict the masked event type and duration from context. This forces the model to learn temporal patterns, session structure, and item affinity without any labels.

**Step 2 -- Supervised fine-tuning on 50K labelled examples**

Replace the pretext head with a binary classification head (purchase intent). Fine-tune the entire network on labelled data with cross-entropy loss. The pre-trained representation reduces the labelled data needed because the model has already learned useful structure.

**Step 3 -- Pseudo-labelling on unlabelled data**

After initial fine-tuning, run inference on all 10M events. Retain predictions with confidence above a threshold (e.g., $p > 0.95$) as pseudo-labels. Retrain or continue fine-tuning on labelled + pseudo-labelled data. This is an expectation-maximisation style procedure.

**Step 4 -- Consistency regularisation**

Apply MixMatch or FixMatch: for each unlabelled example, compute predictions under multiple augmentations (e.g., random subsequence masking, feature dropout). Penalise the model if predictions are inconsistent across augmentations. This encourages confident, stable predictions in low-density regions.

**Step 5 -- Calibration and evaluation**

Evaluate on a held-out labelled set using AUC-ROC and calibration curves. Monitor pseudo-label quality by sampling pseudo-labelled examples for human review. Retract low-quality pseudo-labels.

**Key risks:**
- Confirmation bias: aggressive pseudo-labelling propagates early model errors. Mitigate with a high confidence threshold and periodic human review.
- Distribution shift: the 50K labelled examples may not represent the full distribution of 10M unlabelled events. Stratify sampling during labelling to cover the input space.

---

### Q9. Explain the difference between generative and discriminative models. When does a generative model outperform a discriminative one?

**Answer:**

**Discriminative model**: learns $P(Y \mid X)$ directly -- the conditional distribution of labels given inputs. Examples: logistic regression, SVM, feedforward neural networks. Directly optimised for the classification boundary.

**Generative model**: learns the joint distribution $P(X, Y) = P(X \mid Y) P(Y)$, or equivalently $P(X)$ in the unsupervised case. Examples: Naive Bayes, GMM, VAE, GAN, diffusion models. Classification uses Bayes' rule: $P(Y \mid X) \propto P(X \mid Y) P(Y)$.

**When generative models outperform discriminative ones:**

1. **Small labelled data regime**: Generative models incorporate the full input distribution $P(X)$, which can be estimated from unlabelled data. With few labels, this auxiliary information reduces variance.

2. **Robustness to missing features**: A generative model can marginalise over missing features using $P(X)$; a discriminative model has no principled way to handle inputs with missing dimensions at test time.

3. **Out-of-distribution detection**: A generative model assigns low likelihood to inputs far from the training distribution. Discriminative models can be confidently wrong on OOD inputs.

4. **Data generation**: If the downstream task requires synthesising new examples (data augmentation, simulation, imputation), a generative model is necessary.

5. **Class imbalance**: With very few examples of a rare class, modelling $P(X \mid Y = \text{rare})$ and using Bayes' rule may outperform trying to directly estimate $P(Y = \text{rare} \mid X)$ from very few positive examples.

**When discriminative models win**: With large amounts of labelled data, discriminative models are asymptotically more efficient -- they make fewer modelling assumptions and focus capacity directly on the decision boundary. Naive Bayes (generative) is typically beaten by logistic regression (discriminative) when training data is plentiful because feature independence is rarely satisfied.

---

## Quick Reference Quiz

**Q: Which of the following is a self-supervised learning task?**

A) Predicting house prices from features and historical sales data  
B) Predicting the next word in a sentence using only the sentence itself  
C) Grouping customers by purchase frequency with K-Means  
D) Learning a policy to win at chess through trial and error  

**Answer: B.** The supervisory signal (next word) is derived automatically from the input text -- no human labels are required. (A) is supervised. (C) is unsupervised. (D) is reinforcement learning.

---

**Q: A model trained on daytime driving images performs poorly at night. This is an example of which problem?**

A) Overfitting  
B) Underfitting  
C) Covariate shift  
D) Label noise  

**Answer: C.** The input distribution $P(X)$ has changed between train and test -- covariate shift. The model architecture may be perfectly well-fitted to training data; the issue is distributional mismatch.

---

**Q: Which statement about pseudo-labelling is TRUE?**

A) Pseudo-labels are always more accurate than human labels  
B) Pseudo-labelling can suffer from confirmation bias if the initial model is poor  
C) Pseudo-labelling requires that all unlabelled examples be included regardless of confidence  
D) Pseudo-labelling is a form of transfer learning  

**Answer: B.** Confirmation bias is the key failure mode of pseudo-labelling: a model that makes early systematic errors will assign wrong pseudo-labels, which reinforce those errors on the next training round. This is mitigated by high confidence thresholds and iterative refinement.
