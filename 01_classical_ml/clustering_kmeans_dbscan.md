# Clustering: K-Means and DBSCAN

## Prerequisites
- Euclidean and other distance metrics
- Basic probability: Gaussian mixture models (helpful context)
- Understanding of the bias-variance trade-off (for hyperparameter choice)

---

## Concept Reference

### What is Clustering?

Clustering is an unsupervised learning task: partition a dataset $\{x_i\}_{i=1}^n$ into $K$ groups (clusters) such that points within a cluster are more similar to each other than to points in other clusters. There is no ground-truth label; the notion of "correct" partitioning is task-dependent.

### K-Means Algorithm

K-Means minimises the **within-cluster sum of squares (WCSS)** -- also called inertia:

$$\mathcal{L}(C, \mu) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where $C_k$ is cluster $k$ and $\mu_k$ is its centroid.

**Algorithm (Lloyd's algorithm):**

1. **Initialise**: choose $K$ cluster centroids $\mu_1, \ldots, \mu_K$ (see initialisation strategies below)
2. **Assignment step**: assign each point to the nearest centroid:

$$c_i = \arg\min_k \|x_i - \mu_k\|^2$$

3. **Update step**: recompute each centroid as the mean of its assigned points:

$$\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$$

4. **Repeat** steps 2--3 until convergence (assignments do not change)

**Convergence**: Lloyd's algorithm converges in finite iterations because there are finitely many partitions of $n$ points and WCSS strictly decreases at each step (or stays the same when convergence is reached). However, it converges to a **local minimum**, not necessarily the global minimum.

**Computational complexity**: each iteration is $O(nKd)$ where $d$ is the feature dimension. The number of iterations is typically $O(100)$ in practice.

**K-Means++  Initialisation:**

Choose centroids sequentially: the first centroid is chosen uniformly at random; each subsequent centroid $\mu_k$ is chosen with probability proportional to its squared distance from the nearest existing centroid:

$$P(\text{choose } x_i) \propto \min_{j < k} \|x_i - \mu_j\|^2$$

This spreads the initial centroids and guarantees an expected WCSS within $O(\log K)$ of the optimal. It is the default initialisation in scikit-learn.

**Limitations of K-Means:**
1. Assumes clusters are convex and isotropic (spherical) -- fails for elongated or non-convex clusters
2. Sensitive to outliers: a single outlier can pull a centroid away from the true cluster mean
3. Requires specifying $K$ in advance
4. Distance-based: sensitive to feature scale (standardise features)
5. Non-deterministic: different initialisations give different solutions

### Choosing K -- The Elbow Method

Run K-Means for $K = 1, 2, \ldots, K_{\max}$ and plot WCSS against $K$. The WCSS decreases monotonically with $K$ (more clusters always reduces inertia). Look for an "elbow" -- a point where the marginal decrease in WCSS becomes small:

**Formalised elbow**: fit a piecewise linear function to the WCSS vs. $K$ curve and find the breakpoint. In practice, the elbow is often ambiguous on real data.

**Limitation of the elbow method**: if the data has no strong cluster structure, the WCSS curve decays smoothly with no clear elbow. This is a fundamental limitation -- elbow is a heuristic, not a rigorous criterion.

### Silhouette Score

The silhouette score provides a per-point quality measure that does not require knowing the true clusters:

For point $x_i$ in cluster $C_k$:

$$a(i) = \frac{1}{|C_k| - 1} \sum_{j \in C_k, j \neq i} d(x_i, x_j) \quad \text{(mean intra-cluster distance)}$$

$$b(i) = \min_{k' \neq k} \frac{1}{|C_{k'}|} \sum_{j \in C_{k'}} d(x_i, x_j) \quad \text{(mean distance to nearest other cluster)}$$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Interpretation:**
- $s(i) \approx 1$: point is well-matched to its own cluster, far from others
- $s(i) \approx 0$: point is on the boundary between two clusters
- $s(i) < 0$: point may be misclassified (closer to another cluster than its own)

The **mean silhouette score** over all points can be used to compare different $K$ values. Choose $K$ that maximises the mean silhouette score.

**Advantage over elbow method**: the silhouette score accounts for both cohesion and separation, and works for comparing different algorithms (not just different $K$ values).

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN does not require specifying $K$ in advance. It finds clusters as dense regions separated by low-density regions, and explicitly identifies **noise points** (outliers).

**Two hyperparameters:**
- $\varepsilon$ (eps): radius of the neighbourhood around a point
- $\text{MinPts}$: minimum number of points in an $\varepsilon$-neighbourhood to form a dense region

**Point classifications:**
- **Core point**: at least $\text{MinPts}$ points within distance $\varepsilon$ (including itself)
- **Border point**: fewer than $\text{MinPts}$ neighbours but within $\varepsilon$ of a core point
- **Noise point**: not a core point and not within $\varepsilon$ of any core point

**Algorithm:**

1. For each unvisited point $x_i$:
   a. If $x_i$ is a core point, start a new cluster and expand it:
   - Add all points within $\varepsilon$ of $x_i$ to the cluster
   - For each newly added core point, recursively add their $\varepsilon$-neighbours
   b. If $x_i$ is a border point, assign it to the cluster of a nearby core point
   c. If $x_i$ is a noise point, mark as noise (may later be reassigned as a border point)

**Key properties:**
- Clusters can be arbitrary shape (non-convex, elongated, ring-shaped)
- Automatically determines the number of clusters from the data
- Identifies outliers explicitly as noise points
- Deterministic given the same data and hyperparameters (unlike K-Means)

**Computational complexity**: $O(n^2)$ naively; $O(n \log n)$ with spatial indexing (k-d tree or ball tree for low dimensions).

**Choosing hyperparameters:**

For $\varepsilon$: compute the distance from each point to its $\text{MinPts}$-th nearest neighbour and plot these sorted distances (a $k$-distance graph). The elbow in this graph suggests a good $\varepsilon$.

For $\text{MinPts}$: a common rule of thumb is $\text{MinPts} \geq d + 1$ where $d$ is the dimension, or $\text{MinPts} \geq 2d$ for noisier datasets. Larger $\text{MinPts}$ requires denser core regions and produces fewer, more robust clusters.

**Limitations of DBSCAN:**
1. Struggles when clusters have widely varying densities -- a single $\varepsilon$ cannot define "dense" for all clusters simultaneously
2. Does not perform well in very high dimensions: all distances tend towards the same value (concentration of measure), making $\varepsilon$ hard to tune
3. Border point assignment is non-deterministic in the original paper (scikit-learn assigns them based on the order of processing); this can affect results with very similar densities

### Comparison: K-Means vs. DBSCAN

| Property | K-Means | DBSCAN |
|---|---|---|
| Number of clusters | Must specify $K$ | Determined automatically |
| Cluster shape | Convex, spherical | Arbitrary |
| Outlier handling | Assigns all points to clusters | Explicitly labels noise |
| Cluster density | Assumes uniform | Can handle varying (with care) |
| Scalability | $O(nKd)$ per iteration | $O(n^2)$ naive, $O(n\log n)$ with index |
| Determinism | Non-deterministic (random init) | Deterministic |
| Feature sensitivity | Euclidean distance only | Any metric (via distance matrix) |
| When to use | Known $K$, roughly spherical clusters | Unknown $K$, arbitrary shapes, noise present |

---

## Tier 1 -- Fundamentals

### Q1. Describe the K-Means algorithm step by step. What is the objective function being minimised?

**Answer:**

K-Means minimises the within-cluster sum of squares (WCSS / inertia):

$$\mathcal{L} = \sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i - \mu_k\|^2$$

**Algorithm:**

**Step 1 -- Initialisation**: place $K$ centroids. Naively, pick $K$ random training points. Better: use K-Means++ which picks each centroid proportionally to its squared distance from the nearest existing centroid, spreading them across the data.

**Step 2 -- Assignment**: for every point $x_i$, find the nearest centroid:

$$c_i \leftarrow \arg\min_k \|x_i - \mu_k\|^2$$

This is the E-step (expectation) of the EM analogy.

**Step 3 -- Update**: recompute each centroid as the mean of all its assigned points:

$$\mu_k \leftarrow \frac{1}{|C_k|}\sum_{i:\, c_i = k} x_i$$

This is the M-step (maximisation) of the EM analogy.

**Step 4 -- Convergence check**: if no assignments changed between iterations, stop. Otherwise return to Step 2.

**Guarantee**: WCSS decreases (or stays constant) at each iteration. The algorithm terminates in finite steps. However, convergence is only to a local minimum -- results depend on initialisation. Standard practice: run K-Means multiple times (e.g., 10 runs) with different initialisations and keep the result with lowest WCSS.

---

### Q2. What are the main differences between K-Means and DBSCAN? Give a concrete example where K-Means fails but DBSCAN succeeds.

**Answer:**

Key differences:

**K-Means**: assumes clusters are convex, roughly spherical, and equally sized. All points are assigned to a cluster. Requires $K$ as input.

**DBSCAN**: finds clusters of arbitrary shape based on density. Naturally handles noise (outliers are labelled, not force-assigned). Discovers $K$ automatically.

**Example where K-Means fails but DBSCAN succeeds:**

Consider two concentric rings (annuli) in 2D: an inner ring of points at radius 1 and an outer ring at radius 3. The true structure has $K = 2$ clusters (inner ring, outer ring).

K-Means with $K = 2$ will produce two half-moon shaped clusters rather than the two rings. Because the centroids are constrained to be the cluster means, the mean of a ring is the origin (inside both rings) -- no centroid can sit at the centre of a ring without capturing both. K-Means will instead split the data along a diameter (e.g., left half and right half of both rings combined).

DBSCAN with appropriate $\varepsilon$ (larger than the gap between adjacent ring points, smaller than the inter-ring gap) correctly identifies the two rings as two clusters.

---

### Q3. What is the silhouette score and how is it used to compare clustering solutions?

**Answer:**

The silhouette score for point $x_i$ measures how well it fits its assigned cluster relative to other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is the mean distance to other points in the same cluster (cohesion) and $b(i)$ is the mean distance to the nearest other cluster (separation).

The **mean silhouette score** $\bar{s} = \frac{1}{n}\sum_i s(i)$ summarises the overall quality.

**Usage for model selection:**

Run K-Means (or another algorithm) for $K \in \{2, 3, \ldots, K_{\max}\}$. Compute $\bar{s}$ for each. Choose the $K$ that maximises $\bar{s}$. A score near $1$ indicates well-separated, cohesive clusters; near $0$ indicates overlapping clusters; negative values suggest the clustering is worse than random.

**Advantage over elbow method**: the elbow method uses only WCSS (within-cluster cohesion). The silhouette score accounts for both cohesion ($a$) and separation ($b$), making it more informative -- a good clustering needs both tight clusters and clear gaps between clusters.

**Limitation**: $O(n^2)$ to compute exactly (all pairwise distances). For large $n$, compute on a subsample or use approximate nearest neighbours.

---

## Tier 2 -- Intermediate

### Q4. Prove that the K-Means update rule (recomputing centroids as cluster means) strictly decreases the WCSS objective.

**Answer:**

Let $C_k^{(t)}$ denote the assignment of cluster $k$ at iteration $t$ and $\mu_k^{(t)}$ the corresponding centroid.

**Assignment step** (Step 2 to produce $C^{(t+1)}$ given $\mu^{(t)}$):

Each point $x_i$ is reassigned to minimise its contribution to WCSS:

$$c_i^{(t+1)} = \arg\min_k \|x_i - \mu_k^{(t)}\|^2$$

This can only decrease or maintain the WCSS, because we are minimising over cluster assignment for each point independently. Therefore:

$$\mathcal{L}(C^{(t+1)}, \mu^{(t)}) \leq \mathcal{L}(C^{(t)}, \mu^{(t)})$$

**Update step** (Step 3: compute $\mu^{(t+1)}$ given $C^{(t+1)}$):

For a fixed assignment $C_k$, the centroid that minimises $\sum_{x_i \in C_k}\|x_i - \mu_k\|^2$ is exactly the mean $\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$. This follows from differentiating and setting to zero:

$$\frac{\partial}{\partial \mu_k}\sum_{x_i \in C_k}\|x_i - \mu_k\|^2 = -2\sum_{x_i \in C_k}(x_i - \mu_k) = 0 \;\Rightarrow\; \mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$$

Therefore:

$$\mathcal{L}(C^{(t+1)}, \mu^{(t+1)}) \leq \mathcal{L}(C^{(t+1)}, \mu^{(t)})$$

Combining: $\mathcal{L}(C^{(t+1)}, \mu^{(t+1)}) \leq \mathcal{L}(C^{(t)}, \mu^{(t)})$

Each full iteration decreases the WCSS. Since there are finitely many distinct partitions, the algorithm must converge.

**Note**: the convergence is to a local minimum. The proof shows WCSS is non-increasing, but multiple local minima exist.

---

### Q5. Explain how DBSCAN handles the case where clusters have different densities. What variation addresses this limitation?

**Answer:**

**The problem with DBSCAN and varying densities:**

DBSCAN uses a single global $\varepsilon$ and $\text{MinPts}$ to define "dense." If one cluster is tightly packed (high density) and another is spread out (low density):

- If $\varepsilon$ is set for the sparse cluster: the dense cluster gets correctly identified, but the sparse one might also be correctly identified
- If $\varepsilon$ is set for the dense cluster: the sparse cluster may not have enough local density to form core points -- its points become noise

Example: 1000 points tightly clustered around the origin (dense cluster) and 200 points uniformly spread over a wide area (sparse cluster). No single $\varepsilon$ identifies both correctly with a single $\text{MinPts}$.

**OPTICS (Ordering Points To Identify the Clustering Structure):**

OPTICS produces a **reachability plot** by ordering points along their reachability distance. Unlike DBSCAN, it does not require specifying $\varepsilon$ -- it computes a hierarchical density structure. Clusters at different densities appear as "valleys" in the reachability plot at different depths.

The user can extract clusters at any density level by specifying a minimum steepness threshold on the reachability plot, effectively applying DBSCAN at different $\varepsilon$ values simultaneously.

**HDBSCAN (Hierarchical DBSCAN):**

Extends OPTICS with a more principled hierarchical cluster extraction. It:
1. Defines mutual reachability distance to smooth single-linkage effects
2. Builds a minimum spanning tree on the mutual reachability graph
3. Constructs a cluster hierarchy by progressively removing edges
4. Extracts the most stable (persistent) clusters from the hierarchy using a stability measure

HDBSCAN handles multi-density data well, is more robust to $\varepsilon$ choice, and is the current recommended default for density-based clustering. It is implemented in `hdbscan` (Python) and integrated into scikit-learn from version 1.3.

---

### Q6. You are tasked with clustering 10 million customer transactions for a recommender system. Compare K-Means, DBSCAN, and Gaussian Mixture Models. Which would you choose and why?

**Answer:**

**Scale and performance analysis:**

**K-Means at scale:**
- $O(nKd)$ per iteration, typically $O(100)$ iterations: for $n = 10^7$, $K = 100$, $d = 50$: approximately $5 \times 10^{11}$ operations per run -- feasible with minibatch K-Means
- **MiniBatch K-Means**: update centroids using mini-batches of size $b \ll n$. Each step is $O(bKd)$; convergence requires $O(n/b)$ batches per epoch. Practically $10$--$100\times$ faster than full K-Means with slightly worse inertia
- Suitable for recommendation: customer segments tend to be roughly spherical in a well-designed feature space

**DBSCAN at scale:**
- Naive DBSCAN is $O(n^2)$ -- infeasible for $10^7$ points
- With a ball tree or k-d tree: $O(n\log n)$ -- borderline feasible but slow
- Approximate variants (HDBSCAN with subsampling): potentially usable but tricky to tune at scale
- Not recommended for this use case

**Gaussian Mixture Models (GMM) at scale:**
- EM algorithm is $O(nKd^2)$ per iteration -- the covariance update is the bottleneck
- For $d = 50$, $K = 100$, $n = 10^7$: covariance update costs $O(10^7 \times 100 \times 2500) \approx 2.5 \times 10^{12}$ -- very expensive
- Diagonal covariance GMM reduces to $O(nKd)$ -- similar to K-Means but with soft assignments and probabilistic interpretation
- Soft cluster membership is an advantage for recommendation (partial membership in multiple taste profiles)

**Recommendation for this use case:**

Use **MiniBatch K-Means** as the primary clustering approach:

1. Pre-process: standardise all features, apply PCA to reduce to $d = 30$--$50$ dimensions
2. Tune $K$ using silhouette score on a 100K subsample (to make computation tractable)
3. Train MiniBatch K-Means with `batch_size=10000`, `max_iter=100` over the full 10M dataset
4. Interpret clusters by inspecting centroid feature values and the most representative transactions per cluster

If soft assignments are important for the recommender (e.g., a user can partially belong to "value shoppers" and "luxury shoppers"), run a **diagonal-covariance GMM** on a representative subsample and use the learned model for inference on all 10M points.

---

## Tier 3 -- Advanced

### Q7. What is the relationship between K-Means and the EM algorithm for Gaussian Mixture Models? Show that K-Means is a special case of GMM-EM.

**Answer:**

**Gaussian Mixture Model (GMM):**

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

The EM algorithm maximises the log-likelihood by alternating between:

**E-step**: compute the posterior probability (responsibility) of each component for each point:

$$r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

**M-step**: update parameters using weighted sufficient statistics:

$$\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \Sigma_k = \frac{\sum_i r_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}, \quad \pi_k = \frac{\sum_i r_{ik}}{n}$$

**K-Means as GMM-EM in the limit $\Sigma \to \epsilon I$:**

Consider the GMM-EM with equal mixing weights ($\pi_k = 1/K$) and spherical, equal covariance matrices ($\Sigma_k = \epsilon^2 I$ for all $k$). The responsibility becomes:

$$r_{ik} = \frac{\exp\!\left(-\frac{1}{2\epsilon^2}\|x_i - \mu_k\|^2\right)}{\sum_j \exp\!\left(-\frac{1}{2\epsilon^2}\|x_i - \mu_j\|^2\right)}$$

As $\epsilon \to 0$, the softmax converges to a hard argmin:

$$r_{ik} \to \begin{cases} 1 & \text{if } k = \arg\min_j \|x_i - \mu_j\|^2 \\ 0 & \text{otherwise}\end{cases}$$

This is exactly the K-Means assignment step. The M-step under these hard assignments:

$$\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}} = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$$

This is exactly the K-Means centroid update.

**Conclusion**: K-Means is EM applied to a GMM with equal spherical covariances in the hard-assignment limit. This reveals that K-Means implicitly assumes:
1. Clusters are spherical (isotropic Gaussian)
2. All clusters have equal size (equal mixing weights)
3. All clusters have the same spread (same $\epsilon$)

GMM relaxes all three assumptions, at the cost of more parameters and higher computational complexity.

---

### Q8. Describe the concentration of measure phenomenon and explain why it makes distance-based clustering unreliable in high dimensions. What practical steps mitigate it?

**Answer:**

**Concentration of measure:**

In high dimensions, points drawn from a distribution tend to concentrate on a thin shell at a characteristic distance from the mean. For $x \sim \mathcal{N}(0, I_d)$:

$$\mathbb{E}[\|x\|^2] = d, \quad \text{Var}(\|x\|^2) = 2d$$

The coefficient of variation: $\frac{\sqrt{\text{Var}(\|x\|^2)}}{\mathbb{E}[\|x\|^2]} = \sqrt{\frac{2}{d}} \to 0$ as $d \to \infty$.

**Consequence for pairwise distances:**

For two independent points $x, y \sim \mathcal{N}(0, I_d)$:

$$\mathbb{E}[\|x - y\|^2] = 2d, \quad \text{Var}(\|x - y\|^2) = 4d$$

The relative variance $= \sqrt{4d}/(2d) = 1/\sqrt{d} \to 0$.

All pairwise distances converge to $\sqrt{2d}$ -- they become nearly equal. In high dimensions, the concept of "nearest neighbour" breaks down: the ratio of the nearest to farthest neighbour distance approaches 1 as $d \to \infty$. Points are approximately equidistant, making distance-based clustering meaningless.

**Why this breaks K-Means and DBSCAN:**

- **K-Means**: if all pairwise distances are similar, assigning a point to the nearest centroid is essentially random -- WCSS does not provide a useful signal
- **DBSCAN**: choosing $\varepsilon$ is impossible when all inter-point distances are concentrated in a tiny range; either all points are core points (one cluster) or none are (all noise)

**Mitigation strategies:**

1. **Dimensionality reduction before clustering**: PCA, UMAP, or autoencoders project data to a lower-dimensional manifold where Euclidean distance is more meaningful. Apply clustering in the reduced space.

2. **Feature selection**: identify and retain only the features that carry variance relevant to the clustering task. Many features in high-dimensional data are noise.

3. **Subspace clustering**: algorithms like sparse subspace clustering or projected clustering find clusters in low-dimensional subspaces of the feature space.

4. **Use cosine similarity instead of Euclidean distance**: for high-dimensional sparse data (e.g., TF-IDF vectors), cosine similarity is more robust to the concentration effect because it normalises by vector magnitude.

5. **Use domain-appropriate kernels**: kernels that capture semantic similarity (e.g., string kernels for text) avoid raw Euclidean distance in high-dimensional spaces.

6. **Increase sample size**: the concentration effect means more data is needed to distinguish clusters -- $n$ must scale with $d$ to maintain clustering quality.

---

## Implementation Reference

```python
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Always standardise before distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with K-Means++ initialisation (default)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X_scaled, labels_km):.4f}")

# Elbow method: scan K values
inertias = []
silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, lbl, sample_size=5000))

# MiniBatch K-Means for large datasets
mb_kmeans = MiniBatchKMeans(
    n_clusters=10,
    batch_size=5000,
    n_init=3,
    random_state=42
)
labels_mb = mb_kmeans.fit_predict(X_scaled)

# DBSCAN
# Use k-distance graph to estimate eps:
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])[::-1]
# Plot k_distances to find the elbow -> good eps value

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
labels_db = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise    = np.sum(labels_db == -1)
print(f"DBSCAN clusters: {n_clusters}, noise points: {n_noise}")

# Only compute silhouette for non-noise points
mask = labels_db != -1
if mask.sum() > 1 and len(set(labels_db[mask])) > 1:
    score = silhouette_score(X_scaled[mask], labels_db[mask])
    print(f"Silhouette (non-noise): {score:.4f}")

# HDBSCAN (recommended over DBSCAN for most use cases)
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
labels_hdb = clusterer.fit_predict(X_scaled)
```

---

## Quick Reference Quiz

**Q: K-Means is applied to data consisting of two concentric rings. What will the result most likely look like?**

A) Two clusters corresponding to the two rings  
B) Two clusters splitting each ring in half along a diameter  
C) One cluster containing all points  
D) Many small clusters distributed across the rings  

**Answer: B.** K-Means minimises Euclidean distance to centroids and assumes convex, roughly spherical clusters. For concentric rings, the centroids converge to points near the centre of each half of both rings combined, creating a split along a line of symmetry -- not the two rings. DBSCAN or spectral clustering would find the correct ring structure.

---

**Q: The silhouette score for $K = 5$ clusters is $0.52$ and for $K = 8$ clusters is $0.61$. What can you conclude?**

A) $K = 5$ is better because simpler models are always preferred  
B) $K = 8$ is better because it has a higher silhouette score  
C) Neither is conclusive without also checking the elbow plot  
D) The difference is too small to be meaningful  

**Answer: B.** A higher silhouette score indicates better-defined clusters (higher within-cluster cohesion and between-cluster separation). $K = 8$ has meaningfully higher silhouette score, suggesting the data has 8 natural groups. However, it is good practice to also examine the silhouette plot per point (looking for negative values) and the WCSS curve to ensure $K = 8$ is not overfitting. In this case, if the score difference is $0.09$, it is generally considered meaningful.
