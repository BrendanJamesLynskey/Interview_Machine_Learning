# Problem 02: Receptive Field Calculations

**Difficulty:** Intermediate to Advanced  
**Topic:** Receptive field — plain CNNs, dilated convolutions, pooling, ResNet  
**Skills tested:** Computing effective receptive field size, understanding coverage vs effective coverage, architectural reasoning

---

## Background

The **receptive field** (RF) of a unit in a CNN is the region of the input image that can influence its activation. A larger receptive field means the unit can integrate information from a wider spatial context.

### Receptive Field Formula for a Stack of Layers

For a sequence of convolutional (or pooling) layers, the receptive field grows layer by layer. Let $r_l$ be the receptive field after layer $l$ and $j_l$ be the "jump" (the distance between adjacent receptive field centres, i.e., the product of all previous strides).

Initialise: $r_0 = 1$, $j_0 = 1$.

For each layer $l$ with kernel $K_l$, stride $S_l$:

$$r_l = r_{l-1} + (K_l - 1) \times j_{l-1}$$

$$j_l = j_{l-1} \times S_l$$

This formulation correctly handles the fact that after a stride-2 layer, subsequent kernels cover a wider range of the original input.

For a dilated convolution with dilation $D_l$:

$$r_l = r_{l-1} + (K_l - 1) \times D_l \times j_{l-1}$$

(dilation effectively enlarges the kernel coverage without changing $j_l$).

---

## Part A: Plain CNN

### Question A1

Calculate the receptive field of the final layer in this network:

| Layer | $K$ | $S$ | $D$ |
|---|---|---|---|
| Conv 1 | 3 | 1 | 1 |
| Conv 2 | 3 | 1 | 1 |
| Conv 3 | 3 | 1 | 1 |
| Conv 4 | 3 | 1 | 1 |
| Conv 5 | 3 | 1 | 1 |

---

**Answer:**

Use the iterative formula with all stride-1 layers. When all strides are 1, $j_l = 1$ for all $l$, and the formula simplifies to:

$$r_l = r_{l-1} + (K - 1) = r_{l-1} + 2$$

| Layer | $r_l$ | $j_l$ |
|---|---|---|
| Input | 1 | 1 |
| Conv 1 | $1 + (3-1) \times 1 = 3$ | 1 |
| Conv 2 | $3 + 2 = 5$ | 1 |
| Conv 3 | $5 + 2 = 7$ | 1 |
| Conv 4 | $7 + 2 = 9$ | 1 |
| Conv 5 | $9 + 2 = 11$ | 1 |

**Receptive field: 11.**

**Shortcut for all-stride-1, kernel-$K$ networks:**

$$r = 1 + L(K-1)$$

For $L=5$, $K=3$: $r = 1 + 5 \times 2 = 11$. Verified.

---

## Part B: CNN with Stride and Pooling

### Question B1

Calculate the receptive field at the output of this network (a simplified VGG-like block):

| Layer | $K$ | $S$ | Notes |
|---|---|---|---|
| Conv 1 | 3 | 1 | |
| Conv 2 | 3 | 1 | |
| MaxPool | 2 | 2 | |
| Conv 3 | 3 | 1 | |
| Conv 4 | 3 | 1 | |
| MaxPool | 2 | 2 | |

---

**Answer:**

Tracking $r_l$ and $j_l$:

| Layer | $K$ | $S$ | $r_l = r_{l-1} + (K-1) \times j_{l-1}$ | $j_l = j_{l-1} \times S$ |
|---|---|---|---|---|
| Input | — | — | 1 | 1 |
| Conv 1 | 3 | 1 | $1 + 2 \times 1 = 3$ | $1 \times 1 = 1$ |
| Conv 2 | 3 | 1 | $3 + 2 \times 1 = 5$ | $1 \times 1 = 1$ |
| MaxPool | 2 | 2 | $5 + 1 \times 1 = 6$ | $1 \times 2 = 2$ |
| Conv 3 | 3 | 1 | $6 + 2 \times 2 = 10$ | $2 \times 1 = 2$ |
| Conv 4 | 3 | 1 | $10 + 2 \times 2 = 14$ | $2 \times 1 = 2$ |
| MaxPool | 2 | 2 | $14 + 1 \times 2 = 16$ | $2 \times 2 = 4$ |

**Receptive field: 16.**

**Key insight:** the $2\times2$ max pool (treated as a $K=2$ kernel) adds only 1 to the receptive field directly, but it doubles the jump $j$. This means subsequent layers grow the receptive field much faster. The second pair of convolutions each add $2 \times j = 4$ (not 2 as in the first block). The stride is what amplifies receptive field growth — pooling's main contribution is via increasing the jump, not via its own kernel size.

---

## Part C: ResNet-50 Architecture

### Question C1

Calculate the theoretical receptive field at the end of Stage 4 of ResNet-50.

ResNet-50 architecture (spatial layers only):

| Component | $K$ | $S$ | Repeats |
|---|---|---|---|
| Conv1 | 7 | 2 | 1 |
| MaxPool | 3 | 2 | 1 |
| Stage 1: $1\times1$ | 1 | 1 | $\times 3$ |
| Stage 1: $3\times3$ | 3 | 1 | $\times 3$ |
| Stage 1: $1\times1$ | 1 | 1 | $\times 3$ |
| Stage 2 first: $1\times1$ | 1 | 1 | $\times 1$ |
| Stage 2 first: $3\times3$ | 3 | 2 | $\times 1$ (stride 2 here) |
| Stage 2 first: $1\times1$ | 1 | 1 | $\times 1$ |
| Stage 2 remaining | same as Stage 1 pattern | $\times 3$ |
| Stage 3: similar, stride 2 at first block | 3 | 2 (first) | |
| Stage 4: similar, stride 2 at first block | 3 | 2 (first) | |

Rather than tracing every single layer, use the structure to reason about the cumulative receptive field.

---

**Answer:**

**Step-by-step trace of the major stride-changing layers:**

Initialise: $r_0 = 1$, $j_0 = 1$.

**Conv1** ($K=7$, $S=2$):

$$r = 1 + (7-1) \times 1 = 7, \quad j = 2$$

**MaxPool** ($K=3$, $S=2$):

$$r = 7 + (3-1) \times 2 = 7 + 4 = 11, \quad j = 4$$

**Stage 1** (3 bottleneck blocks, each: $1\times1$ + $3\times3$ + $1\times1$, all stride 1):

Each $3\times3$ conv adds $(3-1) \times j = 2 \times 4 = 8$ to the receptive field. There are 3 such layers ($3\times3$ per block).

After Stage 1: $r = 11 + 3 \times 8 = 11 + 24 = 35, \quad j = 4$

($1\times1$ convolutions add 0: $(1-1) \times j = 0$)

**Stage 2 first block** (stride-2 on $3\times3$):

$1\times1$: $r = 35 + 0 = 35$, $j = 4$

$3\times3$, $S=2$: $r = 35 + 2 \times 4 = 43$, $j = 4 \times 2 = 8$

$1\times1$: $r = 43$, $j = 8$

**Stage 2 remaining blocks** (3 blocks, all stride 1):

Each $3\times3$ adds $2 \times 8 = 16$. After 3 more $3\times3$ layers:

$r = 43 + 3 \times 16 = 43 + 48 = 91, \quad j = 8$

**Stage 3 first block** (stride-2 on $3\times3$):

$3\times3$, $S=2$: $r = 91 + 2 \times 8 = 107$, $j = 16$

**Stage 3 remaining** (5 blocks, $3\times3$ adds $2 \times 16 = 32$ each):

$r = 107 + 5 \times 32 = 107 + 160 = 267, \quad j = 16$

**Stage 4 first block** (stride-2 on $3\times3$):

$3\times3$, $S=2$: $r = 267 + 2 \times 16 = 299$, $j = 32$

**Stage 4 remaining** (2 blocks, $3\times3$ adds $2 \times 32 = 64$ each):

$r = 299 + 2 \times 64 = 299 + 128 = 427, \quad j = 32$

**Theoretical receptive field at the end of Stage 4: approximately 427 pixels** (with $224\times224$ input, this exceeds the image size — every output unit in Stage 4 has access to the full input in theory).

**Critical caveat — effective vs theoretical receptive field:**

The theoretical receptive field tells you the maximum possible region that *could* influence a unit. The **effective receptive field** (ERF) is different: empirically, the central pixels of the theoretical RF have much stronger influence than peripheral ones. The ERF follows a roughly Gaussian distribution and is significantly smaller than the theoretical RF — typically closer to the square root of the theoretical RF in each dimension.

This means that even though a ResNet-50 unit has a 427-pixel theoretical RF, its effective sensitivity is concentrated over a much smaller central region.

---

## Part D: Dilated Convolutions

### Question D1

Compare the receptive field of:
- **Network A**: 6 layers of $3\times3$ conv, stride 1
- **Network B**: 6 layers of $3\times3$ conv, stride 1, with dilations $[1, 1, 2, 4, 8, 16]$

Both networks have the same number of parameters.

---

**Answer:**

**Network A** (all dilation 1, all stride 1):

Using $r = 1 + L(K-1) = 1 + 6 \times 2 = 13$.

**Network B** (dilations $[1, 1, 2, 4, 8, 16]$, all stride 1 so $j_l = 1$ throughout):

For dilated convolutions with $j=1$:

$$r_l = r_{l-1} + (K-1) \times D_l$$

| Layer | $D$ | $r_l$ |
|---|---|---|
| Input | — | 1 |
| Conv 1 | 1 | $1 + 2 \times 1 = 3$ |
| Conv 2 | 1 | $3 + 2 \times 1 = 5$ |
| Conv 3 | 2 | $5 + 2 \times 2 = 9$ |
| Conv 4 | 4 | $9 + 2 \times 4 = 17$ |
| Conv 5 | 8 | $17 + 2 \times 8 = 33$ |
| Conv 6 | 16 | $33 + 2 \times 16 = 65$ |

**Network B receptive field: 65** versus Network A's **13** — a 5x improvement with identical parameter count.

Network B also maintains full spatial resolution (no stride), making it suitable for dense prediction tasks (segmentation, depth estimation) where spatial precision matters. This is the design principle behind DeepLab and WaveNet architectures.

**Gridding issue:** using the same dilation throughout (e.g., all $D=2$) means some pixels are never sampled by any kernel — they are in the gaps between all kernel positions. Exponentially increasing dilations ($1, 2, 4, 8, \ldots$) ensure every pixel is covered by at least one kernel position in the stack.

---

## Summary Table

| Architecture | Layers | Receptive Field | Notes |
|---|---|---|---|
| 5 × $3\times3$, S=1 | 5 | 11 | All same-padding |
| 2 × $3\times3$ + pool + 2 × $3\times3$ + pool | 6 | 16 | VGG block pattern |
| ResNet-50 (up to Stage 4) | ~50 | ~427 | Theoretical; ERF much smaller |
| 6 × $3\times3$, dilations $[1,1,2,4,8,16]$ | 6 | 65 | Full resolution maintained |

**Key takeaways:**
1. Stride and pooling amplify subsequent layer contributions — the jump $j$ is the multiplier.
2. Dilated convolutions achieve large receptive fields without spatial downsampling.
3. Theoretical RF far exceeds practical influence — the effective RF is Gaussian-shaped and smaller.
4. For classification: large theoretical RF sufficient. For segmentation: large effective RF at full resolution is the goal.
