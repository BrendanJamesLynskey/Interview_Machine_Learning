# CNN Architectures: ResNet, EfficientNet, and MobileNet

A deep-dive into the landmark CNN architectures that dominate computer vision, covering residual connections, compound scaling, and efficient deployment. Organised by interview difficulty tier.

---

## Table of Contents

- [Fundamentals](#fundamentals)
- [Intermediate](#intermediate)
- [Advanced](#advanced)
- [Architecture Comparison](#architecture-comparison)
- [Common Mistakes](#common-mistakes)

---

## Fundamentals

### Why Do Very Deep Networks Fail Without Residual Connections?

Naively stacking more layers should make a network at least as good as a shallower one: the extra layers could learn the identity function. In practice, optimisers fail to achieve this. This is the **degradation problem**.

Two contributing causes:

1. **Vanishing gradients**: in very deep networks, gradients become exponentially small as they propagate back through many layers with weights less than 1. Batch normalisation partially mitigates this but does not fully solve it.

2. **Optimisation difficulty**: the loss landscape of deep networks has regions where gradient information is very weak, making it hard for SGD to find a path to a good solution even when one exists.

The result is that a 56-layer plain network performs *worse* than a 20-layer one on the training set — this is not overfitting but a fundamental optimisation failure.

### Residual Connections (Skip Connections)

He et al. (2015) introduced the **residual block**, which adds a shortcut connection that bypasses one or more layers:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

The block learns the **residual** $\mathcal{F}(\mathbf{x}) = \mathbf{y} - \mathbf{x}$ rather than the full mapping $\mathbf{y}$.

**Why this helps:**

1. **Gradient highway**: gradients can flow directly through the skip connection back to early layers without passing through the weight layers. Mathematically, $\partial \mathcal{L}/\partial \mathbf{x} = \partial \mathcal{L}/\partial \mathbf{y} \cdot (1 + \partial \mathcal{F}/\partial \mathbf{x})$ — the $1$ term ensures a non-zero gradient even when the weight layer gradients are small.

2. **Identity initialisation**: at initialisation, if weights are small, $\mathcal{F}(\mathbf{x}) \approx 0$, so the block outputs approximately $\mathbf{x}$. The network effectively starts as a shallower model and learns to use depth incrementally.

3. **Easier target**: learning $\mathcal{F}(\mathbf{x}) = 0$ (the residual for an identity block) is easier than learning $\mathcal{F}(\mathbf{x}) = \mathbf{x}$ (the identity mapping itself), because zero is a natural target for small random weights.

### ResNet Architecture Overview

ResNet (Residual Network) stacks residual blocks into stages that progressively reduce spatial size and increase channel count.

**ResNet-50 stage structure:**

| Stage | Output size | Block | Repeat |
|---|---|---|---|
| Conv1 | $112 \times 112$ | $7\times7$, 64, stride 2 | 1 |
| Pool | $56 \times 56$ | $3\times3$ max pool, stride 2 | 1 |
| Stage 1 | $56 \times 56$ | Bottleneck, 64 | $\times 3$ |
| Stage 2 | $28 \times 28$ | Bottleneck, 128 | $\times 4$ |
| Stage 3 | $14 \times 14$ | Bottleneck, 256 | $\times 6$ |
| Stage 4 | $7 \times 7$ | Bottleneck, 512 | $\times 3$ |
| GAP + FC | $1000$ | Global avg pool + softmax | 1 |

Total depth: $1 + 3(3) + 4(3) + 6(3) + 3(3) + 1 + 1 = 50$ weight layers.

---

## Intermediate

### The Bottleneck Block

ResNet-50/101/152 use a **bottleneck block** to keep computation manageable at deeper widths:

```
Input (256 channels)
      |
  1x1 conv -> 64 channels   (compress)
      |
  3x3 conv -> 64 channels   (spatial mixing)
      |
  1x1 conv -> 256 channels  (expand)
      |
  + skip connection
      |
   ReLU
Output (256 channels)
```

Parameter count for one bottleneck block (64 internal, 256 in/out):
- $1\times1$: $256 \times 64 = 16384$
- $3\times3$: $64 \times 64 \times 9 = 36864$
- $1\times1$: $64 \times 256 = 16384$
- Total: $\approx 70$K

Compare to two $3\times3$ convolutions on 256 channels: $2 \times 256 \times 256 \times 9 \approx 1.2$M. The bottleneck uses ~17x fewer parameters for the weight layers.

**ResNet-18/34** use a simpler two-layer basic block (no bottleneck) because the channel counts are smaller.

### The Projection Shortcut

When the block changes the number of channels or the spatial size (first block of each stage), the skip connection $\mathbf{x}$ cannot be added directly — its shape differs from $\mathcal{F}(\mathbf{x})$.

Two options:
1. **Zero-padding**: pad the skip connection channels with zeros to match the new channel count. No extra parameters.
2. **Projection shortcut**: use a $1\times1$ convolution (with appropriate stride) on the skip path to match dimensions.

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

He et al. found that projection shortcuts give slightly better accuracy, and ResNet uses them at stage transitions.

### MobileNet: Efficient CNN for Embedded Devices

MobileNet (Howard et al., 2017) makes CNNs viable on mobile and embedded hardware by replacing standard convolutions with depthwise separable convolutions throughout the network.

**Compute reduction** (FLOPs per output pixel, kernel $K$, $C$ channels):

$$\text{Standard conv FLOPs} = K^2 C^2$$

$$\text{Depthwise separable FLOPs} = K^2 C + C^2$$

$$\text{Ratio} = \frac{K^2 C + C^2}{K^2 C^2} \approx \frac{1}{C} + \frac{1}{K^2} \approx \frac{1}{9} \text{ for } K=3, C \gg 1$$

MobileNetV2 added **inverted residuals**: the skip connection operates on the compressed (narrow) representation, and the main path expands channels, applies depthwise conv, then compresses back.

```
Input (narrow)
      |
  1x1 expand (6x channels)
      |
  3x3 depthwise conv
      |
  1x1 project (narrow)
      |
  + skip (if same stride/channels)
Output (narrow)
```

This is inverted relative to a bottleneck block: standard bottlenecks compress–convolve–expand, while inverted residuals expand–convolve–compress.

MobileNetV3 further adds **squeeze-and-excitation (SE)** blocks and uses a neural architecture search (NAS) to find efficient layer configurations.

### EfficientNet: Compound Scaling

Previous work scaled CNNs along only one dimension: deeper (more layers), wider (more channels), or higher resolution (larger input). EfficientNet (Tan & Le, 2019) asks: **what is the right way to scale all three dimensions simultaneously?**

**Compound scaling coefficients:**

$$\text{depth}: d = \alpha^\phi, \quad \text{width}: w = \beta^\phi, \quad \text{resolution}: r = \gamma^\phi$$

Subject to the constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (so that total FLOPs scale as $\approx 2^\phi$).

The baseline values $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ were found by NAS on a small grid search. The $\phi$ parameter then uniformly scales the compound to produce the EfficientNet-B0 through B7 family.

**Why $\beta^2$ and $\gamma^2$?** FLOPs of a conv layer scale as $d \cdot w^2 \cdot r^2$: doubling width quadruples FLOPs (more input and output channels), doubling resolution also quadruples FLOPs (four times as many spatial positions).

**EfficientNet-B0 baseline architecture** is built from MBConv blocks (mobile inverted bottleneck convolutions) found by NAS. Each EfficientNet-B$k$ simply scales the baseline by $\phi = k$.

| Model | Top-1 (ImageNet) | Params | FLOPs |
|---|---|---|---|
| EfficientNet-B0 | 77.1% | 5.3M | 0.39B |
| EfficientNet-B4 | 82.9% | 19M | 4.2B |
| EfficientNet-B7 | 84.4% | 66M | 37B |
| ResNet-50 | 76.0% | 25M | 4.1B |

EfficientNet-B4 achieves higher accuracy than ResNet-50 with similar FLOPs but fewer parameters.

---

## Advanced

### Residual Networks as Ensembles

Veit et al. (2016) showed that ResNets behave like an implicit ensemble of networks of varying depth. Unrolling the residual connections reveals $2^n$ paths through an $n$-block ResNet. Most gradient signal during training travels through relatively short paths (fewer than ~10 effective layers), even in a 100-layer network.

This explains why:
- ResNets are robust to removing individual layers at test time (unlike plain networks).
- Stochastic depth (randomly dropping residual blocks during training) regularises ResNets effectively.

### Pre-activation vs Post-activation ResNet

Original ResNet (He 2015): BN $\to$ ReLU applied *after* the addition.

$$\mathbf{y} = \text{ReLU}(\mathcal{F}(\mathbf{x}) + \mathbf{x})$$

Pre-activation ResNet (He 2016): BN $\to$ ReLU applied *inside* the residual branch, *before* the weight layers.

$$\mathbf{y} = \mathcal{F}(\text{BN-ReLU}(\mathbf{x})) + \mathbf{x}$$

Pre-activation gives a cleaner gradient path (the skip connection is a pure identity with no activation in the way) and leads to better performance on very deep networks (ResNet-1001). It is the standard formulation in many modern implementations.

### Squeeze-and-Excitation Networks (SENet)

The SE block recalibrates channel-wise feature responses by explicitly modelling inter-channel dependencies:

$$\tilde{\mathbf{x}} = \mathbf{s} \odot \mathbf{x}$$

where $\mathbf{s} = \sigma(W_2 \delta(W_1 \mathbf{z}))$ is a per-channel scaling vector and $\mathbf{z} = \text{GAP}(\mathbf{x})$ is the globally-pooled feature vector.

The two FC layers form a bottleneck ($r=16$ reduction) that learns which channels to emphasise. Adding SE blocks to ResNet and MobileNet architectures gives consistent accuracy improvements (~1%) at minimal compute cost.

### EfficientNetV2 and Fused MBConv

EfficientNetV2 (Tan & Le, 2021) addresses a training speed bottleneck in EfficientNet-B3+: the large input resolutions cause depthwise convolutions to become memory-bandwidth bound (depthwise convolutions access memory inefficiently on accelerators).

Solution: replace early-stage MBConv (depthwise separable) blocks with **Fused-MBConv** (a single standard $3\times3$ conv instead of $1\times1$ expand + $3\times3$ depthwise):

```
Fused-MBConv:  3x3 conv (expand) -> 1x1 proj -> skip
MBConv:        1x1 expand -> 3x3 depthwise -> 1x1 proj -> skip
```

Fused-MBConv uses more FLOPs per block but executes faster on GPUs due to better memory access patterns. EfficientNetV2 uses Fused-MBConv in early stages and MBConv in later stages.

### Neural Architecture Search (NAS)

Both EfficientNet and MobileNetV3 use NAS to find baseline architectures. The search problem is:

$$\max_{m \in \mathcal{A}} \text{ACC}(m) \quad \text{subject to} \quad \text{FLOPS}(m) \leq \text{target}$$

where $\mathcal{A}$ is a discrete architecture space (number of layers, kernel sizes, expansion ratios, etc.).

NAS methods:
- **Reinforcement learning** (NASNet, MnasNet): a controller network generates architectures; a reward based on validation accuracy trains the controller.
- **Differentiable NAS** (DARTS): relax the discrete architecture choice to a continuous mixture; optimise jointly with network weights.
- **One-shot NAS** (Once-for-All): train a single supernet; architectures are subnetworks that can be extracted without retraining.

The key practical concern is search cost: early NAS approaches required thousands of GPU-hours. Modern methods reduce this to hours.

---

## Architecture Comparison

| Architecture | Year | Key Innovation | Params | Notes |
|---|---|---|---|---|
| AlexNet | 2012 | Deep CNN + ReLU + Dropout | 60M | First modern CNN; 2 GPU split |
| VGG-16 | 2014 | Deep 3x3 stacks | 138M | Simple but large |
| GoogLeNet/Inception | 2014 | Inception module, multi-scale | 6.8M | 1x1 bottlenecks |
| ResNet-50 | 2015 | Residual connections | 25M | Still baseline in 2024 |
| MobileNetV1 | 2017 | Depthwise separable conv | 4.2M | Embedded deployment |
| MobileNetV2 | 2018 | Inverted residuals | 3.4M | Linear bottleneck |
| SENet | 2018 | Channel attention | +10% params | Won ILSVRC 2017 |
| EfficientNet-B0 | 2019 | Compound scaling + NAS | 5.3M | SOTA efficiency |
| EfficientNetV2 | 2021 | Fused-MBConv + progressive training | 22M | Faster training |

---

## Common Mistakes

1. **Saying ResNets "solve" vanishing gradients completely**: they alleviate it significantly via the gradient highway, but deep ResNets still benefit from batch normalisation and careful initialisation.

2. **Confusing skip connections with dense connections**: DenseNet (Huang et al., 2017) connects each layer to *all* subsequent layers; ResNet only connects to the immediate output of the block.

3. **Claiming the shortcut always adds $\mathbf{x}$ unchanged**: when dimensions change, a projection $W_s \mathbf{x}$ is used. Not all skip connections are identity mappings.

4. **Misquoting EfficientNet scaling**: the constraint is $\alpha \beta^2 \gamma^2 \approx 2$, not $\alpha \beta \gamma = 2$. Width and resolution each appear squared because FLOPs scale quadratically with them.

5. **Treating MobileNet depthwise conv as standard conv**: depthwise conv applies *one* kernel per channel. There is no cross-channel mixing until the subsequent pointwise conv.
