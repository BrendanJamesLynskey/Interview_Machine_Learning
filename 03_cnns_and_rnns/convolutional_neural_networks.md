# Convolutional Neural Networks

A comprehensive reference covering convolution arithmetic, pooling, and feature maps, organised by interview difficulty tier.

---

## Table of Contents

- [Fundamentals](#fundamentals)
- [Intermediate](#intermediate)
- [Advanced](#advanced)
- [Common Mistakes](#common-mistakes)
- [Quick Reference](#quick-reference)

---

## Fundamentals

### What is a convolution in the context of CNNs?

A convolution in a CNN is a sliding dot-product operation between a learnable kernel (filter) and a local patch of the input. The kernel is applied at every spatial position, producing a scalar output at each location. The collection of all these scalar outputs forms a **feature map** (also called an activation map).

The key insight is **weight sharing**: the same kernel weights are used at every spatial location. This gives CNNs two critical properties:
- **Translation equivariance**: if a feature shifts in the input, its activation shifts by the same amount in the output.
- **Parameter efficiency**: a $3 \times 3$ kernel has only 9 weights regardless of the input size, versus a fully-connected layer which would need $H \times W$ weights per output unit.

### The Output Dimension Formula

For a 2D convolution with a single spatial dimension (the same formula applies to height and width independently):

$$\text{out} = \left\lfloor \frac{\text{in} + 2P - D(K - 1) - 1}{S} \right\rfloor + 1$$

Where:
- $\text{in}$ = input size (height or width)
- $P$ = padding (number of zeros added to each side)
- $K$ = kernel size
- $S$ = stride
- $D$ = dilation factor

For the common case of $D=1$ (no dilation) this simplifies to:

$$\text{out} = \left\lfloor \frac{\text{in} + 2P - K}{S} \right\rfloor + 1$$

**Example:** Input $28 \times 28$, kernel $3 \times 3$, padding $1$, stride $1$, no dilation.

$$\text{out} = \left\lfloor \frac{28 + 2(1) - 3}{1} \right\rfloor + 1 = \left\lfloor \frac{27}{1} \right\rfloor + 1 = 28$$

The output is $28 \times 28$ — the spatial size is preserved. This is called **same padding**.

### Padding

Padding adds a border of zeros (most commonly) around the input before convolution.

**Why padding matters:**

1. **Preserving spatial size**: without padding, each convolution reduces spatial dimensions. With `same` padding ($P = \lfloor K/2 \rfloor$ for odd $K$, stride 1), the output matches the input size.

2. **Edge information**: without padding, corner and edge pixels are only covered by a small number of kernel positions, making them under-represented in the output. Padding gives the kernel a chance to be fully centred over every input pixel.

**Common padding modes:**
- **Zero padding** (`'same'` or `'valid'`): pads with zeros. By far the most common.
- **Reflect padding**: pads by mirroring the input. Useful for image inpainting to avoid edge artefacts.
- **Replicate padding**: repeats the edge pixel value.

`'valid'` convolution means no padding at all ($P = 0$), so output is strictly smaller than input.

### Stride

Stride $S$ is the step size of the sliding kernel. Increasing stride reduces the output spatial size and is a form of **downsampling**.

- Stride 1: kernel moves one pixel at a time (default).
- Stride 2: kernel skips every other position, halving the spatial dimensions (approximately).

Strided convolution is computationally cheaper than a convolution followed by average pooling and is preferred in modern architectures (e.g., ResNet uses stride-2 convolutions instead of pooling for downsampling between stages).

### Parameter Count

For a convolutional layer:

$$\text{params} = C_{\text{out}} \times (C_{\text{in}} \times K_H \times K_W + 1)$$

The $+1$ is for the bias term per output channel.

**Example:** Input with 3 channels (RGB), 64 output filters, $3 \times 3$ kernels.

$$\text{params} = 64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1792$$

Compare to a fully-connected layer mapping $224 \times 224 \times 3 = 150528$ inputs to 64 outputs: $150528 \times 64 + 64 = 9{,}633{,}856$ parameters. The CNN uses 5000x fewer parameters.

### Pooling

Pooling is a spatial aggregation operation that reduces feature map size without learnable parameters.

**Max pooling**: takes the maximum value in each pooling window.

$$y_{i,j} = \max_{(p,q) \in \mathcal{R}_{i,j}} x_{p,q}$$

**Average pooling**: takes the mean value in each pooling window.

$$y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(p,q) \in \mathcal{R}_{i,j}} x_{p,q}$$

**Output size of pooling** follows the same formula as convolution (with $P=0$, $D=1$ typically):

$$\text{out} = \left\lfloor \frac{\text{in} - K}{S} \right\rfloor + 1$$

For a $2 \times 2$ pooling window with stride 2 (the classic setting): $\text{out} = \text{in} / 2$.

**Max vs Average pooling:**
- Max pooling preserves the most activated feature in a region. It is more robust to exact spatial position and tends to produce sharper gradients during backpropagation.
- Average pooling smooths features. Global average pooling (GAP) — averaging each feature map to a single value — is used at the end of modern CNNs to replace large fully-connected heads.

**Interview question: What is Global Average Pooling and why is it used?**

GAP collapses each $H \times W$ feature map to a single scalar by averaging all spatial positions. Given a feature map of shape $(C, H, W)$, GAP outputs a vector of shape $(C,)$.

Benefits:
1. Eliminates the large FC layers that previously accounted for most parameters (e.g., AlexNet's two $4096$-unit FC layers).
2. Makes the network fully convolutional — the model can accept inputs of any size at inference time.
3. Reduces overfitting by dramatically cutting parameters.
4. Provides a natural correspondence between output classes and feature maps (exploited in Class Activation Mapping).

---

## Intermediate

### Dilation (Atrous Convolution)

Dilation inserts gaps (zeros) between kernel elements, effectively enlarging the receptive field without increasing the number of parameters or reducing spatial resolution.

A dilation factor $D$ means the kernel elements are spaced $D$ pixels apart. A $3 \times 3$ kernel with $D=2$ covers a $5 \times 5$ region; with $D=4$ it covers a $9 \times 9$ region.

The effective kernel size is:

$$K_{\text{eff}} = D(K - 1) + 1$$

**Why this matters:** In tasks like semantic segmentation, you need a large receptive field to understand context but also need to maintain the full spatial resolution for per-pixel predictions. Strided convolutions or pooling would discard spatial information. Dilated convolutions give you both.

**Dilated convolution output size:**

$$\text{out} = \left\lfloor \frac{\text{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

**Example:** Input $64$, $K=3$, $D=4$, $P=4$ (to maintain size), $S=1$:

$$K_{\text{eff}} = 4(3-1) + 1 = 9$$

$$\text{out} = \frac{64 + 8 - 9}{1} + 1 = 64 \quad \checkmark$$

**Gridding artefacts:** A naive stack of dilated convolutions all with the same dilation rate creates a grid pattern where some input pixels never influence the output. This is addressed by using exponentially increasing dilation rates ($1, 2, 4, 8, \ldots$) as in WaveNet and DeepLab.

### Depthwise Separable Convolutions

A standard convolution mixes spatial filtering and channel combination in one step. A depthwise separable convolution factorises this into two steps:

1. **Depthwise convolution**: apply one kernel per input channel independently (no cross-channel mixing).
2. **Pointwise convolution**: apply a $1 \times 1$ convolution to mix channels.

**Parameter comparison** (input $C_{\text{in}}$ channels, output $C_{\text{out}}$ channels, kernel $K \times K$):

Standard: $C_{\text{out}} \times C_{\text{in}} \times K^2$

Depthwise separable: $C_{\text{in}} \times K^2 + C_{\text{in}} \times C_{\text{out}}$

**Reduction ratio:**

$$\frac{C_{\text{in}} K^2 + C_{\text{in}} C_{\text{out}}}{C_{\text{out}} C_{\text{in}} K^2} = \frac{1}{C_{\text{out}}} + \frac{1}{K^2}$$

For $K=3$, $C_{\text{out}} = 256$: ratio $\approx \frac{1}{9} \approx 8.9\times$ fewer parameters. This is the key building block of MobileNet.

### The $1 \times 1$ Convolution

A $1 \times 1$ convolution applies a linear combination across all input channels at each spatial position independently. It does not mix spatial information.

Uses:
1. **Channel dimensionality reduction/expansion**: used as a bottleneck in ResNet and Inception to control computation cost.
2. **Cross-channel interaction**: adds non-linearity between channels without touching spatial structure.
3. **Implementing fully-connected behaviour in a convolutional framework**: a $1 \times 1$ conv is equivalent to applying the same FC layer at every spatial position.

### Feature Maps and Visualisation

Each filter in a convolutional layer produces one feature map. Early layers learn low-level features (edges, corners, colour blobs); deeper layers learn increasingly abstract patterns (textures, object parts, semantic concepts).

This hierarchy arises because each successive layer has access to a larger receptive field and can compose features detected by earlier layers.

**Receptive field** of a neuron: the region of the original input that can influence its activation. For a single conv layer with kernel $K$, the receptive field is $K \times K$. For a stack of layers, the receptive field grows. See the worked problem on receptive field calculation for details.

### Transposed Convolution (Deconvolution)

A transposed convolution (often misleadingly called "deconvolution") is the gradient of a forward convolution — it maps from a smaller spatial size to a larger one. It is used in:
- Decoder parts of U-Net and segmentation networks.
- Generators in GANs.

The output size for a transposed convolution is:

$$\text{out} = S(\text{in} - 1) + D(K-1) + 1 - 2P$$

For the common case of $D=1$, $P=0$:

$$\text{out} = S(\text{in} - 1) + K$$

With $S=2$, $K=4$, $P=1$: $\text{out} = 2(\text{in} - 1) + 4 - 2 = 2 \cdot \text{in}$ — doubling spatial size.

**Common interview trap:** transposed convolution is NOT the inverse of convolution. It has the same shape relationship as the transpose of the convolution matrix but does not undo the values computed by the forward convolution.

---

## Advanced

### Backpropagation Through Convolution

During backpropagation, the gradient with respect to the input of a convolution is itself a convolution. Specifically:

Given a convolution $Y = X * W$ (where $*$ denotes cross-correlation), the gradient of the loss $\mathcal{L}$ with respect to $X$ is:

$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} * W^{\text{rot180}}$$

where $W^{\text{rot180}}$ is the kernel rotated 180 degrees (a full convolution rather than cross-correlation). The gradient with respect to the weights is:

$$\frac{\partial \mathcal{L}}{\partial W} = X * \frac{\partial \mathcal{L}}{\partial Y}$$

This is why convolution layers are efficient to train: the backward pass has the same computational structure as the forward pass.

### Equivariance vs Invariance

CNNs are **translation equivariant**: $f(\text{shift}(x)) = \text{shift}(f(x))$. The output feature map shifts if the input shifts.

They are approximately **translation invariant** due to pooling, but this invariance is imperfect for large shifts. Pooling only provides invariance within the pooling window size.

**CNNs are not rotation or scale equivariant by default.** This is a known limitation. Approaches to address it include:
- Data augmentation with rotations/scales.
- Spatial Transformer Networks (STN).
- Equivariant networks (e.g., Group-Equivariant CNNs).

### Grouped Convolutions

A grouped convolution splits the input channels into $G$ groups and convolves each group independently.

**Parameter count:** $\frac{1}{G}$ of a standard convolution (for matching output channels).

When $G = C_{\text{in}}$, this is a **depthwise convolution**.

ResNeXt uses grouped convolutions as a simple way to increase model capacity without proportionally increasing parameters. AlexNet used them to split computation across two GPUs — an historical precursor.

### The Inductive Biases of CNNs

CNNs encode specific assumptions about data:

1. **Locality**: nearby pixels are more correlated than distant ones. Captured by local kernels.
2. **Translation equivariance**: features should be detectable regardless of position. Captured by weight sharing.
3. **Compositionality**: complex features are built from simpler ones. Captured by depth.

These inductive biases make CNNs data-efficient for images — they need far fewer examples than a Vision Transformer (ViT) to reach the same performance on small datasets. However, they also limit the model's ability to capture global context without many layers (limited receptive field).

### Visualising What CNNs Learn

**Gradient-based saliency maps:** compute $\|\partial \mathcal{L} / \partial X\|$ to see which input pixels most affect the prediction.

**Grad-CAM (Gradient-weighted Class Activation Mapping):**

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

where $\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$ are the importance weights for each feature map $A^k$ with respect to class $c$.

Grad-CAM produces a coarse heatmap highlighting the image regions that drove a particular class prediction, without requiring any modification to the network architecture.

---

## Common Mistakes

1. **Off-by-one in output size**: forgetting the $+1$ in the output dimension formula, or using ceiling instead of floor.
2. **Assuming `same` padding always uses half the kernel size**: `same` padding is $P = (K-1)/2$ only for odd $K$ and $S=1$. For $S>1$, PyTorch's `padding='same'` uses asymmetric padding.
3. **Conflating transposed convolution with inverse convolution**: transposed convolution reverses the shape change but not the values.
4. **Forgetting dilation in receptive field calculations**: a dilated conv with $D=2$, $K=3$ has receptive field $5$, not $3$.
5. **Parameter counting error**: forgetting that bias adds $C_{\text{out}}$ parameters, or forgetting that grouped convolutions reduce parameter count by $G$.

---

## Quick Reference

| Operation | Output size |
|---|---|
| Conv ($P$, $K$, $S$, $D=1$) | $\lfloor(in + 2P - K)/S\rfloor + 1$ |
| Conv with dilation $D$ | $\lfloor(in + 2P - D(K-1) - 1)/S\rfloor + 1$ |
| Max/Avg Pool ($K$, $S$, $P=0$) | $\lfloor(in - K)/S\rfloor + 1$ |
| Transposed Conv ($K$, $S$, $P=0$) | $S(in-1) + K$ |

| Technique | Key benefit |
|---|---|
| Same padding | Preserves spatial size |
| Stride 2 | Halves spatial size, no extra params |
| Dilation | Enlarges receptive field, preserves resolution |
| Depthwise separable | ~$K^2\times$ parameter reduction |
| Global average pooling | Replaces FC head, any input size |
| $1\times1$ conv | Channel mixing, bottleneck compression |
