# Problem 01: Convolution Output Dimensions

**Difficulty:** Fundamentals to Intermediate  
**Topic:** Convolution arithmetic — padding, stride, dilation  
**Skills tested:** Applying the output dimension formula, parameter counting, multi-layer composition

---

## Background

The output size of a convolutional layer along one spatial dimension is:

$$\text{out} = \left\lfloor \frac{\text{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

Where:
- $\text{in}$ = input dimension
- $P$ = padding
- $K$ = kernel size
- $S$ = stride
- $D$ = dilation (default 1)

The number of learnable parameters in a conv layer:

$$\text{params} = C_{\text{out}} \times (C_{\text{in}} \times K_H \times K_W + 1)$$

---

## Part A: Single Layer Calculations

Calculate the output spatial dimensions and parameter count for each configuration.

### Question A1

Input: $32 \times 32$ image, 1 channel  
Layer: 16 filters, $5 \times 5$ kernel, padding 0, stride 1, no dilation

**What are the output spatial dimensions and parameter count?**

---

**Answer:**

Output size:

$$\text{out} = \left\lfloor \frac{32 + 2(0) - 5}{1} \right\rfloor + 1 = \left\lfloor \frac{27}{1} \right\rfloor + 1 = 28$$

Output shape: $\mathbf{16 \times 28 \times 28}$ (channels, height, width).

Parameter count:

$$\text{params} = 16 \times (1 \times 5 \times 5 + 1) = 16 \times 26 = 416$$

**Common mistake:** forgetting the $+1$ bias per output channel gives $16 \times 25 = 400$ — wrong by 16.

---

### Question A2

Input: $128 \times 128$ image, 3 channels  
Layer: 64 filters, $3 \times 3$ kernel, padding 1, stride 2, no dilation

**What are the output spatial dimensions and parameter count?**

---

**Answer:**

Output size:

$$\text{out} = \left\lfloor \frac{128 + 2(1) - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{127}{2} \right\rfloor + 1 = 63 + 1 = 64$$

Output shape: $\mathbf{64 \times 64 \times 64}$.

Note: stride 2 with same-like padding halves the spatial dimensions (from 128 to 64). This is the standard way modern CNNs downsample.

Parameter count:

$$\text{params} = 64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1792$$

---

### Question A3

Input: $64 \times 64$, 32 channels  
Layer: 32 filters, $3 \times 3$ kernel, padding 2, stride 1, dilation 2

**What are the output spatial dimensions? What is the effective kernel size?**

---

**Answer:**

Effective kernel size:

$$K_{\text{eff}} = D(K-1) + 1 = 2(3-1) + 1 = 5$$

A $3 \times 3$ kernel with dilation 2 covers a $5 \times 5$ area, but only samples 9 of the 25 positions.

Output size using the full dilation formula:

$$\text{out} = \left\lfloor \frac{64 + 2(2) - 2(3-1) - 1}{1} \right\rfloor + 1 = \left\lfloor \frac{64 + 4 - 4 - 1}{1} \right\rfloor + 1 = 63 + 1 = 64$$

Output shape: $\mathbf{32 \times 64 \times 64}$ — spatial size is preserved.

**Interpretation:** padding $P = D(K-1)/2 = 2$ is the correct same-padding formula for dilated convolutions.

Parameter count:

$$\text{params} = 32 \times (32 \times 3 \times 3 + 1) = 32 \times 289 = 9{,}248$$

Note: dilation does not affect the parameter count — still 9 values per channel per filter.

---

## Part B: Multi-Layer Stack

### Question B1

Trace the spatial dimensions through this network, given a $224 \times 224$ input:

| Layer | Type | Params |
|---|---|---|
| 1 | Conv | $K=7$, $P=3$, $S=2$, 64 filters |
| 2 | MaxPool | $K=3$, $P=1$, $S=2$ |
| 3 | Conv | $K=3$, $P=1$, $S=1$, 128 filters |
| 4 | Conv | $K=3$, $P=1$, $S=1$, 128 filters |
| 5 | Conv | $K=1$, $P=0$, $S=1$, 256 filters |
| 6 | GlobalAvgPool | — |

**What is the output shape after each layer? What is the final output dimensionality?**

---

**Answer:**

Work through each layer:

**Layer 1** (Conv, $K=7$, $P=3$, $S=2$):

$$\text{out} = \left\lfloor \frac{224 + 6 - 7}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = 112$$

Shape after Layer 1: $64 \times 112 \times 112$

**Layer 2** (MaxPool, $K=3$, $P=1$, $S=2$):

$$\text{out} = \left\lfloor \frac{112 + 2 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{111}{2} \right\rfloor + 1 = 55 + 1 = 56$$

Shape after Layer 2: $64 \times 56 \times 56$

**Layer 3** (Conv, $K=3$, $P=1$, $S=1$):

$$\text{out} = \left\lfloor \frac{56 + 2 - 3}{1} \right\rfloor + 1 = 55 + 1 = 56$$

Shape after Layer 3: $128 \times 56 \times 56$

**Layer 4** (Conv, $K=3$, $P=1$, $S=1$): Same size-preserving conv.

Shape after Layer 4: $128 \times 56 \times 56$

**Layer 5** (Conv, $K=1$, $P=0$, $S=1$):

$$\text{out} = \left\lfloor \frac{56 + 0 - 1}{1} \right\rfloor + 1 = 55 + 1 = 56$$

Shape after Layer 5: $256 \times 56 \times 56$

**Layer 6** (GlobalAvgPool): averages each $56 \times 56$ feature map to a single value.

**Final output shape: $256$** (a vector, one value per channel).

This is a typical ResNet stem followed by a classifier head.

---

## Part C: Parameter Budget Analysis

### Question C1

A team proposes replacing a single $7 \times 7$ convolutional layer (64 filters, 3 input channels) with a stack of three $3 \times 3$ convolutional layers (each with 64 filters). Both process the same 3-channel input.

**Which uses more parameters? What are the trade-offs?**

---

**Answer:**

**Single $7 \times 7$ layer:**

$$\text{params} = 64 \times (3 \times 7 \times 7 + 1) = 64 \times 148 = 9{,}472$$

**Three $3 \times 3$ layers** (first takes 3-channel input, second and third take 64-channel input):

$$\text{params}_1 = 64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1{,}792$$

$$\text{params}_{2,3} = 64 \times (64 \times 3 \times 3 + 1) = 64 \times 577 = 36{,}928 \quad \text{each}$$

$$\text{total} = 1{,}792 + 2 \times 36{,}928 = 75{,}648$$

**The three $3 \times 3$ layers use ~8x more parameters.**

However, the comparison is not fair because:
- The three $3 \times 3$ layers have three non-linearities (ReLUs), the single $7 \times 7$ layer has one — deeper networks with the same receptive field are more expressive.
- The relevant comparison for VGG's claim is **same input channels**: two $3 \times 3$ layers on $C$-channel input have $2 \times C^2 \times 9$ weights versus one $5 \times 5$ layer with $C^2 \times 25$ weights. $18C^2 < 25C^2$ — the stacked $3 \times 3$s are cheaper for the same receptive field when input channels are large.

The $7 \times 7$ stem is only used once (on the 3-channel input), where the channel count is small, so the parameter comparison above is less relevant for typical network designs.

---

## Part D: Transposed Convolution

### Question D1

A decoder network uses a transposed convolution to upsample from $7 \times 7$ to $14 \times 14$.

**What combination of kernel size, stride, and padding achieves this?**

For a transposed convolution the output size formula (with $D=1$) is:

$$\text{out} = S(\text{in} - 1) + K - 2P$$

---

**Answer:**

We need $\text{out} = 14$ from $\text{in} = 7$.

The most common choice: $S=2$ (double the size), $K=4$, $P=1$:

$$\text{out} = 2(7-1) + 4 - 2(1) = 12 + 4 - 2 = 14 \quad \checkmark$$

Alternative: $S=2$, $K=2$, $P=0$:

$$\text{out} = 2(7-1) + 2 - 0 = 12 + 2 = 14 \quad \checkmark$$

Both work, but $K=4$, $P=1$ is preferred in practice because $K=2$ with $S=2$ can produce checkerboard artefacts due to uneven overlap in the kernel.

**Checkerboard artefact explanation:** with $K=2$, $S=2$, each output pixel is covered by exactly one kernel position. With $K=4$, $S=2$, there is overlap, which averages contributions and reduces the grid-like pattern.

A common alternative to transposed convolutions that avoids artefacts entirely: **bilinear upsample followed by a $3 \times 3$ conv** (learned upsampling without the overlap issues).

---

## Summary of Key Formulae

$$\text{Conv output} = \left\lfloor \frac{\text{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

$$\text{Pool output} = \left\lfloor \frac{\text{in} - K}{S} \right\rfloor + 1 \quad (P=0)$$

$$\text{Transposed conv output} = S(\text{in}-1) + D(K-1) + 1 - 2P$$

$$\text{Conv params} = C_{\text{out}} \times (C_{\text{in}} \times K_H \times K_W + 1)$$

$$\text{Effective kernel size (dilation)} = D(K-1) + 1$$
