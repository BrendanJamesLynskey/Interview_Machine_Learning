# Data Augmentation

## Prerequisites
- Supervised learning: training set, validation set, overfitting
- Image representation: tensors of shape $(C, H, W)$, pixel value ranges
- Cross-entropy loss, one-hot labels, soft labels
- Basic familiarity with `torchvision.transforms` and `albumentations`

---

## Concept Reference

### What is Data Augmentation?

Data augmentation artificially expands the effective training set by applying label-preserving transformations to training examples. The key insight is that the model should be invariant (or equivariant) to certain transformations of the input -- a horizontally flipped image of a cat is still a cat. By including augmented versions of training samples, the model learns these invariances explicitly rather than having to learn them from limited data.

Augmentation reduces overfitting by:
- Increasing the effective number of distinct training samples.
- Preventing the model from memorising exact pixel values or specific word orderings.
- Acting as a form of noise injection that improves robustness.

Augmentation is most impactful when labelled data is scarce. With very large datasets (ImageNet at full scale, large web corpora), augmentation still helps but has diminishing returns.

---

### Image Augmentation

#### Geometric Transforms

These preserve pixel values but change spatial arrangement:

- **Random horizontal flip:** Flip left-right with probability 0.5. Standard for most image tasks except those where left/right asymmetry matters (e.g., text recognition, medical imaging with specific orientations).
- **Random crop / resize crop:** Randomly crop a region (e.g., 64--100% of image area) and resize to target size. Teaches the model to recognise objects at various scales and positions. The dominant augmentation in ImageNet training (RandomResizedCrop).
- **Random rotation:** Rotate by $\pm\theta$ degrees. For natural images, small rotations ($\pm15°$) are typical. For medical images or satellite imagery, full 360° rotation is appropriate.
- **Random affine:** Combines translation, rotation, scaling, and shear. More expressive than individual transforms.

#### Appearance Transforms

These preserve spatial structure but alter pixel values:

- **Colour jitter:** Randomly perturb brightness, contrast, saturation, and hue. Parameters: `brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1` (SimCLR defaults). Teaches colour invariance.
- **Grayscale:** Convert to grayscale with probability $p$ (e.g., 0.2). Improves robustness to colour loss.
- **Gaussian blur:** Apply a Gaussian kernel with random sigma. Teaches multi-scale robustness.
- **Gaussian noise:** Add pixel-level Gaussian noise. Improves robustness to sensor noise.

#### Erasing / Occlusion Transforms

- **Random erasing (CutOut / Random Erasing):** Randomly mask a rectangular region of the image with zeros, mean pixel values, or random noise. The model must learn to classify objects from partial views. Effective regulariser for attention-based models.

#### Standard Recipes

**ImageNet classification (supervised):**
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(...),
])
```

**Self-supervised learning (SimCLR):** Two random augmented views of the same image are generated; the model is trained to match their representations. The augmentation pipeline is much more aggressive: `RandomResizedCrop` + `ColorJitter` (stronger) + `RandomGrayscale` + `GaussianBlur`.

---

### Text Augmentation

Text augmentation is harder than image augmentation because natural language does not have the same simple geometric invariances -- synonym replacement or paraphrasing may change subtle meaning.

#### Token-level Augmentation

- **Random deletion:** Remove a token at a random position with probability $p$. The model must be robust to missing words.
- **Random swap:** Swap two adjacent or random tokens. Teaches word-order robustness.
- **Synonym replacement (EDA -- Wei & Zou, 2019):** Replace $n$ non-stopword tokens with WordNet synonyms. Easy Data Augmentation (EDA) combines random deletion, swap, insertion, and synonym replacement.
- **Back-translation:** Translate the sentence to an intermediate language and back. Produces a semantically equivalent paraphrase with different surface form. High-quality but computationally expensive.

#### Sentence-level Augmentation

- **Token masking (like MLM):** Randomly replace tokens with `[MASK]` or random tokens. Used in pre-training objectives but also as a fine-tuning augmentation.
- **Crop:** Take a contiguous span of the input as the augmented example. Effective for long documents.

#### Augmentation vs. Pretrained LLMs

With modern pretrained models (BERT, GPT), the pretrained representations already encode many linguistic invariances. Text augmentation has a smaller relative benefit than image augmentation for fine-tuning tasks. The primary use case remains low-resource scenarios (few-shot, domain-specific tasks with < 1,000 labelled examples).

---

### Mixup

Mixup (Zhang et al., 2018) creates synthetic training examples by linearly interpolating between two training samples and their labels:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha \in [0.2, 1.0]$, and $(x_i, y_i)$, $(x_j, y_j)$ are drawn randomly from the training set.

The mixed label $\tilde{y}$ is a soft probability vector combining both classes. The model is trained to produce outputs that interpolate linearly between the correct predictions for both constituent examples.

**Why Mixup works:**

1. **Vicinal risk minimisation:** Instead of minimising the empirical risk at observed data points, Mixup minimises the risk in the "vicinity" of observed points (convex combinations). This provides implicit regularisation by penalising sharp transitions in the output between training examples.

2. **Label noise robustness:** Mixed labels prevent the model from becoming overconfident on any single training example.

3. **Gradient behaviour:** The linear interpolation in input space encourages linear behaviour in output space. This acts as a constraint that discourages the model from learning highly non-linear decision boundaries that do not generalise.

```python
def mixup_batch(x, y, alpha=0.4):
    """Apply Mixup to a batch. y must be one-hot encoded."""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix
```

Note: Mixup requires a loss function that handles soft labels (e.g., cross-entropy with soft targets, not the standard PyTorch `CrossEntropyLoss` which expects integer class indices).

---

### CutMix

CutMix (Yun et al., 2019) is a variant of Mixup that pastes a rectangular patch from one image into another, with labels mixed in proportion to the area of the patch:

$$\tilde{x} = \mathbf{M} \odot x_i + (1 - \mathbf{M}) \odot x_j$$
$$\tilde{y} = (1 - r_{\text{area}}) y_i + r_{\text{area}} y_j$$

where $\mathbf{M}$ is a binary mask with a rectangular "cut" region and $r_{\text{area}} = |\mathbf{M}| / (H \times W)$ is the fraction of the image covered by the patch.

The cut region has size $rW \times rH$ where $r \sim \text{Beta}(\alpha, \alpha)$, and the cut box location is uniformly sampled.

**CutMix vs. Mixup:**

| Property | Mixup | CutMix |
|---|---|---|
| Input | Pixel-level blend | Rectangular patch replacement |
| Locality | All pixels mixed | Local region of pixels replaced |
| Effect on features | Blurry, non-natural images | Natural-looking images with occlusion |
| Preferred domain | Generally applicable | Image classification, detection |
| Texture/shape bias | Encourages smooth features | Preserves texture boundaries |

CutMix tends to outperform Mixup on image tasks because it produces more natural-looking inputs (no blending artifacts) and teaches the model to focus on salient regions rather than spreading attention across the whole image. CutMix is also effective as a data augmentation strategy for training Vision Transformers.

**AugMix (Hendrycks et al., 2020):** Applies chains of augmentation operations sampled from a fixed set, then mixes the augmented images with the original using random coefficients. AugMix targets robustness to distribution shift rather than raw accuracy improvement.

---

### AutoAugment and RandAugment

**AutoAugment (Cubuk et al., 2019):** Uses reinforcement learning to search for the optimal augmentation policy (a sequence of operations and their magnitudes) for a specific dataset. Achieves state-of-the-art augmentation but requires expensive search (5,000 GPU hours for ImageNet). Transferable policies exist for common datasets.

**RandAugment (Cubuk et al., 2020):** Simplifies AutoAugment by selecting $N$ random augmentation operations from a fixed set and applying them each with a global magnitude $M$. RandAugment has only two hyperparameters ($N$ and $M$) instead of the full per-operation policy. Typical values: $N = 2$, $M = 9\text{--}10$. RandAugment achieves performance close to AutoAugment with trivial search cost and is the standard augmentation pipeline for ViT and EfficientNet training.

---

### Test-Time Augmentation (TTA)

TTA applies augmentations at inference and averages predictions across augmented versions of the same input:

$$\hat{y} = \frac{1}{K}\sum_{k=1}^K f(A_k(x))$$

Common TTA augmentations: horizontal flip, multi-scale crops (5 crops: 4 corners + center), slight colour jitter. TTA typically improves top-1 accuracy by 0.3--1% at the cost of $K\times$ inference time. Used in competitions but less common in production due to latency constraints.

---

## Tier 1 -- Fundamentals

### Question F1
**Why does data augmentation reduce overfitting? Explain the mechanism using the concept of invariance.**

**Answer:**

Overfitting occurs when the model memorises specific features of training examples that are not present in unseen examples -- it learns the noise rather than the signal. Data augmentation reduces overfitting through two related mechanisms.

**Mechanism 1: Explicitly teaching invariances.**

A cat remains a cat whether the image is flipped horizontally, brightened, or slightly rotated. If the training set contains only right-facing cats, the model may learn "right-facing" as a feature of cats. By including horizontally flipped versions in training, the model is forced to classify cats correctly regardless of orientation; it learns the invariance. Without augmentation, the model would need vastly more data to encounter the same diversity of orientations, scales, and lighting conditions.

**Mechanism 2: Increasing effective dataset size.**

Each augmentation of a training example is a distinct data point. With $N$ training examples and $k$ augmentations per example, the effective dataset is $Nk$ examples. This is not $Nk$ independent examples (the augmented versions are correlated), but it is much more diverse than $N$. More diverse data forces the model to learn features that generalise across the diversity rather than features that are specific to the exact examples seen.

**The augmentation must be label-preserving.** An augmentation is only valid if the transformation does not change the label. Horizontal flip is valid for dog/cat classification but invalid for tasks where orientation matters (e.g., "is this a correctly oriented document?"). Overaggressive augmentation (e.g., extreme colour jitter that makes a fire engine look grey) breaks the label-preserving property and can hurt performance.

---

### Question F2
**Explain how Mixup creates training examples. Write out the formula for the mixed input and label, and explain how the mixed label is used in the loss computation.**

**Answer:**

Given two training examples $(x_i, y_i)$ and $(x_j, y_j)$ where $y_i, y_j$ are one-hot label vectors, a mixing coefficient $\lambda \sim \text{Beta}(\alpha, \alpha)$ (with $\lambda \in [0,1]$):

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

**Example:** $x_i$ is a cat image ($y_i = [1, 0]$), $x_j$ is a dog image ($y_j = [0, 1]$), $\lambda = 0.7$.

$$\tilde{x} = 0.7 \cdot \text{cat\_image} + 0.3 \cdot \text{dog\_image}$$
$$\tilde{y} = 0.7 \cdot [1, 0] + 0.3 \cdot [0, 1] = [0.7, 0.3]$$

The mixed image is a translucent blend of both images. The label says "this is 70% cat, 30% dog."

**Loss computation.** The cross-entropy loss with soft targets:

$$\mathcal{L}(\tilde{y}, p) = -\sum_k \tilde{y}_k \log p_k = -0.7 \log p_{cat} - 0.3 \log p_{dog}$$

This is a weighted sum of the cross-entropy for each class. An equivalent implementation (which avoids constructing the soft label vector explicitly):

$$\mathcal{L}_{mixup} = \lambda \cdot \text{CE}(p, y_i) + (1-\lambda) \cdot \text{CE}(p, y_j)$$

```python
# Equivalent and more numerically stable implementation:
loss = lam * F.cross_entropy(logits, labels_i) + (1 - lam) * F.cross_entropy(logits, labels_j)
```

This formulation works directly with integer class labels and PyTorch's `cross_entropy`, avoiding the need to construct explicit one-hot vectors.

---

### Question F3
**What is the difference between Mixup and CutMix? In which scenarios would you prefer each?**

**Answer:**

**Mixup:** Linearly blends all pixels of two images with a scalar $\lambda$. The result is a semi-transparent overlay where both images are visible everywhere in the blended image.

**CutMix:** Copies a rectangular region from image $j$ and pastes it into image $i$. Outside the rectangle, the image is entirely $x_i$; inside the rectangle, it is entirely $x_j$. The label is mixed in proportion to the rectangle's area: $\tilde{y} = (1 - r) y_i + r y_j$ where $r$ is the area fraction of the cut region.

**Key differences:**

| Aspect | Mixup | CutMix |
|---|---|---|
| Visual naturalness | Blurry blended images that don't occur naturally | Natural-looking images with realistic occlusion |
| Feature learning | Encourages distributed features (both images visible everywhere) | Encourages local features (model must use the visible region of each image) |
| Implicit regularisation | Smooth interpolation of output space | Occlusion robustness, attention regularisation |

**When to prefer Mixup:**
- When the model needs to learn smooth, interpolatable representations (contrastive learning, regression tasks).
- For tabular or non-image data (feature vectors naturally interpolate; CutMix's spatial cutting is not meaningful).
- For tasks where global context matters (image-level classification where the whole image contributes).

**When to prefer CutMix:**
- Image classification tasks where CutMix typically outperforms Mixup (ViT, DeiT, ResNet on ImageNet).
- Object detection and segmentation, where localised features matter and Mixup's blending confuses spatial extent.
- When training Vision Transformers: CutMix forces the model to attend to localised patches, counteracting ViT's tendency to over-rely on global texture.

**In practice:** Both are often combined (with random choice between them per batch, or alternate strategies like CutMixup). The timm library's default for ViT training uses a combination of RandAugment, Mixup, CutMix, and label smoothing.

---

## Tier 2 -- Intermediate

### Question I1
**Describe the RandAugment procedure. What are the two hyperparameters, and how do you tune them? How does RandAugment compare to manually designed augmentation pipelines?**

**Answer:**

RandAugment selects $N$ augmentation operations uniformly at random from a predefined set (14 operations in the original paper: Identity, AutoContrast, Equalize, Rotate, Solarise, Color, Posterize, Contrast, Brightness, Sharpness, ShearX, ShearY, TranslateX, TranslateY) and applies them sequentially to the image. Each operation is applied with a magnitude $M$ on a scale of $[0, 30]$ (normalised to $[0, 1]$ per operation):

$$\text{augmented}(x) = \text{Op}_N \circ \cdots \circ \text{Op}_1 (x)$$

**The two hyperparameters:**

1. $N$ (number of operations, typically 1--3): Controls augmentation diversity. More operations = more aggressive transformation. $N = 2$ is the standard for image classification.

2. $M$ (magnitude, typically 5--15 for CIFAR, 9--10 for ImageNet): Controls the strength of each operation. $M = 0$ applies no transformation; $M = 30$ applies maximum transformation. All operations use the same magnitude scale.

**Tuning procedure:**

Grid search over $\{N, M\}$ combinations using validation accuracy. The search space is small ($3 \times 5 = 15$ combinations for typical ranges), making RandAugment vastly cheaper to tune than AutoAugment's RL-based search. The optimal $N$ and $M$ typically increase with model capacity and training duration: larger models trained longer benefit from stronger augmentation to prevent overfitting.

Typical values:
- CIFAR-10/100: $N = 1\text{--}2$, $M = 5\text{--}10$
- ImageNet (ResNet-50): $N = 2$, $M = 9$
- ImageNet (ViT-B): $N = 2$, $M = 15$ (stronger augmentation for larger models)

**Comparison to manually designed pipelines:**

Manual pipelines (e.g., ImageNet's standard `RandomResizedCrop + RandomHorizontalFlip + ColorJitter`) are:
- Expert-designed for specific tasks and require domain knowledge.
- Fixed operations at fixed magnitudes -- no diversity in augmentation strength.
- Often miss beneficial operations (e.g., AutoContrast, Equalize) that are not obvious from first principles.

RandAugment:
- Automatically explores a broader operation set.
- Achieves state-of-the-art results with a single $\{N, M\}$ search.
- Lacks task-specific inductive biases (e.g., a manual pipeline for medical imaging would exclude colour jitter but include elastic deformation; RandAugment does not include domain-specific operations).

**Limitation:** RandAugment treats all operations equally. Some operations (e.g., strong Rotate) may be harmful for a specific task while others (e.g., ColorJitter) are universally beneficial. AutoAugment learns task-specific weights; RandAugment averages over all operations.

---

### Question I2
**How does Mixup interact with label smoothing? Should you apply both simultaneously? Explain the effect on calibration.**

**Answer:**

**Label smoothing** replaces one-hot labels with a soft distribution assigning $1 - \epsilon + \epsilon/K$ to the correct class and $\epsilon/K$ to all others.

**Mixup** creates soft labels as a linear combination of two one-hot (or already-smoothed) labels.

**Applying both:**

If label smoothing is applied before Mixup, the mixed label is:

$$\tilde{y}_k = \lambda y_k^{smooth}(i) + (1-\lambda) y_k^{smooth}(j)$$

For the Mixup example above with $K = 2$, $\lambda = 0.7$, $\epsilon = 0.1$:

$$y_i^{smooth} = [0.95, 0.05], \quad y_j^{smooth} = [0.05, 0.95]$$
$$\tilde{y} = 0.7[0.95, 0.05] + 0.3[0.05, 0.95] = [0.68, 0.32]$$

Without label smoothing: $\tilde{y} = [0.7, 0.3]$.

**The interaction:** Both Mixup and label smoothing discourage over-confident predictions. Applying both is double regularisation that can under-smooth the output:

- Mixup's soft labels already prevent the model from assigning probability 1 to any class (unless $\lambda \in \{0, 1\}$).
- Adding label smoothing on top of Mixup's soft labels further distributes probability mass away from the correct class.

For a 1000-class problem (ImageNet) with $\lambda = 0.7$ and $\epsilon = 0.1$, label smoothing shifts the correct-class probability from $0.7$ to approximately $0.7 - 0.7\epsilon + \epsilon/K \approx 0.63$. The model is now targeting $0.63$ for the dominant class, which is less confident than the natural Mixup target of $0.7$.

**Effect on calibration:**

- Mixup alone improves calibration by preventing the model from outputting probabilities close to 0 or 1 on training examples.
- Label smoothing alone improves calibration by setting a finite target probability for the correct class.
- Both together can over-smooth, causing the model to be systematically under-confident -- its output probabilities are always lower than the true probabilities. This is miscalibration in the other direction.

**Practical recommendation:**

In practice, many top-performing recipes use both (ViT, DeiT use Mixup + CutMix + label smoothing). The effect is empirically small for small $\epsilon$ (0.1) combined with Mixup $\alpha = 0.4$. If calibration is critical (medical or financial applications), use either Mixup or label smoothing but not both, and verify calibration with expected calibration error (ECE) on the validation set.

---

### Question I3
**Design an augmentation pipeline for training a skin lesion classifier on dermoscopy images. The dataset has 10,000 images across 7 classes (significantly imbalanced). What augmentations would you include and why? What augmentations would you avoid?**

**Answer:**

**Domain constraints:**
- Dermoscopy images are taken with a specific illumination protocol; extreme colour distortion is unrealistic.
- Lesions appear at all orientations (the camera can be placed at any angle).
- Lesion size relative to the image varies; the lesion is typically centred.
- The task requires distinguishing fine-grained texture and colour patterns (melanoma vs. benign nevus differs in specific texture features).

**Augmentations to include:**

1. **Random rotation (0--360°, full rotation):** Dermoscopy images have no canonical orientation; the lesion appears at any angle in practice. Full rotation is valid and strongly recommended.

2. **Random horizontal and vertical flip:** Same reasoning as rotation -- no orientation constraint.

3. **Random resized crop (scale 0.8--1.0):** The lesion is approximately centred; aggressive scale changes (scale < 0.7) could crop out the lesion. Mild scale variation models different camera distances.

4. **Random affine (shear, slight scale):** Mild deformations model variation in how the skin is pressed against the dermoscope.

5. **Modest colour jitter (brightness ±15%, contrast ±15%):** Illumination variation between dermoscopes and clinical settings. Keep moderate; extreme colour distortion would change diagnostically relevant hue features.

6. **Gaussian blur (sigma 0.1--0.5):** Models slight focus variation.

7. **Random erasing (probability 0.2, area 5--20%):** Some images have hair or artefacts occluding part of the lesion; random erasing models this and forces the model to make decisions from partial information.

8. **Class-weighted sampling or class-balanced augmentation:** Given severe class imbalance, over-sample minority classes. Apply stronger augmentation (more augmented versions per epoch) to minority classes to increase their effective frequency.

**Augmentations to avoid:**

1. **Strong colour jitter / hue rotation:** Dermoscopy diagnosis is heavily colour-dependent (e.g., blue-white structures are a specific diagnostic feature). Rotating hue by 30° changes a diagnostic colour feature. Avoid hue augmentation; limit saturation changes.

2. **Mixup / CutMix with standard settings:** Mixing two different lesion images creates an image that could be diagnostically misleading. The mixed label (e.g., "50% melanoma, 50% benign") is not grounded in clinical reality. If used, keep $\alpha$ small ($\alpha = 0.1\text{--}0.2$) and monitor whether it hurts sensitivity on the minority class (melanoma).

3. **Extreme scale (crop scale < 0.5):** Cropping out most of the lesion removes the object of interest. Unlike ImageNet objects which appear throughout the image, dermoscopy lesions are always centred.

4. **JPEG compression artefacts:** Dermoscopy datasets are typically stored at high quality; adding JPEG artefacts introduces distribution shift not present in the test data.

**Full pipeline:**

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.15, contrast=0.15,
                           saturation=0.1, hue=0.0),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
```

---

## Tier 3 -- Advanced

### Question A1
**Derive the CutMix label mixing formula from first principles. Show that the expected value of the mixed label is the same as the expected label under a uniform mix of both classes, and explain what this implies for training.**

**Answer:**

**Setup.** Let $x_i \in \mathbb{R}^{C \times H \times W}$ and $x_j \in \mathbb{R}^{C \times H \times W}$ be two images with one-hot labels $y_i$ and $y_j$. A binary mask $\mathbf{M} \in \{0, 1\}^{H \times W}$ is created by setting all positions within a randomly placed $W' \times H'$ rectangle to 1 (the "cut" region) and 0 elsewhere.

The CutMix operation:

$$\tilde{x} = \mathbf{M} \odot x_j + (1 - \mathbf{M}) \odot x_i$$

(image $j$'s pixels appear in the cut region, image $i$'s pixels elsewhere).

Let $r = \frac{W' H'}{WH}$ be the area fraction of the cut region. By convention, $r \sim \text{Beta}(\alpha, \alpha)$ (i.e., $r = 1 - \lambda$ where $\lambda$ is the fraction retained from image $i$).

**Label mixing formula:**

$$\tilde{y} = (1 - r) y_i + r y_j$$

**Derivation from first principles.** The mixed input contains $(1-r)$ proportion of pixels from image $i$ and $r$ proportion from image $j$. If we interpret a linear classifier operating on all pixels, the logit for class $c$ is approximately:

$$z_c(\tilde{x}) = \sum_{h,w} f_c(\tilde{x}_{h,w}) = \sum_{(h,w) \notin \text{cut}} f_c(x_{i,h,w}) + \sum_{(h,w) \in \text{cut}} f_c(x_{j,h,w})$$

$$\approx (1-r) z_c(x_i) + r z_c(x_j)$$

The optimal prediction for the mixed image interpolates between the predictions for each component image in proportion to its pixel area contribution. Therefore, the label should be $\tilde{y} = (1-r) y_i + r y_j$.

**Expected value analysis.** Let $r \sim \text{Beta}(\alpha, \alpha)$ with $\mathbb{E}[r] = 0.5$ (for symmetric Beta). The expected mixed label:

$$\mathbb{E}_r[\tilde{y}] = (1 - \mathbb{E}[r]) y_i + \mathbb{E}[r] y_j = 0.5 y_i + 0.5 y_j$$

This is the mean of the two labels: the expected CutMix prediction is an equal mix of both classes. This is the same as drawing uniformly from $\{y_i, y_j\}$.

**Implication for training.** The gradient with respect to the model's parameters is:

$$\nabla \mathcal{L}_{CutMix} = (1-r) \nabla \mathcal{L}(y_i, p(\tilde{x})) + r \nabla \mathcal{L}(y_j, p(\tilde{x}))$$

The gradient is a weighted sum of the gradients towards predicting $y_i$ (from the uncut region) and $y_j$ (from the cut region). The model is simultaneously trained to:
- Classify the uncut region as class $i$ (proportion $1-r$ of pixels).
- Classify the cut region as class $j$ (proportion $r$ of pixels).

This forces the model to identify discriminative local regions: it cannot rely on any single region to make the correct prediction, because part of the image belongs to a different class. This is the "attention regularisation" effect of CutMix: the model learns to look at multiple discriminative regions rather than a single dominant feature.

---

### Question A2
**Explain the "Augmentation Invariance" property in self-supervised contrastive learning. How does the choice of augmentation policy define what features the model learns, and what are the failure modes when augmentations are too weak or too strong?**

**Answer:**

**Augmentation invariance in contrastive learning.**

In SimCLR and related methods (MoCo, BYOL, DINO), the training objective forces the model to produce similar representations for two different augmented views of the same image:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_1, z_2)/\tau)}{\sum_{k} \exp(\text{sim}(z_1, z_k)/\tau)}$$

where $z_1, z_2$ are embeddings of two augmented views of the same image and the denominator sums over all other images in the batch.

**The augmentation policy defines what features are invariant.** The model learns representations that are invariant to the applied augmentations and discriminative with respect to all else. Formally:

- If the augmentation includes `ColorJitter`, the learned representation is invariant to colour and therefore does not encode colour features. The model cannot use colour to distinguish images after training.
- If the augmentation includes `RandomGrayscale`, the representation is invariant to colour saturation.
- If the augmentation does NOT include `RandomRotation`, the representation is NOT invariant to rotation; the model may use orientation as a discriminative feature.

**This is both a feature and a risk.** The augmentation policy must be carefully chosen to preserve the features needed for downstream tasks:

- For object recognition: colour invariance (via `ColorJitter`) is generally beneficial since objects should be recognised across lighting conditions. Rotation invariance is partially beneficial.
- For fine-grained species classification: colour is often diagnostic (e.g., bird plumage). Too strong `ColorJitter` removes colour features that are essential for downstream performance.
- For texture classification: blur invariance (`GaussianBlur`) removes the texture features that are the primary discriminative signal.

**Failure mode: Augmentations too weak.**

If augmentations are weak (e.g., only slight brightness change), two augmented views of the same image are nearly identical. The contrastive objective is easily satisfied by memorising or slightly transforming input features, rather than learning semantic, high-level representations. The model learns features that are specific to the exact pixel values rather than abstract object identity.

In the extreme: if there are no augmentations, the two "views" are identical, the contrastive loss is trivially satisfied (the model just learns the identity function), and the representation learns nothing.

**Failure mode: Augmentations too strong.**

If augmentations are too aggressive (e.g., extreme colour distortion that changes a red apple to look purple, or a crop so small it only captures background), two augmented views of the same image may not share semantic content. The contrastive objective now asks the model to map semantically different images to the same embedding. This is the "view collapse" or "semantic inconsistency" problem: the model is forced to ignore relevant features to satisfy the invariance constraint.

Concretely: with a `RandomResizedCrop` that can crop as little as 5% of the image area, one view may show only the background (grass) while another shows only the object (dog). Forcing the dog and grass views to have the same representation means the model cannot distinguish dogs from backgrounds -- exactly the wrong inductive bias.

**Optimal augmentation regime.** SimCLR's analysis found that the combination of `RandomResizedCrop` + `ColorJitter` was the most impactful pair for ImageNet transfer performance. The two augmentations are complementary:
- `RandomResizedCrop` forces multi-scale, multi-position invariance (object-level rather than pixel-level).
- `ColorJitter` forces colour-invariant representations, discouraging the model from using easily-changed superficial features.

`GaussianBlur` (used by SimCLR but not MoCo v1) further encourages texture-invariant representations, which helps for recognition but can hurt texture-dependent tasks.

The key principle: **augmentations that destroy information irrelevant to the downstream task improve transfer; augmentations that destroy relevant information hurt transfer.** Since the downstream task is unknown at pre-training time, augmentation design for self-supervised learning requires domain knowledge about what features matter.
