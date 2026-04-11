# Attention and Transformers

## Overview

The attention mechanism and the Transformer architecture are the foundation of modern deep learning. Since 2017, Transformers have replaced RNNs and CNNs across most sequence and (increasingly) image tasks. Every contemporary LLM — GPT, Claude, Gemini, Llama — is built on this architecture. Understanding attention, multi-head attention, positional encodings, and the overall Transformer block is essential for any ML engineering role involving modern models.

---

## Key Concepts

### 1. The Attention Mechanism

**Intuition:** given a query, compute a weighted sum of values, where the weights are determined by how well the query matches each key.

**Formal definition:**

Given queries Q, keys K, values V:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$ — queries (n queries, each of dim $d_k$).
- $K \in \mathbb{R}^{m \times d_k}$ — keys.
- $V \in \mathbb{R}^{m \times d_v}$ — values.
- $\sqrt{d_k}$ — scaling factor to prevent softmax saturation in high dimensions.

**Steps:**

1. **Compute similarity scores:** $QK^T$ — dot product of each query with every key. Shape: $n \times m$.
2. **Scale:** divide by $\sqrt{d_k}$ to keep magnitudes reasonable as $d_k$ grows.
3. **Softmax:** convert scores to weights summing to 1 per query.
4. **Weighted sum:** multiply weights by V, giving each query a value that's a mixture of all V vectors, weighted by relevance.

**Interpretation:**

Attention answers "for each query, which values should it look at, and how much?". The query decides what to look for; the keys describe what each value is; the softmax picks the relevant ones.

---

### 2. Self-Attention

**Self-attention** is attention where Q, K, V are all derived from the same input sequence. Each token attends to all other tokens (including itself).

Given an input sequence $X \in \mathbb{R}^{n \times d}$:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices.

**Why it's powerful:**

- **Direct connections.** Any token can attend to any other in a single step, regardless of distance. Contrast with RNNs, where distant tokens must pass through many recurrent states.
- **Parallel computation.** All positions computed simultaneously. Much faster to train than RNNs.
- **Content-dependent routing.** The model learns what connections matter.

**Example:**

In "The animal didn't cross the street because it was tired", the word "it" attends strongly to "animal". In "The animal didn't cross the street because it was slippery", "it" attends to "street". Self-attention resolves reference dynamically based on content.

---

### 3. Multi-Head Attention

**Idea:** instead of one big attention operation, do several smaller ones in parallel, then concatenate their outputs.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

Each head has its own set of projections $(W_Q^i, W_K^i, W_V^i)$. Typically, $d_k = d/h$ so total parameters stay constant.

**Why multiple heads:**

1. **Different subspaces.** Each head can learn to focus on different kinds of relationships — syntactic, semantic, positional.
2. **Redundancy and robustness.** Losing one head degrades gracefully.
3. **Ensemble-like.** Each head makes independent decisions, then the outputs are combined.

**Typical values:**

- GPT-3: 96 heads of dim 128 each ($d = 12288$).
- BERT-base: 12 heads of dim 64 each ($d = 768$).

---

### 4. The Transformer Block

A **Transformer block** (one layer of the model) consists of:

1. **Multi-head self-attention.**
2. **Residual connection and layer norm.**
3. **Feed-forward network (FFN).**
4. **Residual connection and layer norm.**

**Pseudo-code:**

```
def transformer_block(x):
    a = multi_head_attention(x)
    x = layer_norm(x + a)    # residual + norm

    f = ffn(x)
    x = layer_norm(x + f)    # residual + norm
    return x
```

**FFN** is typically:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

A two-layer MLP applied position-wise (each token independently). The hidden dimension is usually 4x the model dimension — so a model with $d=768$ has an FFN hidden dim of 3072.

**Stacking blocks:** modern LLMs use dozens to hundreds of Transformer blocks. GPT-3 has 96 layers; GPT-4 is estimated to have more.

---

### 5. Positional Encodings

**Problem:** self-attention is permutation-invariant. Without additional information, the model can't distinguish "dog bites man" from "man bites dog".

**Solution:** add positional information to the input embeddings.

**Sinusoidal positional encodings (original Transformer):**

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

For each position $pos$ and each dimension $i$, compute sine or cosine with a specific frequency. The result is added to the token embedding.

**Learned positional embeddings:**

BERT and GPT use a learned embedding per position. Simpler but limited to the maximum sequence length seen during training.

**Rotary positional embeddings (RoPE):**

Used by GPT-NeoX, Llama, etc. Applies a rotation to queries and keys based on their position. Naturally extrapolates to longer sequences and encodes relative position elegantly.

**ALiBi (Attention with Linear Biases):**

Instead of adding to embeddings, subtract a linear bias from attention scores proportional to the distance between query and key. Used in some models for better extrapolation.

---

### 6. Encoder vs Decoder vs Encoder-Decoder

**Encoder-only (BERT-style):**

- Bidirectional self-attention (each token attends to all others).
- Used for classification, feature extraction, embedding.
- Pre-trained with masked language modelling.

**Decoder-only (GPT-style):**

- Causal self-attention (each token only attends to previous tokens).
- Used for text generation.
- Pre-trained with next-token prediction.
- **Most modern LLMs are decoder-only.**

**Encoder-decoder (T5, original Transformer):**

- Encoder processes the input bidirectionally.
- Decoder attends to encoder output via **cross-attention** plus its own causal self-attention.
- Used for translation, summarisation, sequence-to-sequence tasks.

**Causal masking:**

In decoder self-attention, the attention matrix is masked so token $i$ can only attend to tokens $\leq i$. This is implemented by adding $-\infty$ to disallowed positions before the softmax.

---

## Interview Questions

### Q1. Why is the attention score divided by $\sqrt{d_k}$?

**Answer:** Without scaling, the dot products $QK^T$ grow in magnitude as $d_k$ increases. Large values push softmax into its saturation region — most weights become nearly 0 or 1, and gradients through softmax become very small (vanishing gradients). Dividing by $\sqrt{d_k}$ normalises the variance of the dot products (assuming Q and K entries are roughly independent with unit variance), keeping softmax in a well-behaved region.

### Q2. Why use multi-head attention instead of a single larger head?

**Answer:** (1) Multiple heads can attend to different positions or relationships simultaneously. A single head tends to concentrate attention on one pattern. (2) Multi-head attention allows learning different "projection subspaces" — one head might focus on syntax, another on coreference, another on long-range dependencies. (3) Empirically, multi-head outperforms equivalent single-head models with the same parameter budget. (4) It's parallelisable per head.

### Q3. How does the Transformer handle position information?

**Answer:** Since self-attention is permutation-invariant, positional information must be added explicitly. The original paper used sinusoidal positional encodings: sine/cosine functions of varying frequencies added to token embeddings. Modern models use learned embeddings, RoPE (rotary positional embeddings), or ALiBi. Each has trade-offs: learned embeddings don't extrapolate beyond training length; sinusoidal and RoPE generalise better to longer sequences.

### Q4. Why are Transformers more parallelisable than RNNs?

**Answer:** RNNs process tokens sequentially — token $t+1$ depends on the hidden state from token $t$. This creates a sequential bottleneck during training; you can't compute step $t+1$ until $t$ finishes. Transformers compute self-attention over all positions simultaneously — the entire attention matrix is one matrix multiplication. This maps efficiently to GPU parallelism and lets training scale to much larger models. During inference (for decoder-only), generation is still sequential because each token depends on the previous; but training exploits parallelism.

### Q5. What is the computational complexity of self-attention in terms of sequence length?

**Answer:** $O(n^2 \cdot d)$ where $n$ is sequence length and $d$ is embedding dimension. The $n^2$ comes from computing attention scores between every pair of tokens. This is the main scaling bottleneck — doubling sequence length quadruples the memory and compute needed. Long-context models (100K+ tokens) require specialised techniques:

- **FlashAttention:** IO-aware algorithm that avoids materialising the $n^2$ attention matrix.
- **Sparse attention:** only compute attention for a subset of token pairs.
- **Linear attention:** approximate self-attention with $O(n)$ complexity.
- **Sliding window attention:** limit each token's attention to a local window.

### Q6. Explain the role of layer normalisation in Transformers.

**Answer:** Layer normalisation normalises activations across the feature dimension (per token), with learnable scale and shift. Its purposes: (1) **Training stability.** Deep networks with residual connections can have activations growing without bound. LN keeps them in a controlled range. (2) **Gradient flow.** Normalising inputs to each sub-layer prevents vanishing/exploding gradients. (3) **Faster convergence.** Empirically, LN lets Transformers train more efficiently.

**Pre-LN vs post-LN:**

- **Post-LN (original):** `x = LN(x + Sublayer(x))`. Harder to train; requires learning-rate warmup.
- **Pre-LN (modern):** `x = x + Sublayer(LN(x))`. More stable; most modern LLMs use this.

### Q7. Why do decoder-only models dominate current LLMs?

**Answer:** Several reasons: (1) **Unified training and inference.** Next-token prediction is a single, scalable training objective. (2) **Universal task formulation.** Any NLP task can be framed as text generation. (3) **Transfer learning.** Pre-training gives general language understanding; fine-tuning adapts to specific tasks. (4) **Scaling laws.** Decoder-only models scale smoothly with data and compute, reaching emergent capabilities at scale. (5) **Simplicity.** One architecture for everything. Encoder-decoder models (T5) are competitive for specific tasks, and encoder-only (BERT) remains useful for embeddings and classification, but decoder-only has won the general-purpose race.

### Q8. What is KV caching and why is it important for inference?

**Answer:** During autoregressive generation, each new token attends to all previous tokens. Naively, you'd recompute attention over the entire prefix for each new token — $O(n^2)$ per step, $O(n^3)$ total. **KV caching** stores the keys and values for all previous tokens, so generating token $t$ only requires computing the new token's query against cached keys ($O(n)$ per step, $O(n^2)$ total). Trades memory for speed. The KV cache is often the memory bottleneck in serving LLMs — scaling with batch size and context length.

### Q9. What is the difference between self-attention and cross-attention?

**Answer:** **Self-attention:** Q, K, V all come from the same sequence. Each token attends to other tokens in the same sequence. **Cross-attention:** Q comes from one sequence (usually the decoder's hidden states) while K and V come from another (usually the encoder's output). Used in encoder-decoder models to let the decoder "look at" the encoded input. For example, in translation, the decoder's queries ask "what English word corresponds to this French context?" and the keys/values are the encoder's French representations.

### Q10. Explain the "residual connection" in a Transformer block.

**Answer:** A residual connection (or skip connection) adds the input of a sublayer to its output: `output = x + Sublayer(x)`. Purposes: (1) **Gradient flow.** Gradients can flow directly through the addition, bypassing potentially difficult sublayers. Prevents vanishing gradients in deep networks. (2) **Easier optimisation.** The network needs only learn the "residual" adjustment, which is often small and simple. (3) **Identity preservation.** If a sublayer is initialised to zero, the block behaves like an identity function — a safe starting point. Without residual connections, Transformers (and ResNets) wouldn't scale to the depths they do.

### Q11. What does scaling do for Transformer performance (scaling laws)?

**Answer:** Empirically, Transformer performance (measured by validation loss on text) follows a power law in three quantities: **parameters**, **data**, and **compute**. Increasing any one (with the others adequate) predictably improves performance. The OpenAI and DeepMind scaling law papers formalised this — there's no visible "ceiling" within the scales explored (billions to trillions of parameters). Key findings:

1. **Compute-optimal scaling (Chinchilla):** for a given compute budget, ~20 tokens per parameter is optimal.
2. **Emergent abilities.** Some tasks (arithmetic, chain-of-thought reasoning) only work above certain scales.
3. **Diminishing returns.** Log-linear improvements — doubling parameters yields a small, predictable drop in loss.

Scaling laws are the empirical backbone of the current LLM revolution.

### Q12. Explain chain-of-thought prompting and why it works.

**Answer:** **Chain-of-thought (CoT) prompting:** include worked examples in the prompt that show step-by-step reasoning, not just final answers. The model, following the pattern, produces its own reasoning before answering.

**Example:**

Prompt: "Q: Alice has 3 apples. She gives 2 to Bob. How many does Alice have? A: Alice starts with 3. She gives 2 away. 3 - 2 = 1. Alice has 1 apple. Q: ..."

**Why it works:**

1. **Longer computation.** Generating intermediate tokens gives the model more "think time" per question. Internal state reflects partial solutions.
2. **Explicit structure.** The model learns to decompose problems into steps.
3. **Error correction.** Intermediate steps can catch and correct mistakes.
4. **Alignment with training distribution.** Pretraining corpora contain many step-by-step explanations; CoT matches this distribution.

**Emergence:** CoT improves large models but doesn't help small ones. It requires the model to have enough capacity to reason multi-step. A characteristic of "emergent abilities".

### Q13. What is mixture-of-experts (MoE) and how does it relate to Transformers?

**Answer:** **Mixture-of-Experts (MoE):** replace a dense feed-forward layer with multiple "expert" FFNs plus a router that selects which experts to activate for each token. Only a fraction of parameters are active per token (e.g., 2 out of 64 experts).

**Why:**

1. **Parameter efficiency.** Total parameter count is huge (hundreds of billions), but compute per token is small (equivalent to a much smaller dense model).
2. **Specialisation.** Different experts learn different kinds of content — code, math, dialogue.
3. **Scaling.** MoE lets models be larger without proportionally more inference cost.

**Examples:** Mixtral 8x7B (8 experts, 2 active per token), GPT-4 (rumored), Grok-1 (8 experts).

**Challenges:**

1. **Load balancing.** Some experts may get too much traffic; others idle. Auxiliary loss terms encourage balance.
2. **Communication overhead.** Experts are typically sharded across GPUs; routing requires all-to-all communication.
3. **Training instability.** The router can collapse to using one expert. Tricks like expert dropout mitigate.

### Q14. What is attention's O(n²) problem and how does FlashAttention address it?

**Answer:** Standard attention requires materialising the full $n \times n$ attention matrix, which is $O(n^2)$ memory. For long sequences (100K tokens), this is infeasible.

**FlashAttention (Dao et al., 2022):** an IO-aware algorithm that computes attention exactly (not approximate) without materialising the full matrix. Key ideas:

1. **Tile computation.** Process blocks of Q, K, V at a time.
2. **Streaming softmax.** Incrementally update the softmax normaliser as new blocks arrive. No need to store all scores.
3. **SRAM/HBM awareness.** Keep active tiles in fast SRAM, minimising slow HBM accesses.

**Results:** 2-3x faster forward and backward, 10-20x less memory. Supports much longer sequences on the same hardware. Part of PyTorch 2.0+ via `scaled_dot_product_attention`.

**FlashAttention-2, FlashAttention-3** further optimise, especially for H100 GPUs.

### Q15. How does the attention mechanism extend beyond NLP?

**Answer:** Attention and Transformers have spread to many domains:

1. **Vision (Vision Transformers, ViT).** Split an image into patches, treat them as tokens, apply a Transformer. Competitive or better than CNNs on large datasets.
2. **Multimodal models (CLIP, GPT-4V).** Joint text + image Transformers for tasks like captioning, visual Q&A.
3. **Audio (Whisper, AudioLM).** Transformer encoder-decoder for speech recognition and synthesis.
4. **Video.** Extended ViTs for temporal modelling.
5. **Protein structure (AlphaFold 2).** Self-attention over amino acid residues, combined with geometry-aware modules.
6. **Reinforcement learning (Decision Transformer).** Frame RL trajectories as sequences; generate actions autoregressively.
7. **Code generation (Codex, CodeLlama).** Treat code as text; standard Transformers work well.

The attention mechanism is a general-purpose "learnable routing" — wherever you have structured data that benefits from content-dependent connections, a Transformer is a strong candidate.

---

## Interview Tips

- **Know the math.** Write out the attention formula without looking it up. Explain each term.
- **Understand scaling.** The $\sqrt{d_k}$ division is a common interview question.
- **Discuss trade-offs.** Multi-head vs single-head, encoder-decoder vs decoder-only, scaling laws and compute optimality.
- **Know production concerns.** KV caching, FlashAttention, MoE, quantisation. ML engineering roles care about these as much as the math.
- **Modern developments.** RoPE, ALiBi, sliding window attention, RMSNorm vs LayerNorm. Stay current.

Transformers have reshaped deep learning. Every serious ML interview touches them. Understanding the architecture deeply — not just the surface — distinguishes a candidate who has used models from one who can build them.
