# Sequence to Sequence: Encoder-Decoder and Attention

A reference covering the seq2seq framework, the encoder-decoder architecture, and the Bahdanau attention mechanism as a precursor to the Transformer. Organised by interview difficulty tier.

---

## Table of Contents

- [Fundamentals](#fundamentals)
- [Intermediate](#intermediate)
- [Advanced](#advanced)
- [The Road to Transformers](#the-road-to-transformers)
- [Common Mistakes](#common-mistakes)

---

## Fundamentals

### What is Sequence to Sequence?

A **sequence-to-sequence** (seq2seq) model maps an input sequence of arbitrary length to an output sequence of arbitrary (and generally different) length.

This differs from a standard RNN where the output sequence is the same length as the input (e.g., POS tagging) or a single vector (e.g., sentiment classification).

**Core use cases:**
- Neural machine translation (NMT): "The cat sat" $\rightarrow$ "Le chat s'est assis"
- Text summarisation: long document $\rightarrow$ short summary
- Speech recognition: audio frames $\rightarrow$ word sequence
- Code generation: natural language description $\rightarrow$ code
- Question answering (generative): question + context $\rightarrow$ answer

### The Encoder-Decoder Architecture

The classic seq2seq model (Sutskever, Vinyals & Le, 2014) uses two RNNs:

**Encoder**: reads the input sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T$ and produces a fixed-size **context vector** $\mathbf{c}$ (also called the thought vector):

$$\mathbf{h}_t^{(e)} = f_{\text{enc}}(\mathbf{x}_t, \mathbf{h}_{t-1}^{(e)})$$

$$\mathbf{c} = \mathbf{h}_T^{(e)}$$

The context vector is the final hidden state of the encoder.

**Decoder**: an RNN that generates the output sequence one token at a time, conditioned on $\mathbf{c}$:

$$\mathbf{s}_0 = \mathbf{c}$$

$$\mathbf{s}_t = f_{\text{dec}}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1})$$

$$P(\mathbf{y}_t | \mathbf{y}_{<t}, \mathbf{x}) = \text{softmax}(W_o \mathbf{s}_t)$$

At each step, the decoder predicts the next token from the hidden state $\mathbf{s}_t$. The predicted token is fed as input to the next step (or the true token during training — this is **teacher forcing**).

**Key architectural decisions in the original paper:**
- Deep LSTM (4 layers) for both encoder and decoder.
- Input sequence reversed before encoding — shorter paths from encoder inputs to decoder outputs, which helped gradient flow.
- Beam search (beam width 12) at inference for better output quality.

### Training: Teacher Forcing

During training, the decoder receives the **true** previous output token $\mathbf{y}_{t-1}$ rather than its own prediction $\hat{\mathbf{y}}_{t-1}$. This is teacher forcing.

Benefits:
- Faster convergence: the decoder always gets correct context and never cascades errors.
- Stable gradients: avoids the compounding of errors during training.

Drawback — **exposure bias**: at test time the decoder must use its own (imperfect) predictions, which creates a distribution mismatch with training. This can be addressed with scheduled sampling (gradually replacing teacher tokens with model predictions as training proceeds).

### Inference: Greedy Search vs Beam Search

**Greedy search**: at each step, pick the highest-probability token.

$$\hat{\mathbf{y}}_t = \arg\max_y P(y | \mathbf{y}_{<t}, \mathbf{x})$$

Simple but suboptimal: a locally best choice can lead to a globally poor sequence.

**Beam search**: maintain a set of $k$ (beam width) candidate sequences. At each step, expand each candidate by all vocabulary entries, score the expanded sequences, and keep the top-$k$.

$$\text{score}(\mathbf{y}) = \sum_{t=1}^{T} \log P(\mathbf{y}_t | \mathbf{y}_{<t}, \mathbf{x})$$

Beam search is the standard at inference for seq2seq models. Larger beams generally improve quality up to a point (beam width 4-10 is typical), after which they give diminishing returns and can even hurt (the model becomes overly conservative).

**Length normalisation**: beam search is biased toward shorter sequences because each token adds a $\log P(\cdot) \leq 0$ term. Divide the score by $T^\alpha$ (typically $\alpha = 0.6$--$0.7$) to penalise length bias.

---

## Intermediate

### The Bottleneck Problem

The core limitation of the vanilla seq2seq model is that the **entire input sequence must be compressed into a single fixed-size context vector** $\mathbf{c}$. No matter how long the input, the context vector has the same dimensionality.

For long sequences:
- The encoder's final hidden state has difficulty capturing all relevant information from the beginning of the sequence.
- BLEU scores on NMT degrade sharply as input sentence length increases.
- Information from early tokens is further from the context vector in the computation graph, so their influence on the decoder is weaker.

This bottleneck is the primary motivation for the attention mechanism.

### Bahdanau Attention

Bahdanau, Cho & Bengio (2015) proposed **attention**: instead of a single context vector, the decoder creates a different context vector for each output step, formed as a weighted sum over all encoder hidden states.

**The attention mechanism (additive attention):**

Given encoder hidden states $\mathbf{h}_1^{(e)}, \ldots, \mathbf{h}_T^{(e)}$ and the decoder state at step $t-1$, $\mathbf{s}_{t-1}$:

1. **Score** each encoder state against the decoder state:

$$e_{t,j} = \mathbf{v}_a^{\top} \tanh(W_a \mathbf{s}_{t-1} + U_a \mathbf{h}_j^{(e)})$$

2. **Normalise** with softmax to get attention weights:

$$\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T} \exp(e_{t,k})}$$

3. **Compute** the context vector as a weighted sum:

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{t,j} \mathbf{h}_j^{(e)}$$

4. **Update** the decoder state using both $\mathbf{c}_t$ and the previous output:

$$\mathbf{s}_t = f_{\text{dec}}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}_t)$$

**What the attention weights mean:** $\alpha_{t,j}$ is the probability that the $t$-th output token should "attend to" the $j$-th input token. For NMT, learned attention patterns often correspond to word alignments — the model discovers that "chat" in French corresponds to "cat" in English.

**Why this solves the bottleneck:** the decoder can access the full encoder hidden state sequence at every step. Information does not need to be squeezed into a single vector; each output step retrieves exactly the relevant context.

### Luong Attention

Luong et al. (2015) proposed a simpler, computationally cheaper attention variant using the current decoder state (not $\mathbf{s}_{t-1}$ as in Bahdanau):

**Dot-product scoring** (requires equal dimensions):

$$e_{t,j} = \mathbf{s}_t^{\top} \mathbf{h}_j^{(e)}$$

**General scoring** (allows different dimensions):

$$e_{t,j} = \mathbf{s}_t^{\top} W_a \mathbf{h}_j^{(e)}$$

**Comparison:**

| Feature | Bahdanau | Luong |
|---|---|---|
| Scoring function | Additive (MLP) | Multiplicative (dot/general) |
| Query | $\mathbf{s}_{t-1}$ (previous step) | $\mathbf{s}_t$ (current step) |
| Complexity | $O(d)$ per score | $O(1)$ or $O(d^2)$ per score |
| Context use | Input to decoder | Concatenated after decoder |

Luong attention is faster; Bahdanau is slightly more expressive. In practice the difference is small.

### The Copy Mechanism

Standard seq2seq predicts from a fixed vocabulary. For tasks like summarisation or dialogue, the model should sometimes **copy** words directly from the input (proper nouns, numbers, rare words).

The **pointer network** / **copy mechanism** (Gu et al., 2016; See et al., 2017) extends attention: at each step, choose either to generate from the vocabulary or to copy an input token, controlled by a soft switch $p_{\text{gen}} \in [0,1]$:

$$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{j: x_j = w} \alpha_{t,j}$$

This allows the model to handle out-of-vocabulary (OOV) words that appear in the input.

---

## Advanced

### Attention as Key-Value-Query Retrieval

Reframing attention in the language of databases reveals the generalisation to Transformer attention.

Given a query $\mathbf{q}$ (e.g., the decoder state), keys $\mathbf{k}_j$ and values $\mathbf{v}_j$ (from the encoder):

$$\text{Attention}(\mathbf{q}, K, V) = \sum_j \text{softmax}(\text{score}(\mathbf{q}, \mathbf{k}_j)) \cdot \mathbf{v}_j$$

In Bahdanau attention: keys and values are both the encoder hidden states $\mathbf{h}_j^{(e)}$, and the scoring function is additive.

In Transformer **scaled dot-product attention**: keys, values, and query are separate linear projections; the score is $\mathbf{q}^{\top}\mathbf{k}_j / \sqrt{d_k}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$$

The $1/\sqrt{d_k}$ scaling prevents the dot products from growing large in high dimensions, which would push softmax into regions with tiny gradients.

This abstraction shows that RNN-based attention and Transformer attention are the same computational pattern: the Transformer simply removes the recurrence entirely and makes all positions attend to all positions in parallel.

### Sequence-Level Training Objectives

Standard seq2seq is trained with **token-level cross-entropy** (teacher forcing), maximising:

$$\mathcal{L}_{\text{MLE}} = -\sum_{t=1}^{T} \log P(\mathbf{y}_t^* | \mathbf{y}_{<t}^*, \mathbf{x})$$

**Problems:**
1. Exposure bias (train on gold, test on predictions).
2. Mismatch between training objective (token-level likelihood) and evaluation metric (e.g., BLEU, ROUGE), which are defined at sequence level.

**REINFORCE / Minimum Risk Training (MRT)**: train by directly optimising a sequence-level reward $r(\mathbf{y})$:

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\hat{\mathbf{y}} \sim P(\cdot | \mathbf{x})}[r(\hat{\mathbf{y}})]$$

The gradient is estimated by sampling and uses the REINFORCE estimator:

$$\nabla_\theta \mathcal{L}_{\text{RL}} \approx -\frac{1}{K}\sum_{k=1}^{K} (r(\hat{\mathbf{y}}^{(k)}) - b) \nabla_\theta \log P(\hat{\mathbf{y}}^{(k)} | \mathbf{x})$$

where $b$ is a baseline (e.g., greedy reward) to reduce variance.

In practice, models are first warm-started with MLE training, then fine-tuned with REINFORCE. This is also the basis of RLHF in large language models.

### Connectionist Temporal Classification (CTC)

For speech recognition and OCR, the input (audio frames, image columns) and output (phonemes, characters) are not aligned. CTC (Graves et al., 2006) solves this without an explicit attention mechanism.

CTC introduces a blank token $\varnothing$ and allows the model to output many tokens per input frame, then collapses the output:
- Remove consecutive duplicates.
- Remove blank tokens.

The training loss marginalises over all valid alignments:

$$P(\mathbf{y} | \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} P(\pi_t | \mathbf{x})$$

computed efficiently with dynamic programming (forward-backward algorithm).

CTC is still widely used in streaming speech recognition due to its simplicity and ability to produce outputs before the full input is consumed.

---

## The Road to Transformers

The development of seq2seq with attention directly led to the Transformer:

| Component | RNN Seq2Seq | Transformer Seq2Seq |
|---|---|---|
| Sequence encoding | Recurrent (sequential) | Self-attention (parallel) |
| Input representation | Hidden state $\mathbf{h}_t$ | Positional embedding + linear |
| Cross-sequence attention | Bahdanau/Luong attention | Multi-head cross-attention |
| Long-range dependencies | Limited (gradient vanishing) | Direct (any-to-any attention) |
| Training parallelism | Limited (time dependency) | Full (all positions at once) |
| Computational cost | $O(T)$ sequential steps | $O(T^2)$ attention (parallelisable) |

The Transformer ("Attention is All You Need", Vaswani et al., 2017) showed that the recurrence is not necessary — attention alone is sufficient for seq2seq tasks, and without recurrence, training can be fully parallelised over the sequence length, enabling scaling to much larger models and datasets.

The key insight: if attention can reach any position in the sequence directly (in one step), the long-range dependency problem that motivated LSTM and GRU is bypassed entirely.

---

## Common Mistakes

1. **Saying the encoder passes only the final hidden state to the decoder when attention is used**: with attention, the decoder accesses ALL encoder hidden states. The fixed context vector is replaced by a step-specific weighted combination.

2. **Confusing teacher forcing at training vs inference**: at training, the decoder sees ground-truth previous tokens; at inference, it sees its own predictions. This discrepancy is the exposure bias problem.

3. **Forgetting length normalisation in beam search**: raw beam search favours short sequences. Always apply length normalisation when comparing sequences of different lengths.

4. **Conflating attention weights with word importance**: $\alpha_{t,j}$ shows what the model attends to, not necessarily what is linguistically important. Attention weights are not reliable explanations and should not be treated as ground-truth alignments.

5. **Treating Bahdanau and Luong attention as interchangeable implementations of the same thing**: they differ in whether the query is the previous or current decoder state, and in how the context vector is combined with the decoder, which affects the computational graph.
