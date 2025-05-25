---
layout: content
title: "Attention Is All You Need"
date: 2017-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2017 – Attention Is All You Need

**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

**GitHub:** [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) (reference code)

**Project page / blog:**
[Google Research overview](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)

**Conference:** NeurIPS 2017

The central innovation of Vaswani et al. (pp. 1–2) lies not in inventing a new operation.
Instead it rejects recurrent and convolutional biases.
Stacking purely attention-based blocks shortens the path between any two tokens to one step.
This greatly improves gradient flow and enables massive parallelism,
allowing eight P100 GPUs to reach 28.4 BLEU (Bilingual Evaluation Understudy, a translation quality score)
on English→German in only 3.5 days.

At the heart of the architecture is the scaled attention mechanism:

```python
# Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = torch.nn.functional.dropout(weights, p=dropout_p)
    output = torch.matmul(weights, V)
    return output, weights
```

The scaling by \(\sqrt{d_k}\) keeps the logits numerically stable as \(d_k\) grows.
The additive mask inserts \(-\infty\) so the softmax cleanly ignores blocked positions.
Multi-head attention, typically eight heads with \(d_k = d_v = 64\),
preserves information that a single head would compress.

<img src="/assets/images/attention.png" alt="Transformer" style="max-width:60%;margin:1rem auto;display:block;">

### Positional encoding

Without recurrence, positional information comes from sinusoidal encodings:

```python
# PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

These fixed oscillations embed relative order and generalise to longer sequences.
Pre-compute the matrix once, stash it on the device and add it to your token embeddings at each forward pass.

### Efficiency and empirical results

Self-attention offers a constant path length but is quadratic in sequence length.
For translation tasks this is acceptable, and the design parallelises well on GPUs.
The "big" model attains 41.0 BLEU on English→French, far exceeding contemporary CNN and RNN baselines.

---

### Ongoing challenges

Memory grows quadratically with sequence length, limiting contexts beyond a few thousand tokens.
Smaller datasets often still favour RNNs.
Nevertheless, Transformers now underpin BERT, GPT and Vision Transformers.
