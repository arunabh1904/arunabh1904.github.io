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
allowing eight P100 GPUs to reach 28.4 BLEU on English→German in only 3.5 days.

At the heart of the architecture is the scaled attention mechanism

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.
\]

The scaling by \(\sqrt{d_k}\) stabilises gradients at large hidden dimensions.
Multi-head attention, typically eight heads with \(d_k = d_v = 64\),
preserves information that a single head would compress.

<img src="/assets/images/attention.png" alt="Transformer" style="max-width:60%;margin:1rem auto;display:block;">

### Positional encoding

Without recurrence, positional information comes from sinusoidal encodings:

\[
\begin{aligned}
\text{PE}(pos, 2i) &= \sin\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr),\\
\text{PE}(pos, 2i+1) &= \cos\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr).
\end{aligned}
\]

These fixed oscillations embed relative order and generalise to longer sequences with no extra parameters.

### Efficiency and empirical results

Self-attention offers a constant path length but is quadratic in sequence length.
For translation tasks this is acceptable, and the design parallelises well on GPUs.
The "big" model attains 41.0 BLEU on English→French, far exceeding contemporary CNN and RNN baselines.

---

### Ongoing challenges

Memory grows quadratically with sequence length, limiting contexts beyond a few thousand tokens.
Smaller datasets often still favour RNNs.
Nevertheless, Transformers now underpin BERT, GPT and Vision Transformers.
