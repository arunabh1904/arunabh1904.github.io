---
layout: post
title: "Attention Is All You Need"
date: 2017-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2017 – Attention Is All You Need

**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

**GitHub:** [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) (reference code)

**Project page / blog:** [Google Research 
overview](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)

**Conference:** NeurIPS 2017
![Transformer Architecture](/assets/images/attention.png)


**Abstract in a nutshell:** The Transformer removes recurrence and convolutions, relying solely on 
attention to join an encoder and decoder. This allows full parallelisation in training and 
inference while still modelling long-range dependencies. On WMT 2014 machine translation the "big" 
version beats the prior single-model state of the art (28.4 BLEU En→De, 41.8 BLEU En→Fr) and 
trains in 3.5 days on eight GPUs. It also generalises to tasks like English constituency parsing.

**Novel insights:**
- All-attention architecture proves sequence transduction does not need recurrence.
- Multi-head self-attention attends to different positions or representation sub-spaces in parallel.
- Sinusoidal positional encodings inject order without sequential operations.
- Training efficiency comes from a constant-length computation path and GPU-friendly operations.

**Evals / latency benchmarks:**

| Task / Model | BLEU / F1 | Notes |
| ------------ | --------- | ----- |
| WMT 14 En→De (base) | 27.3 BLEU | Comparable to prior SOTA with fewer parameters |
| WMT 14 En→De (big) | 28.4 BLEU | > 2 BLEU over previous best single models; 3.5 days on 8 GPUs |
| WMT 14 En→Fr (big) | 41.8 BLEU | New SOTA with same 3.5-day budget |
| Penn Treebank parsing | 95 F1 | Demonstrates domain transfer beyond MT |

**Critiques & reflections:**
- **What we liked:** Elegant, conceptually simple architecture with strong empirical gains. Highly 
parallelisable; huge speed-up over RNNs/CNNs on modern hardware. Spawned a vibrant ecosystem (BERT, 
GPT, ViT, etc.).
- **What could be better:** Quadratic memory/compute in sequence length makes very long contexts 
costly. Needs large datasets; can underperform RNNs on tiny corpora. Original paper limited to 
\(\leq\) 1k-token contexts, spurring efficient Transformer research.

**Take-away:** By proving that "attention is all you need," this paper redefined sequence 
modelling, unlocked massive parallelism and laid the groundwork for today's pre-trained foundation 
models.

