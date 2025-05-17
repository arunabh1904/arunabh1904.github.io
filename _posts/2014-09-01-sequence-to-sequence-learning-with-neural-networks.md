---
layout: post
title: "Sequence to Sequence Learning with Neural Networks"
date: 2014-09-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2014 – Sequence to Sequence Learning with Neural Networks

**arXiv:** [1409.3215](https://arxiv.org/abs/1409.3215)

**GitHub:** [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) (community implementation)

**Project page / blog:** [Google Research Blog – "A neural network for machine translation"](https://research.googleblog.com/2014/12/a-neural-network-for-machine-translation.html)

**Conference:** NeurIPS 2014
![Seq2Seq Architecture](/assets/images/seq2seq.png)


**Summary (abstract in plain English):**  
The authors propose an end-to-end encoder–decoder approach for mapping a source sequence to a target sequence. A multi-layer LSTM encodes the input into a single vector, and a second LSTM decodes that vector to produce the output one token at a time. Trained solely on parallel text, a 4-layer model achieved 34.8 BLEU on WMT’14 English→French, beating a strong phrase-based SMT baseline. Using an ensemble or reranking pushed BLEU to 36.5.

**Novel insights:**
- A unified encoder–decoder framework laid the groundwork for neural machine translation and other sequence tasks.
- Reversing the input sentence shortened gradient paths and improved BLEU by about four points.
- Large deep LSTMs with dropout performed well with minimal task-specific tuning.
- Reranking an SMT n-best list with the neural model provided additional gains.

**Evals / Latency benchmarks:**

| Dataset | Model | BLEU | Notes |
| ------- | ----- | ---- | ----- |
| WMT’14 En→Fr | 4-layer LSTM (reversed) | 34.8 | Single model, vocab 160k |
| WMT’14 En→Fr | + SMT 1000-best rerank | 36.5 | Ensemble of 5 nets |
| Training speed | – | – | ~1 week on 8 × K40 GPUs, ~3 days to converge |
| Decoding | Beam = 12 | – | ~0.11 s / sentence on CPU |

**Critiques & limitations:**
- **What works well:** Clean architecture without hand-crafted alignments and reproducible hyper-parameters.
- **Limitations:** Fixed-length bottleneck restricts very long sentences and training large vocabularies requires significant memory.
- Lack of an official implementation left early adopters guessing about details.

**Take-home message:**  
Sutskever et al.’s encoder–decoder LSTM showed that a general neural network could outperform traditional systems on machine translation, catalysing modern sequence modelling.
