---
title: Sequence to Sequence Learning with Neural Networks
date: '2014-09-01T04:00:00.000Z'
section: paper-shorts
postSlug: sequence-to-sequence-learning-with-neural-networks
legacyPath: >-
  /paper
  shorts/2014/09/01/sequence-to-sequence-learning-with-neural-networks.html
tags:
  - Other
field: Natural Language Processing
summary: 2014 – Sequence to Sequence Learning with Neural Networks
---
## 2014 – Sequence to Sequence Learning with Neural Networks

**arXiv:** [1409.3215](https://arxiv.org/abs/1409.3215)

**GitHub:** [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) (community implementation)

**Project page / blog:** [Google Research Blog – "A neural network for machine translation"](https://research.googleblog.com/2014/12/a-neural-network-for-machine-translation.html)

**Conference:** NeurIPS 2014
![Seq2Seq Architecture](/assets/images/seq2seq.png)


**Summary:** Sutskever et al. showed that translation could be treated as a general sequence-to-sequence problem. A multi-layer LSTM encodes the source sentence into a single vector, and a second LSTM decodes that vector into the target sentence one token at a time. Trained only on parallel text, a 4-layer model reached 34.8 BLEU on WMT'14 English to French, beating a strong phrase-based SMT baseline. Ensembles and SMT reranking pushed the score to 36.5.

The paper also contains one of those oddly practical details that becomes famous: reversing the input sentence shortened gradient paths and improved BLEU by about four points. The larger contribution, though, was the encoder-decoder abstraction. It gave neural machine translation and later sequence tasks a clean template: encode the input, decode the output, and learn the mapping end to end.

**Evals / Latency benchmarks:**

| Dataset | Model | BLEU | Notes |
| ------- | ----- | ---- | ----- |
| WMT’14 En→Fr | 4-layer LSTM (reversed) | 34.8 | Single model, vocab 160k |
| WMT’14 En→Fr | + SMT 1000-best rerank | 36.5 | Ensemble of 5 nets |
| Training speed | – | – | ~1 week on 8 × K40 GPUs, ~3 days to converge |
| Decoding | Beam = 12 | – | ~0.11 s / sentence on CPU |

**Critiques & limitations:** The architecture was clean because it removed hand-built alignments, but the fixed-length vector became an obvious bottleneck for long sentences. Large vocabularies also made training expensive, and the lack of an official implementation left early adopters guessing about details. Attention mechanisms would soon address the bottleneck directly.

**Take-home message:** The encoder-decoder LSTM showed that a general neural network could outperform traditional translation systems. It did not solve sequence modelling by itself, but it gave the field the scaffold that attention and Transformers later expanded.
