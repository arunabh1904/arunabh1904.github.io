---
layout: post
title: "Attention Is All You Need"
date: 2017-06-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2017 – Attention Is All You Need

**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

**GitHub:**
- [tensor2tensor](https://github.com/tensorflow/tensor2tensor) (official implementation)
- [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

**Project page / blog:** [Google AI Blog – "Transformer: A Novel Neural Network Architecture for Language Understanding"](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)

**Conference:** NeurIPS 2017

**Summary (abstract in plain English):** The paper removes recurrence and convolution from sequence-to-sequence models and relies solely on self-attention. A 6-layer encoder–decoder Transformer surpasses prior RNN and CNN systems on WMT-14 translation while training much faster thanks to full parallelism. The base model reaches 27.3 BLEU on English→German and 38.1 BLEU on English→French. The "big" variant scores 28.4 and 41.8 BLEU after about 3.5 days on 8 GPUs. The architecture also generalises to tasks like constituency parsing.

**Novel insights:**
- All-attention design shows sequence transduction does not require recurrence, enabling massive parallelism.
- Multi-head self-attention lets different heads focus on separate positions or sub-spaces simultaneously.
- Sinusoidal positional encoding injects order information without adding learned parameters.
- Residual connections with layer normalisation stabilise deep stacks.
- Hardware efficiency demonstrates up to an order-of-magnitude faster training than RNNs on the same GPUs.

**Evals / Latency benchmarks:**

| Benchmark | Model | Score | Training-time notes |
| --------- | ----- | ----- | ------------------- |
| WMT-14 En→De | Transformer-Big | 28.4 BLEU | 3.5 days on 8× NVIDIA P100 |
| WMT-14 En→Fr | Transformer-Big | 41.8 BLEU | same runtime as above |
| Penn Treebank parsing | Base | 95.4 F1 | fine-tuned with no architectural changes |
| Hardware efficiency | – | – | Base model trains state-of-the-art NMT in about 12 h on 8 P100s |

**Critiques & limitations:**
- **What works well:** Elegant, modular design that is easy to scale. Speed-ups enabled large-scale pre-training and inspired models like BERT and GPT. The paper shares code and hyper-parameters for reproduction.
- **Limitations:** Quadratic cost in sequence length makes vanilla attention impractical beyond about 1k tokens, leading to efficient Transformer variants. Large datasets are needed; RNNs can still compete on smaller corpora. The paper explored limited ablations for very deep stacks (>12 layers).

**Take-home message:** The Transformer proved that self-attention alone can power high-quality sequence modelling with superior parallelism, laying the groundwork for modern large-scale language models.
