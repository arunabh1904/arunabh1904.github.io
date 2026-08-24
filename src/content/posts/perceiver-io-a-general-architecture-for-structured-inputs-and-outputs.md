---
title: 'Perceiver IO: A General Architecture for Structured Inputs & Outputs'
date: '2021-07-30T00:00:00.000Z'
section: paper-shorts
postSlug: perceiver-io-a-general-architecture-for-structured-inputs-and-outputs
legacyPath: /paper shorts/2021/07/30/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs.html
tags:
  - Other
field: 'Omni-Model Architectures'
summary: "2021 – Perceiver IO: A General Architecture for Structured Inputs & Outputs"
---
## 2021 – Perceiver IO

**arXiv:** [2107.14795](https://arxiv.org/abs/2107.14795)

**Code:** [google-deepmind/deepmind-research/perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)

### Method and reported result

Perceiver IO generalizes the Perceiver idea from flexible inputs to flexible inputs and outputs. It uses cross-attention to pull information from large inputs into a latent array, processes the latent array, then uses output queries to decode structured predictions.

## Summary

> This belongs in a BEV reading list because modern driving models often need to fuse different input shapes and produce structured outputs: grids, vectors, trajectories, agent states, and maps. Perceiver IO is one of the cleanest architectural templates for that kind of input/output mismatch.

## Core Insights

The problem is architectural specialization. Standard networks bake in assumptions about image grids, token sequences, or fixed output formats. Perceiver IO keeps the computationally expensive processing in a latent space, so input cost scales through cross-attention and output size is controlled by queries. Different query sets can request different output structures from the same latent representation.

The evidence spans language, visual understanding, multimodal reasoning, optical flow, and StarCraft II. The headline examples include outperforming a Transformer-based BERT baseline on GLUE without input tokenization and reaching state-of-the-art Sintel optical-flow performance without explicit multiscale correspondence machinery. The caveat is that generality does not remove representation design; good positional encodings, queries, and training recipes still matter.

![Figure 2 from Perceiver IO showing encode, latent processing, and output-query decoding](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-paper-figure.png)
*Figure 2 shows the Perceiver IO template: arbitrary inputs enter a latent workspace, then output queries decode task-specific structured outputs. From the [Perceiver IO paper](https://arxiv.org/abs/2107.14795), via ar5iv. source: [Perceiver IO paper](https://arxiv.org/abs/2107.14795)*

![Figure 1 from Perceiver IO: A General Architecture for Structured Inputs & Outputs](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-source-figure-1.webp)
*Figure 1 The Perceiver IO architecture can be used on domains with a wide variety of input and output spaces, including multi-task language understanding, dense visual tasks like optical flow, hybrid dense/sparse multimodal tasks such as video+audio+class autoencoding, and tasks with symbolic outputs like StarCraft II. See Tables 5 and 6 for details of all domains considered here. source: [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)*

![Figure 4 from Perceiver IO: A General Architecture for Structured Inputs & Outputs](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-source-figure-4.webp)
*Figure 4 Multimodal audio-video-label autoencoding with 88x compression. Side-by-side: inputs on left, reconstructions right. See the supplemental material for example output video and audio. source: [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)*


**What to look at:**
- Cross-attention moves arbitrary inputs into a fixed latent workspace.
- Output queries turn the same latent state into different structured outputs.
- The architecture separates data shape from model shape.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Architecture | Latent bottleneck plus output queries | Handles large inputs and structured outputs. |
| Modalities | Language, vision, multimodal reasoning, games | Tests generality across data types. |
| BEV relevance | Query-based structured decoding | Mirrors later map, trajectory, and planning decoders. |

**Compact result slice:**

| Task | Baseline | Perceiver IO result | What to notice |
| ---- | -------- | ------------------- | -------------- |
| GLUE average | BERT Base at 81.1 | 81.2 with Perceiver IO Base | Comparable language performance with the general architecture. |
| Sintel clean optical flow | RAFT at 1.95 EPE | 1.81 EPE | Strong dense prediction without hand-built multiscale matching. |
| Sintel final optical flow | RAFT at 2.57 EPE | 2.42 EPE | General output queries still handle dense visual output. |

## High-Level Takeaways

- Perceiver IO informs whether compute should scale with raw input size or with a fixed latent bottleneck and a chosen set of output queries. The atomic operations are input-to-latent cross-attention, latent self-attention, and output-query cross-attention; task structure enters through the queries rather than a bespoke head.
- The architecture demonstrates broad modality and output flexibility, but the fixed latent array can discard fine detail before the task reveals what matters. The missing ablation sweeps latent count and output-query density against full attention at matched FLOPs across tasks with different information bottlenecks. At 10× input size, cross-attention remains manageable, but latent capacity becomes the failure point. The claim would fail if task-specific sparse attention preserved accuracy with equal efficiency and less latent tuning.
- Perceiver IO gave researchers a reusable pattern for multimodal models whose inputs and outputs do not fit one simple grid or sequence.
- Use a latent workspace when input size, output size, and output semantics all vary; the important design work moves into queries and embeddings.
