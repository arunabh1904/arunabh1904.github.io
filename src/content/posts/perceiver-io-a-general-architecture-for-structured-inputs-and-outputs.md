---
title: 'Perceiver IO: A General Architecture for Structured Inputs & Outputs'
date: '2021-07-30T04:00:00.000Z'
section: paper-shorts
postSlug: perceiver-io-a-general-architecture-for-structured-inputs-and-outputs
legacyPath: /paper shorts/2021/07/30/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs.html
tags:
  - Other
field: BEV
summary: Perceiver IO uses latent bottleneck cross-attention plus output queries to process arbitrary inputs and produce structured outputs without task-specific heads.
---
## 2021 - Perceiver IO

**arXiv:** [2107.14795](https://arxiv.org/abs/2107.14795)

**Code:** [google-deepmind/deepmind-research/perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver)

**Plain-language summary:** Perceiver IO generalizes the Perceiver idea from flexible inputs to flexible inputs and outputs. It uses cross-attention to pull information from large inputs into a latent array, processes the latent array, then uses output queries to decode structured predictions.

This belongs in a BEV reading list because modern driving models often need to fuse different input shapes and produce structured outputs: grids, vectors, trajectories, agent states, and maps. Perceiver IO is one of the cleanest architectural templates for that kind of input/output mismatch.

## Paper map

The problem is architectural specialization. Standard networks bake in assumptions about image grids, token sequences, or fixed output formats. Perceiver IO keeps the computationally expensive processing in a latent space, so input cost scales through cross-attention and output size is controlled by queries. Different query sets can request different output structures from the same latent representation.

The evidence spans language, visual understanding, multimodal reasoning, optical flow, and StarCraft II. The headline examples include outperforming a Transformer-based BERT baseline on GLUE without input tokenization and reaching state-of-the-art Sintel optical-flow performance without explicit multiscale correspondence machinery. The caveat is that generality does not remove representation design; good positional encodings, queries, and training recipes still matter.

![Figure 2 from Perceiver IO showing encode, latent processing, and output-query decoding](/assets/images/perceiver-io-a-general-architecture-for-structured-inputs-and-outputs-paper-figure.png)
_Figure 2 shows the Perceiver IO template: arbitrary inputs enter a latent workspace, then output queries decode task-specific structured outputs. From the [Perceiver IO paper](https://arxiv.org/abs/2107.14795), via ar5iv._

**What to look at:**
- Cross-attention moves arbitrary inputs into a fixed latent workspace.
- Output queries turn the same latent state into different structured outputs.
- The architecture separates data shape from model shape.

**Evals / Benchmarks / Artifacts:**

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

**Why it mattered:** Perceiver IO gave researchers a reusable pattern for multimodal models whose inputs and outputs do not fit one simple grid or sequence.

**Take-home message:** Use a latent workspace when input size, output size, and output semantics all vary; the important design work moves into queries and embeddings.
