---
title: 'MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction'
date: '2023-08-10T04:00:00.000Z'
section: paper-shorts
postSlug: maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2023/08/10/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: BEV
summary: MapTRv2 strengthens MapTR with auxiliary one-to-many matching and dense supervision, improving convergence and accuracy on nuScenes and Argoverse2.
---
## 2023 - MapTRv2

**arXiv:** [2308.05736](https://arxiv.org/abs/2308.05736)

**Code:** [hustvl/MapTR](https://github.com/hustvl/MapTR)

**Summary:** MapTRv2 keeps MapTR's central idea: represent map elements as permutation-equivalent point sets and learn them with hierarchical queries. The upgrade focuses on making the system train faster and perform better across datasets.

The important additions are auxiliary one-to-many matching and dense supervision. Those extra training signals reduce the fragility of set matching and help the model learn map geometry before the final sparse vector loss has to carry everything.

## Paper Insights

The paper presents MapTRansformer as an end-to-end framework for online vectorized HD map construction. It preserves the unified permutation-equivalent representation and hierarchical bipartite matching from MapTR, then adds auxiliary one-to-many matching and dense supervision to accelerate convergence. The model handles map elements with arbitrary shapes and remains a simple encoder-decoder Transformer.

The evidence spans nuScenes and Argoverse2, where the paper reports state-of-the-art performance and real-time inference. The tradeoff is that MapTRv2 adds more training objectives; the inference story stays clean, but reproduction depends on carefully matching the supervision recipe.

![Figure 4 from MapTRv2 showing the encoder-decoder architecture, hierarchical queries, and matching branches](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-paper-figure.png)
_Figure 4 shows how MapTRv2 keeps the vector-map decoder structured while adding attention variants and richer matching supervision. From the [MapTRv2 paper](https://arxiv.org/abs/2308.05736), via arXiv HTML._

**What to look at:**
- The representation is still point sets plus equivalent permutations.
- Auxiliary one-to-many matching gives more positive training signal than strict one-to-one matching alone.
- Dense supervision helps the model learn spatial structure before final vector outputs are evaluated.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Datasets | nuScenes and Argoverse2 | Tests whether the MapTR idea transfers across map benchmarks. |
| Training recipe | One-to-many matching plus dense supervision | Improves convergence and final accuracy. |
| Output | Real-time vector HD map construction | Keeps the planner-friendly vector representation. |

**Context:** MapTRv2 turned MapTR from a strong baseline into a more robust framework that later online-map papers could compare against.

**Takeaway:** Once the representation is right, the next bottleneck is supervision density: sparse vector labels often need auxiliary training signals.
