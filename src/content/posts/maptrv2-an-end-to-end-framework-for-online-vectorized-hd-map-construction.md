---
title: 'MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction'
date: '2023-08-10T00:00:00.000Z'
section: paper-shorts
postSlug: maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2023/08/10/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2023 – MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction"
---
## 2023 – MapTRv2

**arXiv:** [2308.05736](https://arxiv.org/abs/2308.05736)

**Code:** [hustvl/MapTR](https://github.com/hustvl/MapTR)

### Method and reported result

MapTRv2 keeps MapTR's central idea: represent map elements as permutation-equivalent point sets and learn them with hierarchical queries. The upgrade focuses on making the system train faster and perform better across datasets.

## Summary

> The important additions are auxiliary one-to-many matching and dense supervision. Those extra training signals reduce the fragility of set matching and help the model learn map geometry before the final sparse vector loss has to carry everything.

## Core Insights

The paper presents MapTRansformer as an end-to-end framework for online vectorized HD map construction. It preserves the unified permutation-equivalent representation and hierarchical bipartite matching from MapTR, then adds auxiliary one-to-many matching and dense supervision to accelerate convergence. The model handles map elements with arbitrary shapes and remains a simple encoder-decoder Transformer.

The evidence spans nuScenes and Argoverse2, where the paper reports state-of-the-art performance and real-time inference. The tradeoff is that MapTRv2 adds more training objectives; the inference story stays clean, but reproduction depends on carefully matching the supervision recipe.

![Figure 4 from MapTRv2 showing the encoder-decoder architecture, hierarchical queries, and matching branches](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-paper-figure.png)
*Fig 1: Shows how MapTRv2 keeps the vector-map decoder structured while adding attention variants and richer matching supervision. | source: [MapTRv2 paper](https://arxiv.org/abs/2308.05736)*

![Figure 1 from MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-source-figure-1.webp)
*Fig 2: Speed-accuracy trade-off comparisons. The proposed MapTRv2 outperforms previous state-of-the-art methods in terms of both speed (FPS) and accuracy (mAP). | source: [MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction](https://arxiv.org/abs/2308.05736)*

![Figure 7 from MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction](/assets/images/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction-source-figure-7.webp)
*Fig 3: Convergence curves of MapTRv2 and MapTR with ResNet-50 backbone on nuScenes dataset. MapTRv2 significantly outperforms MapTR with 4x shorter training schedule and 2.8 higher mAP. | source: [MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction](https://arxiv.org/abs/2308.05736)*


**What to look at:**
- The representation is still point sets plus equivalent permutations.
- Auxiliary one-to-many matching gives more positive training signal than strict one-to-one matching alone.
- Dense supervision helps the model learn spatial structure before final vector outputs are evaluated.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Datasets | nuScenes and Argoverse2 | Tests whether the MapTR idea transfers across map benchmarks. |
| Training recipe | One-to-many matching plus dense supervision | Improves convergence and final accuracy. |
| Output | Real-time vector HD map construction | Keeps the planner-friendly vector representation. |

## High-Level Takeaways

- MapTRv2 informs whether sparse one-to-one supervision is sufficient for vector-map queries or should be supplemented with one-to-many assignments and dense auxiliary targets. The training unit remains a map-element query and its point sequence, but extra matches expose more positive supervision during early optimization.
- The gains support supervision density as an optimization lever, not a change in the final map representation. The missing ablation matches the number of positive gradients across one-to-many matching, denoising queries, and dense segmentation auxiliaries. At 10× classes or map elements, duplicate assignments and auxiliary-loss balance become unstable. The claim would fail if longer training or a simpler query-denoising scheme matched convergence and map quality without the extra matching path.
- MapTRv2 turned MapTR from a strong baseline into a more robust framework that later online-map papers could compare against.
- Once the representation is right, the next bottleneck is supervision density: sparse vector labels often need auxiliary training signals.
