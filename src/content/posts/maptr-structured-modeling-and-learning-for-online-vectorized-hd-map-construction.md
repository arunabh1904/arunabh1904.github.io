---
title: 'MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction'
date: '2022-08-30T04:00:00.000Z'
section: paper-shorts
postSlug: maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: BEV
summary: MapTR models HD map elements as permutation-equivalent point sets, making online vectorized map construction trainable with a structured Transformer.
---
## 2022 - MapTR

**arXiv:** [2208.14437](https://arxiv.org/abs/2208.14437)

**Code:** [hustvl/MapTR](https://github.com/hustvl/MapTR)

**Plain-language summary:** MapTR is one of the core papers for online vectorized HD map construction. It replaces dense raster map outputs with structured vector elements such as lane dividers, road boundaries, and pedestrian crossings.

The key modeling move is permutation equivalence. A map element can be represented by a set of points, but the same shape may have several valid point orders. MapTR encodes that ambiguity directly so training does not punish equivalent representations.

## Paper Insights

The paper frames online HD map construction as structured set prediction. Map elements are modeled as point sets with a group of equivalent permutations. A hierarchical query embedding scheme encodes instance-level and point-level structure, and hierarchical bipartite matching assigns predictions to ground-truth map elements during training.

The reported evidence is strong for its time: with camera input on nuScenes, MapTR achieved better accuracy and efficiency than prior vector-map construction methods. The abstract highlights MapTR-nano at 25.1 FPS on an RTX 3090, 8x faster than the existing camera-based state of the art while improving mAP by 5.0. The caveat is that MapTR's clean vector output still depends on supervised map annotations and benchmark geometry, not closed-loop planner value.

![Figure 4 from MapTR showing the encoder-decoder architecture for online vectorized HD map construction](/assets/images/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction-paper-figure.png)
_Figure 4 shows the structured MapTR pipeline: sensor inputs become BEV features, hierarchical queries decode vector map elements, and matching handles point-level ambiguity. From the [MapTR paper](https://arxiv.org/abs/2208.14437), via ar5iv._

**What to look at:**
- Map elements are point sets with multiple equivalent orderings.
- Hierarchical queries mirror the structure of a vector map element.
- Matching is part of the method, not just a training detail.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Representation | Permutation-equivalent point sets | Removes unnecessary label ambiguity. |
| Dataset | nuScenes | Standard public benchmark for online map construction. |
| Efficiency | MapTR-nano reported at 25.1 FPS | Made vectorized maps feel deployable, not only accurate. |

**Compact result slice:**

| Method | Modality | Backbone | nuScenes mAP | FPS |
| ------ | -------- | -------- | ------------ | --- |
| HDMapNet | Camera | EfficientNet-B0 | 23.0 | 0.8 |
| VectorMapNet | Camera | R50 | 40.9 | 2.9 |
| MapTR-nano | Camera | R18 | 45.9 | 25.1 |
| MapTR-tiny | Camera | R50 | 50.3 | 11.2 |

**Why it mattered:** MapTR gave the field a practical transformer baseline for vector maps and made permutation ambiguity a first-class modeling issue.

**Take-home message:** Vector-map learning works better when the loss respects the geometry's symmetries instead of forcing one arbitrary point order.
