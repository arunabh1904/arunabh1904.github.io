---
title: 'Sparse-to-Dense: Depth Prediction from Sparse Depth and RGB'
date: '2017-09-21T04:00:00.000Z'
section: paper-shorts
postSlug: sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb
legacyPath: /paper shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2017 – Sparse-to-Dense: complete sparse runtime range measurements with RGB'
---
## 2017 – Sparse-to-Dense

**arXiv:** [1709.07492](https://arxiv.org/abs/1709.07492)

**Summary:** Sparse-to-Dense feeds RGB and sparse depth into an encoder-decoder that predicts a dense depth map. The sparse values are runtime inputs, not merely training labels. Random sampling simulates different depth densities and lets the paper study how a small number of range measurements changes monocular prediction.

That lifecycle distinction is essential: depth completion is sensor fusion. Removing the range sensor at deployment changes the input distribution and invalidates the advertised completion behavior.

## Paper Insights

The network can use one joint encoder or separate RGB and depth branches. Sparse points anchor metric scale while image features interpolate structure between measurements. The paper reports that 100 samples roughly halve NYU depth error and reduce the cited KITTI error from about 7 m to about 3.5 m; at 500 samples, the fraction of reliable KITTI pixels rises from 59.1% to 93.5%.

| Runtime input | Role | Failure consideration |
| --- | --- | --- |
| RGB | Boundaries and semantic priors | Lighting and texture shortcuts. |
| Sparse depth | Metric anchors | Density, pattern, and missing returns shift. |
| Sampling mask | Indicates observed support | Must not confuse zero with missing. |
| Dense output | Consumer-friendly geometry | Can be overconfident between anchors. |

## Decision Lens

Use depth completion when sparse depth is guaranteed at runtime and downstream modules benefit from a dense surface. If the deployment goal is camera-only, use sparse depth as supervision or distillation instead and remove it from the inference graph deliberately.

Evaluate error by distance, object boundary, surface type, and sampling pattern. Uniform random samples are easier than real scanning geometry and motion distortion.

**Context:** DeepLiDAR adds surface-normal reasoning and learned confidence; GuideFormer replaces convolutional exchange with guided attention.

**Takeaway:** Sparse measurements can dramatically improve dense depth, but the improvement is a runtime sensor dependency, not free privileged supervision.
