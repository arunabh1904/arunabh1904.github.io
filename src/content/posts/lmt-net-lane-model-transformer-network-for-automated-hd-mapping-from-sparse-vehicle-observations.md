---
title: 'LMT-Net: Lane Model Transformer Network for Automated HD Mapping from Sparse Vehicle Observations'
date: '2024-09-19T00:00:00.000Z'
section: paper-shorts
postSlug: lmt-net-lane-model-transformer-network-for-automated-hd-mapping-from-sparse-vehicle-observations
legacyPath: /paper shorts/2024/09/19/lmt-net-lane-model-transformer-network-for-automated-hd-mapping-from-sparse-vehicle-observations.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2024 – LMT-Net: Lane Model Transformer Network for Automated HD Mapping from Sparse Vehicle Observations"
---

**arXiv:** [2409.12409](https://arxiv.org/abs/2409.12409)

## Summary

> LMT-Net constructs an offline lane graph from fleet vehicles' sparse driven traces and observed lane boundaries. Statistical preprocessing aligns and aggregates the observations; a transformer then predicts left/right boundary pairs and directed connections between them. The useful division of labor is that traces suggest where a lane runs, while boundary observations constrain its width. On an internal German-road dataset, the learned graph improves lane-width error and connectivity over geometric heuristics, but does not outperform a constant-width baseline on every position metric.

## Core Insights

### A driven trace is a useful query, not a lane boundary

A vehicle's path lies somewhere inside its lane. Treating that path as the exact centerline would turn driver position into map geometry; treating nearby markings as complete boundaries would inherit missing and false detections. LMT-Net uses both observations without assuming that either is the answer.

An existing preprocessing method aligns traces and boundary observations with an ICP variant, then clusters them into polylines. Bundled traces provide candidate center points. These become decoder queries, while encoded traces and boundaries supply the keys and values. The output for each query is four Cartesian coordinates: one left boundary point and one right boundary point. That pair forms a cross-section of the lane, rather than a complete lane instance.

![LMT-Net uses trace-derived center queries to read encoded traces and lane observations](/assets/images/lmt-net-lane-model-transformer-network-for-automated-hd-mapping-from-sparse-vehicle-observations-paper-figure.png)
*Fig 1: Follow the separate routes into the decoder: observed geometry supplies context, while trace-derived center points specify where to predict a lane pair. A second head connects those pairs into a graph. | source: [LMT-Net, Figure 1](https://arxiv.org/abs/2409.12409)*

The polyline encoder includes each point's successor coordinates to preserve direction, applies within-polyline attention, and max-pools a variable-length sequence into one token. Trace and boundary inputs use type-specific encoders. The transformer exchanges information across these tokens before decoding the lane pairs. Concatenating two decoded tokens lets a separate MLP predict a directed edge, retained above a 0.8 probability threshold. Geometry and connectivity are trained jointly with boundary regression and binary cross-entropy losses; independent edge predictions do not themselves enforce a globally valid road graph.

### Relative lane width and absolute placement answer different questions

The source examples show why aggregation alone is insufficient. Markings can be incomplete on an otherwise regular highway, or noisy where a road changes direction. A learned model can combine several partial observations, but it must also avoid replacing the evidence with a memorized road shape.

![Raw fleet traces, aggregated observations, and human lane boundaries in two road settings](/assets/images/lmt-net-lane-model-transformer-network-for-automated-hd-mapping-from-sparse-vehicle-observations-source-figure-2.webp)
*Fig 2: Yellow raw traces become red driven paths and green observed boundaries; blue ground truth reveals gaps and noise in those inputs. Sparse observations constrain the map without already containing a complete lane model. | source: [LMT-Net, Figure 2](https://arxiv.org/abs/2409.12409)*

Evaluation separates mean boundary-point error (mBPE) from mean lane-width error (mLWE). A pair can have the correct separation while both endpoints are shifted, so low width error does not establish accurate global placement. The authors prioritize relative width partly because localization can compensate for small common offsets.

| Method | Highway / non-highway boundary error | Highway / non-highway width error |
| --- | ---: | ---: |
| Constant 3.2 m lane width | 0.24 / 0.27 m | 0.42 / 0.42 m |
| Nearest perpendicular boundary intersection | 0.39 / 0.40 m | 0.31 / 0.36 m |
| LMT-Net | 0.21 / 0.35 m | 0.15 / 0.31 m |

The highway width gain is substantial, while the non-highway improvement is smaller. Constant width still wins non-highway boundary placement, 0.27 versus 0.35 m. The authors identify limited non-highway training data and possible input/annotation misalignment or road changes as explanations, rather than demonstrating one isolated cause.

Connectivity is more decisive: the nearest-forward heuristic has 96–97% accuracy but only 67% highway and 57% non-highway F1. LMT-Net reaches approximately 99% accuracy with 99% and 94% F1. Accuracy alone obscures how many lane pairs should remain disconnected.

### Trace coverage and map stitching remain outside the learned solution

The dataset covers roughly 10,000 lane-kilometers, two-thirds highway and one-third non-highway, mostly country roads with fewer urban examples. It contains 13,428 independently processed hexagonal minimaps, including 692 evaluation tiles. Each tile averages 14 center queries and 190 polylines; a center point is supported by five to ten traces. These are human-supervised maps, despite the sparse inference inputs.

The architecture ablation favors four decoder layers for complex roads: non-highway width error falls from 0.39 m with one layer to 0.31 m with four, and connectivity F1 rises from 88% to 94%. Six layers worsen those values to 0.38 m and 92%. More capacity does not monotonically repair uncertain geometry.

The remaining failures follow directly from the representation. Two-dimensional alignment cannot separate a bridge from the road beneath it. Sparse sampling loses tight curvature. Tiles have no context margins, leaving stitching unresolved. The output covers lane geometry and connectivity, not every feature required by an HD map. The paper establishes a learned fleet-mapping component, with public-data replication and end-to-end map maintenance still open.

## High-Level Takeaways

- Use driven traces to locate lane queries and boundary observations to constrain geometry; neither should be mistaken for a complete lane model.
- Report placement, width, and connectivity separately. Their baselines and failure modes differ.
- The strongest evidence is improved width and edge F1 under the reported internal-data protocol, not uniform superiority across road settings.
- Three-dimensional alignment, curvature-aware sampling, and tile stitching are necessary extensions before local predictions become a continuous fleet-maintained map.
