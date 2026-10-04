---
title: 'VectorMapNet: End-to-End Vectorized HD Map Learning'
date: '2022-06-17T00:00:00.000Z'
section: paper-shorts
postSlug: vectormapnet-end-to-end-vectorized-hd-map-learning
legacyPath: /paper shorts/2022/06/17/vectormapnet-end-to-end-vectorized-hd-map-learning.html
tags: [Autonomous Driving, Mapping]
field: Mapping
summary: "2022 – VectorMapNet: Detect map instances, then generate variable-length polylines"
---

# VectorMapNet: End-to-End Vectorized HD Map Learning

**Paper:** [2206.08920](https://arxiv.org/abs/2206.08920) · **Project:** [VectorMapNet](https://tsinghua-mars-lab.github.io/vectormapnet/)

## Summary

> VectorMapNet makes vector generation a learned prediction task. A detector first predicts map instances and coarse keypoints; a transformer then emits a variable-length sequence of polyline coordinates for each instance. On nuScenes, its camera model reaches 40.9 Chamfer mAP, or 46.0 after the paper's additional fine-tuning, against a re-evaluated HDMapNet baseline of 23.0. The method removes raster clustering and tracing, but introduces sequential decoding and a mismatch between ground-truth conditioning during training and predicted conditioning at inference.

## Core Insights

### Coarse detection supplies an address; generation supplies the curve

[HDMapNet](/paper%20shorts/2021/07/13/hdmapnet-local-semantic-map-learning.html) predicts dense fields and turns them into vectors through post-processing. VectorMapNet predicts the vector representation directly in two learned stages. The detector uses hierarchical element and keypoint queries to identify each object and its coarse geometry. The generator conditions on those keypoints, the semantic class, and BEV features to predict the detailed polyline.

![VectorMapNet BEV extraction, coarse map-element detection, and autoregressive polyline generation](/assets/images/vectormapnet-source-figure-2.png)
*Fig 1: Coarse keypoints connect detection to generation. The generator emits quantized coordinates until an end token; these output vertices are distinct from the detector's fixed keypoints. | source: [VectorMapNet, Figure 2](https://arxiv.org/abs/2206.08920)*

The camera branch uses ResNet-50 features and inverse perspective mapping onto four height planes: −1, 0, 1, and 2 meters. Concatenating these planes relaxes dependence on a single assumed ground height. The optional LiDAR branch uses pillar features. Their shared BEV tensor has spatial dimensions 200 by 100 with 128 channels.

The detector compares several coarse representations: bounding-box corners, start/middle/end points, and four extremes. These keypoints locate an instance; they are not the final curve. Bounding-box conditioning gives 40.9 Chamfer mAP in the ablation, versus 32.5 for start/middle/end and 33.6 for extremes. Adding more geometric structure to the conditioning target does not automatically improve generation.

### A variable-length polyline becomes a coordinate sequence

The generator discretizes a 60-by-30-meter region into 200-by-100 cells, each 0.3 meters wide. It predicts alternating coordinate tokens and terminates with an end-of-sequence token. Coordinate-axis, vertex-position, and value embeddings distinguish where each token belongs. Different instances can be processed together, but later coordinates within one instance still depend on earlier coordinates.

Polyline simplification reduces redundant annotation vertices before training. In the start/middle/end ablation, curvature-based sampling reaches 32.5 Chamfer mAP versus 17.0 for one-meter fixed intervals. A dense sequence of points on straight segments can dilute the training signal from the few vertices that define a corner. Variable output length helps only when the target serialization allocates that length usefully.

### Training must reconcile the two stages

Hungarian matching assigns detected instances to ground truth. Classification and geometric losses train the detector, while token negative log-likelihood trains the generator. Initial teacher forcing conditions generation on ground-truth coarse keypoints; additional fine-tuning uses predicted keypoints to reduce the mismatch at inference.

The reported training setup uses eight GTX 3090 GPUs, 110 epochs, batch size 32, AdamW, and six-layer detector and generator decoders. The two-stage schedule is a material part of the result. The authors identify both this feature mismatch and the lack of temporal consistency as limitations.

| nuScenes input | HDMapNet, re-evaluated | VectorMapNet | With additional fine-tuning |
| --- | ---: | ---: | ---: |
| Cameras | 23.0 | 40.9 | 46.0 |
| LiDAR | 24.1 | 34.0 | Not reported in Table 1 |
| Cameras + LiDAR | 31.0 | 45.2 | 53.7 |

These are Table 1 Chamfer mAP results for pedestrian crossings, dividers, and road boundaries, averaged over 0.5-, 1.0-, and 1.5-meter thresholds. They are not directly comparable with HDMapNet's original tighter thresholds. The paper also reports Fréchet AP, which preserves point order when comparing curves. Its Argoverse 2 camera experiment reaches 37.9 Chamfer mAP for 2D output and 35.8 for 3D output.

### The map helps forecasting, but does not establish topology correctness

The paper feeds predicted maps to an mmTransformer motion forecaster. Best-of-six minADE improves from 0.909 with past trajectories alone to 0.826 with predicted maps; ground-truth maps reach 0.779. This is downstream evidence that the vectors carry useful scene context, while retaining a measurable gap to the labeled map.

A coherent curve set is still different from a directed lane graph. The main benchmark does not supervise successors, control assignments, persistent identities, or real-world change detection. Predictions in occluded regions can reflect learned scene regularities, and the paper notes that such completion is difficult to interpret.

[MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) changes the next bottleneck: it predicts vertices in parallel and allows equivalent point orders during matching. VectorMapNet establishes learned polyline generation; MapTR removes the requirement to commit to one sequential traversal of an undirected shape.

## High-Level Takeaways

- Separate the detector's coarse keypoints from the generator's final vertices. They solve localization and detailed geometry respectively.
- Learned vectorization removes raster post-processing, but coordinate serialization introduces sequential inference and target-order choices.
- Fine-tuning on predicted coarse inputs matters because teacher-forced training otherwise gives the generator cleaner conditioning than deployment.
- Compare scores under the same distance thresholds and distinguish geometric reconstruction from topology, temporal consistency, and change detection.
