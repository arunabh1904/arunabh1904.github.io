---
title: 'HDMapNet: A Local Semantic Map Learning and Evaluation Framework'
date: '2021-07-13T00:00:00.000Z'
section: paper-shorts
postSlug: hdmapnet-local-semantic-map-learning
legacyPath: /paper shorts/2021/07/13/hdmapnet-local-semantic-map-learning.html
tags: [Autonomous Driving, Mapping]
field: Mapping
summary: "2021 – HDMapNet: Dense BEV supervision followed by instance grouping and vectorization"
---

# HDMapNet: A Local Semantic Map Learning and Evaluation Framework

**Paper:** [2107.06307](https://arxiv.org/abs/2107.06307) · **Project:** [HDMapNet](https://tsinghua-mars-lab.github.io/HDMapNet/)

## Summary

> HDMapNet learns a local semantic map from surround cameras, LiDAR, or both. It predicts three dense BEV fields—semantic classes, instance embeddings, and local directions—then groups and connects their pixels into vectors. On nuScenes, camera–LiDAR fusion reaches 44.5 mean IoU and 30.6 instance mAP, compared with 32.9 and 22.7 for cameras alone under the paper's protocol. The method establishes a useful separation between learning spatial evidence and constructing vector objects; subsequent vector decoders move the latter step into the network.

## Core Insights

### The learned representation is a field; the delivered map is a set of curves

HDMapNet predicts lane dividers, pedestrian crossings, and road boundaries. A semantic mask can locate their pixels, but it cannot identify which pixels belong to one divider. The model therefore adds an instance embedding at every BEV location and a direction distribution that helps connect points along each element.

![HDMapNet camera and LiDAR branches feed semantic, instance, and direction heads before vector post-processing](/assets/images/hdmapnet-source-figure-2.png)
*Fig 1: The three dense heads provide class, grouping, and local orientation. Post-processing combines them into vector elements, so curve construction remains outside the learned decoder. | source: [HDMapNet, Figure 2](https://arxiv.org/abs/2107.06307)*

The camera path uses an EfficientNet-B0 image encoder and a learned multilayer-perceptron view transformation. Camera extrinsics place the transformed features in the ego frame, where features from the six cameras are averaged. The LiDAR path uses dynamic pillar voxelization and a PointNet-style encoder. With both sensors, their BEV features are concatenated before a convolutional decoder. This shared frame allows dense camera appearance and measured LiDAR geometry to support the same output pixels.

### Three losses supervise three different ambiguities

Cross-entropy trains semantic classification. A discriminative embedding loss pulls pixels from one instance toward their cluster center and separates different instance centers. Direction classification predicts a discretized local tangent. Because the annotations do not define forward travel along these curves, both tangent orientations count as positives. The direction head therefore supports curve tracing; it does not predict legal lane direction.

At inference, DBSCAN clusters the instance embeddings. Non-maximum suppression reduces the point set, and a greedy procedure connects points using the predicted directions. Errors can enter at either stage: a good semantic mask may still fragment into several instances, or a direction error may connect the wrong samples. The vector output is consequently not a jointly learned set prediction in the later MapTR sense.

The paper trains with Adam at a learning rate of 0.001. Its learning objective combines semantic, embedding, and direction supervision. A complete training-compute budget is not reported.

### Fusion improves the geometry, but raster and vector scores measure different failures

The nuScenes evaluation uses the three map classes. Mean IoU compares raster masks. Instance mAP instead matches predicted vectors through Chamfer distance and averages thresholds of 0.2, 0.5, and 1.0 meters.

| Input | Mean IoU | Instance mAP |
| --- | ---: | ---: |
| Cameras | 32.9 | 22.7 |
| LiDAR | 29.5 | 11.6 |
| Cameras + LiDAR | 44.5 | 30.6 |

LiDAR alone performs better on road boundaries than the camera model in the semantic evaluation, but worse on dividers and crossings. That pattern is consistent with the sensors' different evidence: physical edges produce geometric structure, while paint requires appearance. Fusion improves all three semantic classes in the reported comparison.

Temporal fusion warps earlier BEV features into the current ego frame and max-pools them. For the camera model, one, two, and four frames produce mean IoU of 32.9, 35.8, and 36.4. This demonstrates a benefit from past observations, but the metric does not establish stable element identities across time.

### The next step was to learn vectorization itself

[VectorMapNet](/paper%20shorts/2022/06/17/vectormapnet-end-to-end-vectorized-hd-map-learning.html) replaces clustering and tracing with element detection followed by autoregressive polyline generation. [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) then predicts structured point sets in parallel while matching equivalent traversals. These methods change the output-learning problem, rather than merely substituting a stronger BEV encoder.

HDMapNet does not evaluate a directed successor graph, stop-line associations, curb height, or detection of real map changes. Its original distance thresholds also differ from the later 0.5/1.0/1.5-meter convention. An mAP number copied across these protocols would obscure the progression it is supposed to measure.

## High-Level Takeaways

- Dense semantics, instance embeddings, and local tangents solve different parts of vector construction; high segmentation IoU does not guarantee correct grouping.
- Bidirectional tangent labels support tracing undirected curves, not inference of legal travel direction.
- Camera–LiDAR fusion improves all three reported semantic classes, while temporal aggregation improves camera segmentation without measuring identity continuity.
- The historical contribution is a learned local BEV mapping pipeline with explicit vector evaluation. Later vector decoders remove its clustering and tracing stages.
