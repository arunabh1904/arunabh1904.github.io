---
title: 'MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction'
date: '2022-08-30T00:00:00.000Z'
section: paper-shorts
postSlug: maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2022 – MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction"
---
## Summary

> A lane divider does not acquire a different shape when its points are listed backward. MapTR builds that equivalence into supervision: it predicts a set of map elements, each represented by connected points, and matches against the valid orderings of the same geometry. This removes an arbitrary learning burden while retaining the structure needed to produce vector maps directly. On nuScenes, its camera-only ResNet-50 model reaches 50.3 mAP after 24 epochs and 58.7 after 110; the smaller ResNet-18 configuration runs at 25.1 FPS on an RTX 3090. These are geometry-reconstruction results, not evidence of improved driving.

## Core Insights

### The correct target is a shape with equivalent orderings

A detector can represent a box with a small, largely standardized parameter vector. A map element is harder: a lane divider is an open polyline, while a pedestrian crossing can be a closed polygon. Picking one starting vertex and one traversal direction introduces choices that need not correspond to any difference in the road. Two predictions can trace the same crossing yet receive a large coordinate loss because their first vertices differ.

MapTR represents an element by its sampled points and the set of equivalent index orderings. For an open polyline, the forward and reversed sequences describe the same geometry. For a closed polygon with $N_v$ points, cyclic shifts and reversal give $2N_v$ equivalent sequences. This is a restricted symmetry: arbitrary permutations would scramble the edges and change the shape. With the default 20 points, a polygon has 40 allowed sequences, not every possible ordering of 20 vertices.

![MapTR Figure 2 shows why lane-divider direction and crossing start points are ambiguous](/assets/images/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction-source-figure-2.webp)
*Fig 1: The divider separates opposite traffic directions, while the crossing has no natural first corner. Neither visual scene supplies a unique point sequence; requiring one adds a target convention unrelated to the geometry. | source: [MapTR, Figure 2](https://arxiv.org/abs/2208.14437)*

The controlled ablation makes this more than a representational preference. Replacing fixed-order supervision with permutation-equivalent modeling raises mAP from 44.4 to 50.3. Pedestrian-crossing AP rises from 34.4 to 46.3, a larger gain than for dividers or boundaries. That pattern fits the ambiguity being addressed: closed polygons have more equivalent starting-point and direction choices than open curves.

### Matching resolves element identity before choosing its point order

The decoder emits a fixed set of candidate elements. Instance-level Hungarian matching assigns those candidates to ground-truth elements using classification and geometric costs; unused candidates receive the no-object label. Within each matched pair, point-level matching selects the allowed ordering with the smallest summed Manhattan distance. It does not freely rematch every vertex to an arbitrary ground-truth vertex.

The distinction matters because the loss must preserve connectivity while ignoring irrelevant serialization choices. The paper also compares two position costs for instance assignment: Chamfer distance gives 47.5 mAP, while the best permitted point-to-point assignment gives 50.3. Nearest-neighbor proximity alone discards correspondence information that this structured representation can use.

Training combines focal classification loss, point-coordinate loss, and an edge-direction loss based on cosine similarity between corresponding edges. Coordinates locate the samples; edge directions supervise how adjacent samples connect. Adding the default direction term improves mAP from 48.2 to 50.3, but doubling its weight reduces the score to 48.3. The result supports balancing local shape supervision with coordinate accuracy, rather than treating a stronger geometric penalty as automatically better.

### Hierarchical queries connect an element's identity to its local geometry

MapTR first transforms surround-view image features into a BEV feature map. Each decoder query then adds an instance embedding to a point embedding shared across instances: $q_{ij}=q_i^{\mathrm{ins}}+q_j^{\mathrm{pt}}$. The instance component groups points into one object; the point component distinguishes locations within that object's representation. Self-attention exchanges information within and between elements, and deformable cross-attention samples BEV features around predicted reference points.

![MapTR Figure 4 illustrates hierarchical queries and the two stages of matching](/assets/images/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction-paper-figure.png)
*Fig 2: Follow the bottom row: instance matching selects the correct map element, then point matching selects an equivalent traversal of that element. The shared point queries support parallel decoding without imposing one ground-truth starting vertex. | source: [MapTR, Figure 4](https://arxiv.org/abs/2208.14437)*

The practical benefit is parallel prediction of all elements and their points, instead of an autoregressive process that emits vertices sequentially. MapTR-tiny uses 50 instance queries, 20 points per element, and six decoder layers. Its point-count ablation gives 48.0 mAP with 10 points, 50.3 with 20, and 50.0 with 40. More samples can represent finer curves, but the experiment does not show an accuracy benefit from simply doubling the default resolution.

### Accuracy and speed belong to specific model configurations

The benchmark covers pedestrian crossings, lane dividers, and road boundaries in a 30-by-60-meter local region. Its AP averages Chamfer-distance thresholds of 0.5, 1.0, and 1.5 meters. It measures reconstructed geometry; it does not directly measure lane connectivity, legal maneuvers, or a planner's response to a mapping error.

| Camera-only model | Backbone | Training epochs | nuScenes validation mAP | RTX 3090 FPS |
| --- | --- | ---: | ---: | ---: |
| VectorMapNet | ResNet-50 | 110 | 40.9 | 2.9 |
| MapTR-nano | ResNet-18 | 110 | 45.9 | 25.1 |
| MapTR-tiny | ResNet-50 | 24 | 50.3 | 11.2 |
| MapTR-tiny | ResNet-50 | 110 | 58.7 | 11.2 |

The headline speed belongs to nano, whose smaller images, coarser BEV grid, and two-layer decoder differ from tiny. It should not be attached to tiny's 58.7 mAP. In tiny's measured runtime, the image backbone consumes 55.5 of 89.3 milliseconds, compared with 21.5 for the decoder. Once vertex generation is parallelized, much of the remaining latency sits upstream of the map representation.

Calibration is another concrete limit. The camera-translation perturbation experiment drops from 50.3 to 49.0 mAP at 0.1-meter Gaussian standard deviation, but reaches 34.0 at 0.5 meters. Rotation noise with 0.02-radian standard deviation gives 42.0 mAP. These tests show degradation under geometric misalignment, even though the source prose describes the latter result as comparable. A structured output does not remove the need for reliable sensor-to-BEV geometry.

## High-Level Takeaways

- Supervision should ignore changes in point order that preserve a map element, while retaining the adjacency that defines its shape. MapTR's allowed permutations do exactly that.
- The largest modeling-ablation gain occurs for closed pedestrian-crossing polygons, where arbitrary starting vertices create especially strong label ambiguity.
- Instance matching and point-order selection solve different problems. Their geometric costs determine which errors the model is trained to correct.
- Parallel vector decoding improves efficiency, but the reported speed depends on backbone, image resolution, BEV resolution, and decoder depth. Nano and tiny represent different accuracy–latency choices.
- Chamfer-based map AP is useful geometric evidence. Connectivity, calibration robustness, and downstream driving value require additional evaluation.
