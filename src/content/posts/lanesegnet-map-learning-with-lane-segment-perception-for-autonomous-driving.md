---
title: "LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving"
date: '2023-12-26T00:00:00.000Z'
section: paper-shorts
postSlug: lanesegnet-map-learning-with-lane-segment-perception-for-autonomous-driving
legacyPath: /paper shorts/2023/12/26/lanesegnet-map-learning-with-lane-segment-perception-for-autonomous-driving.html
tags: ["Autonomous Driving", "HD Maps", "Topology"]
field: Mapping
summary: "2023 – LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving"
---

# 2023 – LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving

**Paper:** [2312.16108](https://arxiv.org/abs/2312.16108)

**Code:** [OpenDriveLab/LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)

## Summary

> LaneSegNet predicts a lane as one structured segment containing its centerline, boundaries, boundary types, and directed connections. On its OpenLane-V2 lane-segment benchmark, it reaches 32.6 mAP versus 28.5 for adapted MapTRv2, at a reported 14.7 FPS on an A100. The central experiment shows why representation matters: a coherent lane-segment target outperforms simply putting centerlines and map elements into the same output branch. The result depends on the new annotation format and does not establish a downstream driving gain.

## Core Insights

### Boundaries and centerlines describe the same lane

[MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) predicts map geometry as sets of points, while [TopoNet](/paper%20shorts/2023/04/11/toponet-graph-based-topology-reasoning-for-driving-scenes.html) predicts directed centerlines and their relationships. LaneSegNet puts both forms of geometry inside one lane instance. A segment contains three ordered 3D polylines, left/right boundary types, and an outgoing connectivity relation to subsequent segments.

A single instance query predicts a centerline and symmetric offsets for its two boundaries. The left boundary is the centerline plus the offset; the right is the centerline minus it. Supervision on all three curves therefore constrains a shared geometric prediction. An instance mask adds supervision over the lane's area, while separate classifiers predict segment and boundary categories.

The representation requires consistent training instances. Dataset processing merges ordinary intermediate lane breakpoints but retains splits at merges, diverges, intersections, and changes in boundary type. A solid-to-dashed transition is thus part of how a lane segment is delimited, not just another label attached to an arbitrary curve.

### One query needs attention distributed along the lane

An object's center is often a useful reference for local image sampling. A long curved lane needs evidence along its length. Lane attention assigns different heads to local regions distributed along the predicted boundaries, giving one instance query access to both distant context and local changes in markings.

The source figure contrasts attention around a single center with attention spread along an elongated lane. The latter preserves local evidence without allocating a full query to every point.

![LaneSegNet source Figure 3 compares object-centered deformable attention with head-specific reference regions along lane boundaries](/assets/images/lanesegnet-source-figure-3.png)
*Fig 1: Each lane-attention head samples around a different region of the predicted lane. Distributed regions cover its length, while local offsets preserve detail near a boundary or marking change. | source: [LaneSegNet, Figure 3](https://arxiv.org/abs/2312.16108)*

The first decoder layer is a deliberate exception: all heads for one query begin at the same learned reference point. Later layers distribute them using the preceding lane prediction. Separating initial location learning from subsequent shape refinement improves convergence. The ablation gives 28.8 mAP without either design, 29.1 with heads-to-regions alone, 30.6 with identical initialization alone, and 32.6 with both.

### Structured sharing works better than mixing output categories

The controlled representation study omits the specialized lane-attention designs. It compares centerline-only prediction, map-element-only prediction, two ways to combine those tasks, and lane-segment prediction.

| Representation ablation | Centerline OLS | Map-element mAP |
| --- | ---: | ---: |
| Centerline + map elements, one branch | 19.3 | 20.4 |
| Centerline + map elements, separate branches | 21.7 | 23.9 |
| Lane segment | 26.5 | 24.8 |

The shared branch can confuse incompatible target structures; sharing alone is not enough. Lane segments encode the relationship between center and boundaries directly, providing a more useful constraint. The paper's OLS in this comparison is computed from centerline detection and lane topology, so it should not be equated with the full traffic-element OLS in other papers.

Training uses Hungarian assignment, L1 curve supervision, cross-entropy and Dice mask supervision, boundary-type classification, and focal topology supervision. The main model uses ResNet-50, a BEVFormer encoder, 200 lane queries, and 24 epochs on eight V100 GPUs. Training and evaluation represent each curve with ten ordered 3D points.

### Annotation and evaluation determine what transfers

On the lane-segment benchmark, LaneSegNet reports 32.6 mAP, 32.3 lane-segment AP, and 32.9 pedestrian-crossing AP. Adapted MapTRv2 reaches 28.5 mAP. These baselines are retrained for the paper's labels; the comparison is not against their original nuScenes map-construction results. The paper's original lane–lane topology score is 8.1, whereas later evaluations of the checkpoint use revised topology metrics and report different values.

The qualitative results still misestimate the number of lateral lanes at intersections with restricted camera views. NuScenes and Waymo are not included because the required lane-segment annotations are unavailable in the reported study. This is where SD-map methods such as SEPT and Score add complementary evidence: changing the output representation cannot reveal a road that remains unobserved.

My follow-up test would retain the same annotations, query count, and runtime budget while comparing shared lane segments with separate center/boundary heads. If the gain disappears under that control or under geographically held-out evaluation, the extra annotation and representation constraints would need reconsideration.

## High-Level Takeaways

- Share geometry through a meaningful lane instance, rather than assuming that a shared output branch automatically produces compatible centerlines and boundaries.
- Distributed lane attention and identical first-layer initialization solve different problems: long-range feature collection and stable initial localization.
- The evaluation uses adapted labels and baselines; preserve that distinction when comparing with MapTR or later topology papers.
- Lane-segment annotation is a substantial data commitment, while occluded lane counts remain unresolved by representation alone.
