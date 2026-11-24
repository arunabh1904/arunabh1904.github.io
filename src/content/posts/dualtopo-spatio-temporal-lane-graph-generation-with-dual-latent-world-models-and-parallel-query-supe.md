---
title: 'DualTopo: Spatio-Temporal Lane Graph Generation with Dual Latent World Models and Parallel Query Supervision'
date: '2026-11-18T09:00:00.000Z'
section: paper-shorts
postSlug: dualtopo-spatio-temporal-lane-graph-generation-with-dual-latent-world-models-and-parallel-query-supe
legacyPath: /paper shorts/2026/11/18/dualtopo-spatio-temporal-lane-graph-generation-with-dual-latent-world-models-and-parallel-query-supe.html
tags: [Paper Radar, Lane Topology, Temporal Modeling]
field: 'BEV Perception & Mapping'
summary: "2026 – DualTopo: Spatio-Temporal Lane Graph Generation with Dual Latent World Models and Parallel Query Supervision"
---

## 2026 – DualTopo: Spatio-Temporal Lane Graph Generation with Dual Latent World Models and Parallel Query Supervision

**Paper:** [Yang et al., arXiv:2507.23325v5](https://arxiv.org/abs/2507.23325v5) · **Code:** [DualTopo](https://github.com/YimingYang23/DualTopo)

## Summary

> Fresh lane queries need to work on the first frame, even when training usually gives them strong competitors from earlier frames. DualTopo lets fresh and temporal pathways match targets separately while sharing decoder weights. It also predicts how stored lane queries and bird's-eye-view features change with ego motion. The combination improves the studied lane graphs at a measured throughput cost.

## Core Insights

### Give the first frame its own training signal

A lane graph needs both geometry and directed links. The SubsetA task predicts each lane's centerline and boundaries, plus links between lanes. In a streaming decoder, propagated queries already carry evidence about specific lanes. In the paper's ordinary streaming baseline, they occupied 33% of the available query slots but got 75.2% of ground-truth matches after the first frame. Fresh queries still have to find lanes when no history is available.

DualTopo updates all fresh queries using the current image features in its first decoder layer. On later frames, the temporal path then replaces the 66 lowest-confidence fresh queries with 66 propagated queries, out of 200 total slots. The fresh path keeps all its queries. The remaining decoder layers and prediction heads share weights, but each pathway independently assigns predictions to targets during training. Thus, a good temporal query cannot take away the fresh pathway's match to that lane.

![Fresh and temporal pathways use the same decoder weights, but each computes its own assignment to ground truth.](/assets/images/dualtopo-parallel-query-supervision.svg)
*Fig 1: Separate assignment gives the fresh path direct lane supervision even when propagated queries are stronger. Original explanatory diagram. | source: [DualTopo v5, Figure 2](https://arxiv.org/html/2507.23325v5#S1.F2)*

### Predict past-to-present state, then check it against the current view

Pose-based warping can resample only where the old bird's-eye-view grid has support. The new grid can include area the vehicle has just moved into. DualTopo gives separate Transformer models the prior lane queries and prior fused scene features, each conditioned on the known change in vehicle pose. One predicts current query features, with an update informed by prior directed links. The other predicts a feature grid, which a gated recurrent unit combines with current image evidence.

The predicted grid is trained against current image-derived features before fusion. The query model learns from prior-frame lane annotations transformed into the current coordinate frame. This setup supplies a motion-conditioned prior, rather than independent evidence for a new lane outside the old view. The model carries state forward to the frame it is processing; it does not forecast a planner's future trajectory.

### The controls favor separate matching, then learned propagation

In the [SubsetA module ablation](https://arxiv.org/html/2507.23325v5#S3.T5.fig1), ordinary streaming with BEV warping scores 34.0 mAP on the temporal output, but its fresh output falls to 29.8 mAP from 32.6 for single-frame training. Keeping warping and adding parallel supervision lifts the temporal and fresh results to 38.6 and 36.4. Adding both world models raises the temporal result to 40.5 mAP and 32.1 lane-to-lane topology, with fresh mAP at 36.9. Here mAP averages lane-segment and pedestrian-crossing AP; the topology score evaluates links after matching predicted segments.

The efficiency trade-off shows up on SubsetB. On the A800, adding DualTopo to RATopo takes the composite OpenLane-V2 score from 50.3 to 55.0 and throughput from 5.6 to 4.8 FPS. Other tested centerline backbones show the same direction of trade-off. These results measure the perception pipeline, not closed-loop planning. The temporal path uses relative ego motion, and these tables do not establish how gracefully it degrades with drift or missing pose updates.

## High-Level Takeaways

- Measure the fresh path after introducing history. An aggregate gain can hide degradation on cold starts.
- Give fresh and temporal queries separate target assignments before spending more compute on a learned transition.
- To test whether the extra compute is worthwhile, compare junction edge errors during first frames, pose dropouts, and newly visible regions at the same inference budget.
