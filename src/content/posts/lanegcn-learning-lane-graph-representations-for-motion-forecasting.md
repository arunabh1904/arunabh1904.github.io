---
title: 'LaneGCN: Learning Lane Graph Representations for Motion Forecasting'
date: '2020-07-27T04:00:00.000Z'
section: paper-shorts
postSlug: lanegcn-learning-lane-graph-representations-for-motion-forecasting
legacyPath: /paper shorts/2020/07/27/lanegcn-learning-lane-graph-representations-for-motion-forecasting.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: LaneGCN made lane topology and actor-map interaction explicit by building a graph from raw map lanes and fusing actor-to-lane, lane-to-lane, lane-to-actor, and actor-to-actor messages.
---
## 2020 - LaneGCN

**arXiv:** [2007.13732](https://arxiv.org/abs/2007.13732)

**Code:** [uber-research/lanegcn](https://github.com/uber-research/lanegcn)

**Summary:** LaneGCN treats the map as a graph, not an image. It constructs lane nodes from raw map data and uses graph convolutions that respect lane connectivity, then fuses those map features with actor motion features.

The useful idea is explicit relational structure. LaneGCN does not only ask "what is near the actor?" It asks how actors, lanes, neighboring lanes, and other actors should pass information to each other.

## Paper Insights

LaneGCN builds a lane graph with several adjacency types and uses dilated lane convolutions to capture long-range dependencies along the road topology. ActorNet encodes observed agent trajectories. FusionNet then applies four interaction blocks: actor-to-lane, lane-to-lane, lane-to-actor, and actor-to-actor.

The paper argues that vector maps should keep their graph structure through the model. Raster encoders can represent road context, but they must infer topology from pixels. LaneGCN exposes topology directly and reports state-of-the-art motion forecasting on Argoverse at publication time. The caveat is that the model assumes access to reasonably accurate lane graph data.

![Figure 1 from LaneGCN showing lane graph construction, actor encoding, actor-map fusion, and trajectory prediction](/assets/images/lanegcn-learning-lane-graph-representations-for-motion-forecasting-paper-figure.png)
_Figure 1 shows the full LaneGCN pipeline: a lane graph and actor trajectories are encoded separately, fused through structured actor-map interactions, and decoded into future trajectories. From the [LaneGCN paper](https://arxiv.org/abs/2007.13732), via ar5iv._

**What to look at:**
- Lane graph edges preserve topology that raster features can blur.
- Along-lane dilation helps messages travel farther than immediate lane neighbors.
- Actor-map fusion is directional and staged, not one generic attention block.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Map encoder | Lane graph convolution | Keeps road topology explicit. |
| Actor encoder | ActorNet over observed trajectories | Encodes dynamic motion separately from static map geometry. |
| Fusion | A2L, L2L, L2A, A2A interactions | Makes actor-map relations first-class. |
| Benchmark | Argoverse motion forecasting | Tests the value of explicit lane structure in a public driving setting. |

## Decision Lens

LaneGCN informs whether road topology should be rasterized into pixels or preserved as an explicit directed lane graph for forecasting. Its atomic units are actor histories and lane-segment nodes; typed lane-to-lane, actor-to-lane, lane-to-actor, and actor-to-actor messages control which interactions are represented.

The graph supplies topology efficiently, but map quality and graph radius become hidden dependencies. The missing factorial ablation holds the trajectory decoder fixed while replacing the lane graph with raster and polyline encoders at equal latency. At 10× agents or map extent, interaction edges and neighborhood expansion dominate. LaneGCN's representation claim would fail if a simpler polyline attention model matched forecasting accuracy and map generalization with fewer graph-specific operations.

**Context:** LaneGCN set up a clean actor-map relational template that later forecasting and planning models kept reusing in denser Transformer forms.

**Takeaway:** Motion forecasting improves when the model can reason over the lane graph as a graph, not as a painted background.
