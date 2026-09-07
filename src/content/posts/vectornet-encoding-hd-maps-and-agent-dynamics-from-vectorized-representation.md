---
title: 'VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation'
date: '2020-05-08T00:00:00.000Z'
section: paper-shorts
postSlug: vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation
legacyPath: /paper shorts/2020/05/08/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2020 – VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation"
---
## 2020 – VectorNet

**arXiv:** [2005.04259](https://arxiv.org/abs/2005.04259)

### Method and reported result

VectorNet is a foundational vectorized-scene paper. Instead of rendering maps and trajectories into bird's-eye-view images, it keeps lanes, crosswalks, traffic elements, and agent histories as vectors grouped into polylines.

## Summary

> That representation matters because autonomous driving scenes are already structured. VectorNet lets the model operate on map and agent geometry directly, first within each polyline and then across the whole scene.

## Core Insights

VectorNet uses a hierarchical graph neural network. A local subgraph network summarizes each polyline, such as one lane segment or one agent trajectory. A global interaction graph then lets those polyline-level nodes exchange information. The paper also adds a masked entity completion objective: the model must reconstruct randomly hidden map entities or agent trajectories from context.

The paper's main contrast is against rasterization. Raster BEV images let standard convolutional networks process the scene, but rendering discards some structure and spends computation on pixels that are not meaningful entities. VectorNet reports comparable or better behavior prediction on an internal benchmark and Argoverse while saving more than 70% of model parameters and roughly an order of magnitude in FLOPs against the rendering baseline.

The masked entity-completion loss is more than a generic pretext task. When a lane polyline or agent history is hidden, the global graph must infer it from neighboring geometry and motion context before the forecasting head sees the scene. In the paper's Argoverse ablation, adding this auxiliary objective improves displacement error at three seconds from 3.84 to 3.67, a modest but direct test that the global representation became more useful for forecasting. The gain does not erase preprocessing assumptions: the vectorizer still decides how curves are segmented and what happens when map entities are missing.

![Figure 2 from VectorNet showing input vectors, polyline subgraphs, a global interaction graph, and trajectory prediction](/assets/images/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation-paper-figure.png)
*Fig 1: Shows the core hierarchy: vectors become polyline features, polyline features interact globally, and agent nodes support map completion and trajectory prediction. | source: [VectorNet paper](https://arxiv.org/abs/2005.04259)*

![Figure 4 from VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](/assets/images/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation-source-figure-4.webp)
*Fig 2: (Left) Visualization of the prediction: lanes are shown in grey, non-target agents are green, target agent’s ground truth trajectory is in pink, predicted trajectory in blue. (Right) Visualization of attention for road and agent: Brighter red color corresponds to higher attention score. | source: [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259)*




## High-Level Takeaways

- VectorNet informs whether HD maps and trajectories should be rasterized or preserved as polylines with explicit geometric identity. The atomic unit is a point feature grouped into a polyline; local aggregation creates polyline embeddings and a global graph models interactions among agents and map elements.
- Vectorization preserves topology and avoids empty pixels, but preprocessing choices determine segment length, coordinate frame, and missing-map behavior. The missing ablation matches encoder capacity and latency against raster and raw-point attention across map quality levels. At 10× map extent, global polyline attention and retrieval dominate. The claim would fail if a compact raster encoder matched forecasting accuracy and cross-city generalization without vector-specific preprocessing.
- VectorNet made vectorized map and agent encoding feel like a primary representation, not a preprocessing trick.
- If the world is already made of lanes, agents, and polylines, the encoder should not have to rediscover those entities from pixels.
