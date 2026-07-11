---
title: 'VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation'
date: '2020-05-08T04:00:00.000Z'
section: paper-shorts
postSlug: vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation
legacyPath: /paper shorts/2020/05/08/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation.html
tags:
  - Other
field: BEV
summary: VectorNet encoded HD maps and agent histories as polylines, using local polyline aggregation and a global interaction graph instead of rasterized BEV images.
---
## 2020 - VectorNet

**arXiv:** [2005.04259](https://arxiv.org/abs/2005.04259)

**Summary:** VectorNet is a foundational vectorized-scene paper. Instead of rendering maps and trajectories into bird's-eye-view images, it keeps lanes, crosswalks, traffic elements, and agent histories as vectors grouped into polylines.

That representation matters because autonomous driving scenes are already structured. VectorNet lets the model operate on map and agent geometry directly, first within each polyline and then across the whole scene.

## Paper Insights

VectorNet uses a hierarchical graph neural network. A local subgraph network summarizes each polyline, such as one lane segment or one agent trajectory. A global interaction graph then lets those polyline-level nodes exchange information. The paper also adds a masked entity completion objective: the model must reconstruct randomly hidden map entities or agent trajectories from context.

The paper's main contrast is against rasterization. Raster BEV images let standard convolutional networks process the scene, but rendering discards some structure and spends computation on pixels that are not meaningful entities. VectorNet reports comparable or better behavior prediction on an internal benchmark and Argoverse while saving more than 70% of model parameters and roughly an order of magnitude in FLOPs against the rendering baseline.

![Figure 2 from VectorNet showing input vectors, polyline subgraphs, a global interaction graph, and trajectory prediction](/assets/images/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation-paper-figure.png)
_Figure 2 shows the core hierarchy: vectors become polyline features, polyline features interact globally, and agent nodes support map completion and trajectory prediction. From the [VectorNet paper](https://arxiv.org/abs/2005.04259), via the arXiv PDF._

**What to look at:**
- Map and motion inputs stay as vectors rather than rendered pixels.
- The model separates local polyline structure from global scene interaction.
- Masked map and trajectory completion make context learning part of training.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Representation | HD maps and trajectories as polylines | Preserves lane and agent geometry without rasterization. |
| Architecture | Local polyline subgraphs plus global graph | Matches the natural hierarchy of driving scenes. |
| Auxiliary task | Masked entity completion | Forces the global graph to use scene context. |
| Evidence | Internal benchmark and Argoverse | Shows vectorized encoding can compete with rendered BEV baselines. |

**Context:** VectorNet made vectorized map and agent encoding feel like a primary representation, not a preprocessing trick.

**Takeaway:** If the world is already made of lanes, agents, and polylines, the encoder should not have to rediscover those entities from pixels.
