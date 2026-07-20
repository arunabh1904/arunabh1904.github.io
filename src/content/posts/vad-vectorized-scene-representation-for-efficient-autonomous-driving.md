---
title: 'VAD: Vectorized Scene Representation for Efficient Autonomous Driving'
date: '2023-03-21T04:00:00.000Z'
section: paper-shorts
postSlug: vad-vectorized-scene-representation-for-efficient-autonomous-driving
legacyPath: /paper shorts/2023/03/21/vad-vectorized-scene-representation-for-efficient-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: VAD replaces dense rasterized planning inputs with vectorized agents and map elements, improving end-to-end planning efficiency and safety constraints.
---
## 2023 - VAD

**arXiv:** [2303.12077](https://arxiv.org/abs/2303.12077)

**Code:** [hustvl/VAD](https://github.com/hustvl/VAD)

**Summary:** VAD argues that end-to-end driving should not have to plan from dense rasterized scene tensors. It represents the scene with vectors: agent motion and map elements stay as instance-level structures, and the planner can use them as explicit constraints.

That design is useful because rasterization can be expensive and can blur the object-level structure that planning cares about. VAD keeps the scene closer to the planner's natural language: agents, lanes, boundaries, and candidate trajectories.

## Paper Insights

The paper proposes an end-to-end vectorized paradigm for autonomous driving. Instead of generating dense occupancy or semantic-map rasters for planning, VAD uses vectorized agent and map representations, query interactions, and vectorized planning constraints. The model aims to improve both safety and speed by avoiding computation-heavy raster operations and hand-designed post-processing.

The paper reports state-of-the-art end-to-end planning performance on nuScenes. The abstract highlights VAD-Base reducing average collision rate by 29.0% while running 2.5x faster than the previous best method, and VAD-Tiny reaching up to 9.3x faster inference with comparable planning performance. The caveat is that open-loop dataset planning metrics still cannot fully prove closed-loop driving robustness.

![Figure 1 from VAD comparing rasterized scene representation with vectorized scene representation](/assets/images/vad-vectorized-scene-representation-for-efficient-autonomous-driving-paper-figure.png)
_Figure 1 shows the representational shift: VAD keeps agents, maps, and ego plans as vectors instead of flattening the scene into dense raster grids. From the [VAD paper](https://arxiv.org/abs/2303.12077), via ar5iv._

**What to look at:**
- Scene representation is vectorized end to end.
- Agent motion and map elements become explicit planning constraints.
- Efficiency gains come from avoiding dense rasterized representations.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Representation | Vectorized agents and map elements | Preserves instance-level structure for planning. |
| Benchmark | nuScenes end-to-end planning | Standard public planning evaluation. |
| Efficiency | 2.5x faster VAD-Base, up to 9.3x faster VAD-Tiny | Makes vectorized planning attractive for deployment constraints. |

**Compact result slice:**

| Method | Avg L2 (m) | Avg collision (%) | Latency (ms) | FPS |
| ------ | ---------- | ----------------- | ------------ | --- |
| UniAD | 1.03 | 0.31 | 555.6 | 1.8 |
| VAD-Tiny | 0.78 | 0.38 | 59.5 | 16.8 |
| VAD-Base | 0.72 | 0.22 | 224.3 | 4.5 |

## Decision Lens

VAD informs whether an ego planner needs a dense raster feature map or can reason over vectorized agents and map elements. Its atomic units are agent vectors, map vectors, and ego trajectory points; vector attention exposes interaction while explicit safety costs rescore plans.

The representation reduces dense BEV compute, but planning quality becomes bounded by vector extraction recall and uncertainty. The missing comparison matches latency and backbone across vector-only, raster-only, and hybrid planners under missed detections and map errors. At 10× actors, pairwise interaction and vector selection dominate. VAD's claim would fail if a low-resolution raster or occupancy interface matched collision and progress metrics while degrading more gracefully when upstream instances are missing.

**Context:** VAD carried vectorized scene understanding beyond map construction and into end-to-end planning.

**Takeaway:** Planning benefits when the model keeps the world as vectors and relations instead of flattening it into dense pixels too early.
