---
title: 'VAD: Vectorized Scene Representation for Efficient Autonomous Driving'
date: '2023-03-21T00:00:00.000Z'
section: paper-shorts
postSlug: vad-vectorized-scene-representation-for-efficient-autonomous-driving
legacyPath: /paper shorts/2023/03/21/vad-vectorized-scene-representation-for-efficient-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2023 – VAD: Vectorized Scene Representation for Efficient Autonomous Driving"
---
## 2023 – VAD

**arXiv:** [2303.12077](https://arxiv.org/abs/2303.12077)

**Code:** [hustvl/VAD](https://github.com/hustvl/VAD)

## Method
VAD argues that end-to-end driving should not have to plan from dense rasterized scene tensors. It represents the scene with vectors: agent motion and map elements stay as instance-level structures, and the planner can use them as explicit constraints.

## Summary

> That design is useful because rasterization can be expensive and can blur the object-level structure that planning cares about. VAD keeps the scene closer to the planner's natural language: agents, lanes, boundaries, and candidate trajectories.

## Core Insights

The paper proposes an end-to-end vectorized paradigm for autonomous driving. Instead of generating dense occupancy or semantic-map rasters for planning, VAD uses vectorized agent and map representations, query interactions, and vectorized planning constraints. The model aims to improve both safety and speed by avoiding computation-heavy raster operations and hand-designed post-processing.

The paper reports state-of-the-art end-to-end planning performance on nuScenes. The abstract highlights VAD-Base reducing average collision rate by 29.0% while running 2.5x faster than the previous best method, and VAD-Tiny reaching up to 9.3x faster inference with comparable planning performance. The caveat is that open-loop dataset planning metrics still cannot fully prove closed-loop driving robustness.

The planner uses 200 BEV queries, 100 map-vector queries with 20 points each, and 300 agent queries over a 60 m by 30 m local range. Its safety constraints are expressed in the same coordinate system as the plan: keep the ego trajectory away from predicted agents, keep it inside the ego boundary, and keep its direction consistent with the lane vector. A map-representation ablation makes the choice more than a visual preference: vectorized maps with all constraints reach 0.72 m average L2 and 0.22% average collision, while a rasterized map reaches 0.74 m and 0.39%.

The closed-loop CARLA result is also more informative than the abstract speed claim. On Town05 Short, VAD-Base reaches driving score 64.29 and route completion 87.26; on Town05 Long it reaches 30.31 and 75.20. VAD-Tiny runs at 9.3 times the reference speed, while VAD-Base is 2.5 times faster than the previous best in the paper’s comparison. The remaining risk is representation recall: a missed agent or map vector disappears before the planner can apply any constraint.

The useful abstraction is therefore a shared coordinate language between perception and planning. Agent vectors expose who can collide with the ego vehicle; boundary and lane vectors expose where the vehicle may travel and in which direction. VAD can apply those three constraints directly to future waypoints, whereas a dense raster would need another step to recover instance identity, geometry, and lane direction. That coupling explains both the efficiency gain and the failure mode: a compact but incorrect vector is more dangerous than a blurry feature map because it can impose the wrong safety constraint with confidence.

Read the architecture figure as four phases rather than one monolithic decoder: image features become a BEV substrate, queries extract vectorized agents and maps, the ego query attends to both, and the planner regularizes its trajectory with the same vectors. The qualitative constraint figure is the companion intuition: its collision, boundary, and lane-direction penalties are geometric checks on the plan, not extra perception labels.

![Figure 1 from VAD comparing rasterized scene representation with vectorized scene representation](/assets/images/vad-vectorized-scene-representation-for-efficient-autonomous-driving-paper-figure.png)
*Fig 1: Shows the representational shift: VAD keeps agents, maps, and ego plans as vectors instead of flattening the scene into dense raster grids. | source: [VAD paper](https://arxiv.org/abs/2303.12077)*

![Figure 2 from VAD: Vectorized Scene Representation for Efficient Autonomous Driving](/assets/images/vad-vectorized-scene-representation-for-efficient-autonomous-driving-source-figure-2.webp)
*Fig 2: The four phases turn multi-view features into agent and map vectors, forecast other agents, and decode a constrained ego plan. The arrows show where vector instances enter the planning query, which is the paper’s efficiency/safety interface. | source: [VAD: Vectorized Scene Representation for Efficient Autonomous Driving](https://arxiv.org/abs/2303.12077)*




## High-Level Takeaways

- VAD informs whether an ego planner needs a dense raster feature map or can reason over vectorized agents and map elements. Its atomic units are agent vectors, map vectors, and ego trajectory points; vector attention exposes interaction while explicit safety costs rescore plans.
- The representation reduces dense BEV compute, but planning quality becomes bounded by vector extraction recall and uncertainty.
- VAD carried vectorized scene understanding beyond map construction and into end-to-end planning.
- Planning benefits when the model keeps the world as vectors and relations instead of flattening it into dense pixels too early.
