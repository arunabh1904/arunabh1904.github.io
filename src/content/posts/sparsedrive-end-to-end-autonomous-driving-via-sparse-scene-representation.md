---
title: 'SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation'
date: '2024-05-30T04:00:00.000Z'
section: paper-shorts
postSlug: sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation
legacyPath: /paper shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html
tags:
  - Other
field: BEV
summary: SparseDrive replaces expensive dense BEV planning features with sparse scene instances, symmetric sparse perception, parallel motion planning, and collision-aware rescoring.
---
## 2024 - SparseDrive

**arXiv:** [2405.19620](https://arxiv.org/abs/2405.19620)

**Code:** [swc-17/SparseDrive](https://github.com/swc-17/SparseDrive)

**Summary:** SparseDrive asks whether end-to-end driving really needs dense BEV features everywhere. It argues for a sparse-centric representation: keep trackable agents and map elements as sparse instances, then predict motion and plan from those instances.

This puts SparseDrive in the same broad family as VAD: the planner should reason over structured scene entities instead of spending most of its budget on a dense grid.

## Paper Insights

SparseDrive has three main parts. Symmetric sparse perception unifies object detection, tracking, and online mapping in a sparse instance representation. A parallel motion planner performs motion prediction and planning together. A hierarchical planning selector and collision-aware rescoring stage choose safer trajectories.

The paper frames dense BEV computation as both an efficiency problem and a planning-safety problem. Sparse scene representations are lighter and keep agent-map instances explicit, but the model must preserve enough context to avoid losing global scene cues. That is the central tradeoff.

![Figure 3 from SparseDrive showing the sparse scene representation pipeline for perception, motion prediction, and planning](/assets/images/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation-paper-figure.png)
_Figure 3 shows SparseDrive's architecture: image features become sparse scene representations, which support symmetric sparse perception and parallel motion planning. From the [SparseDrive paper](https://arxiv.org/abs/2405.19620), via arXiv HTML._

**What to look at:**
- Sparse perception represents agents and maps symmetrically.
- Motion prediction and ego planning run in parallel instead of as a long cascade.
- Collision-aware rescoring injects an explicit safety check into planning selection.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Representation | Sparse scene instances | Avoids dense BEV cost while preserving objects and map elements. |
| Perception | Detection, tracking, and mapping in a symmetric sparse module | Keeps dynamic and static scene structure aligned. |
| Planning | Parallel motion planner | Lets agent prediction and ego planning interact earlier. |
| Safety | Collision-aware rescoring | Makes trajectory selection sensitive to physical conflicts. |

**Context:** SparseDrive sharpened the argument that sparse/vectorized planning can be both faster and more planner-aligned than dense BEV stacks.

**Takeaway:** Dense BEV is powerful, but the planner often wants sparse entities and relations.
