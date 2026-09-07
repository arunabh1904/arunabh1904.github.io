---
title: 'SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation'
date: '2024-05-30T00:00:00.000Z'
section: paper-shorts
postSlug: sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation
legacyPath: /paper shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation"
---
## 2024 – SparseDrive

**arXiv:** [2405.19620](https://arxiv.org/abs/2405.19620)

**Code:** [swc-17/SparseDrive](https://github.com/swc-17/SparseDrive)

## Summary

> SparseDrive replaces a dense BEV-centered stack with sparse agent and map instances, then lets motion prediction and ego planning interact in parallel. Its selection stage first respects the requested driving command, scores multiple trajectory modes, and sets a candidate's score to zero when predicted agent motion collides with it. On nuScenes, SparseDrive-B reports 49.6% detection mAP, 58.8% NDS, 0.58 m average planning L2, and 0.06% average collision rate. SparseDrive-S reaches 9.0 FPS with 20 hours of training, compared with UniAD's 1.8 FPS and 144 hours in the paper's comparison.

## Core Insights

### Keep the scene sparse and symmetric

SparseDrive starts from multi-view, multi-scale image features and maintains two instance sets: surrounding agents with 11-dimensional boxes and static map elements represented as polylines. Detection and online mapping share the same decoder shape, so the representation is symmetric across dynamic objects and road structure. A six-decoder perception stack has one non-temporal decoder and five temporal decoders; an instance memory queue carries features forward, and tracking comes from identity propagation rather than a separate tracking loss. The default uses 900 detection anchors, 100 map polylines, 20 points per map element, and a 0.2 detection-confidence threshold for locking an identity.

The overview shows the intended flow clearly: the encoder feeds sparse instances, those instances feed both perception and the planner, and the memory queue provides the temporal link. The model is sparse in the scene representation, not blind to dense context—the smallest front feature map also initializes the ego instance and supplies a fallback for obstacles that sparse perception misses.

![Figure 2 from SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation](/assets/images/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation-paper-figure.png)
*Fig 1: Multi-view features become symmetric sparse agent/map instances, which then support parallel motion prediction and planning. | source: [SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation, Figure 2](https://arxiv.org/abs/2405.19620)*

### Give prediction and planning the same interaction space

The ego vehicle is initialized from the front feature map rather than a random token. Its position, size, and yaw are known, while velocity is initialized from the previous frame's prediction to avoid leaking ground-truth ego status. Ego and surrounding instances then exchange information through agent-temporal cross-attention, agent-agent self-attention, and agent-map cross-attention. The planner predicts six modes for each surrounding agent over 12 future timestamps and six ego modes over six timestamps, with separate scores for each command branch.

Selection is deliberately hierarchical. First choose the subset corresponding to left, right, or straight; then use the two most confident motion-prediction trajectories to check each ego proposal for collision; finally choose the highest remaining score. The learned proposal scores and an explicit collision-zeroing rule are combined before the final argmax. That keeps the safety check inside the selection stage, while making clear that the zeroing rule itself is hand-coded rather than learned.

![Figure 1 from SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation](/assets/images/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation-source-figure-2.webp)
*Fig 2: The sparse-centric pipeline removes the dense BEV block and connects sparse perception outputs directly to motion and planning. | source: [SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation, Figure 1](https://arxiv.org/abs/2405.19620)*

### The metrics reward both uncertainty and efficiency

Training is staged: symmetric sparse perception learns first, then the planner and perception modules are jointly optimized with no frozen weights. The loss combines detection, mapping, motion, planning, and auxiliary depth terms. Multi-modal trajectories use winner-takes-all assignment, while the collision-aware rescore is tested under a corrected collision metric that accounts for ego yaw and box overlap rather than a 0.5 m occupancy grid.

On nuScenes, SparseDrive-B reaches 0.60 m minADE, 0.96 m minFDE, 13.2% miss rate, and 0.555 EPA for motion prediction. Its planning errors at 1/2/3 seconds are 0.29/0.55/0.91 m, averaging 0.58; collision rates are 0.01/0.02/0.13%, averaging 0.06%. Against VAD, the average L2 improvement is 19.4% and the collision-rate reduction is 71.4%. The six-mode ablation reaches 0.61 m average L2 and 0.07% collision, while one mode reaches 0.69 m and 0.25%, showing why uncertainty is part of the planning problem. The paper's main limitation is scale: its nuScenes corpus contains 1,000 total 20-second scenes, and sparse instances can still miss objects that are outside the learned anchors or hard to detect.

## High-Level Takeaways

- SparseDrive's scene carrier is a learned set of agents and map elements, with temporal identity and geometry kept explicit.
- Parallel prediction and planning let the ego trajectory respond to the same agent/map interactions used to forecast other road users.
- Multi-modal proposals plus collision-aware rescoring improve safety under the paper's corrected box-overlap metric without a post-optimization stage.
- The efficiency result is tied to the reported hardware and input sizes, while the sparse representation still depends on perception coverage and a relatively small nuScenes corpus.
