---
title: 'EMMA: End-to-End Multimodal Model for Autonomous Driving'
date: '2024-10-30T00:00:00.000Z'
section: paper-shorts
postSlug: emma-end-to-end-multimodal-model-for-autonomous-driving
legacyPath: /paper shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – EMMA: End-to-End Multimodal Model for Autonomous Driving"
---
## 2024 – EMMA

**arXiv:** [2410.23262](https://arxiv.org/abs/2410.23262)

**Project:** [Waymo research page](https://waymo.com/research/emma/)

## Summary

> EMMA is Waymo's end-to-end multimodal driving model. It uses camera data plus non-sensor state such as navigation commands and ego status, then predicts driving outputs including trajectories, objects, and road graph elements through task-specific prompts. The striking design choice is to represent many non-sensor inputs and outputs as text. That lets the model reuse the structure and world knowledge of a multimodal language model while training across several driving tasks.

## Core Insights

### One textual output interface serves several driving tasks

EMMA builds autonomous-driving outputs on top of a multimodal foundation model. It maps camera inputs, navigation instructions, ego state, road graph elements, objects, and trajectories into a unified language-like interface with task-specific prompts. The paper's evidence includes strong motion planning on nuScenes and competitive Waymo motion results. The appeal is one model for several driving outputs; the risk is precision. Text-style serialization must still produce exact geometry, calibrated trajectories, and low-latency behavior for safety-critical driving.

The basic planner is deliberately simple: surround-view images, a route intent, and a history of ego waypoints are serialized as text-like inputs, and future $(x,y)$ waypoints are generated autoregressively. The target is self-supervised because only future ego locations are required. EMMA then reuses the same interface for objects and road graphs, with coordinate precision and punctuation treated as part of the output format. Multiple trajectories are sampled on WOMD and clustered by a median-like L2 rule; on nuScenes, the shorter three-second horizon makes top-1 decoding sufficient.

The readable rationale is not an extra human annotation layer. The paper generates scene descriptions with off-the-shelf perception/prediction models and heuristic decisions, then trains EMMA to produce those descriptions before the trajectory. That makes the interface scalable, but it also means the model can inherit upstream errors and the language output should be treated as an inspectable intermediate representation rather than proof of grounded reasoning.

The scale comparison separates foundation-model initialization from data. On nuScenes, EMMA reaches average L2 0.32 m and EMMA+ reaches 0.29 m, compared with 0.37 m for the randomly initialized version and 0.35 m for the self-supervised BEV-Planner baseline. On the internal WOMD benchmark, EMMA+ with chain-of-thought reaches 0.543 m at five seconds, versus 0.610 m without CoT; the same trend is smaller on public data. Sampling helps, with diminishing returns after about 12 trajectories in an experiment evaluated through 24 samples. These results make the unified interface credible for transfer, while leaving text quantization, decoding latency, and long-horizon compounding as real engineering costs.

Read the overview as one prompt/output contract reused across tasks: a task instruction and camera context enter the shared model, textual predictions come out, and small decoders turn them into boxes, road graphs, or waypoints.

![Figure 1: EMMA overview diagram from EMMA: End-to-End Multimodal Model for Autonomous Driving](/assets/images/emma-end-to-end-multimodal-model-for-autonomous-driving-paper-figure.png)
*Fig 1: EMMA frames perception, scene understanding, and driving decisions as language generation from a shared multimodal model instead of separate task-specific heads. | source: [EMMA: End-to-End Multimodal Model for Autonomous Driving paper, Figure 1](https://arxiv.org/abs/2410.23262)*

Sampling makes the ambiguity of the future explicit: several continuations can be consistent with the same history. Figure 3 tracks ADE at five seconds as the sample count grows. The gain flattens after about twelve, so more decoding has diminishing value even before its computational cost is considered.

![Figure 3 from EMMA: End-to-End Multimodal Model for Autonomous Driving](/assets/images/emma-end-to-end-multimodal-model-for-autonomous-driving-source-figure-3.webp)
*Fig 2: ADE@5s improves as EMMA samples more trajectory candidates, then flattens after roughly 12 samples. The curve shows why multi-sampling is valuable for long-horizon multimodal behavior but cannot substitute for a better single-sample decoder indefinitely. | source: [EMMA: End-to-End Multimodal Model for Autonomous Driving, Figure 3](https://arxiv.org/abs/2410.23262)*


## High-Level Takeaways

- EMMA informs whether driving perception, road-structure understanding, and planning can be represented through one language-token interface. Its atomic unit is an autoregressive token conditioned on camera observations and a task prompt; shared parameters produce textual structures and future trajectories in a common format.
- The unified vocabulary simplifies task transfer but quantizes geometry and ties control latency to sequence generation. The missing ablation compares tokenized trajectories with a continuous head while keeping the visual-language backbone and multitask data fixed.
- EMMA is a strong example of the generalist-model thesis entering autonomous driving: one model, many outputs, shared representations.
- Language can be a unifying interface for driving tasks, but EMMA also makes the costs obvious: limited temporal context, no full 3D sensor stack, and heavy compute.
