---
title: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model"
date: "2025-01-27T00:00:00.000Z"
section: paper-shorts
postSlug: spatialvla-exploring-spatial-representations
legacyPath: /paper shorts/2025/01/27/spatialvla-exploring-spatial-representations.html
tags: ["Robotics", "Spatial Reasoning"]
field: "Vision-Language-Action & Robotics"
summary: "2025 – SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model"
---

## 2025 – SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model

**Paper:** [arXiv:2501.15830](https://arxiv.org/abs/2501.15830) · [PDF](https://arxiv.org/pdf/2501.15830)

## Summary

> SpatialVLA adds depth-derived 3D positions to visual tokens and predicts robot motion through adaptive spatial action grids. Pretrained on 1.1 million robot demonstrations, it reaches 78.1% average success after LIBERO fine-tuning versus 76.5% for the reported OpenVLA baseline. Its central contribution is to structure both observation and action spaces; the evidence concerns manipulation, not BEV driving.

## Core Insights

### Give visual tokens a location and action tokens a neighborhood

A PaliGemma2 backbone receives images and a task instruction. ZoeDepth estimates a depth map, and camera intrinsics back-project image locations into camera-relative 3D coordinates. A sinusoidal encoding and learned MLP turn those positions into embeddings that are added to SigLIP features. This avoids requiring robot-to-camera extrinsic calibration, but still depends on camera intrinsics and estimated depth.

The action representation changes too. Seven control dimensions become three autoregressive tokens: one for translation, one for rotation, and one for the gripper. Translation is represented by direction and radius. Rather than assigning the same interval width everywhere, the tokenizer fits Gaussian distributions to action variables and places boundaries at equal-probability intervals. Frequently used movements receive finer resolution. The model predicts a translation cell and rotation cell, then decodes their coordinates into continuous commands.

Follow the two spatial interfaces in the overview. The observation branch adds depth-derived positions; the output branch maps a small token sequence back into robot movement.

![Depth-aware image encoding and adaptive action-token prediction; source Figure 2](/assets/images/spatialvla-source-figure-2.webp)
*Fig 1: SpatialVLA combines 3D position embeddings with an action vocabulary organized around translation, rotation, and gripper state. The two interfaces are trained together. | source: [Paper, Figure 2](https://arxiv.org/abs/2501.15830)*

[View full-size figure](/assets/images/spatialvla-source-figure-2.webp)

### Adapt the grid when the robot changes

The pretraining mixture draws on OXE and RH20T, with 1.1 million demonstrations. Image observations, task instructions, and associated actions supply next-token supervision. The vision encoder and language backbone are trained along with new spatial parameters, while text-token embeddings remain frozen. DROID is removed for the final third of pretraining. The recipe therefore includes data and optimization choices beyond adding coordinates.

For a new robot, the method fits the action distribution again and adapts the spatial embeddings to the resulting grid. This matters because a cell that represented a common displacement during pretraining may represent an uncommon displacement for the target setup. The figure shows why equal-probability bins create unequal geometric intervals.

![Action distributions converted into nonuniform translation and rotation grids; source Figure 3](/assets/images/spatialvla-source-figure-3.webp)
*Fig 2: Adaptive grid boundaries allocate resolution according to observed action frequency. Their geometry changes when the target robot has a different movement distribution. | source: [Paper, Figure 3](https://arxiv.org/abs/2501.15830)*

[View full-size figure](/assets/images/spatialvla-source-figure-3.webp)

Table IV isolates useful effects on the Fractal/Bridge mixture. Removing Ego3D encoding reduces eggplant-placement success from 87.5% to 37.5%, while replacing the learned distribution with uniform discretization gives 54.2%. The gains are not universal: one grasp-carrot submetric improves without Ego3D. On LIBERO, adapted spatial embeddings with LoRA improve the four suites over the unadapted LoRA variant, but SpatialVLA's 78.1% average still includes only 55.5% success on LIBERO-Long.

Compared with [SpatialVLM](/paper%20shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html), the output is now an executable action, and spatial structure enters the action vocabulary as well as the observation. The result supports testing an explicit geometry channel. It does not show that a depth estimate is calibrated, that discretization is harmless, or that three tokens imply low total system latency once depth inference is included.

## High-Level Takeaways

- Treat action tokenization as a distribution-design problem: the same vocabulary size can allocate very different precision to common movements.
- Spatial grounding depends on the observation encoding, action representation, training mixture, and adaptation procedure together.
- A fair driving transfer experiment should compare equal-budget image and geometry-conditioned policies; manipulation success is evidence for the design principle, not a driving result.
