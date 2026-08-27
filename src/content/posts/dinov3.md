---
title: 'DINOv3'
date: '2025-08-13T00:00:00.000Z'
section: paper-shorts
postSlug: dinov3
legacyPath: /paper shorts/2025/08/13/dinov3.html
tags:
  - Self-Supervised Learning
  - Dense Vision
field: 'Vision Foundations'
topics:
  - learning
summary: '2025 – DINOv3'
---

## 2025 – DINOv3

**arXiv:** [2508.10104](https://arxiv.org/abs/2508.10104)

**Code and models:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## Summary

> DINOv3 scales self-supervised vision to a 6.7-billion-parameter transformer trained on 1.7 billion images, but its key technical result is about what scale breaks. During very long DINO/iBOT training, patch features gradually become too similar to the global class token. Classification keeps improving while dense spatial quality degrades. Gram anchoring constrains patch-to-patch relationships to an earlier, spatially healthier teacher.

## Core Insights

![DINOv3 ablations show Gram anchoring restoring dense prediction while retaining ImageNet classification](/assets/images/dinov3-gram-anchoring-paper-figure.png)
*Fig 1: Long training improves global recognition but erodes dense features. Matching the student's patch Gram matrix to an earlier teacher restores segmentation and depth performance with little classification loss. | source: [DINOv3](https://arxiv.org/abs/2508.10104)*

![Figure 1 from DINOv3](/assets/images/dinov3-source-figure-1.webp)
*Fig 2: (a) Evolution of linear probing results on ImageNet1k (IN1k) over the years, comparing fully- (SL), weakly- (WSL) and self-supervised learning (SSL) methods. Despite coming into the picture later, SSL has quickly progressed and now reached the Imagenet accuracy plateau of recent years. | source: [DINOv3](https://arxiv.org/abs/2508.10104)*

![Figure 3 from DINOv3](/assets/images/dinov3-source-figure-3.webp)
*Fig 3: High-resolution dense features. We visualize the cosine similarity maps obtained with DINOv3 output features between the patches marked with a red cross and all other patches. | source: [DINOv3](https://arxiv.org/abs/2508.10104)*


DINOv3 retains the DINO global loss, iBOT patch loss, Sinkhorn target centering, and feature-spreading regularization. The new diagnosis is representational: patch tokens increasingly encode the same global information instead of preserving local relations. The authors save an earlier teacher whose dense features remain strong and add a loss between teacher and student patch Gram matrices,

$$
\mathcal{L}_{\text{Gram}} =
\left\|
X_s X_s^\top - X_g X_g^\top
\right\|_F^2.
$$

Because the loss matches pairwise relations rather than individual coordinates, it does not require the current model to copy the earlier feature basis exactly. In the reported recipe, Gram anchoring is introduced during a refinement stage after the long base run; the Gram teacher is refreshed periodically. High-resolution adaptation and multi-student distillation then produce a family of smaller checkpoints.

| Ablation | ImageNet linear | ADE20K mIoU | NYU depth RMSE ↓ |
| --- | ---: | ---: | ---: |
| Long-training baseline | 88.2% | 50.3 | 0.307 |
| Gram anchoring, 200k refinement ×2 | 88.0% | 55.7 | 0.281 |

This table isolates the main claim better than the largest-model leaderboard. Dense segmentation rises 5.4 points and depth error falls while classification changes by 0.2 points. The result says that one representation can retain both global and local information, but only if the training objective explicitly protects spatial relations late in training.

## High-Level Takeaways

- DINOv3 informs whether continued self-supervised scale is uniformly beneficial. It is not: a proxy such as ImageNet linear accuracy can hide deterioration in patch geometry. The atomic monitoring unit should therefore include both global and dense probes throughout the run, not only at the final checkpoint.
- Gram anchoring depends on choosing a good earlier teacher. That choice introduces a checkpoint-selection oracle: the system must know when dense quality is high enough to preserve. The decisive missing experiment varies teacher age and anchoring onset under a fixed total compute budget. At ten times scale, storing teachers, computing patch Gram matrices, and running dense probes become material costs, though still smaller than wasting a frontier-scale pretraining run.
- DINOv3 also uses post-training rather than one monolithic run: resolution adaptation, distillation, and text alignment extend the base model after self-supervised learning. That modularity connects it to [dino.txt](/paper%20shorts/2024/12/20/dinov2-meets-text-dino-txt.html), where a frozen DINOv2 backbone is aligned to language without relearning its visual geometry.
- DINOv3 scales the DINO/iBOT recipe and introduces Gram anchoring to prevent local patch structure from collapsing into global semantics.
- The 1.7-billion-image corpus and 6.7B model make full replication inaccessible, while anchor-checkpoint selection and web-data composition remain consequential.
- More self-supervised training can improve classification while silently damaging dense vision; DINOv3 makes preserving patch relations an explicit optimization target.
