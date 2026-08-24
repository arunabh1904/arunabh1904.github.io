---
title: 'V-JEPA 2.1: Dense Features in Video Self-Supervised Learning'
date: '2026-03-15T00:00:00.000Z'
section: paper-shorts
postSlug: v-jepa-2-1-dense-video-features
legacyPath: /paper shorts/2026/03/15/v-jepa-2-1-dense-video-features.html
tags: [Video Models, Dense Prediction]
field: 'Video & Interactive World Models'
summary: '2026 – V-JEPA 2.1: Dense Features in Video Self-Supervised Learning'
---

## 2026 – V-JEPA 2.1: Dense Features in Video Self-Supervised Learning

**arXiv:** [2603.14482](https://arxiv.org/abs/2603.14482)

## Summary

> V-JEPA 2.1 adds dense predictive loss, deep self-supervision, and joint image-video training to improve local features without giving up global scene understanding. The paper reports 7.71 mAP on Ego4D interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS, and a 20-point real-robot grasping gain over V-JEPA 2-AC.

## Core Insights

![Dense V-JEPA 2.1 patch features compared with V-JEPA 2 across video frames](/assets/images/v-jepa-2-1-paper-figure-1.png)
*Patch-feature projections remain spatially and temporally coherent in V-JEPA 2.1, while the earlier model is more globally semantic. source: [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)*

![Figure 10 from V-JEPA 2.1: Dense Features in Video Self-Supervised Learning](/assets/images/v-jepa-2-1-dense-video-features-source-figure-10.webp)
*Figure 10 Depth estimation comparison on NYU and KITTI datasets. While V-JEPA 2 captures the overall scene geometry, its predictions lack local consistency and precise boundary structure. In contrast, our V-JEPA 2.1 produces sharper, more coherent, and fine-grained depth maps. source: [V-JEPA 2.1: Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)*

![Figure 9 from V-JEPA 2.1: Dense Features in Video Self-Supervised Learning](/assets/images/v-jepa-2-1-dense-video-features-source-figure-9.webp)
*Figure 9 Planning Navigation Trajectories in Latent Space. V-JEPA 2.1 enables faster and more accurate navigation planning compared to ( Bar et al. 2025 ) . We show PCA visualizations of 8 denoising steps of the planned latent trajectory between a start frame and a goal frame. source: [V-JEPA 2.1: Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)*


The dense loss trains both visible and masked tokens instead of concentrating prediction on a global target. Deep self-supervision applies the objective at intermediate layers, where local structure has not yet been compressed into the final representation.

This directly tests a recurring limitation of global objectives. Strong recognition does not guarantee depth, correspondence, or contact-relevant features. V-JEPA 2.1 improves those outputs by measuring them during representation learning.

## High-Level Takeaways

- Dense predictive targets make local spatial evidence part of self-supervised video learning.
- Intermediate-layer supervision prevents all useful structure from depending on the final representation.
- Global and dense quality can coexist, but only when the objective pressures both.
