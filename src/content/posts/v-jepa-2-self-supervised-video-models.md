---
title: 'V-JEPA 2: Self-Supervised Video Models for Understanding and Planning'
date: '2025-06-11T00:00:00.000Z'
section: paper-shorts
postSlug: v-jepa-2-self-supervised-video-models
legacyPath: /paper shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html
tags: [Video Models, World Models]
field: 'Video & Interactive World Models'
summary: '2025 – V-JEPA 2: Self-Supervised Video Models for Understanding and Planning'
---

## 2025 – V-JEPA 2: Self-Supervised Video Models for Understanding and Planning

**arXiv:** [2506.09985](https://arxiv.org/abs/2506.09985)

## Summary

> V-JEPA 2 learns predictive visual representations from more than one million hours of internet video, then adapts them to language tasks and robot planning. The paper reports 77.3 top-1 on Something-Something v2 and 39.7 recall-at-5 on EPIC-KITCHENS-100. An action-conditioned predictor trained on fewer than 62 hours of DROID video supports zero-shot image-goal planning on Franka arms.

## Core Insights


![Figure 1 from V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](/assets/images/v-jepa-2-self-supervised-video-models-source-figure-1.webp)
*Fig 1: V-JEPA 2 pretrains on internet video and images, then branches into language alignment for video QA, attentive probes for recognition and anticipation, and action-conditioned post-training for robot planning. | source: [V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](https://arxiv.org/abs/2506.09985)*

![Figure 3 from V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](/assets/images/v-jepa-2-self-supervised-video-models-source-figure-3.webp)
*Fig 2: Scaling Ingredients. The effects of scaling interventions on average accuracy across 6 image and video classification tasks (SSv2, Diving-48, Jester, Kinetics, COIN, ImageNet) using a ViT-L/16 model as baseline. | source: [V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](https://arxiv.org/abs/2506.09985)*


V-JEPA 2 predicts representations rather than pixels or text. Action-free web video supplies broad dynamics. Robot trajectories then teach how candidate actions change those representations, and model-predictive control selects actions that approach an image goal.

The separation is useful. Video understanding does not automatically imply control. The action-conditioned stage is what turns passive temporal prediction into a model that can compare interventions.

## High-Level Takeaways

- V-JEPA 2 learns video structure without a language decoder.
- A small robot dataset can add action conditioning to a large passive-video representation.
- Planning depends on intervention-sensitive prediction, not temporal coherence alone.
