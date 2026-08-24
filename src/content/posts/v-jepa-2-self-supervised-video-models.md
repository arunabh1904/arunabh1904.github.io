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

![V-JEPA 2 pipeline from action-free video pretraining to language understanding and robot planning](/assets/images/v-jepa-2-paper-figure-1.png)
*A shared video encoder supports recognition and language alignment. A separate action-conditioned predictor adds the intervention model needed for planning. source: [V-JEPA 2](https://arxiv.org/abs/2506.09985)*

![Figure 1 from V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](/assets/images/v-jepa-2-self-supervised-video-models-source-figure-1.webp)
*Figure 1 V-JEPA 2 Overview. Leveraging 1M hours of internet-scale video and 1M images, we pretrain the V-JEPA 2 video model using a visual mask denoising objective ( Bardes et al. 2024 ; Assran et al. 2023 ) , and leverage this model for downstream tasks such as action classification, object recognition, action anticipation, and Video Question Answering by aligning the model with an LLM backbone. source: [V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](https://arxiv.org/abs/2506.09985)*

![Figure 3 from V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](/assets/images/v-jepa-2-self-supervised-video-models-source-figure-3.webp)
*Figure 3 Scaling Ingredients. The effects of scaling interventions on average accuracy across 6 image and video classification tasks (SSv2, Diving-48, Jester, Kinetics, COIN, ImageNet) using a ViT-L/16 model as baseline. source: [V-JEPA 2: Self-Supervised Video Models for Understanding and Planning](https://arxiv.org/abs/2506.09985)*


V-JEPA 2 predicts representations rather than pixels or text. Action-free web video supplies broad dynamics. Robot trajectories then teach how candidate actions change those representations, and model-predictive control selects actions that approach an image goal.

The separation is useful. Video understanding does not automatically imply control. The action-conditioned stage is what turns passive temporal prediction into a model that can compare interventions.

## High-Level Takeaways

- V-JEPA 2 learns video structure without a language decoder.
- A small robot dataset can add action conditioning to a large passive-video representation.
- Planning depends on intervention-sensitive prediction, not temporal coherence alone.
