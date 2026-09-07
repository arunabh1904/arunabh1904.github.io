---
title: 'SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning'
date: '2024-01-22T00:00:00.000Z'
section: paper-shorts
postSlug: spatialvlm-spatial-reasoning-capabilities
legacyPath: /paper shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html
tags: [Vision-Language Models, Spatial Reasoning]
field: 'Vision-Language Models'
summary: '2024 – SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning'
---

## 2024 – SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning

**arXiv:** [2401.12168](https://arxiv.org/abs/2401.12168)

## Summary

> SpatialVLM argues that a general VLM's weak metric reasoning is primarily a supervision gap. It turns ordinary scene images into spatial question–answer data by combining object captions, segmentation, monocular metric depth, and 3D geometry rules, then fine-tunes a PaLM 2-E model on the generated mixture. On a held-out human benchmark it reaches 75.2% on qualitative spatial predicates, emits a number for 99.0% of quantitative questions, and lands within half to twice the human estimate for 37.2%. The same pipeline also exposes the limit: errors follow the depth estimator and the noisy human reference range.

## Core Insights

### Turn pixels into referable metric objects

The data generator starts by filtering an internet image corpus for scene-level photographs. Region proposal, object-centric captioning, and semantic segmentation produce pixels plus descriptions. ZoeDepth then lifts those pixels into metric 3D points; the system canonicalizes camera coordinates using horizontal surfaces such as a floor or tabletop, so “left,” “above,” distance, and size can be computed in a common frame.

![Figure 2: SpatialVLM data synthesis pipeline](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-2.webp)
*Fig 1: This source Figure 2 follows scene filtering, object-centric context extraction, 2D-to-3D lifting, ambiguity removal, and template-based spatial Q&A synthesis; the supervision is generated from intermediate geometry rather than written as a caption alone. | source: [SpatialVLM, Figure 2](https://arxiv.org/abs/2401.12168)*

Reference resolution is a surprisingly important part of the method. Fixed detector labels such as “cake” can refer to several objects, so the authors use FlexCap to sample one-to-six-word descriptions and reject or augment ambiguous labels with a semantic post-processing algorithm. They synthesize 38 qualitative and quantitative question types, each involving at most two objects. That restriction makes the labels more reliable while keeping the resulting examples composable with ordinary VQA data.

### The synthetic labels buy direct spatial behavior

The paper evaluates on 331 qualitative and 215 quantitative human-labeled WebLI examples that were unseen during training. For binary spatial predicates, SpatialVLM reaches 75.2% accuracy, versus 71.3% for LLaVA-1.5 and 68.0% for GPT-4V accessed in November 2023. For quantitative questions, it outputs a number 99.0% of the time; 37.2% of answers fall in the interval [50%, 200%] of the human reference, compared with 33.9% for PaLM 2-E and 0.0% for GPT-4V in the same table.

The evaluation distinguishes a useful failure mode. A model may understand that a distance question asks for a quantity while still miscalibrating the quantity. SpatialVLM improves both format and rough scale because the training targets contain units and metric relations. The result is not a calibrated sensor: the reference answers are free-form human estimates, and the depth model is itself the source of much of the metric signal.

### Geometry determines the error budget

| Intervention or data condition | Reported result | Interpretation |
| --- | ---: | --- |
| Frozen ViT, [50%, 200%] range | 34.9% | Coarse scale is partly retained by the pretrained visual encoder |
| Unfrozen ViT, [50%, 200%] range | 37.2% | Updating the visual encoder helps metric transfer |
| Frozen ViT, [90%, 110%] range | 5.6% | Fine numerical agreement is difficult |
| Unfrozen ViT, [90%, 110%] range | 8.4% | Spatial fine detail improves, but remains limited |

Noise ablation is similarly instructive. On the robotic manipulation set, adding Gaussian label noise with standard deviation 0.2 yields MSE 0.039 m and 61.1% in the half-to-twice range, close to or better than the zero-noise condition's 0.046 m and 59.0%. The authors interpret this as evidence that the model learns a spatial common sense from many noisy examples, not as proof that noise is beneficial. They also find the system works best for medium-range scenes, roughly 1–10 meters, where the monocular depth estimator is more reliable.

![Figure 6: SpatialVLM as a robotics reward generator](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-6.webp)
*Fig 2: This source Figure 6 uses language-queried distances to color robot-object configurations; reward values rise as the gripper approaches the task object, showing how spatial estimates can become a dense robotics signal. | source: [SpatialVLM, Figure 6](https://arxiv.org/abs/2401.12168)*

The downstream robotics result is therefore a composition argument. SpatialVLM supplies a natural-language distance tool, while a controller or a larger language model supplies planning. The paper demonstrates monotonic reward behavior in the shown manipulation tasks; it does not establish reliable metric control across cameras, scenes, or depth distributions outside the training experts.

## High-Level Takeaways

- SpatialVLM's central contribution is a data engine that converts 2D images into referable 3D supervision at scale.
- The held-out benchmark separates answer formatting from numerical calibration: 99.0% of outputs contain a number, but only 37.2% land within half to twice the human estimate.
- Unfreezing the visual encoder improves fine distance estimation, supporting the claim that a frozen contrastive representation loses useful spatial detail.
- The method inherits depth, segmentation, caption ambiguity, and human-rounding errors; its strongest metric behavior is reported in medium-range and robotic settings with better geometric supervision.
