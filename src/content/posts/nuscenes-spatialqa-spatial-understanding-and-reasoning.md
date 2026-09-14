---
title: "NuScenes-SpatialQA: Spatial Understanding and Reasoning for Driving"
date: "2025-04-04T00:00:00.000Z"
section: paper-shorts
postSlug: nuscenes-spatialqa-spatial-understanding-and-reasoning
legacyPath: /paper shorts/2025/04/04/nuscenes-spatialqa-spatial-understanding-and-reasoning.html
tags: ["Spatial Reasoning", "Autonomous Driving"]
field: "Autonomous Driving: VLMs & Evaluation"
summary: "2025 – NuScenes-SpatialQA: Spatial Understanding and Reasoning for Driving"
---

## 2025 – NuScenes-SpatialQA: Spatial Understanding and Reasoning for Driving

**Paper:** [arXiv:2504.03164](https://arxiv.org/abs/2504.03164) · [PDF](https://arxiv.org/pdf/2504.03164)

## Summary

> NuScenes-SpatialQA separates recognizing a plausible driving situation from measuring its geometry. Qwen2.5-VL-7B-Instruct scores 84.06 on situational reasoning but 19.41 on quantitative spatial understanding. The benchmark makes that gap visible with annotation-derived answers; it evaluates question answering, not the closed-loop driving competence of a trained policy.

## Core Insights

### Spatial answers come from annotations

The benchmark builds camera-specific 3D scene graphs from the nuScenes validation split: 150 scenes, roughly 6,000 keyframes, and six camera views. Nodes carry object position, dimensions, category, and a generated identifying caption. Edges encode distance, lateral and longitudinal offsets, and relative bearing. Templates turn these values into qualitative comparisons, numeric questions, direct reasoning, and situational reasoning.

This changes the source of geometric truth relative to [SpatialVLM](/paper%20shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html). SpatialVLM constructs training targets using estimated geometry from internet images; NuScenes-SpatialQA evaluates models against an existing 3D annotation system. A language model still generates object descriptions, but it does not supply the metric answers. The paper describes the corpus as over 3.3 million questions; its rounded totals vary between the statistics paragraph and comparison table.

Read the pipeline by following one object: its annotated 3D box provides position, its projected image crop provides an identifying description, and the graph supplies the spatial relation used in the answer.

![Annotation and caption pipelines feeding scene graphs and spatial questions; source Figure 2](/assets/images/nuscenes-spatialqa-source-figure-2.webp)
*Fig 1: The benchmark separates object descriptions from geometric answer generation. Captions identify the target, while annotated scene graphs determine the spatial quantities. | source: [Paper, Figure 2](https://arxiv.org/abs/2504.03164)*

[View full-size figure](/assets/images/nuscenes-spatialqa-source-figure-2.webp)

### Separate the metrics before comparing models

Closed-ended questions use answer accuracy. Numeric questions use both mean absolute error and a tolerance test accepting values between 75% and 125% of the target. A high situational score therefore cannot be interpreted as a corresponding percentage of accurate distance estimates.

| Model | Qualitative understanding | Quantitative tolerance accuracy | Direct reasoning | Situational reasoning |
| --- | ---: | ---: | ---: | ---: |
| LLaVA-v1.6-Mistral-7B | 53.30 | 35.48 | 48.51 | 73.50 |
| Qwen2.5-VL-7B-Instruct | 58.02 | 19.41 | 58.18 | 84.06 |
| SpatialRGPT | 59.79 | 14.59 | 45.45 | 80.77 |

These are the paper's Table 3 scores. SpatialRGPT leads this comparison on qualitative understanding while trailing on numeric tolerance accuracy. Specialized spatial training does not transfer uniformly across tasks. The LLaVA size study is similarly mixed: increasing model size does not monotonically improve quantitative estimation. These observations argue for a task-specific evaluation, without establishing an intrinsic parameter threshold for geometry.

The benchmark filters to fully visible objects and sufficiently large boxes to make descriptions reliable. That also removes some of the occlusion and small-object cases a driving system must handle. Questions share scenes, objects, and templates, so millions of QA pairs are not millions of independent traffic situations. Chain-of-thought prompting also reduces scores for several tested models; longer explanations do not repair uncertain visual measurements by themselves.

## High-Level Takeaways

- Budget geometry evaluation separately from driving-language evaluation: use units, coordinate frames, and tolerance curves alongside situational answers.
- Annotation-derived labels reduce dependence on a model judge, but caption ambiguity, object filtering, and template coverage remain part of the measurement contract.
- Use this benchmark to probe a proposed BEV interface, then test whether the policy uses the recovered quantities in closed loop. QA improvements alone do not establish that connection.
