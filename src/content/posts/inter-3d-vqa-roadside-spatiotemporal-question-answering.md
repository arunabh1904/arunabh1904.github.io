---
title: "Inter-3D VQA: Roadside 3D Spatiotemporal Question Answering"
date: "2026-08-28T00:00:00.000Z"
section: paper-shorts
postSlug: inter-3d-vqa-roadside-spatiotemporal-question-answering
legacyPath: /paper shorts/2026/08/28/inter-3d-vqa-roadside-spatiotemporal-question-answering.html
tags: ["Spatial Reasoning", "Roadside Perception"]
field: "Autonomous Driving: VLMs & Evaluation"
summary: "2026 – Inter-3D VQA: Roadside 3D Spatiotemporal Question Answering"
---

## 2026 – Inter-3D VQA: Roadside 3D Spatiotemporal Question Answering

**Paper:** [arXiv:2608.28762](https://arxiv.org/abs/2608.28762) · [PDF](https://arxiv.org/pdf/2608.28762)

## Summary

> Inter-3D VQA evaluates roadside traffic understanding with 407,000 questions grounded in synchronized images, LiDAR, trajectories, and map structure. Its Inter-Geo baseline raises the grounded numerical score from 0.432 for Qwen3-VL-4B to 0.589 by adding object and scene LiDAR features. The gain costs throughput: the paper reports 1.3 FPS for Inter-Geo versus 2.6 for the image-only VLM.

## Core Insights

### Roadside geometry changes the reference frame

An intersection camera sees several approaches, lanes, and agents at once. “Left” in one camera is not a stable description across views, and a linguistically plausible answer can confuse a crossing with a turning lane. The benchmark combines synchronized multi-view imagery, global point clouds, object boxes and trajectories, and HD-map relations. Templates generate questions, language models expand descriptions, and sanity checks and balancing filter the results.

Grounded questions specify spatiotemporal object references; free-form questions identify objects through infrastructure-aware descriptions. The task families cover positions, relationships, motion, and interactions including near-miss-oriented reasoning. Unlike ego-view spatial QA, the reference system is roadside infrastructure and intersection topology.

Follow the metadata into the QA templates in the source figure. The map and trajectory annotations determine what can be asked reliably; language expansion does not create missing measurements.

![Roadside metadata, template questions, and quality-control pipeline; source Figure 3](/assets/images/inter-3d-vqa-source-figure-3.webp)
*Fig 1: Inter-3D VQA constructs questions from synchronized geometry and infrastructure annotations, then applies language expansion and checks. Both reference-frame consistency and annotation quality constrain the benchmark. | source: [Paper, Figure 3](https://arxiv.org/abs/2608.28762)*

[View full-size figure](/assets/images/inter-3d-vqa-source-figure-3.webp)

Inter-Metrics reports textual consistency, numerical accuracy, and semantic correctness separately. Numeric fields use scalar or Euclidean errors before the protocol converts them to a score. Therefore, 0.589 is a numerical benchmark score, not 58.9% collision-free driving and not a distance in meters. The shared scene-level split and five-run evaluation help comparisons, while model-specific adaptation settings still differ.

### Retrieve geometry inside the language decoder

Inter-Geo preserves Qwen3-VL-4B's camera–text path and adds LiDAR through decoder cross-attention. A LION-based backbone with a TransFusion detection head supplies object features and scene BEV features. A dual-query network compresses those branches, and gated adapters inject them into language hidden states. The appendix retains at most sixteen object representations and sixty-four scene representations, with adapters in the top four decoder layers.

The architecture keeps local object attributes and global scene structure separate until retrieval. The model can use one branch for a specific agent and the other for road layout.

![Object and scene LiDAR branches injected through gated decoder adapters; source Figure 5](/assets/images/inter-3d-vqa-source-figure-5.webp)
*Fig 2: Inter-Geo supplies both object-level and scene-level geometric context through decoder cross-attention. This preserves the original camera–text input path while adding a specialized geometry interface. | source: [Paper, Figure 5](https://arxiv.org/abs/2608.28762)*

[View full-size figure](/assets/images/inter-3d-vqa-source-figure-5.webp)

| Model | Grounded text score | Grounded numerical score | Grounded semantic score |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B | 0.827 | 0.432 | 0.736 |
| Qwen3-VL-8B | 0.851 | 0.488 | 0.802 |
| Inter-Geo, 4B backbone | 0.859 | 0.589 | 0.828 |

Table 3's removal of both LiDAR branches returns the 4B image-only scores. The object branch is particularly important for quantitative grounding, while decoder injection beats the reported input-level alternatives. This complements [BEVLM](/paper%20shorts/2026/03/06/bevlm-semantic-distillation-into-bev.html): both expose specialized spatial features to a language model, but Inter-Geo uses dual LiDAR representations and decoder adapters in a roadside setting.

Extra sensors and computation are part of that advantage. Calibration, synchronization, detector coverage, the sixteen-object cap, and the benchmark's annotation vocabulary limit what the interface can represent. The results evaluate questions and predicted quantities; no deployed traffic-control or autonomous-driving safety result follows directly.

## High-Level Takeaways

- Evaluate language consistency and physical quantities separately, especially when objects must be matched across views and infrastructure coordinates.
- Object and scene representations serve different geometric questions; retaining both can outperform simply scaling an image-only backbone.
- Compare gains with added sensing and full inference cost, and test whether crowded scenes exceed the representation's object capacity.
