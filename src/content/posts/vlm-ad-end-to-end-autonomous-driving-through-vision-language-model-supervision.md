---
title: 'VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision'
date: '2024-12-19T00:00:00.000Z'
section: paper-shorts
postSlug: vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision
legacyPath: /paper shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision"
---

**arXiv:** [2412.14446](https://arxiv.org/abs/2412.14446)

**PMLR:** [CoRL 2025 proceedings](https://proceedings.mlr.press/v305/xu25f.html)

**OpenReview:** [JM2vDI6DlP](https://openreview.net/forum?id=JM2vDI6DlP)

## Summary

> VLM-AD uses GPT-4o as a training-time teacher for an ordinary end-to-end driving model. It projects the future ego trajectory onto a front-view image, asks for freeform reasoning and structured action labels, and trains auxiliary heads to align the student's features and behavior with those annotations. The VLM disappears at inference. On nuScenes, the full auxiliary recipe lowers UniAD's average L2 error from 1.03 to 0.88 m and collision rate from 0.31% to 0.19%; on CARLA Town05 Long it raises route completion from 75.20% to 84.14%.

## Core Insights

### The teacher sees time through a projected future path

The annotation problem is that a single image shows the present while planning depends on the near future. VLM-AD projects the ego vehicle's future trajectory onto the initial front-view image, then prompts GPT-4o with three freeform questions: what the vehicle is doing now, what it will do next, and why. A second prompt asks for structured control, turn, and lane labels. The authors tested a composite six-camera input and a front view; they found comparable annotation quality and chose the front view for lower complexity. They also report that feeding seven consecutive images often confuses temporal continuity, which is why the projected path is the central temporal cue.

![VLM-AD teacher and student training pipeline](/assets/images/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision-paper-figure.png)
*Fig 1: The VLM converts a front-view video frame, camera parameters, prompts, and projected future path into freeform and structured labels; an arbitrary end-to-end driver learns from those labels during training. | source: [VLM-AD, Figure 2](https://arxiv.org/abs/2412.14446)*

This is a useful distinction from using a VLM as a runtime planner. The teacher is an annotation instrument. Its errors can still enter the dataset, but its latency and token-generation cost do not enter the deployed driving loop.

### Two auxiliary heads distill different kinds of knowledge

The planning module exposes an ego feature to two plug-in heads. The text-alignment head uses three learnable queries and multi-head cross-attention to align the feature with CLIP-ViT-B/32 embeddings of the current-action, future-action, and reasoning annotations. The structured head uses three separate attention queries to predict one-hot control, turn, and lane labels. The total loss is the normal planning objective plus weighted alignment and action losses; the reported default weights are lambda_1=1 and lambda_2=0.1, with eight attention heads and three cross-attention layers per auxiliary head.

![VLM-AD framework with freeform text alignment and structured action classification](/assets/images/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision-source-figure-1.webp)
*Fig 2: The deployed end-to-end model keeps its original planning path while the auxiliary text-alignment and action-classification heads inject teacher supervision during training. | source: [VLM-AD, Figure 1](https://arxiv.org/abs/2412.14446)*

The two signals do different work. Freeform text supplies a high-dimensional semantic target, while structured labels constrain the behavior at a smaller, more stable vocabulary. In the UniAD ablation, Q1 freeform supervision alone reaches 0.89 m average L2 and 0.24% average collision rate; Q2 structured labels alone reach 0.91 m and 0.29%; combining them reaches 0.88 m and 0.19%. The result supports complementarity, but does not prove that the natural-language rationale itself is faithful to the student's causal computation.

### The gains survive model integration, with protocol caveats

The official UniAD baseline reports 1.03 m average L2 and 0.31% collision rate on nuScenes; full VLM-AD reaches 0.88 m and 0.19%. The method also improves VAD-Base from 0.72 to 0.48 m average L2 in the paper's reproduced integration, while collision rate changes from 0.54% to 0.23%. For SparseDrive-B, average L2 moves from 0.58 to 0.52 m and collision rate from 0.06% to 0.05%. These rows use the corresponding model's open-loop protocol, and the paper notes discrepancies between official and reproduced checkpoints, so they should not be read as one universal ranking.

The closed-loop test integrates VLM-AD into VAD-Base on CARLA Town05. On short routes, driving score rises from 64.29 to 67.78 and route completion from 87.26% to 88.56%. On long routes, the changes are larger: driving score 30.31 to 35.25 and route completion 75.20% to 84.14%. The deployed model still uses the VAD traffic-light branch and navigation inputs; the teacher is absent. That makes the result a useful test of training-time supervision, while leaving open how much the gain depends on GPT-4o's annotation style, the projected-path cue, and the VAD integration.

## High-Level Takeaways

- Use a large VLM offline when its rationale can enrich a compact driver without putting language-model latency in the control loop.
- Projecting future motion into the teacher's visual input is a practical temporal shortcut, but it also means the annotation process already contains privileged trajectory information.
- Freeform alignment and structured action labels complement each other; ablate them separately before claiming “reasoning distillation.”
- The next decisive study should vary teacher models, annotation errors, front-view versus multi-view inputs, and trajectory projection while keeping the student, data, and closed-loop budget fixed.
