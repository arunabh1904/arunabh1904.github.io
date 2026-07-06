---
title: 'VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision'
date: '2024-12-19T05:00:00.000Z'
section: paper-shorts
postSlug: vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision
legacyPath: /paper shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html
tags:
  - Other
field: Vision-Language Models
summary: VLM-AD uses a VLM as a training-time teacher, distilling driving reasoning and structured action labels into an end-to-end driving model without using the VLM at inference time.
---
## 2024 - VLM-AD

**arXiv:** [2412.14446](https://arxiv.org/abs/2412.14446)

**PMLR:** [CoRL 2025 proceedings](https://proceedings.mlr.press/v305/xu25f.html)

**OpenReview:** [JM2vDI6DlP](https://openreview.net/forum?id=JM2vDI6DlP)

**Plain-language summary:** VLM-AD uses a vision-language model as a teacher for end-to-end autonomous driving. The VLM watches driving scenes during dataset construction, produces freeform reasoning and structured action annotations, and those annotations become auxiliary supervision for a smaller driving model.

The important deployment detail is that the VLM is not used at inference time. The runtime model stays an end-to-end driver, but its training objective is shaped by language-style reasoning signals that explain why a maneuver makes sense.

## Paper Insights

The paper studies a missing piece in imitation-heavy end-to-end driving: models can copy trajectories without learning the reasoning behind traffic behavior. VLM-AD addresses that by prompting a VLM to generate two kinds of supervision. The first is freeform reasoning about the scene and intended behavior. The second is structured action information, converted into labels for control, turn, and lane decisions.

During training, an arbitrary end-to-end driving model receives its normal planning loss plus auxiliary text-alignment and action-classification heads. At inference, those heads and the VLM teacher are not needed for the driving loop. The evidence comes from nuScenes open-loop planning and CARLA closed-loop evaluation, where VLM-AD improves planning error, collision rate, route completion, and driving score when added to strong baselines. The main caveat is teacher quality: if the VLM annotations are wrong or biased, the student can inherit that reasoning.

![Figure 2 from VLM-AD showing VLM-generated reasoning and action labels supervising an end-to-end driving model](/assets/images/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision-paper-figure.png)
_Figure 2 shows the training-time teacher setup: the VLM produces freeform reasoning and structured action labels, while the deployed driver remains VLM-free. From the [VLM-AD paper](https://arxiv.org/abs/2412.14446), via arXiv HTML._

**What to look at:**
- VLM reasoning is used as supervision, not as a runtime planner.
- Freeform text supervision and structured action labels teach different parts of the driving representation.
- The method can be attached to existing end-to-end driving models.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Teacher | VLM-generated reasoning and action annotations | Adds traffic rationale beyond trajectory imitation. |
| Student | Arbitrary E2E driving model | Keeps inference practical and model-agnostic. |
| Deployment | No VLM at inference | Avoids putting a slow language model in the control loop. |

**Compact result slice:**

| Setting | Baseline | +VLM-AD | What changed |
| ------- | -------- | ------- | ------------ |
| UniAD on nuScenes | 1.03 avg L2, 0.31 avg collision | 0.88 avg L2, 0.19 avg collision | Lower planning error and fewer collisions. |
| VAD-Base on CARLA Town05 Short | 64.29 DS, 87.26 RC | 67.78 DS, 88.56 RC | Better closed-loop score and route completion. |
| VAD-Base on CARLA Town05 Long | 30.31 DS, 75.20 RC | 35.25 DS, 84.14 RC | Larger gain on the longer interactive route. |

**Why it mattered:** VLM-AD is a clean example of the "language model as teacher" pattern for physical systems. The VLM improves training signal, but the deployed system still respects latency and control constraints.

**Take-home message:** For autonomous driving, VLMs may be most useful offline: generating rationales and labels that teach compact planners what the trajectory alone does not explain.
