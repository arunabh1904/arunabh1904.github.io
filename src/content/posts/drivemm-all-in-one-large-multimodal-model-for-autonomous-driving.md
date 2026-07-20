---
title: 'DriveMM: All-in-One Large Multimodal Model for Autonomous Driving'
date: '2024-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: drivemm-all-in-one-large-multimodal-model-for-autonomous-driving
legacyPath: /paper shorts/2024/12/01/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – DriveMM: All-in-One Large Multimodal Model for Autonomous Driving"
---
## 2024 – DriveMM

**arXiv:** [2412.07689](https://arxiv.org/abs/2412.07689)

**Summary:** DriveMM is an academic generalist model for autonomous driving. It trains across multiple datasets and tasks, including perception, prediction, and planning, with a curriculum that moves from easier visual understanding toward harder planning behavior.

The system takes multi-view driving imagery and produces a unified token sequence that can be decoded into task-specific outputs.

## Paper Insights

DriveMM, presented as RoboTron-Drive in the paper, tries to unify autonomous-driving tasks in one large multimodal model. It handles multiple datasets and task types through a shared model and prompt formulation instead of training separate specialized networks for every perception, prediction, or planning task. The main evidence is broad performance across six datasets and 13 tasks, including zero-shot generalization to unseen datasets. The limitation is that all-in-one benchmark performance does not prove closed-loop driving safety. The paper is best read as a generalization study for driving LMMs, not as a complete autonomy stack.

![Figure 1: RoboTron-Drive achieves SOTA in both general capabilities and generalization ability from DriveMM: All-in-One Large Multimodal Model for Autonomous Driving](/assets/images/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving-paper-figure.jpg)
_Figure 1: RoboTron-Drive achieves SOTA in both general capabilities and generalization ability. From the [DriveMM: All-in-One Large Multimodal Model for Autonomous Driving paper](https://arxiv.org/abs/2412.07689), via arXiv HTML._

**What to look at:**
- One transformer is trained across perception, prediction, and planning tasks.
- Curriculum learning moves from visual comprehension toward harder driving outputs.
- Zero-shot transfer is the evidence to inspect.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Training | Driving curriculum | Stages task difficulty for a generalist model. |
| Inputs | Multi-view driving imagery | Matches surround-camera AV settings. |
| Evidence | Multiple public benchmarks | Tests whether one model can replace specialists. |

## Decision Lens

DriveMM informs whether perception, prediction, and planning should share one multimodal token-processing model and curriculum. Its atomic examples come from different driving tasks; task prompts and output formats route them through a shared backbone while stage-specific data teaches increasingly action-oriented behavior.

The all-in-one model promises transfer, but mixture proportions and gradient conflict determine whether low-resource tasks benefit or disappear. The missing study measures per-task gradient alignment and transfer while matching total parameters against specialized models and a shared-backbone multi-head baseline. At 10× tasks, sequence formats, sampling weights, and negative transfer dominate. The unified claim would fail if specialists achieved better worst-task and closed-loop performance at equal aggregate inference cost.

**Context:** DriveMM pushed against the assumption that every driving subproblem needs a separate specialized network. The paper asks whether shared multimodal representations can support the full stack.

**Takeaway:** End-to-end driving models are becoming multitask foundation models. The hard question is not only performance, but whether shared training improves closed-loop reliability.
