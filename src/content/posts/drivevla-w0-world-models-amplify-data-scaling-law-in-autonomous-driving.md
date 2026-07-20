---
title: 'DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving'
date: '2025-10-14T04:00:00.000Z'
section: paper-shorts
postSlug: drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving
legacyPath: /paper shorts/2025/10/14/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving"
---
## 2025 – DriveVLA-W0

**arXiv:** [2510.12796](https://arxiv.org/abs/2510.12796)

**Code:** [BraveGroup/DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0)

**Summary:** DriveVLA-W0 focuses on a supervision problem. Driving action labels are sparse and low-dimensional, so a large VLA can underuse its capacity if training only asks it to predict future actions.

The paper adds world modeling. By predicting future images, the model receives dense self-supervised feedback about scene dynamics, not only a sparse trajectory or control target.

## Paper Insights

DriveVLA-W0 frames the issue as a "supervision deficit." It instantiates world modeling in two ways: an autoregressive model over discrete visual tokens and a diffusion model over continuous latent features. A lightweight action expert then supports faster inference. The paper evaluates on NAVSIM v1/v2 and a much larger in-house dataset, using the world model as the mechanism that makes data scaling more useful.

The conceptual contribution is broader than one architecture. Dense future-scene prediction can train representations of traffic dynamics, counterfactual actions, and scene evolution before the action head has to emit sparse driving outputs. The tradeoff is that image prediction is expensive and can optimize visual fidelity that is not always planner-critical.

![Figure 2 from DriveVLA-W0 showing autoregressive and diffusion world-model variants for future-scene supervision](/assets/images/drivevla-w0-world-models-amplify-data-scaling-law-in-autonomous-driving-paper-figure.png)
_Figure 2 shows the two world-modeling variants: autoregressive discrete visual-token prediction and diffusion over continuous latent features. From the [DriveVLA-W0 paper](https://arxiv.org/abs/2510.12796), via arXiv HTML._

**What to look at:**
- Sparse action labels are treated as an insufficient supervision signal.
- Future image prediction supplies dense scene-dynamics learning.
- A lightweight action expert separates representation learning from fast action inference.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Problem | Supervision deficit | Explains why bigger VLAs may not scale from sparse actions alone. |
| World model | AR visual-token and diffusion latent variants | Tests two dense prediction routes. |
| Action head | Lightweight action expert | Keeps inference practical after heavy representation learning. |
| Evaluation | NAVSIM v1/v2 and large in-house data | Tests whether world modeling improves data scaling. |

## Decision Lens

DriveVLA-W0 informs whether additional driving data should supervise only sparse actions or also dense future visual dynamics. The training unit couples current observations and actions with future images; a shared representation learns both control and world prediction so each logged frame supplies more than a low-dimensional trajectory label.

The reported scaling result suggests world-model supervision improves the return from more data in the tested NAVSIM and internal regimes. It does not isolate future-image prediction from extra decoder capacity or richer augmentation. A held-out scale sweep with equal parameters and target count is decisive. At 10× data, video redundancy and prediction of irrelevant appearance can consume the budget. The claim would fail if action-only training recovered the same scaling slope after matching auxiliary compute and regularization.

**Context:** DriveVLA-W0 argues that driving VLAs should learn dense world dynamics alongside sparse action imitation.

**Takeaway:** Dense world-model supervision can make scaling useful when action labels are too thin to train a large driving model by themselves.
