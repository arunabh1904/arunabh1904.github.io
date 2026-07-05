---
title: 'DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models'
date: '2024-02-01T05:00:00.000Z'
section: paper-shorts
postSlug: drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models
legacyPath: /paper shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html
tags:
  - Other
field: Autonomous Driving
summary: DriveVLM combined VLM reasoning with hierarchical planning, then paired it with a traditional pipeline for real-time driving.
---
## 2024 - DriveVLM

**arXiv:** [2402.12289](https://arxiv.org/abs/2402.12289)

**Plain-language summary:** DriveVLM uses large vision-language models for scene description, scene analysis, and hierarchical planning in complex driving scenarios. The paper also proposes DriveVLM-Dual, which pairs VLM reasoning with a more traditional autonomous driving pipeline to compensate for spatial precision and latency limits.

That hybrid design is the interesting part. The VLM contributes semantic reasoning about rare or complex situations; the conventional stack keeps the control loop more grounded.

![Figure 1: DriveVLM and DriveVLM-Dual model pipelines from DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](/assets/images/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models-paper-figure.png)
_Figure 1: DriveVLM and DriveVLM-Dual model pipelines. From the [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models paper](https://arxiv.org/abs/2402.12289), via arXiv HTML._

**What to look at:**
- Hierarchical VLM reasoning is split into scene description, analysis, and planning.
- DriveVLM-Dual keeps a traditional stack around the VLM to recover spatial precision.
- Real-time and geometry limits are the main deployment pressure.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Architecture | VLM plus hierarchical planning | Uses language to decompose complex scenes. |
| Hybrid variant | DriveVLM-Dual | Combines VLM semantics with conventional driving modules. |
| Evidence | nuScenes, SUP-AD, vehicle deployment | Tests both public and production-style settings. |

**Why it mattered:** DriveVLM captures the field's tension clearly: VLMs are useful for understanding and explanation, but driving still needs precise geometry and real-time behavior.

**Take-home message:** The near-term role for VLMs in driving may be as semantic planners and critics, not as the only system between sensors and steering.
