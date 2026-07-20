---
title: 'DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models'
date: '2024-02-01T05:00:00.000Z'
section: paper-shorts
postSlug: drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models
legacyPath: /paper shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models"
---
## 2024 – DriveVLM

**arXiv:** [2402.12289](https://arxiv.org/abs/2402.12289)

**Summary:** DriveVLM uses large vision-language models for scene description, scene analysis, and hierarchical planning in complex driving scenarios. The paper also proposes DriveVLM-Dual, which pairs VLM reasoning with a more traditional autonomous driving pipeline to compensate for spatial precision and latency limits.

That hybrid design is the interesting part. The VLM contributes semantic reasoning about rare or complex situations; the conventional stack keeps the control loop more grounded.

## Paper Insights

DriveVLM explores how VLM reasoning can support autonomous driving. It uses language-level scene understanding and hierarchical planning, with a dual variant that combines VLM semantics with conventional driving modules. The evaluation covers public driving datasets and deployment-oriented tests, aiming to show that VLMs can help decompose complex scenes and decisions. The design question is where the VLM belongs: it can reason and explain, but low-level control still needs geometry, timing, and safety constraints. The caveat is standard for driving VLMs: semantic competence is not the same as robust closed-loop planning.

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

## Decision Lens

DriveVLM informs which decisions benefit from explicit language reasoning and which must remain in a fast geometric pipeline. The model decomposes planning into scene description, object analysis, and hierarchical trajectory reasoning, then fuses that output with a conventional planner for real-time control.

The hybrid design acknowledges that semantic breadth and control latency have different requirements, but it obscures whether the VLM adds causal driving information or merely an ensemble prior. The missing closed-loop ablation removes each reasoning stage and the traditional branch at matched latency. At 10× traffic density, token generation and stale object narratives dominate. The VLM branch would fail its purpose if the conventional planner alone matched safety and progress on the scenarios where language reasoning is supposed to help.

**Context:** DriveVLM captures the field's tension clearly: VLMs are useful for understanding and explanation, but driving still needs precise geometry and real-time behavior.

**Takeaway:** The near-term role for VLMs in driving may be as semantic planners and critics, not as the only system between sensors and steering.
