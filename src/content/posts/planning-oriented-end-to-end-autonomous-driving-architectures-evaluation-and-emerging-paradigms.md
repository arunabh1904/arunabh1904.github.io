---
title: 'Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms'
date: '2026-08-20T09:00:00.000Z'
section: paper-shorts
postSlug: planning-oriented-end-to-end-autonomous-driving-architectures-evaluation-and-emerging-paradigms
legacyPath: /paper shorts/2026/08/20/planning-oriented-end-to-end-autonomous-driving-architectures-evaluation-and-emerging-paradigms.html
tags:
  - Autonomous Driving
  - End-to-End Driving
  - Planning
  - Evaluation
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – a planning-oriented taxonomy that treats evaluation protocol as part of the method claim'
---

## 2026 – Planning-Oriented End-to-End Autonomous Driving

**arXiv:** [2608.20111](https://arxiv.org/abs/2608.20111)

## Summary

> This survey argues that end-to-end driving should be classified by what supports the final plan, not by how much visible structure the network removes. It organizes methods along four axes—input representation, planning output, supervision, and evaluation—and follows the field from direct control imitation through structured BEV and vector planners to world models and vision-language-action systems. Its strongest decision rule is that an open-loop displacement score, a NAVSIM non-reactive score, and a Bench2Drive closed-loop score are different claims. The review maps those claims through June 2026; it does not provide a new cross-benchmark experiment or a universal model ranking.

## Core Insights

### End-to-end does not mean structure-free

Planning-oriented systems remain end to end when their intermediate objects are learned or jointly optimized toward driving behavior. A model may carry BEV features, object or map queries, route encodings, world-model latents, language tokens, or explicit safety constraints. The architectural question is whether those objects preserve the geometry, uncertainty, semantics, and action interface the planner needs—not whether the diagram contains named modules.

The survey's four-axis taxonomy prevents a backbone name from standing in for a driving formulation:

| Axis | Representative choices | Decision exposed |
| --- | --- | --- |
| Input representation | Raw sensors, BEV, vectors, object queries, world-state latents, VLM tokens | Which evidence and structure reach the planner? |
| Planning output | Controls, waypoints, trajectories, distributions, action tokens | Can the output express alternatives and accept safety checks? |
| Supervision | Behavior cloning, privileged distillation, auxiliary tasks, RL, world-model prediction, language alignment | Which target teaches recovery, consequence, and route compliance? |
| Evaluation | Open-loop replay, non-reactive real-log simulation, reactive closed loop, long-tail or preference scoring | Which behavior does the reported number actually test? |

The progression is therefore not a clean replacement sequence. Direct behavior cloning minimizes interfaces but suffers from covariate shift. Privileged teachers add structure during training. BEV and vectorized planners expose geometry and multi-task supervision. World models evaluate possible consequences, but only if their imagined futures remain calibrated under action changes. VLA systems add semantic context, yet must bridge text-space reasoning to metric trajectories under a bounded latency budget.

### Evaluation determines the scope of the architecture claim

Open-loop evaluation scores a policy on states visited by the logged expert. It cannot test recovery after the policy causes a deviation, and it may penalize a safe alternative simply because it differs from the recorded path. NAVSIM-style non-reactive evaluation adds real sensor logs and planning-aware safety, progress, and comfort proxies at scale, but other agents do not react to the ego plan. Bench2Drive-style simulation tests interaction and recovery under the policy's induced state distribution, while inheriting CARLA's sim-to-real boundary. WOD-E2E adds rare scenarios and human preferences but remains open loop.

The practical rule is to compare claims within protocols and triangulate across them. A low nuScenes L2 error does not imply a high closed-loop route score. A high NAVSIM score is stronger proxy evidence than displacement alone, but ranking inversions and saturated submetrics prevent it from replacing reactive evaluation. Small leaderboard differences are also uninterpretable without the benchmark version, controller, safety wrapper, sensor configuration, seeds, and metric implementation.

The same standard applies to language and world models. [Inference-time attention steering](/paper%20shorts/2026/08/17/inference-time-attention-steering-for-vision-language-action-driving-models.html) can move a trajectory without establishing safer behavior, while [XCoT-VLA](/paper%20shorts/2026/08/11/xcot-vla-executable-chain-of-thought-for-vision-language-action-driving.html) trains a compact reasoning interface whose gains are still tied to its evaluation setting. Better explanations, plausible futures, and lower displacement error are useful diagnostics; none is a substitute for action-grounded and closed-loop evidence.

### The review is a claim map, not a meta-analysis

The authors use a structured narrative search across scholarly databases, benchmark repositories, leaderboards, code releases, and project pages, with work covered through June 2026. Inclusion criteria span influential driving formulations, planning outputs, benchmarks, world models, VLM/VLA mechanisms, and reproducibility or safety critiques. The resulting scope is broad, but public academic work is overrepresented, recent 2025–2026 methods lack independent reproduction, and metric versions make many numbers protocol-dependent. The paper correctly recommends comparing methods by claim and benchmark family instead of merging their scores.

## High-Level Takeaways

- A structured intermediate representation is compatible with end-to-end learning when the representation and policy are optimized around the final plan.
- Output space, supervision, and evaluation can matter more than the encoder family: a trajectory distribution, a privileged teacher, or a reactive benchmark changes the scientific claim even when the visual backbone stays fixed.
- Open-loop, non-reactive real-log, reactive closed-loop, and preference-aware evaluations test different failure surfaces and should never share one undifferentiated leaderboard.
- World models need calibrated action-conditioned futures; VLA systems need evidence that language improves action quality rather than only explanation quality.
- The survey's thesis would weaken if benchmark-aligned, compute-matched studies found that open-loop rankings reliably predict closed-loop safety and route completion across datasets, controllers, and distribution shifts.
