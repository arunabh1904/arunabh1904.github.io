---
title: 'Are VLMs Ready for Autonomous Driving?'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: are-vlms-ready-for-autonomous-driving-drivebench
legacyPath: /paper shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html
tags:
  - Other
field: Autonomous Driving
summary: DriveBench tested whether VLM driving answers are visually grounded or merely plausible.
---
## 2025 - Are VLMs Ready for Autonomous Driving?

**arXiv:** [2501.04003](https://arxiv.org/abs/2501.04003)

**Plain-language summary:** This paper introduces DriveBench, an empirical study of VLM reliability for autonomous driving. It evaluates models on driving-scene question answering under clean images, corrupted images, and text-only conditions.

The important finding is uncomfortable: models can give confident and plausible answers without grounding them in the visual input. Corruptions expose this because the answer often should change when the evidence changes.

## Paper Insights

DriveBench asks whether VLMs are reliable enough for autonomous driving, not just whether they answer clean image questions. It evaluates visual grounding, robustness, and task metrics across perception, prediction, planning, and text QA. The benchmark corrupts driving inputs with conditions such as brightness shifts, fog, snow, rain, blur, zoom, compression, and bit errors. The central result is that VLMs can look capable on normal scenes while failing under realistic sensor degradation or poor grounding. The paper is useful because it changes the evaluation target from capability demos to reliability under driving-specific stress.

![Figure 1 from DriveBench: benchmark overview across perception, prediction, behavior, and planning](/assets/images/drivebench-paper-figure-1-overview.png)
_Figure 1 from the [DriveBench paper](https://arxiv.org/abs/2501.04003), cropped from the arXiv PDF._

**What to look at:**
- DriveBench compares clean, corrupted, and text-only conditions.
- The key failure is plausible answers that ignore visual evidence.
- Look for grounding metrics, not just QA accuracy.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 19,200 frames / 20,498 QA pairs | Tests multiple driving tasks and question types. |
| Stress test | 17 input conditions | Checks corruption and text-only reliance. |
| Main failure | Weak visual grounding | Models can answer from priors instead of perception. |

**Why it mattered:** Capability benchmarks can flatter VLMs. Driving needs reliability benchmarks that ask whether the model actually looked at the scene.

**Take-home message:** A VLM that sounds right is not necessarily grounded. For driving, grounding under degradation is the benchmark that matters.
