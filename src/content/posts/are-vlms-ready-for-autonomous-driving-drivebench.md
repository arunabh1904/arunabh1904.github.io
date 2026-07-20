---
title: 'Are VLMs Ready for Autonomous Driving?'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: are-vlms-ready-for-autonomous-driving-drivebench
legacyPath: /paper shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: DriveBench tested whether VLM driving answers are visually grounded or merely plausible.
---
## 2025 - Are VLMs Ready for Autonomous Driving?

**arXiv:** [2501.04003](https://arxiv.org/abs/2501.04003)

**Summary:** DriveBench measures VLM reliability for autonomous driving through scene questions under clean-image, corrupted-image, and text-only conditions.

The important finding is uncomfortable: models can give confident and plausible answers without grounding them in the visual input. Corruptions expose this because the answer often should change when the evidence changes.

## Paper Insights

DriveBench asks whether VLMs remain reliable under driving-specific stress rather than stopping at clean-image question answering. It evaluates visual grounding, robustness, and task metrics across perception, prediction, planning, and text QA. The benchmark corrupts driving inputs with brightness shifts, fog, snow, rain, blur, zoom, compression, and bit errors. VLMs can look capable on normal scenes while failing under realistic sensor degradation or poor grounding. DriveBench therefore moves the evaluation target from capability demos to reliability.

![Figure 1 from DriveBench: benchmark overview across perception, prediction, behavior, and planning](/assets/images/drivebench-paper-figure-1-overview.png)
_Figure 1 from the [DriveBench paper](https://arxiv.org/abs/2501.04003), cropped from the arXiv PDF._

**What to look at:**
- DriveBench compares clean, corrupted, and text-only conditions.
- The key failure is plausible answers that ignore visual evidence.
- Treat grounding metrics as a necessary companion to QA accuracy.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 19,200 frames / 20,498 QA pairs | Tests multiple driving tasks and question types. |
| Stress test | 17 input conditions | Checks corruption and text-only reliance. |
| Main failure | Weak visual grounding | Models can answer from priors instead of perception. |

## Decision Lens

DriveBench informs whether a VLM's plausible driving language is grounded enough to support deployment claims. Its atomic evaluation item couples a driving frame with a question and answer, then stresses visual corruption and text-only shortcuts to separate perception from learned priors.

The benchmark shows that fluent answers can survive when visual evidence is removed or degraded, so aggregate QA accuracy is not a grounding metric. The missing validation is replication on new geographies, camera rigs, and freshly collected questions with contamination checks and human disagreement estimates. At 10× scale, annotation consistency and shortcut diversity dominate example count. DriveBench's conclusion would weaken if model rankings and failure modes did not reproduce on that independent set.

**Context:** Capability benchmarks can flatter VLMs. Driving needs reliability benchmarks that ask whether the model actually looked at the scene.

**Takeaway:** A VLM that sounds right is not necessarily grounded. For driving, grounding under degradation is the benchmark that matters.
