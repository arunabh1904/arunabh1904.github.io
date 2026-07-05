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

**Why it mattered:** Capability benchmarks can flatter VLMs. Driving needs reliability benchmarks that ask whether the model actually looked at the scene.

**Take-home message:** A VLM that sounds right is not necessarily grounded. For driving, grounding under degradation is the benchmark that matters.
