---
title: 'DriveMM: All-in-One Large Multimodal Model for Autonomous Driving'
date: '2024-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: drivemm-all-in-one-large-multimodal-model-for-autonomous-driving
legacyPath: /paper shorts/2024/12/01/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: DriveMM trained one multimodal transformer across perception, prediction, and planning with a driving curriculum.
---
## 2024 - DriveMM

**arXiv:** [2412.07689](https://arxiv.org/abs/2412.07689)

**Plain-language summary:** DriveMM is an academic generalist model for autonomous driving. It trains across multiple datasets and tasks, including perception, prediction, and planning, with a curriculum that moves from easier visual understanding toward harder planning behavior.

The system takes multi-view driving imagery and produces a unified token sequence that can be decoded into task-specific outputs.

**Why it mattered:** DriveMM pushed against the assumption that every driving subproblem needs a separate specialized network. The paper asks whether shared multimodal representations can support the full stack.

**Take-home message:** End-to-end driving models are becoming multitask foundation models. The hard question is not only performance, but whether shared training improves closed-loop reliability.
