---
title: "Can LVLMs Obtain a Driver's License?"
date: '2024-09-01T04:00:00.000Z'
section: paper-shorts
postSlug: can-lvlms-obtain-a-drivers-license-idkb
legacyPath: /paper shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html
tags:
  - Other
field: Autonomous Driving
summary: IDKB tested whether vision-language models know explicit driving rules, not just visual scene facts.
---
## 2024 - Can LVLMs Obtain a Driver's License?

**arXiv:** [2409.02914](https://arxiv.org/abs/2409.02914)

**Plain-language summary:** This paper introduces IDKB, an Interactive Driving Knowledge Base for evaluating whether LVLMs understand traffic rules and driving theory. It includes official handbook knowledge, exam-style questions, and applied road scenarios.

The result is a useful warning: a model may recognize cars and pedestrians but still fail rule-based reasoning that every licensed human driver is expected to know.

![Driving VLM loop schematic](/assets/images/driving-vlm-loop-schematic.svg)

**What to look at:**
- IDKB focuses on explicit driving rules and written-test knowledge.
- The benchmark includes handbooks, exams, and scenario QA.
- Fine-tuning gains show that general VLM pretraining does not guarantee domain rules.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 1M+ driving knowledge items | Covers rules, exams, and applied scenarios. |
| Evaluation | 15 LVLMs | Tests whether general models know driving theory. |
| Main failure | Missing specialized rule knowledge | Perception alone is not driving competence. |

**Why it mattered:** Driving competence is not only perception. It is perception plus rule knowledge plus judgment under context.

**Take-home message:** Autonomous driving VLMs need a written test as well as a road test.
