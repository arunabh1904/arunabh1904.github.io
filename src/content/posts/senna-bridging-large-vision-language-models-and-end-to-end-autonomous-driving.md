---
title: 'SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving'
date: '2024-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving
legacyPath: /paper shorts/2024/10/01/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: SENNA split driving into high-level language planning and low-level trajectory control.
---
## 2024 - SENNA

**arXiv:** [2410.22313](https://arxiv.org/abs/2410.22313)

**Plain-language summary:** SENNA uses a hybrid architecture: a vision-language model produces a high-level textual plan, while an end-to-end driving module converts that plan and the sensor input into a precise trajectory. The design avoids asking the LVLM to output exact steering-level control directly.

This makes the language layer inspectable. A planner can say what it intends to do before the control module turns that intent into geometry.

![Driving VLM loop schematic](/assets/images/driving-vlm-loop-schematic.svg)

**What to look at:**
- Senna-VLM produces high-level textual plans.
- Senna-E2E turns those plans into precise trajectories.
- The interface is inspectable and potentially editable.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Decomposition | Language plan plus control module | Separates semantic intent from numeric control. |
| Training | Planning-oriented QA and curriculum | Tunes the VLM for traffic decisions. |
| Caveat | Language plan is not a guarantee | Control still needs safety validation. |

**Why it mattered:** SENNA captures a useful decomposition for safety-critical systems: use language for semantic planning, but keep numeric control in a component designed for precision.

**Take-home message:** The most useful VLM in a driving stack may be the one that thinks out loud at the right abstraction level.
