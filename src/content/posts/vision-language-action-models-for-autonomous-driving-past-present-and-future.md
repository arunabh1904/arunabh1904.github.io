---
title: 'Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future'
date: '2025-12-18T00:00:00.000Z'
section: paper-shorts
postSlug: vision-language-action-models-for-autonomous-driving-past-present-and-future
legacyPath: /paper shorts/2025/12/18/vision-language-action-models-for-autonomous-driving-past-present-and-future.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future"
---
## 2025 – Vision-Language-Action Models for Autonomous Driving

**arXiv:** [2512.16760](https://arxiv.org/abs/2512.16760)

**Awesome list:** [awesome-vla-for-ad](https://github.com/worldbench/awesome-vla-for-ad)

### Method and reported result

Vision-Language-Action Models for Autonomous Driving frames driving VLA as the successor to vision-action systems. Vision-action models map perception to control but often lack structured reasoning and instruction following; VLA systems add language as a reasoning and guidance layer.

## Summary

> The survey is especially useful because it distinguishes end-to-end VLA from dual-system VLA, then breaks down how actions are generated and how language guidance is injected.

## Core Insights

The paper reviews the path from modular perception-decision-action stacks to VA models, world models, and VLA systems. Its central taxonomy splits VLA systems into two paradigms: End-to-End VLA and Dual-System VLA. It also distinguishes textual versus numerical action generators and explicit versus implicit guidance.

That taxonomy complements the other VLA4AD survey. Where the earlier survey is a broad architecture-and-bibliography map, this one is helpful for reasoning about system boundaries: should language be inside the action generator, or should it guide a separate spatial planner?

![Figure 2 from the VLA-for-AD survey summarizing representative VA and VLA models across end-to-end models, world models, and dual systems](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-paper-figure.png)
*Fig 1: Summarizes representative VA and VLA models across end-to-end, world-model, and dual-system families. | source: [survey paper](https://arxiv.org/abs/2512.16760)*

![Figure 7 from Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-source-figure-7.webp)
*Fig 2: Visualization examples of the AutoVLA reasoning/planning results on WOD-E2E dataset. | source: [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](https://arxiv.org/abs/2512.16760)*

![Figure 1 from Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-source-figure-1.webp)
*Fig 3: Outline. This work aims to provide a structured roadmap of the VLA paradigm for autonomous driving. | source: [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](https://arxiv.org/abs/2512.16760)*


**What to look at:**
- End-to-end VLA and dual-system VLA are treated as distinct design paradigms.
- Action generation can be textual, numerical, or mediated by a planner.
- Guidance can be explicit instructions or implicit representation shaping.

**Taxonomy slice:**

| Axis | Options | Why it matters |
| ---- | ------- | -------------- |
| System boundary | End-to-end VLA or dual-system VLA | Decides whether language and planning are one model or cooperating modules. |
| Action generator | Textual action, numerical trajectory, or control signal | Determines how directly the model can drive. |
| Guidance style | Explicit or implicit | Separates prompt-like supervision from representation-level conditioning. |
| Historical line | VA, world model, VLA | Connects new VLA papers to older driving policy and dynamics work. |

## High-Level Takeaways

- This survey informs how to compare vision-action, end-to-end VLA, and dual-system driving models without collapsing them into one label. Its key units are system boundaries: where language enters, how action is represented, whether a world model is present, and whether evaluation is open or closed loop.
- The historical taxonomy is useful if it predicts engineering tradeoffs rather than merely ordering papers. A common benchmark that fixes sensors, data, action horizon, and latency across the three paradigms is the missing test. As the field scales, proprietary data and inconsistent closed-loop protocols make architectural claims hard to compare. The taxonomy would fail if data scale and evaluator choice explained outcomes better than the proposed model evolution.
- This survey gives a cleaner vocabulary for comparing monolithic driving VLAs against hybrid systems such as DriveVLM-Dual-style designs.
- "Driving VLA" is not one architecture; it is a set of choices about where language reasoning sits relative to spatial planning and action generation.
