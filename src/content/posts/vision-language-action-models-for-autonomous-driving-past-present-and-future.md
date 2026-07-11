---
title: 'Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future'
date: '2025-12-18T05:00:00.000Z'
section: paper-shorts
postSlug: vision-language-action-models-for-autonomous-driving-past-present-and-future
legacyPath: /paper shorts/2025/12/18/vision-language-action-models-for-autonomous-driving-past-present-and-future.html
tags:
  - Other
field: Autonomous Driving
summary: This survey traces autonomous-driving VLA work from vision-action models to end-to-end and dual-system VLA paradigms, with attention to action generators and guidance styles.
---
## 2025 - Vision-Language-Action Models for Autonomous Driving

**arXiv:** [2512.16760](https://arxiv.org/abs/2512.16760)

**Awesome list:** [awesome-vla-for-ad](https://github.com/worldbench/awesome-vla-for-ad)

**Summary:** This survey frames driving VLA as the next step after vision-action models. Vision-action systems map perception to control, but they often lack structured reasoning and instruction following. VLA systems add language as a reasoning and guidance layer.

The survey is especially useful because it distinguishes end-to-end VLA from dual-system VLA, then breaks down how actions are generated and how language guidance is injected.

## Paper Insights

The paper reviews the path from modular perception-decision-action stacks to VA models, world models, and VLA systems. Its central taxonomy splits VLA systems into two paradigms: End-to-End VLA and Dual-System VLA. It also distinguishes textual versus numerical action generators and explicit versus implicit guidance.

That taxonomy complements the other VLA4AD survey. Where the earlier survey is a broad architecture-and-bibliography map, this one is helpful for reasoning about system boundaries: should language be inside the action generator, or should it guide a separate spatial planner?

![Figure 2 from the VLA-for-AD survey summarizing representative VA and VLA models across end-to-end models, world models, and dual systems](/assets/images/vision-language-action-models-for-autonomous-driving-past-present-and-future-paper-figure.png)
_Figure 2 summarizes representative VA and VLA models across end-to-end, world-model, and dual-system families. From the [survey paper](https://arxiv.org/abs/2512.16760), via arXiv HTML._

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

**Context:** This survey gives a cleaner vocabulary for comparing monolithic driving VLAs against hybrid systems such as DriveVLM-Dual-style designs.

**Takeaway:** "Driving VLA" is not one architecture; it is a set of choices about where language reasoning sits relative to spatial planning and action generation.
