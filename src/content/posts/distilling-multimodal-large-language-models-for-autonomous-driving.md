---
title: 'Distilling Multi-modal Large Language Models for Autonomous Driving'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: distilling-multimodal-large-language-models-for-autonomous-driving
legacyPath: /paper shorts/2025/01/01/distilling-multimodal-large-language-models-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: DIMA used a large multimodal driving model as a teacher, distilling its behavior into a faster vision-based planner.
---
## 2025 - Distilling Multi-modal Large Language Models for Autonomous Driving

**arXiv:** [2501.09757](https://arxiv.org/abs/2501.09757)

**Plain-language summary:** This paper, referred to as DIMA in the generated report, tackles the latency problem directly. A large multimodal LLM planner may reason well, but it is too slow and expensive for deployment. The proposed approach trains a smaller vision-based student to imitate the larger teacher's planning behavior and intermediate reasoning signals.

The result is a model that keeps more of the teacher's traffic knowledge while avoiding a full LLM in the runtime loop.

![Figure from DiMA: long-tail and zero-shot driving scenarios compared against prior planners](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-paper-figure.png)
_Source figure from the [DiMA paper](https://arxiv.org/abs/2501.09757), via arXiv HTML._

**What to look at:**
- A large multimodal planner is used as an offline teacher.
- The runtime model is a smaller vision-based student.
- This is mainly about latency and deployability.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Teacher | Multimodal LLM planner | Provides richer traffic reasoning during training. |
| Student | Vision-only planner | Keeps inference cheaper and faster. |
| Reported signal | Lower trajectory error and collisions | Measures whether distilled reasoning survives compression. |

**Why it mattered:** Distillation is a plausible path from impressive VLM demos to deployable autonomy components. The expensive model teaches; the small model acts.

**Take-home message:** LLMs may enter driving stacks indirectly, as offline teachers that shape compact planners.
