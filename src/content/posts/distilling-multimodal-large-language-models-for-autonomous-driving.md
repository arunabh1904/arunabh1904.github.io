---
title: 'Distilling Multi-modal Large Language Models for Autonomous Driving'
date: '2025-01-16T00:00:00.000Z'
section: paper-shorts
postSlug: distilling-multimodal-large-language-models-for-autonomous-driving
legacyPath: /paper shorts/2025/01/01/distilling-multimodal-large-language-models-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2025 – Distilling Multi-modal Large Language Models for Autonomous Driving"
---
## 2025 – Distilling Multi-modal Large Language Models for Autonomous Driving

**arXiv:** [2501.09757](https://arxiv.org/abs/2501.09757)

### Method and reported result

DIMA tackles planner latency by distilling a large multimodal LLM into a smaller vision-based student. The teacher may reason well but remains too slow and expensive for deployment, so the student learns both its planning behavior and intermediate reasoning signals.

## Summary

> The result is a model that keeps more of the teacher's traffic knowledge while avoiding a full LLM in the runtime loop.

## Core Insights

DiMA distills an expensive multimodal or language-enhanced driving planner into an efficient LLM-free planner. The teacher contributes world knowledge and long-tail reasoning; the student learns to reproduce useful planning behavior without calling an LLM at deployment time. The target setting is rare or difficult maneuvers where language reasoning can help, such as overtaking and three-point turns. The key benefit is latency and compute reduction. The caveat is that distillation freezes the teacher's behavior into the student, so the deployed model cannot ask new questions or recover reasoning traces at test time.


![Figure 2 from Distilling Multi-modal Large Language Models for Autonomous Driving](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-2.webp)
*Fig 1: DiMA feeds multi-view image sequences to a scene encoder that produces BEV, ego, agent, and map tokens; a multimodal language model supervises the planning transformer during training while deployment remains vision-only. | source: [Distilling Multi-modal Large Language Models for Autonomous Driving](https://arxiv.org/abs/2501.09757)*

![Figure 1 from Distilling Multi-modal Large Language Models for Autonomous Driving](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-1.webp)
*Fig 2: Long-tail nuScenes examples show DiMA-VAD navigating overtakes and three-point turns more successfully than VAD, PARA-Drive, and TOKEN; the three-point turn appears only in validation. | source: [Distilling Multi-modal Large Language Models for Autonomous Driving](https://arxiv.org/abs/2501.09757)*


**What to look at:**
- A large multimodal planner is used as an offline teacher.
- The runtime model is a smaller vision-based student.
- This is mainly about latency and deployability.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Teacher | Multimodal LLM planner | Provides richer traffic reasoning during training. |
| Student | Vision-only planner | Keeps inference cheaper and faster. |
| Reported signal | Lower trajectory error and collisions | Measures whether distilled reasoning survives compression. |

## High-Level Takeaways

- DIMA informs whether an expensive multimodal driving reasoner should run online or serve as a training-time teacher for a compact vision planner. The atomic supervision unit pairs a driving observation with teacher-produced reasoning or action targets; the student absorbs that signal but executes without the teacher.
- Distillation can preserve semantic structure at low latency, but gains may come from extra labels rather than teacher reasoning. The missing factorial ablation compares teacher actions, rationales, intermediate features, and equal-volume human annotations under one student architecture. At 10× teacher size, label-generation cost and systematic teacher errors dominate. The approach would fail if a student trained on simpler privileged labels matched closed-loop safety and progress without the multimodal teacher.
- Distillation is a plausible path from impressive VLM demos to deployable autonomy components. The expensive model teaches; the small model acts.
- LLMs may enter driving stacks indirectly, as offline teachers that shape compact planners.
