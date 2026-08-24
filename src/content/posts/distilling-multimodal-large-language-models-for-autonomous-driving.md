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

![Figure from DiMA: long-tail and zero-shot driving scenarios compared against prior planners](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-paper-figure.png)
*Source figure from the [DiMA paper](https://arxiv.org/abs/2501.09757). source: [DiMA paper](https://arxiv.org/abs/2501.09757)*

![Figure 2 from Distilling Multi-modal Large Language Models for Autonomous Driving](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-2.webp)
*Figure 2 Overview of DiMA. The input to the framework is a multi-view image sequence and a question text prompt. The vision-based end-to-end planner consists of a scene encoder and a planning transformer. The scene encoder learns structured latent representations in the form of b ird’s-eye-view, e go, a gent, and m ap ( ) token embeddings and acts as a trainable tokenizer for the multi-modal large language model (MLLM). The planning transformer is trained under standard planning constraints Jiang et al. 2023 ; Hu et al. source: [Distilling Multi-modal Large Language Models for Autonomous Driving](https://arxiv.org/abs/2501.09757)*

![Figure 1 from Distilling Multi-modal Large Language Models for Autonomous Driving](/assets/images/distilling-multimodal-large-language-models-for-autonomous-driving-source-figure-1.webp)
*Figure 1 Comparison of planning performance in long-tail scenarios from nuScenes: DiMA-VAD demonstrates greater robustness compared to VAD Jiang et al. 2023 in long-tail navigation scenarios such as overtaking a vehicle and performing a 3-point turn. DiMA-VAD also outperforms recent vision-planner PARA-Drive Weng et al. 2024 and LLM planner TOKEN Tian et al. 2024a . Notably, the 3-point turn is a zero-shot scenario that is only present in the validation set. source: [Distilling Multi-modal Large Language Models for Autonomous Driving](https://arxiv.org/abs/2501.09757)*


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
