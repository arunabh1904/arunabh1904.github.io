---
title: 'ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation'
date: '2025-03-25T15:18:43.000Z'
section: paper-shorts
postSlug: orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation
legacyPath: /paper shorts/2025/03/25/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation"
---
## 2025 – ORION

**arXiv:** [2503.19755](https://arxiv.org/abs/2503.19755)

## Summary

> ORION couples long-horizon scene history, language-model reasoning, and a generative trajectory planner in one end-to-end framework. QT-Former aggregates the history, an LLM produces driving reasoning, and the planner turns the combined state into precise trajectories. On Bench2Drive, the paper reports a 77.74 driving score and 54.62% success rate—improvements of 14.28 score points and 19.61 percentage points over its stated prior best. The abstract does not report latency, number of seeds, or a reasoning-ablation matched to the planner.

## Core Insights

ORION is organized around a specific interface problem: semantic reasoning and numeric trajectories inhabit different spaces. The model aligns them while jointly optimizing VQA and planning, so the reasoning path can condition action generation instead of remaining a post-hoc explanation. QT-Former supplies longer temporal context before that reasoning step, which is important for interactive driving decisions that a single frame cannot determine.

![ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation source figure: The comparison of different E2E paradigms.](/assets/images/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation-paper-figure.webp)
_The comparison of different E2E paradigms. Source: [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755), Figure 1, via arXiv HTML._


The reported closed-loop improvement is stronger evidence than an open-loop trajectory score, but it does not by itself isolate why the system wins. The abstract omits the training corpus, loss balance between VQA and planning, action representation, and a test that substitutes uninformative or counterfactual reasoning while preserving the visual input. Without those controls, semantic reasoning, history aggregation, and the generative planner remain coupled interventions.

## High-Level Takeaways

- ORION joins a history encoder, an LLM reasoning path, and a generative planner to make semantic context an input to trajectory generation.
- Its reported Bench2Drive result is meaningful closed-loop evidence, though the abstract does not separate the contributions of reasoning, temporal context, and planning architecture.
- The decisive falsification replaces the reasoning trace with matched-length irrelevant or counterfactual traces; the reasoning-action claim weakens if safety and success do not change in the predicted direction.
