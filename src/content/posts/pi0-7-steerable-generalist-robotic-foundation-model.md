---
title: 'π0.7: A Steerable Generalist Robotic Foundation Model'
date: '2026-04-16T00:00:00.000Z'
section: paper-shorts
postSlug: pi0-7-steerable-generalist-robotic-foundation-model
legacyPath: /paper shorts/2026/04/16/pi0-7-steerable-generalist-robotic-foundation-model.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2026 – π0.7: A Steerable Generalist Robotic Foundation Model'
---

## 2026 – π0.7: A Steerable Generalist Robotic Foundation Model

**arXiv:** [2604.15483](https://arxiv.org/abs/2604.15483)

## Summary

> π0.7 conditions a generalist VLA on more than the task instruction. Episode metadata, subgoal images, and strategy descriptions tell the policy how a task should be performed. The paper evaluates out-of-the-box language following, cross-embodiment transfer, dexterous tasks, and compositional generalization across several robot platforms.

## Core Insights


![Figure 2 from π0.7: A Steerable Generalist Robotic Foundation Model](/assets/images/pi0-7-steerable-generalist-robotic-foundation-model-source-figure-2.webp)
*Fig 1: Architecture overview. The model is a 5B-parameter VLA consisting of a 4B VLM backbone, a MEM -style video history encoder and a 860M parameter action expert. | source: [π0.7: A Steerable Generalist Robotic Foundation Model](https://arxiv.org/abs/2604.15483)*

![Figure 5 from π0.7: A Steerable Generalist Robotic Foundation Model](/assets/images/pi0-7-steerable-generalist-robotic-foundation-model-source-figure-5.webp)
*Fig 2: Illustration of selected evaluation tasks. We evaluate on a number of tasks, and two of the more longer-horizon ones are visualized here. | source: [π0.7: A Steerable Generalist Robotic Foundation Model](https://arxiv.org/abs/2604.15483)*


The central decision is context conditioning. A language command says what to do. Metadata can describe data quality or strategy. Subgoal images specify a desired intermediate state. This lets one policy reuse demonstrations, imperfect autonomous data, and non-robot sources without pretending they are equivalent.

The extra controls improve steerability but also expand the prompt contract. Performance depends on producing reliable metadata and subgoals at deployment. A high-level policy and world model therefore become part of the action system.

## High-Level Takeaways

- π0.7 separates task identity from strategy and desired intermediate states.
- Multimodal conditioning makes heterogeneous data more usable without erasing quality differences.
- The policy gains flexibility while depending on reliable high-level context generation.
