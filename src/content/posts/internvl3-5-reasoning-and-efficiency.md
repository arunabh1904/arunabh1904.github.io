---
title: 'InternVL3.5: Multimodal Reasoning and Efficiency'
date: '2025-08-25T00:00:00.000Z'
section: paper-shorts
postSlug: internvl3-5-reasoning-and-efficiency
legacyPath: /paper shorts/2025/08/25/internvl3-5-reasoning-and-efficiency.html
tags: [Vision-Language Models, Reinforcement Learning]
field: 'Vision-Language Models'
summary: '2025 – InternVL3.5: Multimodal Reasoning and Efficiency'
---

## 2025 – InternVL3.5: Multimodal Reasoning and Efficiency

**arXiv:** [2508.18265](https://arxiv.org/abs/2508.18265)

## Summary

> InternVL3.5 combines two-stage reinforcement learning with dynamic visual resolution and decoupled vision-language serving. Offline reinforcement learning supplies stable initialization, then online reinforcement learning refines reasoning. The paper reports up to a 16 percent overall reasoning gain and a 4.05 times inference speedup over InternVL3.

## Core Insights

![InternVL3.5 comparison across multimodal, reasoning, text, and agentic benchmarks](/assets/images/internvl3-5-paper-figure-1.png)
_The reported aggregate spans several capability families, so the endpoint combines model, post-training, and serving changes. Source: [InternVL3.5](https://arxiv.org/abs/2508.18265), Figure 1._

The visual resolution router allocates different token budgets to different inputs. Decoupled deployment places the vision encoder and language model on separate devices to balance their workloads. These are serving decisions as much as model decisions.

Cascade reinforcement learning changes behavior after pretraining, while routing and deployment change cost. Because all three move together, headline gains should not be read as evidence for one isolated technique.

## High-Level Takeaways

- InternVL3.5 stages offline and online reinforcement learning rather than treating post-training as one pass.
- Visual routing makes input complexity a compute-allocation decision.
- The combined recipe improves quality and speed, but individual contributions require matched ablations.
