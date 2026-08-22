---
title: 'Visual-RFT: Visual Reinforcement Fine-Tuning'
date: '2025-03-03T00:00:00.000Z'
section: paper-shorts
postSlug: visual-rft-visual-reinforcement-fine-tuning
legacyPath: /paper shorts/2025/03/03/visual-rft-visual-reinforcement-fine-tuning.html
tags: [Vision-Language Models, Reinforcement Learning]
field: 'Alignment & Post-Training'
summary: '2025 – Visual-RFT: Visual Reinforcement Fine-Tuning'
---

## 2025 – Visual-RFT: Visual Reinforcement Fine-Tuning

**arXiv:** [2503.01785](https://arxiv.org/abs/2503.01785)

## Summary

> Visual-RFT applies reinforcement learning with verifiable rewards to visual perception tasks. The reward changes with the output contract, including intersection-over-union for detection. With roughly 100 examples, the paper reports a 24.3 percent improvement over the baseline in one-shot fine-grained classification and gains of 21.9 on two-shot COCO detection and 15.4 on LVIS.

## Core Insights

![Visual-RFT comparison with supervised fine-tuning across classification, detection, and grounding](/assets/images/visual-rft-paper-figure-1.png)
_Verifiable visual rewards let one post-training framework optimize several perception tasks without relying only on answer text. Source: [Visual-RFT](https://arxiv.org/abs/2503.01785), Figure 1._

The method samples reasoning and answers, scores outputs with task-specific visual checks, then updates the model with group-relative policy optimization. This makes location or category correctness part of the reward rather than an incidental property of fluent text.

The gains show data-efficient adaptation in the reported few-shot settings. They do not establish that one reward transfers across tasks. Every new output contract needs a verifier that is difficult to exploit and cheap enough to evaluate repeatedly.

## High-Level Takeaways

- Visual-RFT makes perception measurable inside reinforcement fine-tuning.
- Verifiable rewards can outperform supervised fine-tuning in low-data settings.
- Reward design remains task-specific and can optimize only what the verifier exposes.
