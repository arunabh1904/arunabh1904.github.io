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
*Verifiable visual rewards let one post-training framework optimize several perception tasks without relying only on answer text. source: [Visual-RFT](https://arxiv.org/abs/2503.01785)*

![Figure 4 from Visual-RFT: Visual Reinforcement Fine-Tuning](/assets/images/visual-rft-visual-reinforcement-fine-tuning-source-figure-4.webp)
*Figure 4 Qualitative results of Fine-Grained Image Classification. The thinking process significantly improves the reasoning ability of LVLMs, leading to higher image classification performance. source: [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)*

![Figure 1 from Visual-RFT: Visual Reinforcement Fine-Tuning](/assets/images/visual-rft-visual-reinforcement-fine-tuning-source-figure-1.webp)
*Figure 1 Our Visual R einforcement F ine- T uning (Visual-RFT) performs better than previous Supervised Fine-Tuning (SFT) on a variety of tasks, such as Open Vocabulary(OV)/Few-shot Detection, Reasoning Grounding, and Fine-grained Classification. source: [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)*


The method samples reasoning and answers, scores outputs with task-specific visual checks, then updates the model with group-relative policy optimization. This makes location or category correctness part of the reward rather than an incidental property of fluent text.

The gains show data-efficient adaptation in the reported few-shot settings. They do not establish that one reward transfers across tasks. Every new output contract needs a verifier that is difficult to exploit and cheap enough to evaluate repeatedly.

## High-Level Takeaways

- Visual-RFT makes perception measurable inside reinforcement fine-tuning.
- Verifiable rewards can outperform supervised fine-tuning in low-data settings.
- Reward design remains task-specific and can optimize only what the verifier exposes.
