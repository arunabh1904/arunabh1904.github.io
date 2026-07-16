---
title: 'Genie: Generative Interactive Environments'
date: '2024-02-23T09:00:00.000Z'
section: paper-shorts
postSlug: genie-generative-interactive-environments
legacyPath: /paper shorts/2024/02/23/genie-generative-interactive-environments.html
tags: [World Models]
field: Omni-Models
summary: '2024 – Genie: a latent-action generative environment learned from unlabeled Internet video.'
---

## 2024 – Genie: Generative Interactive Environments

**arXiv:** [2402.15391](https://arxiv.org/abs/2402.15391)  
**Project:** [Genie project page](https://sites.google.com/view/genie-2024/home)  
**Conference:** Technical report

**Summary:** Genie learns an interactive generative environment from unlabeled Internet videos. The 11B-parameter model combines a spatiotemporal video tokenizer, an autoregressive dynamics model, and a learned latent-action model, then generates worlds that can be controlled frame by frame.

## Paper Insights

Genie is valuable because it makes latent actions a trainable interface rather than requiring labeled controls. The learned action space also supports imitating behaviors from unseen videos. For a world-model program, its contribution is the separation of visual compression, dynamics prediction, and action representation.

| Component | Job |
| --- | --- |
| Video tokenizer | Compresses observations into a temporal representation. |
| Dynamics model | Predicts the next latent state. |
| Latent-action model | Supplies an action-conditioned control interface without action labels. |

**Limits:** Interactive plausibility in generated environments is not evidence of metric accuracy, safety, or causal fidelity in a physical robot or vehicle setting.

**Takeaway:** A world model needs action-conditioned consequences; realistic video alone is not enough.

