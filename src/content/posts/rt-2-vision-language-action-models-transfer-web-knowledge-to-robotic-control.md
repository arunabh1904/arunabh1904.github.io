---
title: 'RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control'
date: '2023-07-28T09:00:00.000Z'
section: paper-shorts
postSlug: rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control
legacyPath: /paper shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html
tags:
  - Robotics
  - Vision-Language-Action
field: 'Vision-Language-Action & Robotics'
summary: 2023 – RT-2 co-trains web vision-language tasks and tokenized robot actions in one VLA model.
---

## 2023 – RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

**arXiv:** [2307.15818](https://arxiv.org/abs/2307.15818)

**Project:** [robotics-transformer2.github.io](https://robotics-transformer2.github.io/)

RT-2 turns robot control into another output language for a vision-language model. Continuous action dimensions are discretized into tokens, serialized as text-like outputs, and co-fine-tuned with Internet-scale VQA and image-language tasks.

## Paper Insights

The shared token interface allows semantic knowledge from web data to influence action selection. Across roughly 6,000 evaluation trials, RT-2 improves generalization to novel objects and instructions and exhibits behaviors such as selecting an improvised tool. The system demonstrates transfer from semantic pretraining into control, not a general solution to contact dynamics.

Co-training creates an important retention problem: robot data must ground the model without erasing its visual-language priors, while web data must not dominate action learning. Encoding actions as text simplifies the architecture but inherits autoregressive latency and quantization.

| Shared element | Benefit | Risk |
| --- | --- | --- |
| Token vocabulary | One decoder for language and action | Token likelihood may poorly reflect physical error. |
| Transformer parameters | Web knowledge can transfer to control | Gradients from abundant web data can dominate robot data. |
| Co-training mixture | Preserves semantic capability during robot tuning | Mixture ratios become a critical hidden variable. |

## Decision Lens

RT-2 informs whether semantic transfer is worth using a common language/action interface. Its unit is a multimodal sequence ending in either text or action tokens. The model shares nearly the entire transformer; discretization is the bridge between continuous control and next-token prediction.

The evaluations establish semantic generalization in the studied manipulation domain. A missing ablation would compare shared tokens, a continuous action head, and a separate action expert under matched VLM retention and latency. At ten times the control frequency, autoregressive decoding becomes the bottleneck. The central claim weakens if semantic probes improve while closed-loop task success under geometric perturbations does not.

**Context:** RT-2 coined the practical VLA recipe: reuse a VLM, represent actions in its interface, and co-train semantics with control.

**Limits:** Web knowledge does not supply metric state, force awareness, or recovery data.

**Takeaway:** A shared vocabulary transfers meaning efficiently; it does not make language-token probability a complete control objective.
