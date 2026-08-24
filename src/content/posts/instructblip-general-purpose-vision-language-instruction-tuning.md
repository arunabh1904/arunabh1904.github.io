---
title: 'InstructBLIP: General-Purpose Vision-Language Instruction Tuning'
date: '2023-05-11T00:00:00.000Z'
section: paper-shorts
postSlug: instructblip-general-purpose-vision-language-instruction-tuning
legacyPath: /paper shorts/2023/05/11/instructblip-general-purpose-vision-language-instruction-tuning.html
tags: [Vision-Language Models, Instruction Tuning]
field: 'Vision-Language Models'
summary: '2023 – InstructBLIP: General-Purpose Vision-Language Instruction Tuning'
---

## 2023 – InstructBLIP: General-Purpose Vision-Language Instruction Tuning

**arXiv:** [2305.06500](https://arxiv.org/abs/2305.06500)

**Code:** [salesforce/LAVIS](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

## Summary

> InstructBLIP turns 26 public vision-language datasets into a common instruction format and makes BLIP-2's Q-Former instruction-aware. Training on 13 held-in datasets produces state-of-the-art zero-shot performance across 13 held-out datasets. Fine-tuning also reaches 90.7 percent on ScienceQA with image context.

## Core Insights

![InstructBLIP examples across visual understanding, reasoning, description, and dialogue](/assets/images/instructblip-paper-figure-1.png)
*The same model follows different visual instructions without changing its external interface. source: [InstructBLIP](https://arxiv.org/abs/2305.06500)*

![Figure 4 from InstructBLIP: General-Purpose Vision-Language Instruction Tuning](/assets/images/instructblip-general-purpose-vision-language-instruction-tuning-source-figure-4.webp)
*Figure 4 Comparison of instruction tuning and multitask training based on BLIP-2 FlanT5 XL backbone. For held-in evaluation, we compute the average score across all held-in datasets. For held-out evaluation, we compute the average score across GQA, TextVQA, VSR, HatefulMemes, IconQA, ScienceQA, iVQA, VizWiz. source: [InstructBLIP: General-Purpose Vision-Language Instruction Tuning](https://arxiv.org/abs/2305.06500)*

![Figure 2 from InstructBLIP: General-Purpose Vision-Language Instruction Tuning](/assets/images/instructblip-general-purpose-vision-language-instruction-tuning-source-figure-2.webp)
*Figure 2 Tasks and their corresponding datasets used for vision-language instruction tuning. The held-in datasets are indicated by yellow and the held-out datasets by white. source: [InstructBLIP: General-Purpose Vision-Language Instruction Tuning](https://arxiv.org/abs/2305.06500)*


The key change from BLIP-2 is that the instruction reaches the visual connector. The Q-Former does not extract one generic image summary before reading the task. It selects visual features in the context of what the user asks.

The held-out evaluation supports transfer across task formats, but the training mixture still defines the behavior. Instruction tuning teaches response policy and task routing. It does not guarantee that the frozen visual encoder preserved every detail needed by a new instruction.

## High-Level Takeaways

- InstructBLIP makes visual feature selection conditional on the instruction.
- A common instruction format turns many datasets into assistant behavior training.
- Better task following cannot recover evidence discarded by the frozen image encoder.
