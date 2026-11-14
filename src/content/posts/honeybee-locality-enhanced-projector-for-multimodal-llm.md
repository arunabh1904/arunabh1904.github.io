---
title: "Honeybee: Locality-enhanced Projector for Multimodal LLM"
date: '2023-12-11T00:00:00.000Z'
section: paper-shorts
postSlug: honeybee-locality-enhanced-projector-for-multimodal-llm
legacyPath: /paper shorts/2023/12/11/honeybee-locality-enhanced-projector-for-multimodal-llm.html
tags:
  - Multimodal
  - Visual Projectors
field: "Vision-Language Models"
summary: "2023 \u2013 Honeybee: Locality-enhanced Projector for Multimodal LLM"
---

## 2023 – Honeybee: Locality-enhanced Projector for Multimodal LLM

**Paper:** [Cha et al., arXiv:2312.06742v2](https://arxiv.org/abs/2312.06742v2) · **Code and models:** [Honeybee](https://github.com/kakaobrain/honeybee)

## Summary

> Honeybee makes visual-token compression locality-aware. Its convolutional abstractor reaches a normalized spatial-task average of 53.5 with 144 visual tokens, compared with 43.9 for a same-length resampler and 52.6 for a 256-token linear projector. The controlled projector comparison is more informative than the full model's headline gains, which also depend on instruction-data balancing and formatting.

## Core Insights

[LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) maps each image feature into the language model without reducing the sequence. Query-based connectors can choose a smaller output budget, but the tested resampler concentrates on salient regions and can discard details needed for spatial questions. Honeybee changes the compression operation itself.

Its two projectors preserve locality in different ways.

![Honeybee source Figure 4 compares convolutional and deformable-attention abstractors.](/assets/images/honeybee-source-figure-4.png)
*Fig 1: C-Abstractor pools between convolutional blocks; D-Abstractor retrieves features around spatial reference points. Both reduce the visual sequence before the language model, while giving local evidence an explicit path through compression. | source: [Honeybee, Figure 4](https://arxiv.org/html/2312.06742v2#S3.F4)*

[Open the figure at full size](/assets/images/honeybee-source-figure-4.png)

C-Abstractor applies three ResNet blocks, adaptive average pooling, and three more blocks. Pooling chooses a square output grid. D-Abstractor uses six deformable-attention blocks; pooled image features initialize queries, and reference points begin spread across the image. The latter remains adaptive while discouraging every query from starting at the same salient region.

### A token budget changes the reader's work

Table 3 compares six spatial subtasks from MME, MMBench, and SEED-Bench. At 144 tokens, C-Abstractor improves the normalized average by 9.6 points over the resampler. Against the 256-token linear baseline, it slightly improves that average while reducing measured pretraining step time from 3.04 to 2.23 seconds. These are training-step timings, not answer latency. At 256 tokens, the convolutional projector reaches 56.3, so compression still has an accuracy cost.

Appendix B tests a simpler explanation: adding position embeddings to the ordinary resampler barely changes its broader normalized score, from 64.2 to 64.4. The architectural locality is doing more than merely announcing a token's position. This comparison supports testing how evidence is aggregated, alongside how many tokens survive.

### The training recipe is a separate contribution

Honeybee uses CLIP ViT-L/14 features and Vicuna-v1.5 at 7B or 13B. First, image-caption prediction trains the abstractor with both endpoints frozen, sampling COYO100M and BLIP-CapFilt equally. Instruction tuning then updates the abstractor and full language model while keeping CLIP frozen. The final schedule uses 200,000 alignment steps at batch size 256 and 10,000 instruction steps at batch size 128; ablations use shorter schedules.

Instruction data combine captioning, open and multiple-choice VQA, referring expressions, and visual/text conversations. Dataset-specific templates distinguish answer conventions; multi-turn examples remove duplicate targets that could encourage copying. In Table 5, deduplicated multi-turn training reaches 70.6 normalized average versus 69.6 without deduplication. More template variety alone does not consistently improve results.

The final benchmarks include MMBench-dev, image-only SEED-Bench, MME, and GPT-4-judged LLaVA-Bench. They measure different abilities and answer formats. Appendix results also show that improved general visual scores do not guarantee less hallucination: the 7B model remains below LLaVA-1.5 on POPE. For a new connector, I would hold data and reader adaptation fixed, then measure small-object and spatial errors at several token budgets before adopting the complete Honeybee recipe.

## High-Level Takeaways

- Locality-aware aggregation can preserve spatial detail at a smaller visual-token budget than a direct projector.
- The decisive comparison holds output length fixed; a full-model leaderboard cannot isolate the connector.
- Dataset balance, answer templates, and multi-turn deduplication materially affect the trained interface.
- Training-step efficiency and stronger visual averages do not establish lower serving latency or uniformly reduced hallucination.
