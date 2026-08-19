---
title: "Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models"
date: '2026-08-10T00:00:00.000Z'
section: paper-shorts
postSlug: chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models
legacyPath: /paper shorts/2026/08/10/chain-of-spatial-thoughts-modality-agnostic-spatial-grounding-for-vision-language-models.html
tags:
  - Vision-Language Models
  - Spatial Reasoning
  - 3D Geometry
field: 'Vision-Language Models'
summary: "2026 – Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models"
---

## 2026 – Chain of Spatial Thoughts: Modality-Agnostic Spatial Grounding for Vision Language Models

**arXiv:** [2608.10278](https://arxiv.org/abs/2608.10278)

## Summary

> Chain of Spatial Thoughts adds continuous spatial tokens to a VLM without adding a separate inference-time spatial encoder. The tokens distill scene-level 3D geometry and object-centric attributes, remain available inside the model's chain of thought, and can be decoded to check whether they contain the intended geometry. On VSI-Bench, the paper reports a 4.3% improvement for Qwen3-VL-8B and 1.3% for SenseNova-SI-1.3, with especially strong object-size and room-size results.

## Core Insights

Most VLM spatial reasoning methods either ask a language model to infer geometry from visual tokens or add a task-specific geometry module. Space Tokens changes the interface: a spatial encoder produces continuous tokens that are inserted into the ordinary multimodal reasoning stream. The same token format can represent scene geometry and object relations, while auxiliary decoders provide an interpretability check.

The paper's object-level results make the claim concrete. The reported VSI-Bench table gives Qwen3-VL-8B an average score of 59.8 before the method and an improvement of 4.3 percentage points; object-size accuracy reaches 79.2% and room-size estimation 75.7%. These gains are not uniform across all spatial questions, so the method should be read as a targeted representation intervention rather than a general replacement for visual reasoning.

![Space Tokens three-stage pipeline for distilling and using continuous spatial representations](/assets/images/space-tokens-pipeline-paper-figure.png)
_The method learns spatial slots, integrates them into multimodal reasoning, and decodes them to verify geometry. Source: [Space Tokens](https://arxiv.org/abs/2608.10278)._

The architecture-agnostic claim is bounded by fine-tuning configuration. The paper uses parameter-efficient adaptation and a learned token interface, but it does not show that the same tokens transfer unchanged across arbitrary VLMs or modalities. A stronger test would hold the language model frozen and compare continuous tokens with equally sized learned visual prompts and a dedicated geometry encoder.

## High-Level Takeaways

- Space Tokens inform whether explicit continuous geometry should enter the reasoning sequence as a reusable intermediate object.
- The training unit is a scene or object paired with spatial latent tokens and reasoning supervision; the tokens are also decoded for geometric verification.
- The method keeps inference architecture simple, but it moves complexity into token supervision and the spatial teacher used during training.
- The falsification test is a model-matched comparison against prompt tokens, geometry adapters, and extra visual tokens across unseen spatial tasks. The conclusion would weaken if the gain is only a parameter-count or fine-tuning artifact.
