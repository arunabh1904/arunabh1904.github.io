---
title: 'InternVL3: Native Multimodal Pretraining'
date: '2025-04-14T00:00:00.000Z'
section: paper-shorts
postSlug: internvl3-native-multimodal-pretraining
legacyPath: /paper shorts/2025/04/14/internvl3-native-multimodal-pretraining.html
tags: [Vision-Language Models, Multimodal Pretraining]
field: 'Vision-Language Models'
summary: '2025 – InternVL3: Native Multimodal Pretraining'
---

## 2025 – InternVL3: Native Multimodal Pretraining

**arXiv:** [2504.10479](https://arxiv.org/abs/2504.10479)

## Summary

> InternVL3 uses “native multimodal” to describe joint learning during the main pretraining stage, not random initialization. A pretrained ViT and language model are connected through an MLP and optimized together on text and multimodal data. Variable Visual Position Encoding (V2PE) assigns visual tokens fractional position increments so high-resolution and multi-image context consumes less of the language position axis. InternVL3-78B reports 72.2 on MMMU in the abstract, and 79.5 on the aggregate OpenCompass Academic measure; later SFT, preference optimization, and test-time selection are part of the full system.

## Core Insights

### Multimodal data enters the language pretraining stage

InternVL3 keeps the familiar ViT–MLP–LLM stack but changes when the pieces learn together. Its ViT and language model begin from pretrained weights, and the visual path uses pixel unshuffle so a 448 × 448 tile becomes 256 visual tokens before entering the language model. “Native” therefore means that text-only, image-text, video-text, and interleaved examples appear in the main pretraining mixture while all model parameters are jointly optimized. It does not mean that the visual or language abstractions were learned from scratch in isolation.

![InternVL3 multimodal benchmark table across InternVL3, Qwen2.5-VL, and other models](/assets/images/internvl3-native-multimodal-pretraining-source-figure-1.webp)
*Fig 1: The report's comparison table places InternVL3 variants beside Qwen2.5-VL, earlier InternVL models, and other multimodal systems across MMMU, MathVista, charts, documents, OCR, and video. | source: [InternVL3, Figure 1](https://arxiv.org/abs/2504.10479)*

The position mechanism addresses the next bottleneck. Ordinary positions advance by one for every token, so a high-resolution image can consume the same position range that the text needs for reasoning. V2PE advances textual positions by 1 but gives each image a smaller increment δ, selected per image from {1, 1/2, …, 1/256}. Relative order inside the image is preserved while the image occupies less of the position axis; δ = 1 recovers the conventional scheme. This is context allocation, not free computation: the visual encoder and attention still process the tokens.

The loss makes the “conditioning” claim precise. InternVL3 computes the autoregressive objective on text tokens, while visual tokens provide context for those predictions rather than becoming direct reconstruction targets. This lets the visual pathway receive learning signal through language grounding without requiring the model to reproduce every visual code. It also means that the quality of captions, questions, and structured multimodal labels determines which visual distinctions are rewarded.

![InternVL3 OpenCompass academic leaderboard versus model scale and competing multimodal LLMs](/assets/images/internvl3-native-multimodal-pretraining-source-figure-2.webp)
*Fig 2: The OpenCompass comparison plots academic scores against model scale, including the InternVL3 family and competing multimodal LLMs. | source: [InternVL3, Figure 2](https://arxiv.org/abs/2504.10479)*

The sampling study gives the main pretraining tradeoff a measurable shape. The report uses a 1:3 language-to-multimodal ratio under a fixed total budget, roughly 50B language tokens and 150B multimodal tokens. That mixture preserves language competence while giving the visual pathway enough exposure. The final scores then add supervised fine-tuning, mixed preference optimization, and test-time scaling; for reasoning, best-of-N responses are selected with a visual process reward model. A leaderboard number can therefore include extra inference compute beyond one forward pass.

The two benchmark views should be read with those distinctions attached. The 72.2 score is for the individual MMMU benchmark; 79.5 is the aggregate OpenCompass Academic score in Table 1. These measure different things, and comparisons still require the same model variant and evaluation protocol. The figures support the integrated pipeline—initialization, V2PE, data mixture, post-training, and test-time selection—not an isolated causal claim that native pretraining alone explains the gain.

| Decision | InternVL3's answer | Boundary |
| --- | --- | --- |
| Initialization | Pretrained ViT and language-model weights | Native multimodal training is not random initialization. |
| Position budget | V2PE assigns visual tokens fractional position increments | Visual attention and encoder cost still scale with visual input. |
| Loss | Predict text tokens conditioned on visual tokens | Direct visual reconstruction is not the training target. |
| Data ratio | Roughly 1:3 language to multimodal tokens | The mixture trades language retention against visual exposure. |
| Reasoning | SFT + MPO + best-of-N visual process reward | Reported inference scores may use extra test-time compute. |

## High-Level Takeaways

- InternVL3's native claim is about joint optimization and exposure during main pretraining, while strong pretrained base models remain part of the recipe.
- V2PE preserves visual order while slowing visual position growth, helping high-resolution and multi-image context fit inside a language window.
- The 1:3 mixture and text-only loss explain how the model protects language ability while learning visual grounding, but they make supervision quality central.
- MMMU accuracy of 72.2 and the aggregate OpenCompass Academic score of 79.5 measure different scopes; neither is an isolated ablation of the pretraining recipe.
- The clean causal test would hold base weights, data, post-training, and test-time sampling fixed while changing only the multimodal schedule and position encoding.
