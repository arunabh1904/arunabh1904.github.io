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

### Method and reported result

InternVL3 trains language and multimodal data together in one main pretraining stage, adds Variable Visual Position Encoding (V2PE) for long mixed contexts, and then applies supervised fine-tuning, mixed preference optimization, and test-time scaling. The architecture remains a ViT–MLP–LLM stack. InternVL3-78B reports 72.2 on MMMU in the abstract; the report's local OpenCompass table lists a 79.5 score for the 78B model.

## Summary

> “Native multimodal” describes when joint learning happens, not whether every parameter starts from zero. InternVL3 initializes its ViT and language model from pretrained base models, then optimizes them together on text and multimodal data. V2PE compresses visual position growth so more visual evidence can fit inside a finite language context.

## Core Insights

![InternVL3 multimodal benchmark table across InternVL3, Qwen2.5-VL, and other models](/assets/images/internvl3-native-multimodal-pretraining-source-figure-1.webp)
*Fig 1: The report compares multimodal benchmark results across InternVL3 variants, Qwen2.5-VL, earlier InternVL models, and closed-source systems. | source: [InternVL3, Figure 1](https://arxiv.org/abs/2504.10479)*

![InternVL3 OpenCompass academic leaderboard versus model scale and competing MLLMs](/assets/images/internvl3-native-multimodal-pretraining-source-figure-2.webp)
*Fig 2: OpenCompass academic scores are plotted against parameter count, showing the InternVL3 family scaling from small models to the 78B point. | source: [InternVL3, Figure 2](https://arxiv.org/abs/2504.10479)*

The architectural detail that makes the training schedule usable is V2PE. A normal positional sequence advances by one for every token, so a high-resolution image can consume the same position range that text needs for reasoning. V2PE advances textual positions by 1 but visual positions by a smaller δ, chosen per image from a set of fractional values. Relative order inside an image is preserved, while the image occupies less of the position axis. At inference, δ can be selected based on input length; δ = 1 recovers the conventional scheme. This is a context-allocation mechanism, not a claim that visual tokens become computationally free—the attention and ViT still process them.

InternVL3 also clarifies a common ambiguity around native multimodal pretraining. The authors use pretrained ViT and language-model base weights to reduce cost; “native” means that text-only, image-text, video-text, and interleaved samples are jointly exposed during the main pretraining stage instead of adding vision only after a completed language model. The loss is autoregressive but computed only on text tokens. Visual tokens provide conditioning context and receive gradients through their role in predicting text, while the model is not asked to reproduce every visual token directly.

The sampling study is unusually concrete. The report finds a 1:3 language-to-multimodal ratio under a fixed total budget, with approximately 50B language tokens and 150B multimodal tokens. That mixture helps preserve language competence while giving the visual pathway enough supervision. Every model parameter is jointly optimized during this stage, but SFT, MPO, and test-time scaling also contribute to the reported endpoint. For reasoning evaluation, the test-time procedure uses best-of-N responses selected by a visual process reward model, so a leaderboard score may include additional inference compute.

The benchmark figures show a strong scaling curve, but they should be read with the recipe attached. InternVL3 changes the position scheme, data mixture, post-training, inference-time selection, and infrastructure together. The results support the integrated pipeline; they do not isolate how much of the gain comes from native pretraining versus V2PE, data, MPO, scale, or test-time compute.

| Decision | InternVL3's answer | Boundary |
| --- | --- | --- |
| Training schedule | Joint text and multimodal pretraining | Strong base models are still used for ViT and LLM initialization. |
| Position budget | V2PE assigns visual tokens fractional position increments | Visual attention and encoder cost still scale with visual input. |
| Loss | Predict text tokens conditioned on visual tokens | Direct visual reconstruction is not the training target. |
| Reasoning | SFT + MPO + best-of-N visual process reward | Reported inference scores may use extra test-time compute. |

## High-Level Takeaways

- InternVL3's “native” claim is about joint optimization and exposure, not training a multimodal model from random initialization.
- V2PE preserves visual order while slowing visual position growth, which is a practical way to fit high-resolution and multi-image context into a language window.
- The 1:3 language-to-multimodal mixture and text-only loss explain how the recipe protects language ability while learning visual grounding.
- The 72.2 MMMU headline and 79.5 OpenCompass table entry are different evaluations; preserve that distinction when comparing models.
- A clean ablation would hold base weights, data, post-training, and test-time sampling fixed while changing only the multimodal training schedule and position encoding.
