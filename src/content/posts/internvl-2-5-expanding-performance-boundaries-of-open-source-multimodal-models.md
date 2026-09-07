---
title: 'InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models'
date: '2024-12-06T00:00:00.000Z'
section: paper-shorts
postSlug: internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models
legacyPath: /paper shorts/2024/12/01/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models"
---

## 2024 – InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models

**arXiv:** [2412.05271](https://arxiv.org/abs/2412.05271)

**Project:** [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)

## Summary

> InternVL 2.5 is a study of multimodal scaling as a coupled system. It keeps a ViT–MLP–LLM interface, improves dynamic resolution and data mixtures, optionally adapts the vision encoder, and uses strict filtering before full-model instruction tuning. The resulting 1B–78B family improves reasoning, OCR, multilingual, grounding, and video benchmarks; the most persuasive evidence is that large visual encoders reduce data needs and that clean data makes test-time reasoning usable. Benchmark leadership still depends on model size, frame count, prompting, and the OpenCompass slice being measured.

## Core Insights

### The architecture spends tokens where the image needs them

![Figure 1 from InternVL 2.5 showing model performance on the OpenCompass leaderboard](/assets/images/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models-paper-figure.png)
*Fig 1: The model family improves the eight-benchmark OpenCompass VQA aggregate across sizes. The plotted ranking measures that evaluation slice; it does not cover the full range of multimodal behavior. | source: [InternVL 2.5, Figure 1](https://arxiv.org/abs/2412.05271)*

InternVL 2.5 keeps the ViT–MLP–LLM pattern rather than introducing a new fusion block. Its InternViT-6B or InternViT-300M encoder produces visual tokens, a randomly initialized two-layer MLP maps them into the language model’s input space, and the LLM generates the response with next-token prediction. A 448 × 448 tile initially yields 1,024 visual tokens; pixel unshuffle reduces that to 256 before the language model sees it. The reduction is what makes dynamic high resolution practical: the system can tile a wide or tall image without sending every raw patch into the LLM.

The tile allocator matches the input aspect ratio to a set of 448-pixel grids. A single image can receive up to the dataset’s maximum tile count and an optional square thumbnail supplies a global view. For multi-image examples, the tile budget is divided across images and each is tagged as Image-1, Image-2, and so on. Video uses one 448-pixel tile per frame; 32 or 64 frames already produce 8,192 or 16,384 visual tokens. Figure 1 is a leaderboard rather than an architecture diagram, so the right interpretation is not that one scalar summarizes “multimodal intelligence.” It is evidence that the same token interface can scale from single-image OCR to multi-image and video inputs, with context length and tile policy as explicit constraints.

The 6B vision encoder is itself an incremental model. InternViT-6B-448px-V2.5 has 45 layers and 5.5B parameters after removing three layers that were more tuned to CLIP’s global alignment objective. The 300M variant has 24 layers and 0.3B parameters and is used with smaller language models. The paper’s model family swaps in InternLM2.5 or Qwen2.5 backbones, so model size changes both the visual and language capacity. That matters when comparing the 1B–78B curve: it is a family scaling result, not a pure language-model sweep.

### Staged training makes scale reusable

Training is split into an MLP warmup, an optional vision-encoder stage, and full-model instruction tuning. In Stage 1, only the projector is trained while InternViT and the LLM are frozen. The purpose is to map visual features into the LLM’s input space without destabilizing either pretrained component. Stage 1.5 trains InternViT and the projector with a lower learning rate, adding multilingual OCR, charts, and other underrepresented visual domains without discarding existing features. Stage 2 unfreezes the whole model on high-quality multimodal instruction data.

The progressive scaling variant runs the vision stage with a smaller LLM, then reuses the shared vision weights with a larger LLM. The paper reports about 120 billion training tokens for InternVL2.5-78B (roughly 76B in Stage 1 and 44B in Stage 2), compared with 1.4 trillion cumulative tokens for Qwen2-VL. This is a training-efficiency claim enabled by reuse, not proof that fewer tokens are universally sufficient: the data mixture, vision encoder, and stage boundaries differ.

The loss and data plumbing are part of the model. Random JPEG compression at quality 75–100 simulates internet image degradation. A square-averaging loss weight balances token-heavy and token-light responses, avoiding a pure token average that overweights long answers and a pure sample average that favors short ones. Data packing concatenates samples into fuller contexts while preserving per-sample attention and position indices, improving GPU utilization without letting one example attend to another.

### Quality filtering protects reasoning and user behavior

![Figure 5 from InternVL 2.5 showing data configuration by modality](/assets/images/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models-source-figure-5.webp)
*Fig 2: Dataset configuration. In InternVL 2.0 and 2.5, data augmentation is applied selectively, enabled for image datasets and disabled for videos and text. | source: [InternVL 2.5, Figure 5](https://arxiv.org/abs/2412.05271)*

The fine-tuning mixture grows from 5.1M samples in InternVL 1.5 to 7.3M in InternVL 2.0 and 16.3M in InternVL 2.5. By token count, single-image data contribute 45.92%, multi-image 9.37%, video 39.79%, and pure text 4.92%. Figure 2 shows why one augmentation policy is inappropriate: image datasets use JPEG augmentation and up to 6–36 tiles, multi-image datasets use larger tile budgets, videos disable JPEG augmentation and use one tile per frame, and text has no tile dimension.

The paper’s most human-facing finding is that a few thousand repetitive or anomalous samples can make a fully trainable model loop in long answers and chain-of-thought. The filter therefore uses three text paths: domain-specific LLM quality scoring, repetition detection with manual review, and heuristic rules for abnormal lengths, repeated lines, or long zero sequences. Multimodal data use repetition detection and heuristics with manual verification. The authors do not claim that filtering solves all repetition; they say it reduces the failure enough to make test-time scaling more useful.

### Reasoning gains depend on test time

![Figure 10 from InternVL 2.5 showing LongVideoBench performance as input frames increase](/assets/images/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models-source-figure-10.webp)
*Fig 3: Performance on LongVideoBench with varying input video frames. | source: [InternVL 2.5, Figure 10](https://arxiv.org/abs/2412.05271)*

On MMMU validation, InternVL2.5-78B reaches 70.1% when the better of direct answering and chain-of-thought (CoT) is reported, compared with 62.7% for InternVL2-Llama3-76B. Figure 3 makes the test-time contract visible for video: InternVL 2.5 continues to benefit from more frames in the reported LongVideoBench sweep, whereas InternVL 2.0 tends to peak around 16–32 and degrade as frames increase. More frames increase the visual context and cost; the curve is evidence for a scalable input policy, not a claim that 128 frames is always the correct setting.

| Capability | InternVL2.5-78B | Protocol boundary |
| --- | ---: | --- |
| MMMU validation | 70.1 | Higher of direct answer and CoT |
| MathVista test-mini | 72.3 | Multimodal math evaluation |
| DocVQA test | 95.1 | ANLS document metric |
| TextVQA validation | 83.4 | OCR and visual reasoning |
| RefCOCO average | 92.3 | RefCOCO/+/g grounding |
| Video-MME, without/with subtitles | 72.1 / 74.0 | Best among 16–64 frame settings |
| MVBench | 76.4 | 16-frame evaluation |
| LongVideoBench | 63.6 | Best among 16–64 frame settings |
| Pure-language average | 72.9 | OpenCompass language suite |

The family scaling is broad. InternVL2.5-4B reaches 52.3 MMMU validation, 84.0 ChartQA, and 91.6 DocVQA; 8B reaches 56.0 MMMU and 93.0 DocVQA; 38B reaches 63.9 MMMU and 95.3 DocVQA. At 2B, the paper’s 300M vision encoder trails Qwen2-VL-2B on TextVQA, DocVQA, and InfoVQA despite a larger language component, which is evidence that visual capacity matters more at a small total budget. At the largest scale, multilingual results remain close to Qwen2-VL-72B and the authors argue that much of the language-side multilingual ability is inherited from the underlying LLM.

The causal story is therefore narrower than the leaderboard. InternVL 2.5 improves InternVL 2.0 by changing the vision encoder, LLM, data scale, filtering, frame sampling, and inference prompts together. The paper’s controlled ablations and model family show which combinations work, while the benchmark tables show the end-to-end payoff. They do not isolate a single “scale law” for multimodal intelligence.

### Scale is coupled to data quality and inference budget

InternVL 2.5 exposes two budgets that a parameter count hides: how much visual evidence enters the context and how much answer generation is allowed. Dynamic tiles and frame sampling control the first; direct answering and reasoning prompts change the second. Filtering matters because a longer response budget is useful only if the model can spend it without falling into repetition. The benchmark gains therefore belong to the combined training and inference recipe. Reusing a stronger vision encoder makes that recipe cheaper to extend across model sizes, but does not isolate one universal scaling relationship.

## High-Level Takeaways

- InternVL 2.5 keeps a ViT–MLP–LLM core but makes visual token count an explicit dynamic-resolution and frame-budget decision.
- Progressive vision training lets one adapted InternViT serve larger language models, reducing redundant multimodal pretraining.
- Strict filtering targets repetitive and anomalous samples because tiny amounts of noise can destabilize full-model instruction tuning and CoT.
- The 78B model reaches 70.1 MMMU validation and 63.6 LongVideoBench under reported test-time settings; those scores include protocol and inference choices.
