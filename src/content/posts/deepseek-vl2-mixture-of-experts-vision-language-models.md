---
title: 'DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding'
date: '2024-12-13T00:00:00.000Z'
section: paper-shorts
postSlug: deepseek-vl2-mixture-of-experts-vision-language-models
legacyPath: /paper shorts/2024/12/01/deepseek-vl2-mixture-of-experts-vision-language-models.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding"
---

## 2024 – DeepSeek-VL2

**arXiv:** [2412.10302](https://arxiv.org/abs/2412.10302)

**GitHub:** [deepseek-ai/DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)

## Summary

> DeepSeek-VL2 makes both sides of a VLM conditional. Dynamic tiling adapts visual bandwidth to image resolution and aspect ratio, while DeepSeekMoE activates only a subset of language experts and Multi-head Latent Attention compresses the key-value cache. The released family has 3B, 16B, and 27B total LLM parameters with 0.57B, 2.4B, and 4.1B activated LLM parameters in the report's configuration table. The full model reaches 93.3 on DocVQA, 86.0 on ChartQA, 84.2 on TextVQA, and 811 on OCRBench; the efficiency claim is strongest when those scores are read alongside irregular visual-token counts and routing costs.

## Core Insights

### Tile the image and route language computation separately

The vision path replaces DeepSeek-VL's fixed 384 × 384 and 1024 × 1024 views with dynamic tiling. The model chooses a candidate resolution made from 384-pixel tiles, minimizes padding while preserving the original aspect ratio, and adds one global thumbnail to the local tiles. Each 384 × 384 tile produces a 27 × 27 grid, or 729 visual embeddings, from the SigLIP-SO400M-384 encoder. The 2 × 2 pixel-shuffle adaptor then compresses each tile to 14 × 14 = 196 language-model-facing visual tokens, with extra newline and separator tokens added to assemble the full sequence. Thus 729 is the pre-adaptor encoder count, not the token count consumed by the language model. For more than two images, the report disables dynamic tiling to control context growth. This is a concrete way to preserve small text and unusual layouts, but it makes every image a different sequence-length and batching case.

![DeepSeek-VL2 dynamic tiling turns one high-resolution image into local tiles plus a global view](/assets/images/deepseek-vl2-mixture-of-experts-vision-language-models-source-figure-3.webp)
*Fig 1: Dynamic tiling divides an image into local 384 × 384 tiles, encodes them with a shared vision encoder, and flattens the merged features with separator tokens. | source: [DeepSeek-VL2, Figure 3](https://arxiv.org/abs/2412.10302)*

The architecture keeps a LLaVA-style path—vision encoder, vision-language adaptor, then a decoder-only language model—but changes the compute profile inside the language model. DeepSeekMoE routes each token to a sparse expert subset, and MLA stores a compressed latent key-value state rather than a full KV cache. The model family therefore separates total capacity from active computation, while the visual path turns a variable grid of image tiles into a variable-length visual sequence.

![DeepSeek-VL2 architecture with dynamic visual tokens, a VL adaptor, and a sparse MoE language model](/assets/images/deepseek-vl2-mixture-of-experts-vision-language-models-source-figure-2.webp)
*Fig 2: The source architecture connects dynamically tiled image tokens through a VL adaptor to the DeepSeek-MoE language model, which predicts text tokens. | source: [DeepSeek-VL2, Figure 2](https://arxiv.org/abs/2412.10302)*

The training schedule preserves the asymmetry between conditioning and output. Stage 1 trains the vision encoder and adaptor while freezing the language model. Stage 2 unfreezes all components for roughly 800B image-text training tokens, and stage 3 performs supervised fine-tuning with all parameters unlocked. The next-token loss is computed on text tokens, so visual tokens condition the answer rather than becoming direct reconstruction targets. The SFT data adds regenerated VQA responses, cleaned OCR/document examples, detailed reasoning data, grounding coordinates, and text-only instructions to preserve language ability.

The benchmark results show where dynamic resolution matters. In the full model's OCR table, DeepSeek-VL2 reaches 93.3 DocVQA, 86.0 ChartQA, 78.1 InfoVQA, 84.2 TextVQA, and 811 OCRBench with 4.5B activated parameters including the vision encoder. DeepSeek-VL2-Small reaches 92.3 DocVQA and 834 OCRBench with 2.8B activated parameters. The general table reports 51.1 MMMU, 83.1 MMBench, 61.3 MMStar, and 62.8 MathVista for the full model. These comparisons support strong capability per active parameter, but they do not make dense and sparse inference costs identical.

![DeepSeek-VL2 average benchmark performance versus activated parameters](/assets/images/deepseek-vl2-mixture-of-experts-vision-language-models-paper-figure.png)
*Fig 3: The paper compares average multimodal performance against activated parameters across the DeepSeek-VL2, InternVL2, and Qwen2-VL families; its average combines MMBench v1.1, MMStar, MMMU (validation), MathVista (test-mini), AI2D (test), and OCRBench scaled to 0–100. | source: [DeepSeek-VL2, Figure 1](https://arxiv.org/abs/2412.10302)*

The efficiency boundary is operational. Dynamic tile counts can make batches uneven, expert routing can create load imbalance, and the report describes separate pipeline strategies for text-only and image batches. The authors also report that the 3B, 16B, and 27B variants can fit on single GPUs with 10GB, 40GB, and 80GB memory respectively. These are deployment facts for the reported implementation, not guarantees across arbitrary serving stacks.

| Decision | DeepSeek-VL2's answer | Boundary |
| --- | --- | --- |
| Image resolution | Candidate 384-pixel grids with local tiles and a global thumbnail | More tiles mean more visual tokens and variable batching cost. |
| Language capacity | Sparse DeepSeekMoE experts | Routing and expert imbalance affect throughput. |
| Attention memory | MLA compresses the KV cache into a latent vector | Cache savings do not remove visual encoder or token costs. |
| Training target | Text-only next-token supervision after visual conditioning | OCR and grounding quality depend on data and coordinate formats. |

## High-Level Takeaways

- DeepSeek-VL2 combines adaptive visual bandwidth with sparse language computation; either optimization alone would leave the other half of serving cost exposed.
- Dynamic tiling is most persuasive on documents, charts, OCR, and grounding, where fixed resizing can erase the evidence needed for the answer.
- The 93.3 DocVQA, 86.0 ChartQA, and 811 OCRBench results are paired with a 4.5B activated-parameter configuration, but active parameters are not the same as end-to-end latency.
- The report's practical limitations are few-image context, blurry or unseen objects, and weaker reasoning, which matter alongside the headline efficiency curve.
- A production comparison should match image-token budgets, expert load balance, memory, and latency before claiming that sparse routing or dynamic tiling is the cheaper choice.
