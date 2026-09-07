---
title: 'Qwen2.5-VL Technical Report'
date: '2025-02-19T00:00:00.000Z'
section: paper-shorts
postSlug: qwen2-5-vl-technical-report
legacyPath: /paper shorts/2025/02/19/qwen2-5-vl-technical-report.html
tags: [Vision-Language Models, Dynamic Resolution]
field: 'Vision-Language Models'
summary: '2025 – Qwen2.5-VL Technical Report'
---

## 2025 – Qwen2.5-VL Technical Report

**arXiv:** [2502.13923](https://arxiv.org/abs/2502.13923)

## Summary

> Qwen2.5-VL extends adaptive visual bandwidth along two axes: native-resolution images and dynamically sampled video. Window attention keeps most visual processing local, while a few full-attention layers exchange information across windows; absolute-time M-RoPE then distinguishes elapsed time from frame rank. The 72B model reports 70.2 on MMMU, 74.8 on MathVista, 96.4 on DocVQA, 885 on OCRBench, and 70.4 on MVBench. Those scores arrive with a 4T-token curriculum, structured targets, and long-context training, so the endpoint is a systems result rather than a single attention trick.

## Core Insights

### Local windows and absolute time control different costs

Qwen2.5-VL redesigns the visual encoder around high-resolution cost. Patch features retain their native spatial layout; window attention handles most layers locally, and only four layers use full attention to exchange information across windows. The report uses windows up to 112 × 112 pixels, corresponding to 8 × 8 patches. This preserves small cells in a chart or document without paying global quadratic cost at every layer, though the number of patches and the resulting language context still grow with the input.

![Qwen2.5-VL's vision encoder, window-attention blocks, dynamic FPS sampling, and absolute-time position IDs](/assets/images/qwen2-5-vl-technical-report-source-figure-1.webp)
*Fig 1: The source architecture diagram connects native-resolution image and video tokens to window attention, dynamic FPS sampling, and temporal IDs aligned to elapsed seconds. | source: [Qwen2.5-VL, Figure 1](https://arxiv.org/abs/2502.13923)*

The temporal change is more than a new frame sampler. Dynamic FPS chooses how many frames to keep, while M-RoPE's temporal coordinates are aligned to absolute time. Two clips with the same number of frames can therefore represent different durations, and an event at 12 seconds is not confused with “the twelfth sampled frame.” The encoding cannot recover an event the sampler never observes, but it prevents the sampler's rank from becoming the model's only clock.

The training table explains why the model can support these behaviors. Visual pretraining uses about 1.5T tokens; multimodal pretraining adds about 2T tokens of pure text, interleaved data, VQA, video, grounding, and agent examples; long-context pretraining adds roughly 0.6T tokens and raises the sequence length to 32,768. The ViT is trained first, then the ViT and language model are jointly trained, and the final stage adds long-video, long-document, and long-agent data. Structured outputs—coordinates, points, document elements, and UI actions—make spatial relations part of the supervision rather than an afterthought.

![Qwen2.5-VL benchmark comparison across document, reasoning, video, and general multimodal tasks](/assets/images/qwen2-5-vl-paper-figure-1.jpg)
*Fig 2: This table-derived view collects the paper's reported comparisons for Qwen2.5-VL-72B, Qwen2.5-VL-32B, Qwen2-VL-72B, and reference systems; it is a visual rendering of reported benchmark values, not a source-paper figure. | source: [Qwen2.5-VL, benchmark tables](https://arxiv.org/abs/2502.13923)*

The reported scores show the breadth of the recipe: Qwen2.5-VL-72B reaches 70.2 on MMMU, 74.8 on MathVista, 96.4 on DocVQA, 885 on OCRBench, 70.4 on MVBench, and 50.9 Charades-STA mIoU. These metrics span reasoning, documents, OCR, video understanding, and temporal localization. They are useful evidence that the integrated system works across modalities; they do not establish that window attention, dynamic FPS, or absolute time is individually responsible for each gain.

| Design choice | Benefit | Operational risk |
| --- | --- | --- |
| Native resolution | Preserves document and chart detail | Long inputs create memory and batching variance. |
| Window attention | Reduces most visual-encoder cost | Global interactions are limited to selected layers. |
| Dynamic FPS | Allocates frames to video content | Sampling can miss short events. |
| Absolute-time M-RoPE | Distinguishes elapsed time from frame index | Temporal position alone does not guarantee temporal reasoning. |
| Structured targets | Teaches grounding and UI actions directly | Target formats and annotation quality affect generalization. |

## High-Level Takeaways

- Qwen2.5-VL extends adaptive visual bandwidth in both space and time: resolution for images, frame rate for video.
- The four full-attention layers and local windows are a cost allocation inside the encoder, not a guarantee that high-resolution serving is cheap.
- Absolute time helps when clips differ in duration or sampling rate, but it cannot repair missed frames or weak temporal supervision.
- The 1.5T + 2T + 0.6T curriculum, structured targets, and 32,768-token stage are material parts of the benchmark result.
- The decisive deployment test is a matched compute ablation: native versus fixed resolution and dynamic versus fixed FPS under the same token budget and worst-case latency.
