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

### Method and reported result

Qwen2.5-VL extends Qwen2-VL's variable-resolution design with a redesigned vision transformer, window attention, dynamic FPS sampling, and multimodal rotary positions aligned to absolute time. It also trains on structured document, grounding, and agent data and expands the long-context stage to 32,768 tokens. The paper reports a model family that handles high-resolution documents and long videos while retaining the Qwen2.5 language backbone.

## Summary

> Qwen2.5-VL turns “how much visual input should I keep?” into both a spatial and temporal decision. Native-resolution images get variable token budgets; videos choose their sampling rate; absolute time keeps an event at 12 seconds distinct from the twelfth sampled frame.

## Core Insights

![Qwen2.5-VL framework with native-resolution images, dynamic FPS video sampling, and absolute-time M-RoPE](/assets/images/qwen2-5-vl-paper-figure-1.jpg)
*Fig 1: The framework maps native-resolution images and dynamically sampled video frames to variable-length token sequences, while M-RoPE aligns temporal IDs to absolute time. | source: [Qwen2.5-VL, Figure 1](https://arxiv.org/abs/2502.13923)*

The visual encoder is redesigned around the cost of high resolution. Patch features retain native size, window attention makes most layers scale roughly with local windows rather than a global quadratic map, and only a few layers use full attention to exchange information across windows. The input is therefore not resized into one canonical square before perception. The model can preserve the small cells of a chart or the layout of a document, but the sequence and memory footprint grow with what the input contains.

The temporal extension is the more interesting part for video. Dynamic FPS sampling chooses how many frames to keep, while the temporal coordinates in M-RoPE are aligned to absolute time. Two clips with the same number of frames can therefore represent different durations, and an event's position is not confused with its rank in a sampled list. This is a useful distinction for localization and long videos, although it does not by itself recover events that the sampler never observes.

![Qwen2.5-VL's vision encoder, window-attention blocks, and absolute-time position IDs](/assets/images/qwen2-5-vl-technical-report-source-figure-1.webp)
*Fig 2: The source architecture diagram shows variable image and video token sequences, window attention in the visual transformer, dynamic FPS sampling, and time IDs aligned to elapsed seconds. | source: [Qwen2.5-VL, Figure 1](https://arxiv.org/abs/2502.13923)*

The training table makes the compute tradeoff explicit. Visual pretraining uses about 1.5T tokens with image caption, knowledge, and OCR data; multimodal pretraining uses about 2T tokens with pure text, interleaved data, VQA, video, grounding, and agent tasks; long-context pretraining adds about 0.6T tokens and raises sequence length to 32,768. The ViT is trained first, then ViT and language model are jointly trained, and the final stage adds long video, long document, and long-agent data. These phases explain why the endpoint cannot be attributed to dynamic resolution alone.

The model's structured outputs are part of the same design. Grounding coordinates, points, document elements, and UI actions become language-compatible targets, so the visual encoder is trained to preserve not only appearance but also spatial relations and affordances. That broadens the use case from “answer a question about the frame” to “identify, localize, and act,” while making the visual token budget and the supervision format jointly responsible for behavior.

| Design choice | Benefit | Operational risk |
| --- | --- | --- |
| Native resolution | Preserves document and chart detail | Long inputs create memory and batching variance. |
| Window attention | Reduces most visual-encoder cost | Global interactions are limited to selected layers. |
| Dynamic FPS | Allocates frames to video content | Sampling can miss short events. |
| Absolute-time M-RoPE | Distinguishes elapsed time from frame index | Temporal position alone does not guarantee temporal reasoning. |

## High-Level Takeaways

- Qwen2.5-VL extends adaptive visual bandwidth along two axes: image resolution and video sampling rate.
- Absolute time is useful because frame order is not a clock, especially when clips have different FPS or duration.
- The 1.5T + 2T + 0.6T training stages, structured targets, and long-context curriculum are material parts of the result.
- Serving quality should be measured as an accuracy–latency curve with worst-case documents and long videos, not only an average benchmark score.
- The key ablation is matched visual evidence and compute: fixed resolution versus native resolution, and fixed FPS versus dynamic FPS, under the same token budget.
