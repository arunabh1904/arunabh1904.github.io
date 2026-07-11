---
title: 'VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: videollama-3-frontier-multimodal-foundation-models
legacyPath: /paper shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html
tags:
  - Other
field: Vision-Language Models
summary: VideoLLaMA 3 showed that strong image understanding can be the foundation for efficient video understanding.
---
## 2025 - VideoLLaMA 3

**arXiv:** [2501.13106](https://arxiv.org/abs/2501.13106)

**GitHub:** [DAMO-NLP-SG/VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)

**Summary:** VideoLLaMA 3 takes a vision-centric route to image and video understanding. It first adapts the vision encoder for variable-resolution images, aligns image-text data at scale, then adds video-specific training and token merging for temporal inputs.

The key claim is that high-quality image-text learning carries a lot of the load for video. Video data still matters, but the model does not need to learn all semantics from video clips alone.

## Paper Insights

VideoLLaMA 3 is a vision-centric model for image and video understanding. Its training recipe treats high-quality image-text data as the base for video capability, then adds video-specific tuning instead of treating video as a separate problem. The framework also uses visual-token efficiency techniques so longer videos do not overwhelm the language context. The evidence compares image and video benchmarks against prior MLLMs. The caveat is that benchmark videos are still cleaner and shorter than many real temporal reasoning tasks. The takeaway is that video MLLMs need both temporal data and strong visual representation design.

![Figure 1: Performance Comparison of VideoLLaMA3 with the previous advanced image/video MLLM on various representative benchmarks from VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding](/assets/images/videollama-3-frontier-multimodal-foundation-models-paper-figure.png)
_Figure 1: Performance Comparison of VideoLLaMA3 with the previous advanced image/video MLLM on various representative benchmarks. From the [VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding paper](https://arxiv.org/abs/2501.13106), via arXiv HTML._

**What to look at:**
- High-quality image-text alignment is treated as the foundation for video.
- Variable-resolution visual encoding helps preserve image detail.
- Token merging makes longer video contexts cheaper.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Training stages | Image alignment then video tuning | Uses image semantics before temporal specialization. |
| Efficiency | Dynamic token merging | Compresses redundant visual tokens across frames. |
| Evidence | Image and video benchmarks | Checks whether video gains preserve image understanding. |

**Context:** It connects the image VLM and video VLM stories. If static visual grounding is strong, video becomes a temporal extension rather than a separate world.

**Takeaway:** Video VLMs are constrained by visual token budgets. Good image features plus careful temporal compression are the practical path.
