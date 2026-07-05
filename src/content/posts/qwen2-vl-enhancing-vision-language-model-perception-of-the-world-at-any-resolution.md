---
title: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
date: '2024-09-01T04:00:00.000Z'
section: paper-shorts
postSlug: qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution
legacyPath: /paper shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html
tags:
  - Other
field: Vision-Language Models
summary: Qwen2-VL made resolution and video length more flexible by letting visual token count scale with the input.
---
## 2024 - Qwen2-VL

**arXiv:** [2409.12191](https://arxiv.org/abs/2409.12191)

**Project:** [qwen2.org/vl](https://qwen2.org/vl/)

**Plain-language summary:** Qwen2-VL focuses on a practical weakness in many VLMs: images are forced into fixed sizes, even when the useful information lives in small text, documents, or high-resolution details. Its dynamic-resolution approach lets the model allocate more or fewer visual tokens depending on the input.

The model family also handles images and video with a shared multimodal position encoding, making it useful for OCR-heavy tasks, documents, visual reasoning, and longer temporal inputs.

**Why it mattered:** Resolution is not cosmetic. If a model cannot preserve the evidence, the language model hallucinates around it. Qwen2-VL showed that flexible tokenization can make generalist VLMs much more usable.

**Take-home message:** VLMs need adaptive visual bandwidth. A receipt, a street scene, and a video clip should not all be squeezed through the same fixed visual bottleneck.
