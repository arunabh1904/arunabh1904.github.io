---
title: 'PaliGemma: A Versatile 3B VLM for Transfer'
date: '2024-07-10T04:00:00.000Z'
section: paper-shorts
postSlug: paligemma-a-versatile-3b-vlm-for-transfer
legacyPath: /paper shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html
tags:
  - Other
field: Vision-Language Models
summary: PaliGemma combines SigLIP and Gemma into a compact open VLM meant for fine-tuning across many vision-language tasks.
---
## 2024 - PaliGemma

**arXiv:** [2407.07726](https://arxiv.org/abs/2407.07726)

**Plain-language summary:** PaliGemma is a small open VLM built for transfer, not a giant visual chat assistant. It combines a SigLIP-So400m vision encoder with a Gemma-2B language model, then exposes many vision tasks as text generation.

That framing is practical: instead of building a separate head for captioning, VQA, detection, segmentation, remote sensing, and document tasks, the model learns a shared prefix-to-suffix interface that can be fine-tuned.

## Paper map

The architecture is intentionally simple. Image tokens from SigLIP go through a linear projection into Gemma's token space. A task prefix describes what to do, and the decoder autoregressively generates the answer, caption, box tokens, or segmentation tokens.

The training recipe has stages: reuse unimodal checkpoints, run multimodal pretraining, upcycle to higher image resolutions, then transfer to individual tasks. The release includes checkpoints at 224px, 448px, and 896px, which matters because many OCR, chart, document, and segmentation tasks are resolution sensitive.

![Figure 1 from PaliGemma showing the SigLIP image encoder feeding a Gemma decoder language model](/assets/images/paligemma-a-versatile-3b-vlm-for-transfer-paper-figure.png)
_Figure 1 shows PaliGemma's core architecture: a SigLIP image encoder feeds a Gemma decoder language model through a projection layer. From the [PaliGemma paper](https://arxiv.org/abs/2407.07726), via arXiv HTML._

**What to look at:**
- The model is a base VLM for transfer, not primarily an instruction-tuned assistant.
- Detection and segmentation are represented as generated token strings.
- Higher-resolution checkpoints are part of the design, not an afterthought.
- None of the pretraining tasks rely on outputs from a larger commercial VLM.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Vision backbone | SigLIP-So400m | Reuses a strong contrastive visual encoder. |
| Language backbone | Gemma-2B | Keeps the total model under roughly 3B parameters. |
| Transfer suite | Almost 40 tasks | Tests whether the base model is broadly adaptable. |
| Resolution family | 224px, 448px, 896px | Lets users trade compute for fine visual detail. |

**Compact result slice:**

| Finding | Evidence |
| ------- | -------- |
| Broad transfer | The paper reports strong results across VLM benchmarks, remote sensing, video tasks, and referring segmentation. |
| Low-data usability | Most tasks reach within 10% of full-data performance with 4k examples and within 20% with 256 examples. |
| MMVP | PaliGemma-224 reaches 47.3% paired accuracy, compared with 38.7% for GPT-4V and 40.7% for Gemini in the paper's comparison. |
| Practical training | A final Stage1 run takes just under 3 days on TPUv5e-256; Stage2 resolution increases take about 15 hours each. |

**Why it mattered:** PaliGemma made the "small open VLM as a transferable base model" story concrete. It is useful because it is inspectable, fine-tunable, and broad enough to cover more than chat.

**Take-home message:** PaliGemma is a compact VLM workhorse: simple architecture, many task interfaces, and enough resolution control to make transfer practical.
