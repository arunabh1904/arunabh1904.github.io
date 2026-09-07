---
title: 'LLaVA-OneVision: Easy Visual Task Transfer'
date: '2024-08-06T00:00:00.000Z'
section: paper-shorts
postSlug: llava-onevision-easy-visual-task-transfer
legacyPath: /paper shorts/2024/08/06/llava-onevision-easy-visual-task-transfer.html
tags: [Vision-Language Models, Video Understanding]
field: 'Vision-Language Models'
summary: '2024 – LLaVA-OneVision: Easy Visual Task Transfer'
---

## 2024 – LLaVA-OneVision: Easy Visual Task Transfer

**arXiv:** [2408.03326](https://arxiv.org/abs/2408.03326)

## Summary

> LLaVA-OneVision keeps the LLaVA interface—SigLIP, a two-layer MLP projector, and a Qwen-2 language model—and makes the visual sequence carry single images, image sets, and video frames. Higher AnyRes preserves image detail with bilinear pooling; a 3.2M single-image curriculum builds broad visual skills before a 1.6M mixed OneVision stage adds multi-image and video data. The crucial design is a shared token budget: up to 7,290 tokens for a single image, 8,748 for 12 images, and 6,272 for 32 video frames. This makes transfer plausible, while the comparisons also show that some video questions can be answered from a single frame and therefore do not prove temporal understanding.

## Core Insights

The paper's central idea is interface continuity. A single image, a sequence of images, and a video frame stream all become visual tokens inserted before the same language model. The architecture remains deliberately small: SigLIP encodes visual features, a two-layer MLP maps them into the Qwen-2 embedding space, and the language model predicts the answer conditioned on the visual sequence.

![LLaVA-OneVision network architecture](/assets/images/llava-onevision-easy-visual-task-transfer-source-figure-1.webp)
*Source Figure 1. The same projector-and-LLM interface consumes a single image, multiple images, or video frames; the visual signal changes while the language path stays shared. [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)*

The less obvious decision is how to keep those modalities comparable. A 384×384 SigLIP input yields 729 tokens. For a high-resolution single image, Higher AnyRes splits the image into crops, encodes them, then bilinearly interpolates the crop features if their total would exceed a threshold. For multi-image input, each image is padded into a 384×384 frame and receives 729 tokens. For video, each frame is encoded and then pooled to 196 tokens so more frames fit. The authors choose the maximum visual sequences to be roughly equal: $(1+9)\times729=7,290$ for a single image, $12\times729=8,748$ for multi-image, and $32\times196=6,272$ for video.

![LLaVA-OneVision Higher AnyRes representation](/assets/images/llava-onevision-easy-visual-task-transfer-source-figure-2.webp)
*Source Figure 2. Higher AnyRes preserves more spatial crops and uses bilinear interpolation to fit them into a bounded sequence; original AnyRes resizes and splits more aggressively. The tradeoff is resolution versus token count, not a new temporal encoder. [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)*

The curriculum follows the same logic. Stage 1 aligns the projector on 558K image-text pairs and updates only the projector. Stage 1.5 adds 4M high-quality knowledge examples and trains the full model. Stage 2 first trains on 3.2M single-image instructions, then continues with a 1.6M OneVision mixture containing 560K multi-image, 350K video, and 800K sampled single-image examples. The vision encoder learning rate is set five times lower than the language/projector rate in the full-model stages. This is a useful separation: learn a broad image interface before asking it to coordinate several visual signals.

The benchmark table shows that transfer is asymmetric. The 7B model rises from the single-image checkpoint to the full OneVision checkpoint on multi-image tasks: MI-VQA goes from 60.3 to 90.2, NLVR2 from 75.9 to 89.4, and Spot-the-Diff from 7.9 to 39.2. Video also improves: EgoSchema goes from 52.9 to 60.1 and VideoMME from 55.0/59.1 to 58.2/61.5 for the short/long settings. These are capability changes after mixed-modality training, not zero-shot image-to-video transfer alone.

The authors' own analysis supplies the most human insight. On ActivityNet-QA, LLaVA-OV-7B trained only on images scores 55.1, close to 56.6 after OneVision training. Many questions ask about a static property such as the color of a ball visible throughout the clip. A model can answer those from one frame. By contrast, EgoSchema and multi-view tasks benefit more from the extra modality training because they require comparison, ordering, or context across frames. “Video understanding” therefore contains both frame recognition and temporal reasoning; the transfer story is strongest where the question actually demands the latter.

The emerging-capability examples make the composition concrete. A model trained separately on chart and diagram data can combine them in a multi-image insurance calculation; OCR and referring skills transfer to set-of-mark GUI instructions; and image, multi-image, and video skills combine for visual prompting in videos and image-in-video referring. These are plausible compositions, but the paper describes them through qualitative examples, so they should be read as capability demonstrations rather than prevalence estimates.

The boundary is visible in the data as well. Of the 1.6M OneVision examples, the paper reports 43.0% multi-image, 25.9% video, and 31.2% single-image. The model is still trained mostly on single-image instructions overall, and the visual token budget limits the number of frames or images that can be represented. The high-level architecture makes transfer easy to attempt, but it does not by itself enforce object identity across frames, causal order, or long-range temporal state.

## High-Level Takeaways

- OneVision's main mechanism is a shared visual-token interface; the projector and language model can reuse image skills when images, image sets, and frames are serialized compatibly.
- Higher AnyRes spends tokens on spatial detail, then keeps the maximum sequence near 7–9K tokens across modalities: 7,290 for one image, 8,748 for 12 images, and 6,272 for 32 frames.
- The full 7B OneVision stage improves MI-VQA 60.3 → 90.2, NLVR2 75.9 → 89.4, EgoSchema 52.9 → 60.1, and VideoMME 55.0/59.1 → 58.2/61.5 relative to the image-only checkpoint.
- The ActivityNet-QA control is the warning label: 55.1 image-only versus 56.6 after mixed training shows that frame-answerable questions can make video transfer look stronger than temporal reasoning.
- Task transfer is a promising recipe for multi-image and video skills, but temporal persistence and causal understanding still need evaluations designed so one frame cannot answer the question.
