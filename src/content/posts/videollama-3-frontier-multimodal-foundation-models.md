---
title: 'VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding'
date: '2025-01-22T00:00:00.000Z'
section: paper-shorts
postSlug: videollama-3-frontier-multimodal-foundation-models
legacyPath: /paper shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html
tags:
  - Other
field: 'Video & Interactive World Models'
summary: "2025 – VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding"
---

**arXiv:** [2501.13106](https://arxiv.org/abs/2501.13106)<br>
**GitHub:** [DAMO-NLP-SG/VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)

## Summary

> VideoLLaMA3 makes a practical bet: a strong image-language system can carry much of the semantic load for video, if the vision stack preserves detail and compresses redundant frames carefully. Four training stages build from variable-resolution image encoding to video-centric tuning; Any-resolution Vision Tokenization retains spatial detail, while Differential Frame Pruner removes similar temporal patches. The 2B model reports 59.6 VideoMME without subtitles, 68.0 PerceptionTest, and 65.4 MLVU-dev.

## Core Insights

### Image quality is the first video capability

![VideoLLaMA3 four-stage vision-centric training paradigm](/assets/images/videollama-3-frontier-multimodal-foundation-models-source-figure-2.webp)
*Fig 1: VideoLLaMA3 moves from vision-encoder adaptation and image-text alignment to multi-task and video-centric fine-tuning, with the data mixture shown at each stage. | source: [VideoLLaMA 3, Figure 2](https://arxiv.org/abs/2501.13106)*

The report’s curriculum is deliberately asymmetric. Stage 1 adapts the vision encoder to variable-resolution images. Stage 2 aligns the vision encoder, projector, and LLM on detailed scene, document, chart, grounding, and text data. Stage 3 adds multi-task image SFT plus general video data, and Stage 4 concentrates on video, streaming, temporal grounding, image-only, and text-only examples. The reported stage sizes are 15.57M, 21.97M, 19.05M, and 5.71M examples.

That order reflects an annotation economy. High-quality image-text pairs are easier to curate than long video explanations, and many video questions still depend on static visual skills such as OCR, chart reading, and fine-grained object recognition. VideoLLaMA3 therefore asks temporal tuning to specialize an already useful visual interface, rather than asking noisy video captions to teach all visual semantics from scratch.

### Variable resolution and token pruning solve different bottlenecks

![VideoLLaMA3 benchmark comparison across image and video understanding tasks](/assets/images/videollama-3-frontier-multimodal-foundation-models-paper-figure.png)
*Fig 2: The released comparison chart places VideoLLaMA3 against image and video MLLM baselines on representative image, general-video, perception, and long-video benchmarks. | source: [VideoLLaMA 3, Figure 1](https://arxiv.org/abs/2501.13106)*

Any-resolution Vision Tokenization replaces fixed positional embeddings with 2D RoPE so images with different aspect ratios can be encoded without forcing every input into one square shape. This addresses spatial information loss. For videos, the model first applies 2×2 spatial downsampling, then Differential Frame Pruner compares corresponding patches in consecutive frames using pixel-space L1 distance; patches below the 0.1 threshold are treated as redundant and later patches are pruned. This addresses context length.

The two mechanisms should not be conflated. AVT changes how much spatial evidence enters the encoder. DiffFP is a temporal compression heuristic that assumes near-identical neighboring patches are less useful. It is cheap and interpretable, but it is not a learned event detector: a small pixel difference can still carry the decisive temporal cue, while a camera change can make redundant content look different.

### Benchmark gains expose both the value and the boundary

For the 2B model, Table 7 reports 59.6 on VideoMME without subtitles and 63.4 with subtitles, 68.0 on PerceptionTest, 65.5 on MVBench, 58.2 on ActivityNet-QA, 65.4 on MLVU-dev, 57.1 on LongVideoBench, 41.6 on LVBench, 63.4 on TempCompass, and 81.1 on NextQA. The model is evaluated with at most 180 frames and 16K visual tokens, greedy decoding, and benchmark-specific prompts. Charades-STA temporal grounding is scored by extracting predicted start/end times and computing mIoU, rather than by a dedicated temporal decoder.

![VideoLLaMA3 chart-understanding case study](/assets/images/videollama-3-frontier-multimodal-foundation-models-source-figure-6.webp)
*Fig 3: A chart case study shows the model comparing a price trend and then reasoning about model performance versus activated parameters, beyond OCR. | source: [VideoLLaMA 3, Figure 6](https://arxiv.org/abs/2501.13106)*

The image-to-video transfer claim is visible in the mix of results: the same 2B model reaches 69.4 on InfoVQA, 59.2 on MathVista, and 67.3 on RealWorldQA in the report’s image evaluation. The chart example illustrates why the authors emphasize image data: reading the axes and interpreting a trend are prerequisites for many video questions too.

The caveat is computational and temporal. The report acknowledges that high-resolution, long-video inference is not optimized for real-time use, and video annotations remain less diverse and reliable than image data. The 0.1 pixel-similarity threshold is also not a guarantee that a pruned patch was semantically irrelevant. A decisive follow-up would hold visual-token count and compute fixed while varying event duration, camera motion, and rare actions; if DiffFP removes the only frames that disambiguate event order, the efficiency gain is misleading.

## High-Level Takeaways

- VideoLLaMA3’s main design decision is to make image understanding the foundation and reserve video tuning for temporal specialization.
- AVT preserves spatial evidence across arbitrary shapes; DiffFP reduces temporal redundancy with a cheap pixel-space rule. They solve different parts of the token-budget problem.
- The 2B results are broad, covering general video, long video, temporal reasoning, OCR, charts, and math, but the evaluation still uses fixed frame and token caps.
- Pixel similarity is an efficiency signal, not a causal notion of importance. Long, rare, or subtle events remain the clearest stress test for the pruning strategy.
