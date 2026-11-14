---
title: "Unbundling Multimodal Vision Encoding"
date: '2026-11-12T00:00:00.000Z'
section: paper-shorts
postSlug: unbundling-multimodal-vision-encoding
legacyPath: /paper shorts/2026/11/12/unbundling-multimodal-vision-encoding.html
tags:
  - Multimodal
  - VisualEncoders
field: "Vision-Language Models"
summary: "2026 – Unbundling Multimodal Vision Encoding"
---

## 2026 – Unbundling Multimodal Vision Encoding

**Paper:** [arXiv:2611.17334v1](https://arxiv.org/abs/2611.17334v1)

**Evidence:** [Controlled protocol, Section 2](https://arxiv.org/html/2611.17334v1#S2) · [Discussion and limitations, Section 8](https://arxiv.org/html/2611.17334v1#S8)

## Summary

> A vision encoder bundles a pretrained visual prior, dedicated parameters, and computation before text–image fusion. Controlled training separates those choices: dedicated vision capacity helps, while separate pretraining and pre-fusion placement are dispensable in the tested regime. Moving that capacity into the backbone's early layers matches a scratch encoder; reallocating it to shared parameters does not.

## Core Insights

### What the comparison holds fixed

The main experiments train Qwen3-style dense backbones and their visual components from scratch, using common text and image sources. “Lifetime compute” charges encoder pretraining, any initial text training, and multimodal training to one budget; it excludes the instruction-tuning stage used for evaluation. Equal FLOPs consequently need not mean equal tokens. This is a budget-allocation experiment, not a ranking of released checkpoints.

Multimodal training uses a 70:30 text-only/image–text mixture, 224-pixel images, and 256 visual tokens. Each checkpoint then receives three epochs of Cambrian-7M instruction tuning at 448 pixels. Average VQA first averages within four benchmark categories, then across those categories. All recipes use the same evaluation procedure; the mixture was selected using the scratch-encoder recipe, leaving recipe-specific mixture optimization open.

Separate encoder pretraining with contrastive, self-supervised, or captioning objectives does not improve the matched-budget result. Larger pretraining allocations can hurt by consuming compute otherwise available for joint training. This conclusion is bounded by the shared image pool: a released encoder trained on additional data could supply information absent from these runs.

### Matched-compute capacity

[Table 2](https://arxiv.org/html/2611.17334v1#S4.T2) gives the decisive parameter control. With a 1.7B backbone at $2.6\times10^{21}$ lifetime FLOPs, the scratch encoder's 0.4B vision parameters reach 44.4 average VQA. Encoder-free scores 39.0; putting the same extra parameters into a wider or deeper shared backbone yields 38.6–38.8. Total parameter count does not explain the encoder's advantage.

[Table 3](https://arxiv.org/html/2611.17334v1#S5.T3) then moves the same budget inside the backbone. A vision-only FFN branch in the first eight layers reaches 45.0. Giving that branch to every token reaches 42.3; putting vision-only branches in the last eight layers reaches 35.1. The useful combination is early computation reserved for visual tokens. The proposed explanation—reducing competition with text while patch features are still being transformed—is an interpretation of these controls, not a separately measured causal mechanism.

The full **in-fusion** model adds vision-only query, key, value, and output projections as well as FFN branches. Their outputs add to the shared computation; vision and text still participate in joint attention. Zero-initialized added projections preserve the shared model's initial behavior. This differs from replacing the shared weights with entirely separate modality streams. The placement ablation above uses FFN-only branches, so its 45.0 score should not be presented as a measurement of the full architecture.

### Scaling and serving

The scaling ladder tests whether the distinction survives changes in model size.

![Source Figure 7 compares vision-question-answering scores and caption loss for encoder-free, scratch-encoder, and in-fusion recipes.](/assets/images/unbundling-vision-encoding-source-figure-7.png)
*Fig 1: Scratch encoders (blue) and in-fusion models (pink) track closely across the tested sizes, while encoder-free models (orange) require more compute for comparable visual results. Displayed points use 64 token-equivalents per backbone parameter; fits use the wider training sweep. | source: [Paper, Figure 7](https://arxiv.org/html/2611.17334v1#S5.F7)*

[Open the figure at full size](/assets/images/unbundling-vision-encoding-source-figure-7.png)

Across 0.6B–6.4B backbones, the fitted visual-performance curves put encoder-free at roughly three times the scratch encoder's compute requirement, while in-fusion tracks the encoder. Appendix G fits each recipe's 17 ladder runs across sizes and budgets, with held-out-size checks. This is a measured-regime scaling result, not a universal multiplier for encoder-free training.

The serving advantage is narrower. [Table 36](https://arxiv.org/html/2611.17334v1#A8.T36) reports 210 ms time to first token for the scratch encoder versus 182 ms for full in-fusion at 1.7B and 0.4B visual parameters: a 13% reduction. The setup uses one H200, 16 concurrent requests, one 448-pixel image and 256 text tokens per request, and an eager implementation. The FFN-only variant reaches 171 ms. These timings depend on implementation and workload; under mixed continuous batching, in-fusion modules also enter forward passes containing text-only requests.

The clean follow-up would hold dedicated capacity fixed while testing high-resolution, text-rich images and sparse backbones. Those settings could change where visual specialization pays off. My [short blog discussion](/blog/2026/11/14/what-a-vision-encoder-actually-buys.html) connects this result to the earlier connector-based VLM designs.

## High-Level Takeaways

- Separate visual pretraining, dedicated visual capacity, and pre-fusion placement are independently testable choices.
- Matching total parameters does not recover the visual benefit of reserving early capacity for image tokens.
- In-fusion preserves shared multimodal computation while adding vision-only branches; FFN-only ablations isolate their placement.
- The scaling and serving results support this allocation within the tested dense-model regime. Extra pretraining data, high-resolution training, sparse experts, and video remain open.
