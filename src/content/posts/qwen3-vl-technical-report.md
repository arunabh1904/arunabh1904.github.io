---
title: 'Qwen3-VL Technical Report'
date: '2025-11-26T00:00:00.000Z'
section: paper-shorts
postSlug: qwen3-vl-technical-report
legacyPath: /paper shorts/2025/11/26/qwen3-vl-technical-report.html
tags: [Vision-Language Models, Long Context]
field: 'Vision-Language Models'
summary: '2025 – Qwen3-VL Technical Report'
---

## 2025 – Qwen3-VL Technical Report

**arXiv:** [2511.21631](https://arxiv.org/abs/2511.21631)

## Summary

> Qwen3-VL treats long multimodal context as a representation problem as well as a context-window problem. Its vision encoder keeps dynamic native resolution, DeepStack sends features from three ViT levels into the first three language-model blocks, and text timestamps give video patches an explicit temporal address. The flagship family supports a native 256K-token window; in the paper's needle test it reaches 100% through 30 minutes/256K tokens and 99.5% at roughly 1M tokens after YaRN extrapolation. That last number is an evaluation result under one retrieval task, not a guarantee that every long video is equally usable.

## Core Insights

### Put visual evidence back into the language stream

The framework has three pieces: a SigLIP2-based vision encoder, an MLP merger, and a Qwen3 decoder. The encoder accepts dynamic native-resolution images and videos, while the merger compresses each 2×2 visual feature group into one visual token. The unusual step is DeepStack. Instead of asking the final vision layer to carry every local and semantic cue, Qwen3-VL takes visual tokens from three ViT depths and adds them to the first three language-model layers through dedicated mergers.

![Figure 1: Qwen3-VL framework with dynamic visual tokens, DeepStack, and temporal timestamps](/assets/images/qwen3-vl-technical-report-source-figure-1.webp)
*Fig 1: This source Figure 1 shows the visual encoder, 2×2 token merger, multi-level DeepStack injections, and timestamp tokens that connect visual patches to the language decoder. | source: [Qwen3-VL Technical Report, Figure 1](https://arxiv.org/abs/2511.21631)*

This is a useful architectural division of labor. Early ViT layers still have fine local structure; later layers are more semantic. Injecting both lets the language model recover a small detail without forcing a single terminal feature map to preserve it and also summarize it. The ablation supports the mechanism under a controlled recipe: with an internal 15B-A2B language model, 200B pretraining tokens, and no post-training, DeepStack raises the average score from 74.7 to 76.0. InfoVQA moves from 71.9 to 74.2 and DocVQA from 89.5 to 91.1, which is the pattern expected when fine visual detail matters.

### Long context needs a temporal index, not only more tokens

Qwen3-VL extends interleaved multimodal rotary position encoding so time, height, and width are distributed across frequency bands rather than assigned a single undifferentiated position. For video, each temporal patch is also prefixed with a textual timestamp such as `<3.0 seconds>`; the training data uses both seconds and hour–minute–second forms. The timestamp is cheap relative to a full temporal coordinate system, but it gives the decoder a symbol it can copy into an answer or use while comparing events.

![Figure 3: Qwen3-VL long-video needle-in-a-haystack heatmap](/assets/images/qwen3-vl-technical-report-source-figure-3.webp)
*Fig 2: This source Figure 3 measures whether Qwen3-VL-235B-A22B-Instruct can locate a salient frame at different positions in videos from 0 to 120 minutes; the right half is extrapolation beyond the 256K training context. | source: [Qwen3-VL Technical Report, Figure 3](https://arxiv.org/abs/2511.21631)*

The protocol samples video at 1 FPS and adjusts frame resolution to keep a constant visual-token budget. The model is perfect on the reported clips through 30 minutes, corresponding to 256K tokens, and retains 99.5% accuracy at about 1M tokens, or roughly two hours, with YaRN-based positional extension. That is a strong retrieval result, but it is narrower than “the model understands two hours”: the needle is salient, the task asks for locating and answering about one inserted frame, and the token budget is controlled. Long-context serving still pays for the visual tokens and attention state.

### The training recipe carries as much of the result as the architecture

Pretraining moves through 8K, 32K, and 256K sequence lengths. The report's four stages freeze only the merger in S0, unfreeze the full model at roughly 1T tokens in S1, continue at 32K in S2, then use a 100B-token 256K stage for long documents and videos. Post-training adds a 1.2M-sample SFT set: one third text-only and two thirds image-text or video-text, trained first at 32K and then at 256K. Strong-to-weak text distillation improves the smaller models, and reasoning RL uses about 30K filtered queries. The visual-agent branch starts from about 10K grounding examples and expands to about 120K multi-turn tool interactions.

The flagship non-thinking model reports 89.3/88.9 on MMBench-EN/CN and 79.2 on RealWorldQA in the report's comparison table; those numbers summarize the whole data and post-training system. DeepStack is the cleaner architectural test. Separating those two kinds of evidence matters when deciding whether a smaller model needs the mechanism, the training scale, or both.

## High-Level Takeaways

- DeepStack reduces the burden on a single final visual representation by injecting three ViT feature levels into early language-model blocks.
- Text timestamps make temporal grounding explicit, while enhanced interleaved MRoPE keeps time, height, and width distinguishable inside long sequences.
- The 256K context is native; the reported 99.5% at roughly 1M tokens uses YaRN extrapolation and a controlled needle task.
- The report's gains are a system result spanning staged pretraining, 1.2M-example SFT, distillation, RL, and architecture, so DeepStack's 74.7→76.0 ablation is the most direct evidence for its individual contribution.
