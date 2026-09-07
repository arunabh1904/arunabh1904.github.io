---
title: 'Molmo 2: Video Understanding and Grounding'
date: '2026-01-15T00:00:00.000Z'
section: paper-shorts
postSlug: molmo-2-video-understanding-and-grounding
legacyPath: /paper shorts/2026/01/15/molmo-2-video-understanding-and-grounding.html
tags: [Vision-Language Models, Video Grounding]
field: 'Video & Interactive World Models'
summary: '2026 – Molmo 2: Video Understanding and Grounding'
---

**arXiv:** [2601.10611](https://arxiv.org/abs/2601.10611)

## Summary

> Molmo2 treats video grounding as a data and interface problem. Its open recipe adds dense captions, long-form QA, pointing, and tracking data without closed-model distillation, then teaches one VLM to emit language plus timestamped coordinates and object IDs. The 8B model reports 35.5 accuracy on Molmo2-VideoCount, 38.4 F1 on video pointing, and 56.2 J&F on the academic tracking average.

## Core Insights

### Grounding is built into the data interface

![Molmo2 data and model overview for dense captions, QA, pointing, tracking, and spatio-temporal localization](/assets/images/molmo-2-video-understanding-and-grounding-source-figure-1.webp)
*Fig 1: Molmo2 constructs open video data for captions, QA, pointing, tracking, and temporal localization, then trains a model to answer in language or grounding coordinates. | source: [Molmo2, Figure 1](https://arxiv.org/abs/2601.10611)*

Molmo2’s strongest idea is easy to miss if the paper is read as another benchmark table: the output space is designed around evidence. A caption says what happened, but a point says where and when it happened; a track adds identity across frames. The authors build seven new video datasets and two multi-image datasets, including 520k pointing and tracking instances, 104k videos with dense captions, 212k long-form QA instances, and roughly 1.3M long-video QA instances. The data are constructed without distilling from closed VLMs.

The caption pipeline uses human speech rather than forcing annotators to type long descriptions. Workers record detailed descriptions of clips, the audio is transcribed, and frame-level visual details from Molmo are used to enrich the transcript. The pointing pipeline asks for spatial-temporal coordinates; the tracking format adds an integer object ID so the model can follow multiple entities. These choices make the requested evidence explicit in the training target instead of asking a generic language loss to discover it accidentally.

The representation is deliberately simple: normalized x/y coordinates, timestamps or image indices, and IDs are serialized as text. A tracking answer can therefore name the same object at several times, while a counting answer can be learned as a set of points rather than as a naked integer.

### Efficient multimodal training keeps the evidence usable

Molmo2 uses SigLIP 2 So400m/14 at 384px with a Qwen3 or OLMo language model. Images can use up to 8 overlapping training crops (24 at inference); video is sampled at 2 FPS with at most 128 frames, or 384 frames during long-context training. A 3×3 attentional pool reduces per-frame video tokens, and timestamps are interleaved with the visual tokens.

The three-stage recipe is intentionally short and compositional: 32k steps of image captioning/pointing pre-training, 30k steps of joint multimodal SFT at sequence length 16,384, then 2k steps at length 36,864 with up to 384 frames. The loss assigns fixed weight 0.1 to video captions and 0.2 to pointing because their long outputs would otherwise dominate short-answer tasks; other examples use a square-root answer-length heuristic.

![Molmo2 message-tree attention mask for packed multimodal examples](/assets/images/molmo-2-video-understanding-and-grounding-source-figure-3.webp)
*Fig 2: The message-tree encoding lets several annotations share visual input while a custom mask prevents one annotation branch from attending to another. | source: [Molmo2, Figure 3](https://arxiv.org/abs/2601.10611)*

The packing system is more than an implementation footnote. It selects examples jointly under token and crop budgets; message trees linearize about four annotations per example while preserving branch isolation. The authors report 3.8 examples per 16,384-token sequence and 15× training efficiency. This is how the recipe can afford dense, long answers without letting padding and repeated visual encoding consume the budget.

### The matched ablations explain the gains

On the 8B model, Molmo2 reaches 35.5 accuracy on its VideoCount set and 38.4 F1 on its VideoPoint set; the academic tracking average is 56.2 J&F, 57.1 F1, and 57.5 HOTA. These numbers are evaluated with different metrics for different output interfaces: J&F summarizes masks after point predictions are converted with SAM 2, F1 measures points at 1 FPS, and HOTA measures association when IDs are available. They should not be collapsed into one generic “tracking score.”

The ablations make the causal story more concrete. For counting, “point then count” reaches 34.5 MVC versus 28.1 for directly predicting the count, while the BVC result is 61.5 versus 61.3. Pointing supplies a visual inventory before the language model commits to a number. For video modeling, removing bidirectional visual-token attention changes QA/caption F1 from 64.8/39.5 to 64.4/38.5; removing timestamp tokens drops caption F1 to 37.4. Enlarging the video pool from 3×3 to 4×4 slightly lowers QA and drops caption F1 to 37.0, a reminder that fine detail and token budget are in tension.

Long-context SFT improves the long-video QA average from 64.4 to 67.4, but caption F1 falls from 42.3 to 39.9 and short-video QA changes from 69.6 to 69.4. Longer context is therefore a specialization with a measurable cost, not a free upgrade. The paper also notes that benchmark prompting and frame counts are not always public for baselines, so cross-model comparisons deserve that qualification.

## High-Level Takeaways

- Molmo2 makes grounding inspectable: a good answer can carry language, time, coordinates, and persistent object identity together.
- The data pipeline is part of the model. Spoken dense captions, points before counts, and tracking IDs teach evidence that a generic video QA mixture does not provide.
- Packing, message trees, bidirectional visual attention, timestamps, and token weighting are practical mechanisms that preserve this supervision at training scale.
- Grounding still stops short of geometry or control. SAM 2 is used to turn points into masks for evaluation, and long-context gains trade against caption detail; neither result establishes metric depth or action-conditioned prediction.
