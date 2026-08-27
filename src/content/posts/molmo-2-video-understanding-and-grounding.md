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

## 2026 – Molmo 2: Video Understanding and Grounding

**arXiv:** [2601.10611](https://arxiv.org/abs/2601.10611)

## Summary

> Molmo 2 extends open-weight visual assistants to multi-image and video grounding. Nine new datasets cover dense captioning, question answering, pointing, and tracking without labels from closed VLMs. The 8B model reports 35.5 accuracy on video counting, 38.4 F1 on video pointing, and 56.2 J&F on tracking.

## Core Insights


![Figure 3 from Molmo 2: Video Understanding and Grounding](/assets/images/molmo-2-video-understanding-and-grounding-source-figure-3.webp)
*Fig 1: Attention mask for a packed sequence with two examples. The first contains two QA pairs for one image. | source: [Molmo 2: Video Understanding and Grounding](https://arxiv.org/abs/2601.10611)*

![Figure 1 from Molmo 2: Video Understanding and Grounding](/assets/images/molmo-2-video-understanding-and-grounding-source-figure-1.webp)
*Fig 2: Molmo 2 trains on open video-centric data for dense captions, long-video question answering, pointing, and tracking; it accepts images or videos and produces language plus spatio-temporal grounding. | source: [Molmo 2: Video Understanding and Grounding](https://arxiv.org/abs/2601.10611)*


Molmo 2 makes temporal evidence inspectable. A caption can summarize an event, but a point or track must identify where the relevant entity appears across time. The training recipe adds bidirectional attention over visual tokens and token weighting for dense grounded outputs.

The released data improves reproducibility, but video grounding remains annotation-heavy. Points and tracks expose correspondence and persistence; they still do not provide metric depth or action-conditioned prediction.

## High-Level Takeaways

- Molmo 2 carries point-based grounding from images into video and multi-image inputs.
- Open datasets make the training recipe inspectable without proprietary-teacher labels.
- Tracking tests temporal identity, not physical geometry or controllable dynamics.
