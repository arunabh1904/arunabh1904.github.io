---
title: 'DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving'
date: '2024-11-22T00:00:00.000Z'
section: paper-shorts
postSlug: diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2024/11/22/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving.html
tags: [Other]
field: 'Autonomous Driving: VLA & Planning'
summary: '2024 – DiffusionDrive: real-time multimodal planning with truncated diffusion'
---
## 2024 – DiffusionDrive

**arXiv:** [2411.15139](https://arxiv.org/abs/2411.15139)

**Code:** [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)

### Method and reported result

DiffusionDrive uses diffusion to generate multiple ego trajectories without starting from pure Gaussian noise. It clusters human trajectories into a small anchor vocabulary, adds limited noise around those anchors, and uses a cascade decoder to iteratively condition candidates on scene features. Because the initial proposals already lie near plausible driving modes, inference needs only two denoising steps.

## Summary

> The deployment insight is to truncate both the noise distribution and the denoising horizon. Anchors carry the broad action modes; diffusion refines them instead of discovering the driving manifold from scratch at inference.

## Core Insights

With a matched ResNet-34 TransFuser backbone on NAVSIM, the paper reports 88.1 PDMS. Compared with a vanilla diffusion conversion, truncation reduces denoising from 20 steps to 2, and the complete model reports a six-fold FPS increase while improving planning quality and mode diversity. The paper also reports 45 FPS on an RTX 4090 for its real-time configuration.

| Variant | Denoising steps | Planning implication |
| --- | ---: | --- |
| Vanilla diffusion policy | 20 | Broad generation but high runtime. |
| Truncated diffusion | 2 | Starts from plausible anchored modes. |
| Cascade decoder | 2 per cascade stage | Refines candidates through repeated scene interaction. |

The evaluation is still benchmark-bound. NAVSIM's PDMS scores the selected trajectory in a non-reactive simulation, and the paper notes that top-one PDMS plus a diversity metric cannot fully characterize the quality of the complete candidate distribution.

## High-Level Takeaways

- Diffusion is useful here for multimodality, not because a driving policy needs a long generative chain.
- A learned or clustered trajectory prior can cut most denoising steps while preserving distinct maneuver modes.
- Candidate quality and candidate scoring are separate failure points; diverse trajectories do not help if the ranker selects the wrong one.
- Closed-loop reactive evaluation remains necessary before treating open-loop speed and PDMS as deployment evidence.
