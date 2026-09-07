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

## Summary

> DiffusionDrive uses diffusion to generate multiple ego trajectories without starting from pure Gaussian noise. It clusters human trajectories into a small anchor vocabulary, adds limited noise around those anchors, and uses a cascade decoder to iteratively condition candidates on scene features. Because the initial proposals already lie near plausible driving modes, inference needs only two denoising steps. The deployment insight is to truncate both the noise distribution and the denoising horizon. Anchors carry the broad action modes; diffusion refines them instead of discovering the driving manifold from scratch at inference.

## Core Insights

### Anchors carry the maneuver modes before denoising

With a matched ResNet-34 TransFuser backbone on NAVSIM, the paper reports 88.1 PDMS. Compared with a vanilla diffusion conversion, truncation reduces denoising from 20 steps to 2, and the complete model reports a six-fold FPS increase while improving planning quality and mode diversity. The paper also reports 45 FPS on an RTX 4090 for its real-time configuration.

The anchored prior is learned from the demonstrated waypoint distribution. During training, each of 20 anchor trajectories is perturbed with Gaussian noise and the decoder learns both a reconstructed trajectory and a confidence score; at inference, the number of samples can be changed independently of the 20 training anchors. Each cascade layer first samples BEV or perspective features at the current trajectory coordinates, then attends to agent/map queries and predicts an offset and score. The highest-scoring candidate is selected only after this interaction, so multimodality and ranking are separate learned problems.

The matched roadmap isolates where the gain comes from. Vanilla DDIM keeps 20 steps and reaches 84.6 PDMS at 7 FPS; truncated diffusion reaches 85.7 at 27 FPS; adding the cascade decoder reaches 88.1 at 45 FPS, with 2 steps taking 7.6 ms. The design ablation rises from 85.1 with agent/map attention alone to 87.4 with both, then 88.1 after cascading. Six candidate modes are enough to saturate the reported score, while simply increasing samples from 10 to 20 gives the main jump and 40 adds little. This is why the paper’s speedup comes from a better starting distribution and shared refinement, not from making diffusion itself free.

The important distinction is between diversity in the initial samples and diversity created by denoising. Pure Gaussian starts tend to collapse onto similar trajectories after the planner sees the same scene; anchored noise begins near several demonstrated driving patterns, so two denoising steps can preserve a straight path and a lane-change alternative at the same time. The cascade decoder then re-reads scene features at the current waypoint locations, making refinement spatially selective rather than a blind update of the whole trajectory.

The architecture begins with noisy anchors, then samples scene features at their current coordinates before updating and scoring them. That spatial feedback is why a second refinement can be useful even when the initial maneuver is already plausible: a lane-change anchor still needs to fit this scene’s agents and boundaries.

![DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving source figure: Overall architecture of DiffusionDrive.](/assets/images/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving-paper-figure.webp)
*Fig 1: DiffusionDrive samples noisy trajectories from an anchored Gaussian prior, conditions iterative denoising on perception, agents, and maps, and scores diverse trajectory proposals at each decoder step. | source: [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving, Figure 4](https://arxiv.org/abs/2411.15139)*

The qualitative panel separates the selected path from the other candidates. Several plausible maneuvers can survive the denoising process, while a confidence score decides which one becomes the ego plan. Diversity is therefore only useful if ranking preserves the safest appropriate choice.

![Figure 2 from DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](/assets/images/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving-source-figure-2.webp)
*Fig 2: The panel shows several trajectory candidates over the same scene; the selected high-score path continues straight while other candidates preserve lane-change alternatives. The visual supports the claim that the decoder ranks a distribution rather than emitting one deterministic curve. | source: [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving, Figure 2](https://arxiv.org/abs/2411.15139)*

The paradigm comparison locates the trade-off. Regression commits to one path; a fixed vocabulary offers several paths with limited geometric freedom; full diffusion learns a flexible distribution through a long chain. Anchored diffusion retains a maneuver vocabulary while allowing scene-dependent refinement around each mode.

![Figure 1 from DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](/assets/images/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 3: The paradigms progress from single-mode regression to discrete vocabulary sampling, full-step diffusion, and DiffusionDrive’s truncated diffusion policy. | source: [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving, Figure 1](https://arxiv.org/abs/2411.15139)*


The evaluation is still benchmark-bound. NAVSIM's PDMS scores the selected trajectory in a non-reactive simulation, and the paper notes that top-one PDMS plus a diversity metric cannot fully characterize the quality of the complete candidate distribution.

## High-Level Takeaways

- Diffusion is useful here for multimodality, not because a driving policy needs a long generative chain.
- A learned or clustered trajectory prior can cut most denoising steps while preserving distinct maneuver modes.
- Candidate quality and candidate scoring are separate failure points; diverse trajectories do not help if the ranker selects the wrong one.
- Closed-loop reactive evaluation remains necessary before treating open-loop speed and PDMS as deployment evidence.
