---
title: "LaPla: Continuous Actions from Discrete Minds"
date: "2026-09-03T00:00:00.000Z"
section: paper-shorts
postSlug: lapla-latent-aligned-planning-for-autonomous-driving
legacyPath: /paper shorts/2026/09/03/lapla-latent-aligned-planning-for-autonomous-driving.html
tags: ["Action Representations", "Autonomous Driving"]
field: "Autonomous Driving: VLA & Planning"
summary: "2026 – LaPla: Continuous Actions from Discrete Minds"
---

## 2026 – LaPla: Continuous Actions from Discrete Minds

**Paper:** [arXiv:2609.04070](https://arxiv.org/abs/2609.04070) · [PDF](https://arxiv.org/pdf/2609.04070)

## Summary

> LaPla predicts continuous action latents through a frozen residual VQ-VAE decoder, avoiding sequential action-token generation. In sixty AlpaSim scenarios it reaches 61.67% success versus 28.33% for action-only AutoVLA, but fault collisions rise from 5.00% to 11.67%. Its result is faster, more successful progress with a collision trade-off, not an unconditional safety improvement.

## Core Insights

### Use quantization to learn a prior, then bypass it for future actions

LaPla first trains a residual VQ-VAE on trajectory features containing local displacement and heading. Residual codebooks capture coarse structure and successive corrections. The decoder is then frozen as a learned motion prior. During policy training, an Emu3 backbone receives multi-view images, historical actions, and navigation instructions; learned action queries predict continuous latents that the decoder converts into trajectory features.

Historical actions and future actions deliberately use different interfaces. History passes through the frozen encoder and quantizer before projection into the backbone. Future latents are continuous outputs from the action queries, bypassing discrete codebook selection. This removes output quantization and autoregressive action decoding, while retaining the decoder's learned bias toward the training trajectories.

The overview shows the boundary between learning the trajectory representation and learning the policy. The action loss can differentiate through the decoder even though its weights remain fixed.

![Residual VQ-VAE pretraining followed by continuous latent action prediction; source Figure 2](/assets/images/lapla-source-figure-2.webp)
*Fig 1: LaPla freezes a pretrained trajectory decoder and learns continuous latents that produce the desired waypoints. The decoder supplies a learned motion prior rather than a formal dynamics constraint. | source: [Paper, Figure 2](https://arxiv.org/abs/2609.04070)*

[View full-size figure](/assets/images/lapla-source-figure-2.webp)

The next figure isolates the asymmetry: discrete historical features summarize observed motion, while continuous future features preserve output precision.

![Discrete historical action representation and continuous future latent representation; source Figure 3](/assets/images/lapla-source-figure-3.webp)
*Fig 2: Historical motion uses quantized latent tokens, while future motion bypasses discrete selection. This avoids output quantization without discarding the pretrained decoder. | source: [Paper, Figure 3](https://arxiv.org/abs/2609.04070)*

[View full-size figure](/assets/images/lapla-source-figure-3.webp)

A Smooth L1 waypoint loss trains the future projection through the decoder. No separate latent-matching target is required. VQA answers are appended after the action queries during training and receive an auxiliary language loss. LoRA adapts the backbone; action queries and projection parameters are trainable, while the VQ-VAE remains frozen. Relative to [SpatialVLA](/paper%20shorts/2025/01/27/spatialvla-exploring-spatial-representations.html), this changes where discretization is paid: SpatialVLA predicts action-grid tokens, while LaPla keeps a discrete representation for history and predicts continuous future latents.

### Separate oracle open-loop selection from closed-loop execution

On nuScenes, the paper uses twenty trajectory candidates and an oracle best-of-N selector. The reported 0.60 m average UniAD-style L2 error is therefore not a single-trajectory deployment result. It also uses a different aggregation convention from the ST-P3 columns. Baseline numbers are quoted from earlier work. LaPla's average collision metric of 0.52% in the UniAD convention is worse than the listed OpenDriveVLA-7B value of 0.25%, despite its lower L2 error.

The ablation supports freezing the prior: average UniAD-style L2 falls from 1.13 m with a trainable latent decoder to 0.60 m with the frozen decoder. Removing VQA increases it to 0.63 m, a much smaller change. These comparisons suggest the action representation and its preservation account for more of the observed gain than the auxiliary question-answering task.

Closed-loop evaluation adds 20,000 frames of fine-tuning and tests sixty scenarios. It removes the perturbations and best-of-N procedure for deterministic execution. The planner runs on Ascend 910B3 hardware and communicates with AlpaSim over gRPC; the simulation uses four RTX 4090 GPUs.

| AlpaSim system | Fault collision | Route completion | Success | Inference FPS |
| --- | ---: | ---: | ---: | ---: |
| AutoVLA, action only | 5.00% | 33.33% | 28.33% | 0.532 |
| AutoVLA, official CoT checkpoint | 1.67% | 15.00% | 15.00% | 0.215 |
| LaPla | 11.67% | 71.67% | 61.67% | 1.582 |

The authors attribute the higher collision rate to more proactive driving. That explanation does not remove the regression. The model makes more progress and succeeds more often, while also causing more fault collisions. A smooth learned decoder does not guarantee feasible or safe behavior outside its training support.

## High-Level Takeaways

- A frozen trajectory decoder can supply a reusable action prior while allowing continuous future predictions.
- Report oracle best-of-N evaluation separately from deterministic closed-loop execution; they answer different planning questions.
- Evaluate collisions alongside progress and latency. LaPla's strongest deployment result includes a material collision-rate regression.
