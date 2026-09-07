---
title: 'Driving on Registers'
date: '2026-01-08T00:00:00.000Z'
section: paper-shorts
postSlug: driving-on-registers
legacyPath: /paper shorts/2026/01/08/driving-on-registers.html
tags: [Other]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – Driving on Registers: compact camera-aware tokens for end-to-end planning'
---
## 2026 – Driving on Registers

**arXiv:** [2601.05083](https://arxiv.org/abs/2601.05083)

## Summary

> Driving on Registers introduces DrivoR, a camera-only planner built around a pretrained vision transformer. Camera-aware register tokens compress thousands of multiview image tokens into a small scene representation. One lightweight decoder generates candidate trajectories; another scores them with interpretable safety, comfort, and efficiency components. DrivoR makes compression task-aware. It does not ask a small token set to reconstruct every geometric fact; it asks the registers to preserve enough scene evidence to generate and rank driving trajectories.

## Core Insights

### Registers compress the scene for proposal and scoring

The architecture separates proposal from evaluation. That matters because multimodal planning needs both a diverse candidate set and a scorer that can reject unsafe or inefficient candidates. At inference, score weights can be changed to alter behavior without retraining the visual backbone. DrivoR adds 16 camera-specific registers to each of four camera views, retrieves those tokens from a pretrained DINOv2 ViT, and passes the resulting 64 scene tokens to lightweight trajectory and scoring decoders. The scorer receives detached trajectory embeddings, so it cannot backpropagate through the proposal details or quietly reuse the generator's latent shortcut.

The first figure is a diagnostic of the compression itself. Darker entries mean lower cosine similarity: front-camera registers separate into distinct scene roles, while back-camera registers tend to collapse. That pattern is a useful warning against counting tokens alone; the registers have to specialize to preserve planning evidence.


![Figure 4 from Driving on Registers](/assets/images/driving-on-registers-source-figure-4.webp)
*Fig 1: Cosine similarity between scene tokens. Darker indicates lower cosine similarity. | paper Figure 4; source: [Driving on Registers](https://arxiv.org/abs/2601.05083)*

The architecture separates proposing a path from judging it. Learned queries generate multiple trajectories, but the scorer receives a fresh, detached embedding of each candidate. This makes the scoring task depend on the proposed geometry and scene evidence rather than a hidden generator state. Runtime weights then choose how the predicted safety, comfort, and efficiency components affect selection.

![Figure 1 from Driving on Registers](/assets/images/driving-on-registers-source-figure-1.webp)
*Fig 2: DrivoR architecture. The proposed architecture is composed of three transformer blocks: one encoder (perception) and two decoders (trajectory and scoring). | paper Figure 1; source: [Driving on Registers](https://arxiv.org/abs/2601.05083)*


The compression ablations make the efficiency claim causal. Register-based compression reaches 90.0 PDMS on NAVSIM-v1 with 64 scene tokens, close to the 90.2 of a model that consumes roughly 16,000 tokens, while adding only 0.6M parameters. A pooling baseline reaches 89.7, and a transformer decoder with the same 64-token budget reaches 89.3. Pretraining matters even more: the pretrained ViT is over 15 PDMS points better than the same design trained from scratch. Increasing trajectory proposals from 1 to 64 raises PDMS from 80.1 to 90.0, after which the curve saturates.

The reported deployment numbers expose what is and is not compressed. On NAVSIM-v2, DrivoR reaches 48.3 EPDMS with 41M parameters, 351 GFLOPs, 0.5 GB peak memory, and 110 ms throughput on an A100; the visual backbone still dominates runtime. On the HUGSIM test, the no-fine-tuning model obtains 49.8 road completion and 35.7 HD-Score, so the simulator result should not be read as real-world driving. Runtime score weights are a useful control surface, but the paper also shows a traffic-light case where the scorer attends to the rear camera, reminding us that interpretable subscores do not prove causal grounding.

| Design | Role |
| --- | --- |
| Camera-aware registers | Compress multiview ViT features. |
| Generation decoder | Produce candidate trajectories. |
| Scoring decoder | Predict interpretable candidate subscores. |
| Runtime score weights | Tune the safety-comfort-efficiency trade-off. |

The compression is downstream compression, not a claim that the camera encoder has stopped seeing detail. The registers learn a small task-specific bottleneck and the trajectory heads do the cheap reasoning afterward; the A100 profile still assigns most latency to the image backbone. This distinction matters when transferring the idea to a larger VLM: fewer planning tokens can reduce decoder cost while leaving visual encoding and grounding as the dominant bottleneck.

## High-Level Takeaways

- Camera-specific registers preserve planning-relevant evidence with roughly 64 scene tokens, approaching dense-feature performance while reducing downstream compute.
- Detached trajectory re-embedding separates proposal diversity from score learning and permits runtime safety/comfort/efficiency reweighting.
- Pretraining and enough candidate trajectories matter: the reported PDMS rises sharply over scratch/pooling baselines and saturates near 64 proposals.
- The A100 and HUGSIM measurements leave backbone latency, score calibration, and reactive physical driving as open deployment tests.
