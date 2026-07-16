---
title: 'Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)'
date: '2025-02-27T09:00:00.000Z'
section: paper-shorts
postSlug: openvla-oft-optimizing-speed-and-success
legacyPath: /paper shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html
tags:
  - Robotics
  - Fine-Tuning
field: 'Vision-Language-Action & Robotics'
summary: 2025 – OpenVLA-OFT replaces slow autoregressive action tokens with fast continuous parallel chunks.
---

## 2025 – Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**arXiv:** [2502.19645](https://arxiv.org/abs/2502.19645)

**Project:** [openvla-oft.github.io](https://openvla-oft.github.io/)

OpenVLA-OFT shows that the fine-tuning interface can matter more than preserving a VLA's pretraining objective. It replaces autoregressive discrete action-token decoding with parallel continuous action chunks and trains them with a simple L1 regression loss.

## Paper Insights

The paper ablates three coupled decisions: serial versus parallel decoding, discrete versus continuous actions, and next-token versus regression/diffusion objectives. The resulting recipe raises OpenVLA's reported LIBERO average from 76.5% to 97.1% and increases action-generation throughput by 26×. In real ALOHA evaluations it supports higher-frequency bimanual control and beats the compared default VLA recipes and from-scratch imitation policies.

The surprising result is that a simple L1 head can match diffusion fine-tuning in the studied setting. Pretrained semantics remain useful even when the action interface and training loss change completely.

| Adaptation choice | OFT selection | Reason |
| --- | --- | --- |
| Decoding | Parallel action chunks | Removes sequential token latency. |
| Representation | Continuous actions | Avoids quantization error. |
| Objective | L1 regression | Fast convergence and inference in the tested tasks. |

## Decision Lens

OpenVLA-OFT informs which parts of a pretrained VLA should be treated as reusable semantics and which should be replaced for deployment. Its unit is an observation paired with a continuous action chunk. The VLM backbone is shared; the action head abandons the language-token interface.

The results establish a strong speed–success recipe on LIBERO and ALOHA, not universal superiority of L1 regression. A missing stress test varies multimodality, perturbation frequency, and chunk length at equal control rate. At ten times the behavioral ambiguity, L1 can average valid modes. The recipe would fail if a diffusion or flow head wins consistently once tasks require multiple precise strategies rather than one dominant trajectory.

**Context:** OpenVLA-OFT is the practical SFT baseline that later RL post-training papers improve.

**Limits:** Near-saturated LIBERO success leaves little room to measure robustness and recovery.

**Takeaway:** Reuse the representation, not necessarily the pretraining decoder; control latency can justify a different action objective.
