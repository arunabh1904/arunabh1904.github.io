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

### Method and reported result

Driving on Registers introduces DrivoR, a camera-only planner built around a pretrained vision transformer. Camera-aware register tokens compress thousands of multiview image tokens into a small scene representation. One lightweight decoder generates candidate trajectories; another scores them with interpretable safety, comfort, and efficiency components.

## Summary

> DrivoR makes compression task-aware. It does not ask a small token set to reconstruct every geometric fact; it asks the registers to preserve enough scene evidence to generate and rank driving trajectories.

## Core Insights

The architecture separates proposal from evaluation. That matters because multimodal planning needs both a diverse candidate set and a scorer that can reject unsafe or inefficient candidates. At inference, score weights can be changed to alter behavior without retraining the visual backbone.

On the paper's NAVSIM-v2 efficiency comparison, DrivoR reports 48.3 EPDMS with 41 million parameters, 351 GFLOPs, 0.5 GB peak memory, and 110 ms throughput on a single A100. The trajectory heads account for roughly 3 ms; the image backbone still dominates runtime. On NAVSIM-v1, mapping one token to a complete trajectory reports 90.0 PDMS versus 83.9 when a separate token predicts each pose.

| Design | Role |
| --- | --- |
| Camera-aware registers | Compress multiview ViT features. |
| Generation decoder | Produce candidate trajectories. |
| Scoring decoder | Predict interpretable candidate subscores. |
| Runtime score weights | Tune the safety-comfort-efficiency trade-off. |

The paper notes ambiguous attention in some cases, including a traffic-light scene where the scorer focuses on the rear camera. Interpretable subscores expose part of the decision, but they do not guarantee the underlying visual evidence is causally correct.

## High-Level Takeaways

- Compact tokens can be sufficient when the downstream contract is candidate generation and ranking, not universal scene reconstruction.
- Separating generation from scoring exposes a useful control surface for planner behavior.
- Most reported compute remains in the visual encoder, so token compression mainly reduces downstream trajectory evaluation cost.
- Planning scores are only as trustworthy as their training targets and visual grounding; readable labels do not by themselves provide calibrated safety guarantees.
