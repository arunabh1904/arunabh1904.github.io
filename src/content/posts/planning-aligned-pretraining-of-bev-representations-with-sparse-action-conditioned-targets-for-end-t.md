---
title: Planning-Aligned Pretraining of BEV Representations with Sparse Action-Conditioned Targets for End-to-End Autonomous Driving
date: '2026-09-19T09:00:00.000Z'
section: paper-shorts
postSlug: planning-aligned-pretraining-of-bev-representations-with-sparse-action-conditioned-targets-for-end-t
legacyPath: /paper shorts/2026/09/19/planning-aligned-pretraining-of-bev-representations-with-sparse-action-conditioned-targets-for-end-t.html
tags:
- Autonomous Driving
- Research
field: 'Autonomous Driving: VLA & Planning'
summary: 2026 – Planning-Aligned Pretraining of BEV Representations with Sparse Action-Conditioned Targets for End-to-End Autonomous Driving
---

## 2026 – Planning-Aligned Pretraining of BEV Representations with Sparse Action-Conditioned Targets for End-to-End Autonomous Driving

**Paper:** [arXiv:2609.22868](https://arxiv.org/abs/2609.22868) · [Full text and technical supplement](https://arxiv.org/html/2609.22868v1) · [PAVER project](https://archiiive99.github.io/PAVER)

## Summary

> PAVER pretrains a camera BEV encoder to predict occupied and unobserved evidence along candidate ego motions, using a temporary 10,258-parameter head and targets constructed from one LiDAR sweep. VAD-Tiny's average nuScenes collision rate falls from 0.513% to 0.187% after transfer, but VAD-Base's collision rate increases. The contribution is a compact, planning-indexed pretraining target, with camera-only downstream inference; it is not a learned collision-probability model or a guarantee of safer driving across architectures.

## Core Insights

### A small target can ask a spatially demanding question

PAVER asks what evidence lies across the vehicle's width at positions visited by rule-based candidate motions. It does not reconstruct RGB, a dense semantic occupancy volume, or a future video. Candidate acceleration and yaw-rate combinations, perturbed during training, generate trajectories from the current ego speed at 0.5 s intervals. These motions identify where supervision is needed; they are not expert action labels.

A synchronized LiDAR sweep is rasterized into free, occupied, and unknown evidence. Ray interiors mark free cells; return endpoints mark occupied cells; untouched cells remain unknown. Five lateral samples across the vehicle footprint query that raster. The fractions of valid samples that are occupied or unknown become two soft targets. Out-of-range queries are discarded rather than clamped to a boundary.

Consider a candidate crossing a partially observed road edge. Occupied samples indicate measured geometry in its corridor; unknown samples indicate missing ray evidence. Neither target predicts how another actor will move. In particular, “unknown” is not a calibrated confidence estimate, and “risk” is not a future collision probability.

The source diagram shows the complete training path. LiDAR constructs targets, cameras build the BEV representation, and a temporary prediction branch connects them.

![Source Figure 4: PAVER LiDAR evidence targets, action-corridor masking, and prediction head](/assets/images/october-2609.22868-sx2-f4.webp)
*Fig 1: Candidate motions index sparse LiDAR evidence. A masked camera-BEV branch predicts two target ratios, and the entire auxiliary branch is removed after pretraining. | source: [PAVER, Figure 4](https://arxiv.org/html/2609.22868v1#Sx2.F4)*

[Open figure at full resolution](/assets/images/october-2609.22868-sx2-f4.webp)

### The projector is a pointwise readout, not a larger scene decoder

Six synchronized surround-view images pass through ResNet-50, FPN, and a BEV encoder, producing a 256-channel, 100 × 100 representation. The reported BEV range is 30 m by 60 m. Pretraining uses one frame and disables downstream detection, mapping, motion, and planning decoders.

In a temporary copy of the BEV, cells touched by the action corridor are replaced by a shared learned mask token. A **1 × 1 projector** transforms the features without spatial aggregation. The head bilinearly samples at the candidate action center, concatenates the ego-frame state $[x,y,\psi,v]$, and predicts risk and unknown logits through an MLP. Coordinate conversion matters: raster lookup uses the calibrated BEV/sensor frame, while the concatenated action state remains ego-relative.

The objective is risk BCE plus **0.5 times unknown BCE**, averaged over valid entries. The 10,258 auxiliary parameters include the mask token, projector, and readout. The encoder retains the useful representation; auxiliary head size is not model size. A 30K variant widens the readout, while a 90K variant uses a 3 × 3 projector. Larger heads help some metrics but do not give the lowest collision rate.

Masking also has a precise limitation. It removes selected local feature vectors after the BEV has been encoded; it does not erase information previously mixed into other features. The paper therefore tests input dependence rather than claiming the mask guarantees a particular reasoning strategy.

### Transfer has a reproducible data and optimizer contract

The experiment uses 28,130 nuScenes training keyframes and 6,019 validation keyframes. LiDAR is privileged **training-target** information; no task annotations or learned LiDAR teacher are required for pretraining. All transferred backbone, neck, and BEV parameters are optimized.

| Component of the recipe | Reported setting |
| --- | --- |
| Pretraining | 20 epochs; 4 GPUs × 4 samples per GPU |
| Optimizer | AdamW, learning rate $5\times10^{-5}$, weight decay 0.01 |
| Backbone rate | 0.2 multiplier |
| Schedule | 500-step linear warmup, cosine decay, minimum LR ratio $10^{-3}$ |
| Gradient clipping | Norm 35 |
| Downstream setup | Fresh task decoders, optimizer, and scheduler; learning rate $2\times10^{-4}$; three-frame queue |

After pretraining, the target builder, action sampler, mask token, projector, and prediction MLP are discarded. The image backbone, FPN, BEV queries/positions, camera and level embeddings, CAN-bus MLP, and BEV Transformer initialization transfer. There is no PAVER-specific inference module.

Twenty pretraining plus thirty downstream epochs give estimated VAD-Tiny training time of 13.6 hours versus 21.3 hours for sixty scratch epochs. These totals are estimated from mean iteration times on four RTX 5090 GPUs, not a universal speedup across hardware or training recipes.

### The counterexamples are part of the result

| nuScenes model | Average planning L2, scratch → PAVER | Average collision %, scratch → PAVER |
| --- | ---: | ---: |
| VAD-Tiny | 0.66 → 0.60 m | 0.51 → 0.19 |
| VAD-Base | 0.74 → 0.56 m | 0.31 → 0.40 |
| GenAD | 0.59 → 0.54 m | 0.37 → 0.21 |

These are averages over one-, two-, and three-second horizons. VAD-Base demonstrates why better trajectory distance cannot stand in for fewer collisions. GenAD's map mAP also decreases. Likewise, pseudo-LiDAR from frozen UniK3D gives lower VAD-Tiny L2 than measured LiDAR, but collision rises to 0.560%, compared with 0.513% for scratch and 0.187% for measured-LiDAR PAVER. Target quality changes the meaning of the transfer gain.

The component ablation is unusually informative: masking alone yields 0.523% collision and action conditioning alone 0.550%; combining them reaches 0.187%. The conjunction matters. Frozen-predictor tests on 192 validation examples additionally shuffle scenes, spatial positions, or action states; all increase both prediction losses. Yet an action/time prior explains substantial target variation, with residual $R^2$ below 0.05. The evidence supports measurable scene dependence without implying the auxiliary task is free of shortcuts.

Only 2.66% of BEV cells receive direct sparse targets, but frozen probes improve outside the sampled corridor. Shared camera and BEV parameters offer a plausible mechanism for that broader transfer. Probe gains establish changed representations, not direct semantic supervision of every region.

### Closed-loop transfer remains a small experiment

On nine Bench2Drive Town05 Long routes, with one repetition per route, UniAD-Tiny's driving score increases from 48.45 to 58.79 and route completion from 60.96 to 79.06. Its infraction score decreases from 0.85 to 0.79. Three previous timeouts become completions, which explains part of the aggregate improvement without making every safety component better.

The supplement explicitly states that each configuration is trained once. Scene-bootstrap intervals measure variation across sampled scenes, not independent training seeds. My decision would be to test this inexpensive pretraining branch before adopting a large reconstruction decoder, but retain per-model collision checks and repeated closed-loop trials as the acceptance gate. Sparse current geometry cannot resolve occluded intent or future dynamics by itself.

## High-Level Takeaways

- PAVER turns one LiDAR sweep into sparse occupied/unknown ratios along rule-based motions; the candidate actions select supervision locations rather than supply imitation targets.
- The pointwise projector and 10K head exist only during pretraining. The complete camera BEV initialization is what transfers.
- Masking and action conditioning work together in the reported ablation; neither alone reproduces the collision improvement.
- VAD-Base and pseudo-LiDAR results show why lower L2 is insufficient evidence of lower collision risk.
- The next decision-changing evidence is repeated, architecture-specific closed-loop testing under difficult geometry and agent dynamics, not a larger auxiliary head alone.
