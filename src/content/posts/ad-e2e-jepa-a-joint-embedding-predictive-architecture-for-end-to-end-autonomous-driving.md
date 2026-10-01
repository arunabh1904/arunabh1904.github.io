---
title: 'AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving'
date: '2026-09-28T09:00:00.000Z'
section: paper-shorts
postSlug: ad-e2e-jepa-a-joint-embedding-predictive-architecture-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/09/28/ad-e2e-jepa-a-joint-embedding-predictive-architecture-for-end-to-end-autonomous-driving.html
tags: ["Autonomous Driving", "Research"]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving'
---

## 2026 – AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving

**Paper:** [arXiv:2609.34085](https://arxiv.org/abs/2609.34085) · [Full text](https://arxiv.org/html/2609.34085v1)

## Summary

> AD-E2E-JEPA compresses DINOv3 patch features before action-conditioned latent prediction, reducing 256-candidate planning from roughly 92–101 seconds to 0.8 seconds per scene in its sampled NAVSIM comparison. Its strongest 72.9 EPDMS result instead searches 8,192 candidates and takes 18.2 seconds. The zero-shot experiment is given a ground-truth future image as its goal; a separate imitation-learning experiment transfers the projector and improves EPDMS from 80.2 to 85.4. These are different uses of the learned representation.

## Core Insights

### Predict a latent consequence, then search for the action that reaches a goal

This is an action-conditioned visual world model, not a language-conditioned VLA. Four front-camera frames and recent poses provide history. Relative pose changes supply actions, and a predictor forecasts the next visual representation. At planning time, candidate trajectories are rolled through that model; the selected trajectory is the one whose terminal prediction is closest to the supplied goal-image representation.

The distinction from a reactive imitation policy is the role of recorded actions. World-model training conditions on human trajectories to learn transitions, rather than directly teaching a policy to output the human action. “No policy training” therefore does not mean no driving data or no demonstrated trajectories. It describes the learned component and objective.

The source architecture separates world-model training, search, and supervised transfer. The projector is common to the first two and is the component carried into the third.

![Source Figure 2: AD-E2E-JEPA compression, action-conditioned prediction, trajectory search, and downstream transfer](/assets/images/october-2609.34085-s3-f2.webp)
*Fig 1: A shared projector compresses current and target visual features. Planning searches candidate actions through the learned dynamics; a separate experiment transfers the projector to an imitation model. | source: [AD-E2E-JEPA, Figure 2](https://arxiv.org/html/2609.34085v1#S3.F2)*

[Open figure at full resolution](/assets/images/october-2609.34085-s3-f2.webp)

### The projector removes spatial tokens as well as channel width

DINOv3 ViT-L is frozen during world-model learning. The proposed projector uses **two convolutional layers, each with stride 2 × 2**. Together they reduce the number of spatial patch embeddings by 16× and their width from 1,024 to 256. The same learned projection is applied to history and future frames. A linear action embedding conditions an AdaLN-style predictor with rotary position embeddings.

This is not a vision-to-language connector. It defines the compressed state space in which dynamics and planning operate. That distinction makes the anti-collapse objective essential: because the projector is trainable on both sides, predicting a constant could make latent MSE small without preserving useful information.

The target projection is detached in the prediction loss. SIGReg additionally encourages an isotropic Gaussian distribution, applied across the batch independently at each spatial location and time, then averaged. The appendix uses 1,024 random projection directions and a 17-knot approximation over [0, 3]. The goal is to prevent an uninformative compressed state while retaining the computational benefit of fewer, narrower tokens.

### One-step prediction and rollout consistency are different recipes

The base loss predicts the next projected frame. The optional rollout recipe covers eight future frames, a four-second horizon at 2 Hz, using four history frames. It combines teacher-forced prediction with additional losses on autoregressive predictions. After each rollout step, the oldest context leaves and the new predicted state enters; gradients through earlier autoregressive context are stopped on subsequent calls. This is truncated backpropagation, not unrestricted gradient flow through the entire simulated future.

All settings train for 30 epochs with AdamW, one warmup epoch, and cosine decay. The 10-hour navtrain configuration uses one A100, batch 128, learning rate $10^{-4}$, and SIGReg weight 0.09, taking 20 hours without rollout or 46 hours with it. Scaling to 70 hours from the training portion of trainval uses four A100s. The rollout version uses batch 256, learning rate $1.4\times10^{-4}$, and takes 98 hours. Training on more video and adding rollout losses must therefore be separated from the architectural compression effect.

The full video source is sampled at 2 Hz. The downstream imitation appendix specifies four front-camera tensors of 3 × 256 × 512; it adds temporal and spatial embeddings, a driving-command representation, cross-attention, and an MLP trajectory predictor. That model discards the world predictor and fine-tunes the retained encoder/projector with a trajectory MSE objective.

### The zero-shot benchmark supplies the future goal image

Planning searches a vocabulary of 8,192 clustered driving trajectories, subsampling by angular ordering when fewer candidates are used. Each candidate supplies action conditioning to the latent rollout. The target is the **ground-truth frame four seconds ahead**, encoded as the goal. This is a controlled test of goal-conditioned dynamics and selection; an ordinary deployed vehicle does not receive that future observation.

The full benchmark contains 12,146 NAVSIM test scenes. Expensive dense-feature baselines are evaluated on only 100 sampled scenes, while efficient models also run on the full set. The small subset omits extended comfort because neighboring scenes are unavailable. Consequently, its EPDMS is not directly interchangeable with the full-set aggregate.

| Full-set configuration | Candidates | EPDMS ↑ | Final-position error ↓ | Planning time, A100 ↓ |
| --- | ---: | ---: | ---: | ---: |
| LeWM, navtrain | 256 | 39.8 | 14.6 m | 0.7 s |
| AD-E2E-JEPA, navtrain | 256 | 63.5 | 6.3 m | 0.8 s |
| AD-E2E-JEPA, trainval + rollout | 256 | 67.3 | 4.0 m | 0.8 s |
| AD-E2E-JEPA, trainval + rollout | 8,192 | 72.9 | 2.8 m | 18.2 s |

The paper also reports EPDMS†, which removes multiplicative safety terms. It should not replace the safety-inclusive score. Its hit-rate diagnostic appends the ground-truth trajectory to the candidate set and asks whether latent distance ranks it in the top one or five. This tests ranking reliability under a defined candidate pool, rather than the frequency of successful real-world trips.

The roughly 100× speedup is the **256-candidate comparison** against dense DINO-WM/JEPA-WM, not the runtime of the highest-scoring 8,192-candidate variant. More candidates improve endpoint precision while raising latency and lowering the probability that the exact ground-truth candidate remains top-ranked.

### Transfer tests a more directly usable representation

The separate imitation experiment replaces a randomly initialized projector with the pretrained one in the same downstream architecture. EPDMS improves from 80.2 to 85.4. This is the cleaner deployment-facing result because the action model does not need a future goal image. The comparison still concerns a pretrained initialization, not proof that the world model can enforce driving safety during search.

The source distinguishes metrics before and after a NAVSIM human-filter bug fix; the 80.2/85.4 comparison uses the corrected column. Cross-paper rows with the older score should not be silently ranked against it. My next test would hold the imitation budget fixed while varying projector compression and pretraining objectives, then evaluate full closed-loop behavior. That would reveal whether the gain comes from predictive structure, general pretraining, or the particular compressed geometry.

## High-Level Takeaways

- Two strided convolutions reduce both token count and channel width; SIGReg addresses the collapse risk introduced by a trainable target projection.
- Zero-shot planning is conditioned on a future goal image and searches a fixed trajectory vocabulary; it is not ordinary online driving without privileged information.
- The 0.8-second runtime and 72.9 EPDMS belong to different candidate budgets.
- The matched downstream initialization comparison, 80.2 → 85.4 EPDMS, provides a distinct argument for projector pretraining.
- Safety-inclusive metrics, benchmark version, goal availability, and rollout cost must remain visible when deciding whether the representation is useful.
