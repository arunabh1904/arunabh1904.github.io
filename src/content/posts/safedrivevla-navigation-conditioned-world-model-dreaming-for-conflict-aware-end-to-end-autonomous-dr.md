---
title: >-
  SafeDriveVLA: Navigation-Conditioned World Model Dreaming for Conflict-Aware
  End-to-End Autonomous Driving
date: '2026-10-14T09:00:00.000Z'
section: paper-shorts
postSlug: >-
  safedrivevla-navigation-conditioned-world-model-dreaming-for-conflict-aware-end-to-end-autonomous-dr
legacyPath: >-
  /paper
  shorts/2026/10/14/safedrivevla-navigation-conditioned-world-model-dreaming-for-conflict-aware-end-to-end-autonomous-dr.html
tags:
  - Autonomous Driving
  - Planning
field: 'Autonomous Driving: VLA & Planning'
summary: >-
  2026 – SafeDriveVLA: Navigation-Conditioned World Model Dreaming for
  Conflict-Aware End-to-End Autonomous Driving
---
## 2026 – SafeDriveVLA: Navigation-Conditioned World Model Dreaming for Conflict-Aware End-to-End Autonomous Driving

**Paper:** [arXiv:2610.20609](https://arxiv.org/abs/2610.20609) · [Full text, v1](https://arxiv.org/html/2610.20609v1)

## Summary

> SafeDriveVLA combines a navigation-conditioned latent world model with a driving-mode prediction that can permit cautious execution or fallback. It reaches 83.0% navigation compliance on the authors' CARLA-F benchmark, versus 55.6% for the strongest compared baseline, and improves conflict-route driving over SimLingo variants. The key ablation separates those effects: mode prediction improves conflict handling, while imagined futures add compliance but slightly reduce conflict-route Driving Score. The evidence comes from constructed simulator benchmarks; the latent rollout is useful context, not a verified safety check.

## Core Insights

### Imagine the requested maneuver before choosing how to execute it

Following “change left” and refusing an unsafe lane change are different capabilities. A policy can appear safe by ignoring instructions, or obedient by taking dangerous actions. SafeDriveVLA supplies an imagined future for the requested maneuver and asks the policy to select among strict execution, cautious execution, and fallback before emitting motion.

How does a language instruction become a future that the driving policy can inspect? In the source figure, the world model operates alongside the current visual observation. Its predicted features become extra language-model tokens rather than rendered images or candidate trajectories scored by an explicit collision checker.

![SafeDriveVLA source Figure 6 showing navigation-conditioned latent dreaming and mode-conditioned action generation](/assets/images/safedrivevla-source-figure-6.webp)
*Fig 1: A frozen latent world model imagines the navigation maneuver. Projected future features join the current observation and instruction, and the VLA predicts a driving mode before its action sequence. | source: [SafeDriveVLA, Figure 6](https://arxiv.org/html/2610.20609v1#S4.F6)*

[View the figure at full resolution](/assets/images/safedrivevla-source-figure-6.webp).

The world model uses a frozen V-JEPA 2 ViT-L encoder. Two 256-square real frames, separated by 0.25 seconds, provide its observed latent state. An eight-layer, eight-head Transformer predictor with width 512 learns future latent features from action and ego-state conditioning. The action input describes displacement and yaw; ego state includes speed, longitudinal and lateral acceleration, and yaw rate. Future ego-state inputs are masked so training cannot depend on information unavailable during rollout.

At inference, a rule-based parser maps the navigation instruction to a maneuver anchor. Most anchors average 20,000 training trajectories per command. Lane-change anchors receive a manually imposed 3.5-metre lateral displacement because averaging the logged examples produces a weak maneuver. Speed requests use the follow-road anchor, so the dreaming pathway does not explicitly simulate the requested target speed. Five autoregressive predictions at half-second intervals cover 2.5 seconds.

A learned projector compresses each predicted $16\times16\times1024$ feature map into one 896-dimensional language token. These five tokens accompany the current front-camera image, represented by 3,072 visual tokens, and the navigation instruction in InternVL3-1B. The model emits a driving-mode token, an action marker, and ten discrete action tokens from a 2,048-entry motion codebook. A separate MLP at the action marker predicts twenty path points. Lateral control follows that path, while integrated action tokens determine target speed. This is a hybrid discrete-action and continuous-path interface.

### The two training stages supervise different things

The world predictor trains for twenty epochs on half of the [SimLingo](</paper shorts/2025/03/12/simlingo-vision-only-closed-loop-autonomous-driving-with-language-action-alignment.html>) data, with normalized latent L1 losses for teacher-forced and autoregressive prediction weighted equally. The autoregressive horizon grows from two to five steps during the first 40% of training. AdamW uses learning rate $4.25\times10^{-4}$, weight decay 0.04, and batch size eight per GPU. V-JEPA remains frozen.

During VLA training, the entire world model stays frozen, including predictor and normalization; the projector into language space learns with the driving policy. InternVL3-1B receives rank-32 LoRA updates with scaling 64 and dropout 0.1. Training uses 407,717 PDM-Lite driving samples from the SimLingo setup, fourteen epochs on eight RTX PRO 6000 Blackwell GPUs, batch size twelve, and AdamW at $3\times10^{-5}$ with weight decay 0.1. Command, waypoint, and language navigation formats are sampled uniformly. The objective adds action-token cross-entropy, mode cross-entropy, path MSE, and text cross-entropy weighted by 0.5.

The modes are derived from expert motion over the following 2.5 seconds, using speed, braking, turning progress, and lateral displacement rules. They are not human annotations of whether an instruction was safe. For example, sustained cruising can label strict execution, while stopping or hard braking can label fallback. Those rules supply a trainable behavioral distinction but can also confuse traffic-induced slowing with refusal. Mode accuracy on a balanced-command, 600-sample audit is 90.2%; left turns reach only 77%, and mode classes remain imbalanced within commands.

### Separate compliance tests from conflict tests

CARLA-F contains 210 routes reconstructed from Bench2Drive starts and towns. The authors remove background traffic and make lights green, then generate feasible command sequences from the road graph. This deliberately isolates instruction following. Lane-change compliance uses lane occupancy near the instruction window's end; turns use distance to the target endpoint. Once an instruction fails, later instructions count as failures. Average navigation compliance pools lateral and turning commands and excludes speed instructions.

B2D-C supplies 150 routes with manually authored instructions that conflict with the scene at selected events. B2D-Adv adds adversarial instruction variants on fifteen routes, paired with clean instructions. NAVSIM-C contains 300 curated unsafe turn or lane-change cases, including road departures and collisions. It diagnoses other policies; SafeDriveVLA itself is not evaluated there. These construction choices matter when interpreting claims of general conflict awareness.

| Evaluation | Comparison | SafeDriveVLA |
| --- | --- | --- |
| CARLA-F average navigation compliance | Best compared baseline: 55.6% | 83.0% |
| B2D-C Driving Score | SimLingo-Safe: 72.8 | 85.9 |
| B2D-C success rate | SimLingo-Safe: 38.0% | 67.3% |
| Standard Bench2Drive Driving Score | SimLingo: 85.07 | 88.16 |

The same-backbone ablation is more informative about mechanism than the headline comparisons. Adding mode prediction raises CARLA-F compliance from 55.1% to 59.8% and B2D-C Driving Score from 79.2 to 87.3. Adding the world model raises compliance to 83.0%, but Driving Score falls to 85.9. Dreaming contributes substantially to obedience; the mode pathway supplies most of this measured conflict-handling gain.

Speed control remains a weakness. CARLA-F speed error increases from 2.62 to 4.05 and then 4.28 across those ablations. That metric excludes runs which never reach the measurement endpoint, so it also needs its completion denominator. SafeDriveVLA is not uniformly best on every command subset, and adversarial instructions still reduce its paired Driving Score from 94.7 to 83.9.

The latent-model probe reports hazard AUC around 0.70–0.74 on 5,095 held-out examples, above frozen-feature baselines around 0.63–0.68. This supports encoding some hazard information, but does not turn the imagined state into a calibrated guarantee about a counterfactual action. A useful next test would combine route completion, compliance, and conflict severity in reactive traffic while independently varying anchor quality. It would show whether the policy uses accurate imagined consequences or mostly learns command-conditioned behavioral cues.

## High-Level Takeaways

- SafeDriveVLA turns a parsed navigation command into five latent future tokens, then conditions mode and action prediction on those tokens.
- The world model learns latent dynamics first and stays frozen during VLA training. Driving modes come from expert-trajectory rules, which limits their interpretation as explicit safety judgments.
- The ablation separates benefits: mode prediction improves conflict handling, while dreaming supplies most of the additional compliance and introduces some score and speed-control regressions.
- CARLA-F removes traffic to isolate obedience; the conflict suites are separately constructed. Their results support further investigation in reactive traffic rather than a broad deployment claim.
