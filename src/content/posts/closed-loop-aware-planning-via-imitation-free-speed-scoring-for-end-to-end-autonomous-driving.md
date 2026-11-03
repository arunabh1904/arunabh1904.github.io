---
title: >-
  Closed-Loop Aware Planning via Imitation-Free Speed Scoring for End-to-End
  Autonomous Driving
date: '2026-10-26T09:00:00.000Z'
section: paper-shorts
postSlug: >-
  closed-loop-aware-planning-via-imitation-free-speed-scoring-for-end-to-end-autonomous-driving
legacyPath: >-
  /paper
  shorts/2026/10/26/closed-loop-aware-planning-via-imitation-free-speed-scoring-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Planning
field: 'Autonomous Driving: VLA & Planning'
summary: >-
  2026 – Closed-Loop Aware Planning via Imitation-Free Speed Scoring for
  End-to-End Autonomous Driving
---
## 2026 – Closed-Loop Aware Planning via Imitation-Free Speed Scoring for End-to-End Autonomous Driving

**Paper:** [arXiv:2610.36079](https://arxiv.org/abs/2610.36079) · [Full text, v1](https://arxiv.org/html/2610.36079v1) · [Project](https://clap-ad.github.io/)

## Summary

> CLAP-AD learns a driving path from demonstrations, then scores several speeds against collision, traffic-light, and stop-sign constraints. This lets a controller change its target speed without retraining the policy. On the 220 Bench2Drive routes, it raises UniAD's Driving Score from 45.81 to 74.29 and VAD's from 42.35 to 75.04, but lowers the stronger TFv6 baseline from 94.70 to 92.30. The value is explicit speed control and better supervision where expert speed choices are ambiguous, with a measurable trade-off against strong imitation.

## Core Insights

### One path can admit several acceptable speeds

A demonstrated trajectory specifies both where to drive and how quickly to get there. Those choices need not have the same ambiguity. On a lane-constrained path, several speeds may be acceptable even though the dataset records only one. CLAP-AD retains imitation for the spatial path and replaces speed imitation with independent scores for seven candidates: 0, 2.5, 5, 7, 8.5, 10, and 12 m/s.

For a candidate speed, the collision target uses logged future actor motion and the minimum swept bounding-box distance along the predicted ego path. Distances below 0.1 m receive zero; distances above 1 m receive one, with interpolation between them. Traffic-light and stop-sign targets instead depend on required braking, $a_{\mathrm{req}}=v^2/(2d_{\mathrm{stop}})$. Speeds permitting comfortable braking score well; speeds demanding more than the maximum deceleration score zero. Logged futures and rule-derived labels supervise training. At inference, the network predicts scores from observed scene features.

The method therefore depends on a distinction between supervision and guarantees. A predicted collision score is a learned estimate of a rule-derived target. It is not an online collision checker with access to the future.

### Give each criterion the evidence it needs

The planning head first combines navigation and scene features to predict a path. [UniAD](</paper shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html>) and VAD use a GRU path decoder; TFv6 retains its compatible linear decoder and three target points. Separate speed branches then attend to dynamic-agent features, traffic-light features, or stop-sign features. Each branch has its own path encoder. The path is detached before entering those branches so speed losses do not pull the path predictor away from its imitation objective.

The architecture is easiest to follow by tracing the path once, then watching it enter three different scoring branches.

![CLAP-AD source Figure 2 showing the ego-path predictor and separate collision, traffic-light, and stop-sign scoring branches](/assets/images/clap-ad-source-figure-2.webp)
*Fig 1: A shared predicted path conditions criterion-specific speed heads. Each branch receives the scene evidence needed for its own target instead of relying on one collapsed planning query. | source: [CLAP-AD, Figure 2](https://arxiv.org/html/2610.36079v1#S3.F2)*

[View the figure at full resolution](/assets/images/clap-ad-source-figure-2.webp).

The controller accepts a speed only when every relevant head clears a threshold. It can choose the median acceptable speed, the maximum, or the one nearest a user request. If none qualifies, it selects zero. Separate lateral and longitudinal PID controllers convert path and speed into steering, throttle, and braking. This moves the preference decision outside the learned head while retaining its constraint estimates.

Training uses different upstream systems rather than a common language backbone. UniAD and VAD supply object and motion queries; camera–LiDAR TFv6 supplies a dense BEV map pooled into branch inputs. UniAD additionally needs traffic-light color and applicability supervision, including a separately trained, then frozen classifier. That adaptation matters when attributing gains to the head alone.

UniAD and VAD start from official Bench2Drive perception checkpoints and receive one epoch of end-to-end fine-tuning. TFv6 trains perception first and then the complete model. Appendix A specifies AdamW, learning rate $2\times10^{-4}$, batch size 8, weight decay 0.01, clipping at 35, and cosine scheduling with 500 warmup iterations. UniAD/VAD use a 12 m path and one-second collision horizon; TFv6 uses 24 m and two seconds. These are backbone-specific interfaces, not a new universal camera-resolution or frame-history contract.

### Separate controllability from driving quality

Bench2Drive training uses 1,000 expert clips across 44 CARLA scenarios; its main closed-loop evaluation uses 220 routes. Bench2Drive-Speed is a separate 48-route suite that supplies desired speeds and measures Speed-Adherence alongside driving quality. A baseline that cannot accept requests can still obtain an adherence score by coincidence, so that score alone does not establish controllability.

On Bench2Drive-Speed, adding CLAP-AD to TFv6 improves adherence from 69.64 to 82.53 while Driving Score falls from 92.59 to 88.47. On the separate 220-route suite, the TFv6 loss is smaller, 94.70 to 92.30. These are different route sets; mixing the two deltas would conceal the trade-off.

The controlled Dev10 ablation is especially useful. Changing UniAD's speed objective to multi-criteria scoring within its original head reduces Driving Score from 45.12 to 40.50. Giving the same objective CLAP-AD's criterion-specific head raises it to 59.44. Path augmentation and horizon changes then raise it to 72.71, followed by 75.19 with separate path encoders. The objective needs a representation that can route its evidence. It is not a drop-in loss improvement.

NAVSIM v2 supplies a different test: real logged observations and synthetic perturbed viewpoints, rather than fully reactive traffic rollouts. With an LTFv6 backbone and only the collision branch, CLAP-AD raises navhard EPDMS from 31.9 to 38.0. The four-second trajectory is reconstructed from a confidence-weighted speed and bounded acceleration. Collision-related scores improve, while progress and extended comfort decline. This supports a collision–progress trade-off on that protocol, not a general safety result.

The unresolved decision is how much rule-derived supervision to substitute for expert behavior. The method covers three criteria and urban speed candidates; it does not include highway speed limits, and complicated maneuvers may resist a simple scoring rule. A matched-backbone study with the same expert data, auxiliary supervision, and repeated closed-loop routes would distinguish the benefit of explicit constraints from data quality and perception changes. The TFv6 regression already shows why that control matters.

## High-Level Takeaways

- CLAP-AD separates path imitation from speed scoring, allowing user preferences to change at inference without retraining.
- Its training unit is a path-conditioned candidate speed with rule-derived targets; future actor trajectories are supervision, not inference inputs.
- Dedicated evidence branches are essential: the scoring objective alone hurts the original UniAD head in the reported ablation.
- Gains depend on the backbone and evaluation protocol. Stronger speed adherence can cost Driving Score, progress, or comfort, and learned scores do not guarantee safe execution.
