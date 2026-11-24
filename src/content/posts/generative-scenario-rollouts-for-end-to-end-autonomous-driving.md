---
title: 'Generative Scenario Rollouts for End-to-End Autonomous Driving'
date: '2026-11-18T09:00:00.000Z'
section: paper-shorts
postSlug: generative-scenario-rollouts-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/11/18/generative-scenario-rollouts-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Vision-Language-Action
  - Motion Planning
field: "Autonomous Driving: VLA & Planning"
summary: "2026 – Generative Scenario Rollouts for End-to-End Autonomous Driving"
---

## 2026 – Generative Scenario Rollouts for End-to-End Autonomous Driving

**Paper:** [Yasarla et al., arXiv:2601.11475v2](https://arxiv.org/abs/2601.11475v2)

## Summary

> GeRo trains a vision-language-action planner to predict what its own car and nearby traffic participants will do over several future steps. It rolls forward in a learned token space, not in rendered video. The resulting planner performs better on the authors' Bench2Drive baselines even with the rollout loop switched off at inference. Turning it on improves the reported scores further, but nearly triples the tested ORION variant's latency.

## Core Insights

### Give the planner a state it can advance

A waypoint head trained on an expert trajectory can learn where the ego vehicle should go without learning how nearby agents move over successive steps. GeRo first trains a shared representation to support both tasks. Camera observations and language enter a common embedding space; the backbone produces latent tokens for the ego vehicle and surrounding agents. The ego head predicts waypoints, the motion head predicts agent trajectories with auxiliary box supervision, and the language head works on scene descriptions and visual question answering.

The second stage asks the model to predict the **next** ego and agent tokens, conditioned on a scenario description and an ego-action question. The same heads decode trajectories and the language response. The generated tokens then become inputs for the following step. The standard training setup starts from four frames and rolls out eight steps, with a stride of one to four frames. Eight model steps therefore need not mean eight consecutive video frames.

![Original schematic: starting tokens feed a next-token predictor; its outputs are decoded and passed back for another step.](/assets/images/gero-latent-rollout.svg)
*Fig 1: This original schematic shows how predicted ego and agent tokens become the input for the next step while trajectory and answer heads read the evolving state. | source: [Yasarla et al., Figure 3 and Section 3.4](https://arxiv.org/html/2601.11475v2#S3.F3)*

[Open the diagram at full size](/assets/images/gero-latent-rollout.svg)

### Supervise the loop before scoring it

The model sees its own predicted state during rollout, so a small early error can affect every subsequent prediction. GeRo anchors each step with a KL-divergence penalty against reference latents, plus planning and motion losses. The relevant distinction is between predicting a future from current observations and constructing a training target: the pretrained model can see the corresponding future image sequence when supplying reference tokens or pseudo-labels. Those future observations are not supplied as evidence to the autoregressive prediction.

The authors then use Group Relative Policy Optimization (GRPO). They sample eight ego rollout candidates per scene, forecast surrounding-agent trajectories, and score candidates by collision, time-to-collision, and agreement with the reference language response. A candidate is judged relative to the other candidates in its group. These are training signals over predicted interactions, not certification that a chosen plan will be safe.

### Separate training gains from extra computation at test time

On 220 simulator routes, the Qwen2.5 baseline scores 63.9 on Bench2Drive and succeeds on 31.6% of routes. GeRo with the same backbone scores 79.6 and succeeds on 57.8%. The ORION comparison moves from 77.74/54.62% to 81.90/60.10%. The Qwen ablation is useful here: adding temporal consistency to the pretrained variant raises driving score to 67.8 and success to 40.1%, before the later rollout objectives and GRPO terms. The full gain belongs to the training recipe, not to any single reward.

The paper disables inference-time rollout by default. In its separate latency comparison, enabling rollouts raises GeRo (ORION) from 81.9 to 85.7 driving score and from 60.1% to 63.9% success, while latency rises from 0.306 to 0.940 seconds per frame. Results with the SimLingo controller and geometric waypoint supervision are a different configuration; they should not be folded into the matched ORION comparison.

The controlled nuScenes ablation supplies a separate open-loop check: average displacement error falls from 0.96 to 0.31 with the full Qwen recipe. Transferring the Bench2Drive-trained model without nuScenes training yields 0.45 in that protocol. This is evidence of transfer on logged scenes, not closed-loop performance in real traffic. NAVSIM-v1 likewise uses non-reactive evaluation, where the camera-only Qwen PDMS increases from 84.9 to 89.7.

### Forecasts are not independent decision-makers

GeRo plans actions for the ego vehicle. Other agents are predictions, not separate policies that can strategically respond to that plan. Better benchmark results or more fluent explanations do not show that the rollout distribution covers plausible alternatives for a fixed scene. A useful follow-up would hold the camera observations fixed, vary the ego maneuver, and measure both calibration of agent responses and the safety of the selected ego plan under those alternatives.

## High-Level Takeaways

- Train ego planning, agent motion, and language heads on a shared state before rolling that state forward.
- Distinguish future frames used to form training targets from the observations available to a rollout.
- Report training-only gains separately from the accuracy and latency of optional test-time rollouts.
- Treat predicted agent motion as a forecast, not as evidence of strategic reactions or real-road safety.
