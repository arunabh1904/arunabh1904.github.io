---
title: "BEVDriver: Leveraging BEV Maps in LLMs for Closed-Loop Driving"
date: "2025-03-05T00:00:00.000Z"
section: paper-shorts
postSlug: bevdriver-bev-maps-for-closed-loop-driving
legacyPath: /paper shorts/2025/03/05/bevdriver-bev-maps-for-closed-loop-driving.html
tags: ["BEV", "Autonomous Driving"]
field: "Autonomous Driving: VLA & Planning"
summary: "2025 – BEVDriver: Leveraging BEV Maps in LLMs for Closed-Loop Driving"
---

## 2025 – BEVDriver: Leveraging BEV Maps in LLMs for Closed-Loop Driving

**Paper:** [arXiv:2503.03074](https://arxiv.org/abs/2503.03074) · [PDF](https://arxiv.org/pdf/2503.03074)

## Summary

> BEVDriver feeds camera–LiDAR BEV features through a Q-Former into a language model and predicts waypoints for CARLA driving. Its Llama-7B variant reaches 48.9 Driving Score on LangAuto versus 31.3 for LMDrive with the same named backbone. The result supports BEV-conditioned planning, while distance-based instruction failures and a harmful encoder-unfreezing ablation expose what the representation does not solve.

## Core Insights

### Predict waypoints from latent scene features

BEVDriver adapts an InterFuser-style encoder to four camera views and LiDAR. Detection, traffic-light classification, semantic segmentation, and temporal contrastive alignment supervise perception. The deployment path discards the perception decoder and passes latent BEV features onward. Unlike LMDrive's use of pre-predicted waypoint tokens, the language-model stage receives features without that trajectory-specific preprocessing.

A Q-Former with 32 learned queries aligns the features with navigation instructions. LoRA adapts the language backbone; a GRU head reads its final hidden state and predicts five future waypoints. An MLP predicts instruction completion, and PID controllers convert waypoints into steering, throttle, and braking. This distinction matters: the controller does not execute free-form language directly.

Read the architecture through the final waypoint head and PID controller. Those components define the action interface, while the Q-Former defines how the planner accesses spatial perception.

![BEV perception, Q-Former, language backbone, waypoint head, and PID controller; source Figure 2](/assets/images/bevdriver-source-figure-2.webp)
*Fig 1: Camera and LiDAR features enter a shared BEV representation before language alignment. Learned waypoint and completion heads translate the language model state into executable driving outputs. | source: [Paper, Figure 2](https://arxiv.org/abs/2503.03074)*

[View full-size figure](/assets/images/bevdriver-source-figure-2.webp)

The training data includes LMDrive's 15,000 sequences and 2,000 additional sequences with semantic labels. Training uses eight public CARLA towns; held-out weather and time-of-day conditions test environmental variation. The paper also reports a Town05 exclusion experiment, which should be distinguished from results where that town appears in training. Language-model training takes 72 hours on eight A100 GPUs, following separate perception pretraining.

### Completion and infractions can move in opposite directions

| Llama-7B system | LangAuto Driving Score | Short-route Driving Score | Tiny-route Driving Score |
| --- | ---: | ---: | ---: |
| LMDrive, reported baseline | 31.3 | 42.8 | 52.2 |
| BEVDriver | 48.9 | 66.7 | 70.2 |

Table I averages three repetitions per route. The architecture, added semantic data, and training recipe change together, so this is not an isolated proof that BEV alone produces the gain. Backbone size is also not a monotonic predictor: the reported Llama-3.1-8B-Instruct variant scores lower than Llama-7B on the long-route benchmark.

The encoder ablation is especially instructive. Unfreezing it raises tiny-route completion from 81.3 to 85.6, but reduces infraction score from 0.87 to 0.64. Driving Score falls from 70.2 to 55.2. That run also changes traffic-light supervision, so it does not establish that freezing is universally better. It does show why route completion alone can conceal a worse driving policy.

The authors report that commands such as turning after a specified distance can trigger a turn at the next opportunity. A spatially structured input does not automatically ground the timing or distance semantics of an instruction. Compared with [EMMA](/paper%20shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html), BEVDriver retains specialized spatial perception and a continuous waypoint head rather than serializing all driving outputs as text. The sensor and control differences preclude a direct score comparison.

## High-Level Takeaways

- BEVDriver is a direct precedent for language-conditioned planning from learned BEV features, including a concrete waypoint-to-control interface.
- Camera-only systems must test whether they can recover the same useful features without LiDAR; this paper does not establish that transfer.
- Keep completion, infractions, distance-conditioned instructions, and total sensor-to-action latency separate in evaluation. A higher aggregate score can hide a specific grounding failure.
