---
title: 'DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving'
date: '2025-05-22T06:23:04.000Z'
section: paper-shorts
postSlug: drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving
legacyPath: /paper shorts/2025/05/22/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving"
---
## 2025 – DriveMoE

**arXiv:** [2505.16278](https://arxiv.org/abs/2505.16278)

**Project:** [DriveMoE](https://thinklab-sjtu.github.io/DriveMoE/)

## Summary

> DriveMoE adds two sparse-routing decisions to a Drive-$\pi_0$ VLA baseline. A scene-specialized Vision MoE selects camera evidence for the driving context, while a skill-specialized Action MoE selects behavior modules for different maneuvers. The paper reports state-of-the-art Bench2Drive closed-loop performance. Its abstract does not give the number of experts, routing load, action latency, or a matched dense-capacity baseline.

## Core Insights

The two routers act at different stages. Vision routing allocates perception compute across cameras; action routing allocates policy capacity across driving skills. That separation targets two sources of averaging: processing every view equally and asking one action function to cover heterogeneous maneuvers. It also creates two ways to fail: the system can suppress the camera that contains the hazard, or select an unsuitable action expert before the maneuver is fully visible.

![DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving source figure: Framework of DriveMoE.](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-paper-figure.webp)
*Fig 1: DriveMoE routes scene-dependent camera views through a Vision MoE and driving skills through an Action MoE, then converts predicted trajectories into steering, braking, and acceleration. | source: [DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](https://arxiv.org/abs/2505.16278)*

![Figure 3 from DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-source-figure-3.webp)
*Fig 2: The scene-specialized Vision MoE uses a GPS target and supervised camera router to select dynamic views, concatenate them with fixed-view features, and condition the language model. | source: [DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](https://arxiv.org/abs/2505.16278)*

![Figure 1 from DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](/assets/images/drivemoe-mixture-of-experts-for-vision-language-action-model-in-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 3: Comparison of Different Vision and Action Modeling Strategies in VLA-based End-to-End Driving. (a) Vanilla visual token encoding 47 processes all surround-view images through a vision tower, leading to token redundancy and increased computational cost. (b) Query-based token extraction 53 (e.g., Q-Former 35 ) selects a subset of visual tokens from each image, but loses spatial structure and requires additional pretraining. | source: [DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](https://arxiv.org/abs/2505.16278)*


The result cannot establish which router accounts for the reported improvement without controlled ablations. The abstract does not disclose routing regularization, load balance, expert utilization, data mixture, or whether rare maneuvers receive enough examples to train their specialist. A useful scale test would keep total active FLOPs and demonstrations fixed while comparing a dense model, vision-only routing, action-only routing, and both routers under long-tail route splits.

## High-Level Takeaways

- DriveMoE uses sparse specialization twice: once to choose visual evidence and once to choose a driving behavior module.
- The reported closed-loop result motivates expert routing, but the abstract does not yet isolate its efficiency, calibration, or rare-event benefits from the larger composite architecture.
- The key rejection test holds active compute and data fixed; if a dense conditional policy matches safety and long-tail success, the extra routing complexity is not justified.
