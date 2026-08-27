---
title: 'WPT: World-to-Policy Transfer via Online World Model Distillation'
date: '2025-11-25T00:00:00.000Z'
section: paper-shorts
postSlug: wpt-world-to-policy-transfer-via-online-world-model-distillation
legacyPath: /paper shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html
tags: [Other]
field: 'Autonomous Driving: VLA & Planning'
summary: '2025 – WPT: training-time world-model reasoning for a lightweight driving policy'
---
## 2025 – WPT

**arXiv:** [2511.20095](https://arxiv.org/abs/2511.20095)

**Paper:** [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_WPT_World-to-Policy_Transfer_via_Online_World_Model_Distillation_CVPR_2026_paper.html)

### Method and reported result

WPT uses a world model during training to forecast how candidate actions change future agents, occupancy, and road structure. A trainable reward model turns those rollouts into imitation and simulation signals for safety, progress, drivable-area compliance, time to collision, and comfort. Query and reward distillation then transfer that supervision into a smaller policy that runs without the world model.

## Summary

> The world model is a training instrument, not a mandatory runtime component. WPT spends predictive computation while learning, then distills the resulting decision criteria into a deployable student policy.

## Core Insights

On nuScenes, the teacher reports 0.61 m average L2 error and 0.11% collision rate. The distilled student reports 0.66 m and 0.24%, compared with 0.88 m and 1.06% for the undistilled baseline. On Bench2Drive, the teacher reaches a 79.23 driving score and 54.54% success rate; the student retains 72.61 and 45.45%. The paper reports 64 ms planning latency for the student versus 312 ms for the teacher, a 4.9× difference.

![WPT: World-to-Policy Transfer via Online World Model Distillation source figure: Overview of WPT framework.](/assets/images/wpt-world-to-policy-transfer-via-online-world-model-distillation-paper-figure.webp)
*Fig 1: WPT trains a world-model-enhanced teacher that proposes and scores multimodal trajectories, then distills both policy features and world-model rewards into a lighter student policy. | source: [WPT: World-to-Policy Transfer via Online World Model Distillation](https://arxiv.org/abs/2511.20095)*

![Figure 1 from WPT: World-to-Policy Transfer via Online World Model Distillation](/assets/images/wpt-world-to-policy-transfer-via-online-world-model-distillation-source-figure-1.webp)
*Fig 2: WPT is positioned against imitation learning, direct world-model integration, and simulator-based reinforcement learning. Its teacher and student both consult the world model during training, transferring planning and reward knowledge without requiring it at student inference. | source: [WPT: World-to-Policy Transfer via Online World Model Distillation](https://arxiv.org/abs/2511.20095)*


| Model | Avg. L2 | Collision rate | Runtime role |
| --- | ---: | ---: | --- |
| Baseline policy | 0.88 m | 1.06% | Student architecture without transfer. |
| WPT teacher | 0.61 m | 0.11% | World model and reward scoring. |
| WPT student | 0.66 m | 0.24% | World-model-free inference. |

The results also show a comfort-efficiency trade-off, and the student does not retain every teacher gain. More importantly, simulation rewards inherit the world model's forecast errors; distillation can transfer a teacher's blind spots as readily as its useful structure.

## High-Level Takeaways

- A large predictive model can improve a policy without remaining in the deployed inference graph.
- Reward decomposition makes the transferred decision criteria more inspectable than opaque feature matching alone.
- Teacher and student must be evaluated separately because distillation changes both safety metrics and latency.
- Training-time rollouts only help when the world model is accurate on the rare interactions that dominate driving risk.
