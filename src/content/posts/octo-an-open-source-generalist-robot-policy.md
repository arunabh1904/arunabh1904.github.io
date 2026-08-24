---
title: 'Octo: An Open-Source Generalist Robot Policy'
date: '2024-05-20T00:00:00.000Z'
section: paper-shorts
postSlug: octo-an-open-source-generalist-robot-policy
legacyPath: /paper shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html
tags:
  - Robotics
  - Generalist Policies
field: 'Vision-Language-Action & Robotics'
summary: "2024 – Octo: An Open-Source Generalist Robot Policy"
---

## 2024 – Octo: An Open-Source Generalist Robot Policy

**arXiv:** [2405.12213](https://arxiv.org/abs/2405.12213)

**Project:** [octo-models.github.io](https://octo-models.github.io/)

## Summary

> Octo treats adaptation interfaces as part of the foundation-model design. A transformer policy pretrained on 800,000 Open X-Embodiment trajectories accepts language or goal-image tasks, supports flexible camera and proprioceptive inputs, and can be fine-tuned to new action spaces on consumer GPUs.

## Core Insights

![Octo architecture tokenizing task and observation inputs with flexible blockwise attention and readout action heads](/assets/images/octo-an-open-source-generalist-robot-policy-paper-figure.png)
*Figure 0 shows Octo's adaptation interface: task and observation tokens share one transformer, while blockwise attention and readout tokens let fine-tuning add observations or action spaces without rewriting the pretrained backbone. source: [Octo](https://arxiv.org/abs/2405.12213)*

![Figure 4 from Octo: An Open-Source Generalist Robot Policy](/assets/images/octo-an-open-source-generalist-robot-policy-source-figure-4.webp)
*Figure 4 Fig. 4: Model Scaling. The performance of Octo improves with larger model sizes on both UR5 and WidowX tasks. Success rates are averaged over 10 trials on one language-conditioned task per robot. source: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)*

![Figure 2 from Octo: An Open-Source Generalist Robot Policy](/assets/images/octo-an-open-source-generalist-robot-policy-source-figure-2.webp)
*Figure 2 Fig. 2: Evaluation Tasks. We evaluate Octo on 9 real robot setups across 4 institutions. Our evaluations capture diverse object interactions (e.g., “WidowX BridgeV2”), long task horizons (e.g., “Stanford Coffee”) and precise manipulation (e.g., “Berkeley Peg Insertion”). We evaluate Octo’s capabilities to control robots in environments from the pretraining data out-of-the-box and to efficiently finetune to new tasks and environments with small target domain datasets. source: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)*


Octo uses tokenized observations and tasks but a diffusion action head, separating semantic/temporal representation from continuous action generation. Its modular tokenizers and readouts allow new sensors or controllers without rebuilding the backbone. Experiments across nine platforms study both out-of-the-box behavior and downstream adaptation.

The model is an open research baseline rather than a claim of zero-shot universal control. The most useful contribution is inspectability: architecture, data, checkpoints, and adaptation code make it possible to test which priors transfer.

| Interface | Octo design | Why it matters |
| --- | --- | --- |
| Task | Language or goal image | Supports semantic and visual goal conditioning. |
| Observation | Modular camera/proprio tokens | New sensors can be added during adaptation. |
| Action | Diffusion readout | Handles continuous multimodal trajectories. |

## High-Level Takeaways

- Octo informs whether a generalist policy should optimize for zero-shot breadth or for cheap, modular adaptation. Its atomic unit is a heterogeneous robot trajectory; shared transformer tokens carry task and observation context while the action readout remains control-specific.
- The experiments show that diverse pretraining can provide a useful initialization across nine platforms. A missing comparison would equalize pretraining compute and target demonstrations against per-robot policies and VLM-backed VLAs. At ten times the sensor/action interfaces, modularity may become a routing and configuration burden. The claim would fail if pretraining does not reduce target data or training time after controlling for architecture and optimizer.
- Octo is the clean open baseline for studying what cross-embodiment pretraining buys during adaptation.
- A flexible interface cannot compensate for low-quality or incompatible pretraining trajectories.
- For robot foundation models, adaptation cost is a first-class metric—not an afterthought to zero-shot success.
