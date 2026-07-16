---
title: 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion'
date: '2023-03-07T09:00:00.000Z'
section: paper-shorts
postSlug: diffusion-policy-visuomotor-policy-learning-via-action-diffusion
legacyPath: /paper shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html
tags:
  - Robotics
  - Diffusion
field: 'Vision-Language-Action & Robotics'
summary: 2023 – Diffusion Policy models multimodal continuous action sequences through iterative denoising.
---

## 2023 – Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**arXiv:** [2303.04137](https://arxiv.org/abs/2303.04137)

**Project:** [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)

Diffusion Policy represents a visuomotor policy as a conditional denoising process over action trajectories. Instead of regressing toward the average of several valid behaviors, it learns the score of a multimodal action distribution and samples a coherent sequence at inference time.

## Paper Insights

The policy conditions a diffusion model on observations, starts from noisy action sequences, and refines them across denoising steps. Receding-horizon control executes only the near part of each sampled trajectory before observing again. This combination gives diffusion enough horizon to coordinate motion while retaining closed-loop replanning.

Across 15 tasks from four manipulation benchmarks, the paper reports an average 46.9% improvement over the compared state of the art. The important mechanism is distributional expressivity: high-dimensional action sequences and multiple valid strategies are represented without an explicit mixture model. The cost is iterative sampling and a likelihood interface that is less convenient for policy-gradient or preference optimization.

| Representation | Strength | Post-training complication |
| --- | --- | --- |
| Per-step regression | Cheap and explicit | Averages incompatible actions |
| Autoregressive tokens | Native likelihood | Quantization and sequential latency |
| Diffusion trajectory | Multimodal continuous behavior | Iterative inference and denoising-step credit assignment |

## Decision Lens

Diffusion Policy informs whether action multimodality is important enough to justify iterative decoding. Its atomic unit is an action trajectory corrupted at a diffusion timestep; the loss predicts denoising information conditioned on visual state. Temporal compression comes from predicting a sequence and executing it receding-horizon.

The benchmark establishes a strong imitation-learning Pareto point, not that diffusion remains optimal under strict latency or online RL. A missing experiment matches end-to-end control frequency and compute against flow, autoregressive, and parallel regression heads. At ten times the horizon, denoising cost and model error across the unused tail grow. The representation claim fails if a simpler continuous chunk policy matches robustness and multimodality at the same closed-loop rate.

**Context:** Diffusion Policy made the policy distribution—not only the backbone—a central robot-learning decision.

**Limits:** Strong offline imitation results do not automatically provide tractable action log-probabilities for RL.

**Takeaway:** Diffusion is valuable when the action distribution has several precise modes; its sampling interface must still fit the control and post-training loop.
