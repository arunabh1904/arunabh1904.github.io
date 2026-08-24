---
title: 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion'
date: '2023-03-07T00:00:00.000Z'
section: paper-shorts
postSlug: diffusion-policy-visuomotor-policy-learning-via-action-diffusion
legacyPath: /paper shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html
tags:
  - Robotics
  - Diffusion
field: 'Vision-Language-Action & Robotics'
summary: "2023 – Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
---

## 2023 – Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**arXiv:** [2303.04137](https://arxiv.org/abs/2303.04137)

**Project:** [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)

## Summary

> Diffusion Policy represents a visuomotor policy as a conditional denoising process over action trajectories. Instead of regressing toward the average of several valid behaviors, it learns the score of a multimodal action distribution and samples a coherent sequence at inference time.

## Core Insights

![Diffusion Policy overview with observation-conditioned denoising of action sequences using convolutional or transformer backbones](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-paper-figure.png)
*Figure 2 shows why the output is a trajectory distribution rather than a point action: Gaussian action sequences are iteratively denoised while visual observations condition every convolutional block or transformer decoder layer. source: [Diffusion Policy](https://arxiv.org/abs/2303.04137)*

![Figure 10 from Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-source-figure-10.webp)
*Figure 10 Realworld Sauce Manipulation. [Left] 6DoF pouring Task. The robot needs to \raisebox{-0.9pt}{1}⃝ dip the ladle to scoop sauce from the bowl, \raisebox{-0.9pt}{2}⃝ approach the center of the pizza dough, \raisebox{-0.9pt}{3}⃝ pour sauce, and \raisebox{-0.9pt}{4}⃝ lift the ladle to finish the task. [Right] Periodic spreading Task The robot needs to \raisebox{-0.9pt}{1}⃝ approach the center of the sauce with a grasped spoon, \raisebox{-0.9pt}{2}⃝ spread the sauce to cover pizza in a spiral pattern, and \raisebox{-0. source: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)*

![Figure 4 from Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-source-figure-4.webp)
*Figure 4 Velocity v.s. Position Control. The performance difference when switching from velocity to position control. While both BCRNN and BET performance decrease, Diffusion Policy is able to leverage the advantage of position and improve its performance. source: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)*


The policy conditions a diffusion model on observations, starts from noisy action sequences, and refines them across denoising steps. Receding-horizon control executes only the near part of each sampled trajectory before observing again. This combination gives diffusion enough horizon to coordinate motion while retaining closed-loop replanning.

Across 15 tasks from four manipulation benchmarks, the paper reports an average 46.9% improvement over the compared state of the art. The important mechanism is distributional expressivity: high-dimensional action sequences and multiple valid strategies are represented without an explicit mixture model. The cost is iterative sampling and a likelihood interface that is less convenient for policy-gradient or preference optimization.

| Representation | Strength | Post-training complication |
| --- | --- | --- |
| Per-step regression | Cheap and explicit | Averages incompatible actions |
| Autoregressive tokens | Native likelihood | Quantization and sequential latency |
| Diffusion trajectory | Multimodal continuous behavior | Iterative inference and denoising-step credit assignment |

## High-Level Takeaways

- Diffusion Policy informs whether action multimodality is important enough to justify iterative decoding. Its atomic unit is an action trajectory corrupted at a diffusion timestep; the loss predicts denoising information conditioned on visual state. Temporal compression comes from predicting a sequence and executing it receding-horizon.
- The benchmark establishes a strong imitation-learning Pareto point, not that diffusion remains optimal under strict latency or online RL. A missing experiment matches end-to-end control frequency and compute against flow, autoregressive, and parallel regression heads. At ten times the horizon, denoising cost and model error across the unused tail grow. The representation claim fails if a simpler continuous chunk policy matches robustness and multimodality at the same closed-loop rate.
- Diffusion Policy made the policy distribution—not only the backbone—a central robot-learning decision.
- Strong offline imitation results do not automatically provide tractable action log-probabilities for RL.
- Diffusion is valuable when the action distribution has several precise modes; its sampling interface must still fit the control and post-training loop.
