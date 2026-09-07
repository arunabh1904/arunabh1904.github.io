---
title: 'DPPO: Diffusion Policy Policy Optimization'
date: '2024-09-01T00:00:00.000Z'
section: paper-shorts
postSlug: dppo-diffusion-policy-policy-optimization
legacyPath: /paper shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html
tags:
  - Robotics
  - Reinforcement Learning
field: 'Robot Post-Training & Evaluation'
summary: "2024 – DPPO: Diffusion Policy Policy Optimization"
---

**arXiv:** [2409.00588](https://arxiv.org/abs/2409.00588)

**Project:** [diffusion-ppo.github.io](https://diffusion-ppo.github.io/)

## Summary

> DPPO makes policy-gradient fine-tuning well-defined for a diffusion policy. It treats each stochastic denoising transition as an action in an inner MDP, nests that process inside the robot’s environment MDP, and applies PPO to the resulting chain. The payoff is structured exploration around the imitation data manifold; the cost is a longer credit-assignment path and a sampler whose training details matter.

## Core Insights

![Diffusion Policy MDP unrolling denoising states inside each environment action step for policy-gradient optimization](/assets/images/dppo-diffusion-policy-policy-optimization-paper-figure.png)
*Fig 1: The reduction that makes the update tractable: each denoising transition has a Gaussian likelihood, while environment reward is paid only after the final action is executed. | source: [DPPO, Figure 3](https://arxiv.org/abs/2409.00588)*

### The policy gradient follows denoising time, not just robot time

A Diffusion Policy first samples noise and repeatedly denoises an action chunk. The robot then executes only part of that chunk before observing the next state. DPPO keeps both clocks explicit. For environment state $s_t$, the inner chain contains $a_t^K, a_t^{K-1}, \ldots, a_t^0$; the denoising transition from $a_t^{k+1}$ to $a_t^k$ is Gaussian, so its log-likelihood can be evaluated exactly. The final action marginal $p_\theta(a_t^0\mid s_t)$ is not available as one tractable density; DPPO instead optimizes the explicit Gaussian transitions along the denoising path. Only the final $a_t^0$ advances the physical environment and receives its reward. The resulting trajectory is therefore a chain of inner denoising MDPs joined by environment transitions.

PPO is applied to those inner transitions. An environment discount handles future robot rewards, while a denoising discount downweights earlier, noisier denoising steps. The value estimator depends on the environment state rather than also conditioning on the partially denoised action; the authors report that this choice is more stable on difficult tasks. That detail matters: the method is not “PPO on the final action” with a diffusion wrapper. Its optimization unit is the stochastic path that produced the final action chunk.

### The practical recipe is to update the tail of the sampler

![Figure 7 from DPPO: Diffusion Policy Policy Optimization](/assets/images/dppo-diffusion-policy-policy-optimization-source-figure-7.webp)
*Fig 2: State and pixel comparisons on the difficult Square and Transport tasks from ROBOMIMIC. | source: [DPPO, Figure 7](https://arxiv.org/abs/2409.00588)*

The paper does not fine-tune every pretraining denoising step by default. A policy may use up to 100 DDPM steps, but DPPO can freeze the early steps and update only the last $K'$ steps. This reduces memory and training time without sacrificing final performance in the reported settings. For pixel policies and long-horizon furniture tasks, the authors use DDIM with five denoising steps during fine-tuning; stochasticity is retained for training by setting its noise parameter to one and removed for evaluation. They also clip the exploration noise to a task-dependent floor, typically $0.01$–$0.1$, and use at least $0.1$ when evaluating likelihoods to avoid unstable large log-probabilities.

Figure 2 isolates the policy-class comparison. For state input, DPPO reaches roughly 100% on the easier Lift and Can tasks, and above 90% on the harder Transport task; its MLP and UNet variants are similar, with the MLP training somewhat faster. With pixel input, DPPO improves Square more quickly and to a higher level than the Gaussian ViT baseline, while Gaussian does not improve from its 0% pretrained Transport success. The comparison uses the same action execution horizon, $T_a=4$ for Lift, Can, and Square and $T_a=8$ for Transport, so the result is about the stochastic policy parameterization rather than a longer controller horizon.

### The long-horizon result is a sim-to-real test of the exploration claim

![Figure 4 from DPPO: Diffusion Policy Policy Optimization](/assets/images/dppo-diffusion-policy-policy-optimization-source-figure-4.webp)
*Fig 3: The bimanual ROBOMIMIC Transport task and the One-leg, Lamp, and Round-table FURNITURE-BENCH tasks used for long-horizon evaluation. | source: [DPPO, Figure 4](https://arxiv.org/abs/2409.00588)*

The data regime is deliberately modest. ROBOMIMIC policies use 300 state or 100 pixel demonstrations, with sparse reward equal to one on task completion. FURNITURE-BENCH policies use 50 simulated human demonstrations, execute eight actions from each predicted chunk, and receive sparse stage-completion reward. On the six furniture settings, DPPO improves from its pretrained policy while Gaussian-MLP collapses to zero in all three medium-randomness tasks and in low-randomness Round-table. The paper’s most concrete deployment result is One-leg: after simulation fine-tuning, DPPO succeeds on 16 of 20 hardware trials (80%) zero-shot, whereas Gaussian reaches 88% in simulation but 0% on hardware. Adding behavior-cloning regularization to Gaussian avoids total collapse but limits it to 53% simulation and 50% hardware.

The qualitative explanation is more specific than “diffusion is robust.” In the hardware rollouts, the fine-tuned policy corrects a small peg-to-hole misalignment before releasing, while the pretrained policy sometimes pushes the peg down and lets it topple. The paper’s Avoid study likewise shows noise being injected through multiple denoising steps while the denoising process keeps samples near the demonstrated action manifold. That is a plausible reason for transfer, but the paper presents it as a mechanism supported by diagnostic experiments, not as a guarantee for arbitrary real-world dynamics.

## High-Level Takeaways

- DPPO supplies the missing likelihood for policy-gradient updates when a robot action is generated by a diffusion trajectory. Treating denoising as an inner MDP preserves multimodality while making PPO’s ratio computable.
- The strongest evidence is controlled: the difficult Transport task improves from a 0% pretrained pixel baseline, and One-leg reaches 80% on hardware versus 0% for a Gaussian policy that looks strong in simulation.
- The method’s efficiency depends on updating only a denoising tail or a short DDIM chain. A diffusion policy is therefore a deployment contract with sampler steps, action chunk horizon, noise floors, and execution frequency, not only a neural network checkpoint.
- On-manifold exploration helps when pretraining covers useful success modes. The authors also observe that DPPO can underperform Gaussian on at least one setting where more aggressive, unstructured exploration may be useful.
- The evidence is concentrated in simulated tasks plus one zero-shot furniture transfer. It does not establish that denoising-time credit assignment is preferable when rewards are dense, hardware interaction is cheap, or a simpler actor matches the same pretrained distribution.
