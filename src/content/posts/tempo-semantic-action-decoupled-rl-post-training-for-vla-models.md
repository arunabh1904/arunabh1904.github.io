---
title: 'TEMPO: Semantic-Action Decoupled RL Post-Training for VLA Models'
date: '2026-08-07T15:09:51.000Z'
section: paper-shorts
postSlug: tempo-semantic-action-decoupled-rl-post-training-for-vla-models
legacyPath: /paper shorts/2026/08/07/tempo-semantic-action-decoupled-rl-post-training-for-vla-models.html
tags:
  - Robotics
  - Vision-Language-Action
  - Reinforcement Learning
  - Post-Training
field: 'Robot Post-Training & Evaluation'
summary: '2026 – TEMPO: Semantic-Action Decoupled RL Post-Training for VLA Models'
---

## 2026 – TEMPO: Semantic-Action Decoupled RL Post-Training for VLA Models

**arXiv:** [2608.07314](https://arxiv.org/abs/2608.07314)

**Project:** [TEMPO](https://anonymous.4open.science/w/tempo-page/)

## Summary

> TEMPO treats a modular VLA as two coupled adaptation problems. It freezes the vision-language backbone, updates the semantic projection layer slowly, and updates the action expert quickly through separate TD3 loops. On CALVIN ABC→D, a 5:1 action-to-semantic update ratio raises five-task-chain success from 77.8% for the pretrained FLOWER policy to 81.7%; equal-frequency dual-loop training reaches only 77.7%. The scheduling result is controlled more tightly than the broad leaderboard, but the paper does not report a compute-matched comparison for the extra critics and replay machinery.

## Core Insights

![TEMPO with a frozen VLM, slowly updated semantic projection, quickly updated action expert, and separate replay buffers](/assets/images/tempo-two-timescale-post-training-framework.png)
*Fig 1: The semantic projection and action expert share environment rollouts but use separate replay buffers, critics, and actor updates. The slow semantic loop limits latent drift while the fast action loop absorbs control feedback. | source: [TEMPO paper](https://arxiv.org/abs/2608.07314)*

### One RL clock is a hidden architectural assumption

FLOWER maps a frozen VLM representation through a semantic projection into a latent action, then maps that latent through an action expert into an executable chunk. A joint actor update changes both the representation and its consumer at once. TEMPO hypothesizes that this moving interface destabilizes action learning.

The method assigns each module a twin-critic TD3 loop. Semantic updates stop gradients before the frozen VLM and action expert. Action updates recompute the current latent but stop gradients before the semantic projection. The two loops share rollouts while keeping their optimization targets separate.

TEMPO then controls the effective actor-update ratio

$$
\rho = \frac{f_{\mathrm{action}}}{f_{\mathrm{semantic}}}.
$$

The expensive design choice is not only which parameters to expose to RL. It is how often each exposed interface is allowed to move.

### The ratio ablation carries the mechanism claim

CALVIN ABC→D trains on environments A, B, and C and evaluates five-instruction chains in the unseen environment D. TEMPO reaches 81.7% success on all five tasks and an average completed-chain length of 4.59. DeFI, the strongest external baseline in the table, reaches 81.2% and 4.51; the margin is 0.5 points and 0.08 tasks. Against FLOWER, which supplies TEMPO's starting policy, the gains are 3.9 points and 0.10 tasks.

| Action : semantic update ratio | Five-task success | Average chain length |
| --- | ---: | ---: |
| 1:1 | 77.7% | 4.46 |
| 5:1 | 81.7% | 4.59 |
| 10:1 | 81.2% | 4.57 |

Equal-frequency training is 0.1 point below the pretrained FLOWER result. The 5:1 and 10:1 schedules improve it. That comparison supports the timescale hypothesis more directly than the cross-paper leaderboard because the architecture and post-training framework stay closer to fixed.

Component ablations are smaller but consistent. Updating only the action expert reaches 79.8% five-task success; updating only the semantic projection reaches 79.6%; the full method reaches 81.7%. The two modules contribute complementary changes under this reward, although the study does not isolate whether two critics, two buffers, or the actor frequency itself supplies each part of the gain.

### More post-training tasks do not automatically help

With a fixed 10:1 ratio, post-training on one selected CALVIN subtask reaches 81.2% five-task success. Training on all 34 reaches 79.2%. The result warns against treating online-task count as a monotone scaling axis: a fixed interaction budget spread across more tasks changes both coverage and update density.

The physical study uses two multi-stage drawer tasks, 60 demonstrations per task, three random seeds, and 20 evaluation trials per checkpoint. TEMPO reaches and maintains higher late-training rewards than FLOWER-RL. This supports feasibility on hardware, but two tasks cannot establish general semantic retention, and the reported curves do not price the extra online data, critics, or update loops.

## High-Level Takeaways

- TEMPO shows that module update frequency is part of a VLA's post-training architecture. A stable semantic interface can matter as much as exposing more parameters to RL.
- The 5:1 versus 1:1 ablation supports timescale separation; the broader state-of-the-art margin is small and mixes different pretrained models and recipes.
- Sparse terminal rewards still govern both loops. The paper leaves dense progress rewards and safer real-robot credit assignment open.
- A decisive comparison would match environment steps, accelerator time, replay ratio, critic capacity, and wall-clock convergence against one-loop RL. Reject the two-timescale system if an equally budgeted joint actor reaches the same chain success or if semantic slowing prevents adaptation under genuine task shifts.
