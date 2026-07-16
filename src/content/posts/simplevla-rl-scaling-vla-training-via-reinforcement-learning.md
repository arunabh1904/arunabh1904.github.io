---
title: 'SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning'
date: '2025-09-11T09:00:00.000Z'
section: paper-shorts
postSlug: simplevla-rl-scaling-vla-training-via-reinforcement-learning
legacyPath: /paper shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html
tags:
  - Robotics
  - Reinforcement Learning
field: 'Robot Post-Training & Evaluation'
summary: 2025 – SimpleVLA-RL adapts group-relative RL and parallel rollouts to OpenVLA-OFT.
---

## 2025 – SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

**arXiv:** [2509.09674](https://arxiv.org/abs/2509.09674)

**GitHub:** [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)

SimpleVLA-RL treats VLA reinforcement learning as a systems problem as much as an objective problem. Built on veRL and OpenVLA-OFT, it adds robot-specific trajectory sampling, parallel environments, multi-environment rendering, and optimized loss computation around a group-relative policy update.

## Paper Insights

The framework reports state-of-the-art results on LIBERO and strong RoboTwin 1.0/2.0 performance, including large gains when SFT data are scarce. The paper also reports a “pushcut” behavior not present in the demonstrations, using it as evidence that RL can discover action patterns beyond imitation.

The claim depends on exploration. Group-relative methods produce no useful gradient when every rollout in a group receives the same reward. The paper introduces exploration-enhancing strategies, but the same mechanism can favor simulator-specific shortcuts if reward and environment diversity are weak.

| Layer | VLA-specific requirement |
| --- | --- |
| Rollouts | Parallel control environments and policy-version tracking |
| Advantages | Reward variation within task-conditioned groups |
| Loss | Correct masks and likelihoods over action chunks |
| Evaluation | Seen tasks, held-out shifts, and real-world confirmation |

## Decision Lens

SimpleVLA-RL informs whether scaling rollout infrastructure can extract more from a strong SFT policy than scaling demonstration data. Its atomic unit is a group of task-conditioned trajectories whose relative rewards produce advantages. The policy remains an OpenVLA-OFT continuous chunk model, so likelihood and masking must align with that action interface.

The results establish that RL can outperform SFT in several simulation and real settings. A missing control measures total environment, tuning, and compute cost against failure-targeted SFT. At ten times the worker count, policy staleness and correlated environments become critical; at ten times the task count, reward homogeneity causes advantage collapse. The claim fails if newly discovered behaviors do not transfer outside the training simulator.

**Context:** SimpleVLA-RL is the reference implementation for fleet-like parallel VLA rollouts and group-relative updates.

**Limits:** High success on benchmark rewards can hide changes in smoothness, safety, and strategy diversity.

**Takeaway:** VLA RL scales only when rollout diversity produces informative advantages and the infrastructure preserves their provenance.
