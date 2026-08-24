---
title: 'SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning'
date: '2025-09-11T00:00:00.000Z'
section: paper-shorts
postSlug: simplevla-rl-scaling-vla-training-via-reinforcement-learning
legacyPath: /paper shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html
tags:
  - Robotics
  - Reinforcement Learning
field: 'Robot Post-Training & Evaluation'
summary: "2025 – SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning"
---

## 2025 – SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

**arXiv:** [2509.09674](https://arxiv.org/abs/2509.09674)

**GitHub:** [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)

## Summary

> SimpleVLA-RL treats VLA reinforcement learning as a systems problem as much as an objective problem. Built on veRL and OpenVLA-OFT, it adds robot-specific trajectory sampling, parallel environments, multi-environment rendering, and optimized loss computation around a group-relative policy update.

## Core Insights

![SimpleVLA-RL loop comparing limited offline supervised trajectories with on-policy rollouts grouped for advantage estimation](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-paper-figure.png)
*Figure 2 shows where the additional signal comes from: the SFT policy interacts with the environment, produces groups of trajectories and rewards, and updates from relative advantages instead of remaining bounded by the offline demonstrations. source: [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)*

![Figure 1 from SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-source-figure-1.webp)
*Figure 1 Overview of SimpleVLA-RL . SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “ pushcut ” new-action phenomenon, and strengthens spatial/object/goal generalization. source: [SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](https://arxiv.org/abs/2509.09674)*

![Figure 4 from SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-source-figure-4.webp)
*Figure 4 (c) Higher Rollout Temperature. source: [SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](https://arxiv.org/abs/2509.09674)*


The framework reports state-of-the-art results on LIBERO and strong RoboTwin 1.0/2.0 performance, including large gains when SFT data are scarce. The paper also reports a “pushcut” behavior not present in the demonstrations, using it as evidence that RL can discover action patterns beyond imitation.

The claim depends on exploration. Group-relative methods produce no useful gradient when every rollout in a group receives the same reward. The paper introduces exploration-enhancing strategies, but the same mechanism can favor simulator-specific shortcuts if reward and environment diversity are weak.

| Layer | VLA-specific requirement |
| --- | --- |
| Rollouts | Parallel control environments and policy-version tracking |
| Advantages | Reward variation within task-conditioned groups |
| Loss | Correct masks and likelihoods over action chunks |
| Evaluation | Seen tasks, held-out shifts, and real-world confirmation |

## High-Level Takeaways

- SimpleVLA-RL informs whether scaling rollout infrastructure can extract more from a strong SFT policy than scaling demonstration data. Its atomic unit is a group of task-conditioned trajectories whose relative rewards produce advantages. The policy remains an OpenVLA-OFT continuous chunk model, so likelihood and masking must align with that action interface.
- The results establish that RL can outperform SFT in several simulation and real settings. A missing control measures total environment, tuning, and compute cost against failure-targeted SFT. At ten times the worker count, policy staleness and correlated environments become critical; at ten times the task count, reward homogeneity causes advantage collapse. The claim fails if newly discovered behaviors do not transfer outside the training simulator.
- SimpleVLA-RL is the reference implementation for fleet-like parallel VLA rollouts and group-relative updates.
- High success on benchmark rewards can hide changes in smoothness, safety, and strategy diversity.
- VLA RL scales only when rollout diversity produces informative advantages and the infrastructure preserves their provenance.
