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

**arXiv:** [2509.09674](https://arxiv.org/abs/2509.09674)

**GitHub:** [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)

## Summary

> SimpleVLA-RL treats VLA reinforcement learning as an interaction and systems problem. It samples action tokens in parallel environments, assigns each completed trajectory a binary success reward, and updates OpenVLA-OFT with group-relative policy optimization. Dynamic mixed-outcome groups, a wider clip range, and temperature 1.6 make sparse-reward exploration productive enough to scale across LIBERO and RoboTwin.

## Core Insights

![SimpleVLA-RL loop comparing limited offline supervised trajectories with on-policy rollouts grouped for advantage estimation](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-paper-figure.png)
*Fig 1: The extra signal comes from closed-loop trajectories: the policy samples actions, the environment returns success or failure, and group-relative advantages update the action-token distribution. | source: [SimpleVLA-RL, Figure 2](https://arxiv.org/abs/2509.09674)*

### The rollout engine is part of the algorithm

The paper starts from an OpenVLA-OFT variant with an autoregressive action-token head. The implementation configures a 256-token discretized action vocabulary; that is a token-space setting, not a 256-step rollout. During rollout, the policy interacts with the environment until success or a horizon limit, rather than generating a static text-like sequence. Eight trajectories are sampled per input group, and the binary terminal reward is propagated uniformly to the action tokens in that trajectory. GRPO normalizes each trajectory against the mean and standard deviation of its group, so a group of all successes or all failures produces no useful advantage.

SimpleVLA-RL addresses that failure mode explicitly. Dynamic sampling rejects homogeneous groups and continues sampling until each retained group contains both outcomes. It widens the GRPO ratio clip from $[0.8,1.2]$ to $[0.8,1.28]$, allowing low-probability tokens more room to increase, and raises rollout temperature from 1.0 to 1.6. The objective removes KL regularization, reducing memory and avoiding a fixed reference policy that could suppress new action patterns. These choices are coupled: more exploration only helps when the sampler preserves mixed outcomes and the optimizer can reinforce the rare successful trajectory.

![Figure 2 from SimpleVLA-RL: benchmark, data-scarcity, pushcut, and generalization overview](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-source-figure-1.webp)
*Fig 2: The source paper’s Figure 2 summarizes the four claims: data-scarce LIBERO improvement, RoboTwin and real-world gains, emergent pushcut behavior, and held-out spatial/object/goal generalization. | source: [SimpleVLA-RL, Figure 2](https://arxiv.org/abs/2509.09674)*

### The reported gains span horizons and data regimes

The framework uses 8 NVIDIA A800 80GB GPUs, learning rate $5\times10^{-6}$, batch size 64, eight samples per group, mini-batch size 128, clip ratios $0.2/0.28$, temperature 1.6, and greedy evaluation repeated three times. The action-chunk length is eight in LIBERO and 25 in RoboTwin1.0/2.0. This implementation uses parallel decoding and a token cross-entropy head rather than the official continuous L1 head. Because the input and output interfaces differ, the authors cannot reuse the official checkpoints: “SFT from scratch” means re-running supervised fine-tuning from the pretrained OpenVLA/OFT initialization with the official datasets and hyperparameters, not training a foundation model from random weights.

On LIBERO, OpenVLA-OFT rises from 91.0% average SFT success to 99.1%; LIBERO-LONG rises from 86.5% to 98.5%. On RoboTwin1.0, the mean rises from 39.8% to 70.4%. Across 12 RoboTwin2.0 tasks, the mean rises from 38.3% to 68.8%, above π0 at 49.2% and RDT at 33.3%. The horizon breakdown matters: gains remain +43.6 points on short tasks, +25.4 on medium tasks, and +22.4 on long and extra-long tasks rather than appearing only on easy episodes.

The data-scarcity experiment is sharper than the headline average. With one demonstration trajectory per task, OpenVLA-OFT averages 48.9% across LIBERO; after SimpleVLA-RL it reaches 96.9%, including LIBERO-LONG 17.3%→91.7%. Full 500-trajectory SFT followed by RL reaches 99.1%, so one-trajectory RL trails the full-data RL result by only 2.2 points. That is a claim about simulation exploration from a nonzero prior, not a claim that RL can learn a task from no competence.

![Figure 3 from SimpleVLA-RL: higher rollout temperature improves LIBERO-LONG learning](/assets/images/simplevla-rl-scaling-vla-training-via-reinforcement-learning-source-figure-4.webp)
*Fig 3: The source paper’s Figure 3 shows that temperature 1.6 reaches a higher LIBERO-LONG plateau about 15 points above temperature 1.0 and gets there earlier. | source: [SimpleVLA-RL, Figure 3](https://arxiv.org/abs/2509.09674)*

### RL can discover a shortcut, but it still needs a capable prior

The “pushcut” examples are revealing because the demonstrations always use grasp–move–place. After RL, the policy pushes a can into the pot or pushes object A next to object B when the binary outcome reward does not care which successful strategy was used. The behavior is a genuine change in the action sequence, not merely a smoother imitation of demonstrations. Yet Table 7 shows the boundary: with zero trajectory SFT, all five sampled RoboTwin2.0 tasks remain at 0% after RL because no successful rollout supplies a positive advantage. With 100 trajectories, mean success rises 7.3%→25.4%; with 1,000, it rises 28.2%→50.4%.

The real-world RoboTwin2.0 test uses 1,000 simulated trajectories for SFT and 1,000 scenarios for RL, then 50 trials per task on two AgileX Piper arms with unseen tabletops. Average success rises from 17.5% to 38.5%, while RDT scores 23.5%. This is encouraging sim-to-real evidence, but it remains dependent on a capable initial policy, a simulator that rewards the right behavior, and outcome labels that do not penalize unsafe shortcuts.

## High-Level Takeaways

- SimpleVLA-RL’s main contribution is an end-to-end rollout stack that makes group-relative RL informative for closed-loop robot trajectories: parallel environments, mixed-outcome sampling, token likelihoods, and explicit exploration controls.
- The strongest data result is one-trajectory LIBERO training: average success rises 48.9%→96.9%, and LIBERO-LONG rises 17.3%→91.7%. The model still needs a nonzero task prior.
- Temperature 1.6 is not a cosmetic hyperparameter. In the reported LIBERO-LONG curve it lifts the plateau by about 15 points, because rare successful action-token paths need to be sampled before GRPO can reinforce them.
- “Pushcut” shows that outcome reward can remove procedural constraints and discover a shorter strategy, while also exposing a safety boundary: reward-equivalent shortcuts need independent checks for collisions, smoothness, and object handling.
- The 38.5% real-world average comes from simulation-trained policies evaluated on four tasks and 50 trials each. Larger embodiment and reward shifts remain open tests for whether the infrastructure scales beyond these benchmarks.
