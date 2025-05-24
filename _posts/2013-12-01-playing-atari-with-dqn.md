---
layout: content
title: "Playing Atari with Deep Reinforcement Learning"
date: 2013-12-01 00:00:00 -0500
categories: ["Paper Shorts"]
field: Reinforcement Learning
---

## 2013 – Playing Atari with Deep Reinforcement Learning

**arXiv:** [1312.5602](https://arxiv.org/abs/1312.5602)

**GitHub:** [DeepMind-Atari-Deep-Q-Learner](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner) (unofficial re-implementation)

**Project page / DeepMind blog:** [Deep Reinforcement Learning](https://deepmind.com/blog/deep-reinforcement-learning)

**Conference:** NIPS Deep Learning Workshop 2013 (expanded Nature version published 2015)

**Summary (abstract in plain English):** The paper introduces the Deep Q-Network (DQN) — a convolutional neural network that learns, via Q-learning, to map raw Atari-2600 screen pixels directly to action values. Two key stabilisation tricks make deep RL feasible:
1. **Experience replay:** store transitions and sample them randomly to break temporal correlations.
2. **Target network:** hold a slowly updated copy of the Q-network to compute stable learning targets.
Using a single CNN, the authors trained agents on seven Atari games without game-specific tweaks; the same weights and hyper-parameters generalised across titles.

**Novel insights:**
- End-to-end pixel-to-action learning proved possible, eliminating hand-crafted state representations.
- Replay memory and target network became standard building blocks for stabilising value-based deep RL.
- Demonstrated that depth (three conv layers + two fully connected) can handle high-dimensional RL state spaces.
- Kick-started a wave of research into distributional, prioritised, and double DQN variants.

**Evals / Benchmarks:**

| Game (subset of 7) | DQN vs. prior SOTA | DQN vs. human expert |
| ------------------ | ------------------ | -------------------- |
| Breakout | +560% | Super-human |
| Enduro | +134% | Super-human |
| Pong | Wins 93% | Matches |

Overall, DQN beat previous algorithms on 6 / 7 games and exceeded human scores on 3. Training cost: approximately 10 million frames per game (~2–4 GPU-days in 2013).

**Critiques & limitations:**
- **What I liked:** Elegant, single-architecture baseline across games. Replay and target network tricks still in use a decade later.
- **What I didn’t like / open issues:** Sample-inefficient, unstable without careful tuning, struggles on sparse-reward titles such as Montezuma’s Revenge, and Q-value overestimation later required fixes like Double DQN and prioritized replay.

**Take-home message:** DQN’s simple CNN, replay memory, and target network showed, for the first time, that deep RL could learn complex control policies directly from raw pixels, laying the foundation for the modern deep-RL toolbox.
