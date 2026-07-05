---
title: Playing Atari with Deep Reinforcement Learning
date: '2013-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: playing-atari-with-dqn
legacyPath: /paper shorts/2013/12/01/playing-atari-with-dqn.html
tags:
  - Other
field: Reinforcement Learning
summary: DQN combined Q-learning, replay, and target networks to make pixel-based Atari control work with deep networks.
---
## 2013 – Playing Atari with Deep Reinforcement Learning

**arXiv:** [1312.5602](https://arxiv.org/abs/1312.5602)

**GitHub:** [DeepMind-Atari-Deep-Q-Learner](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner) (unofficial re-implementation)

**Project page / DeepMind blog:** [Deep Reinforcement Learning](https://deepmind.com/blog/deep-reinforcement-learning)

**Conference:** NIPS Deep Learning Workshop 2013 (expanded Nature version published 2015)

![Figure 1 provides sample screenshots from five of the games used for training from Playing Atari with Deep Reinforcement Learning](/assets/images/playing-atari-with-dqn-paper-figure.png)
_Figure 1 provides sample screenshots from five of the games used for training. From the [Playing Atari with Deep Reinforcement Learning paper](https://arxiv.org/abs/1312.5602), via arXiv HTML._

**Summary:** DQN made a blunt claim feel plausible: a single neural network could learn control policies directly from Atari pixels, without hand-built state features for each game. The model was a convolutional Q-network trained from raw Atari-2600 frames, and two stabilisation tricks kept the learning problem from collapsing:

1. **Experience replay:** store transitions and sample them randomly to break temporal correlations.
2. **Target network:** hold a slowly updated copy of the Q-network to compute stable learning targets.

With those pieces in place, the authors trained one CNN architecture across seven Atari games using the same weights and hyperparameters. That result mattered because it moved deep RL away from game-specific feature engineering and toward end-to-end pixel-to-action learning. Replay memory and target networks also became durable building blocks for value-based deep RL, while the paper's limitations helped motivate later variants such as Double DQN, prioritized replay, and distributional RL.

**Why it mattered:** DQN showed that a relatively simple network, three convolutional layers followed by two fully connected layers, could handle high-dimensional visual state in a reinforcement learning loop. The important contribution was not just the score table. It was the recipe: pair Q-learning with enough neural capacity, then add just enough machinery to make the targets less volatile.

**Evals / Benchmarks:**

| Game (subset of 7) | DQN vs. prior SOTA | DQN vs. human expert |
| ------------------ | ------------------ | -------------------- |
| Breakout | +560% | Super-human |
| Enduro | +134% | Super-human |
| Pong | Wins 93% | Matches |

Overall, DQN beat previous algorithms on 6 of the 7 reported games and exceeded human scores on 3. The cost was still high for the time: roughly 10 million frames per game, or about 2 to 4 GPU-days in 2013.

**Critiques & limitations:** The cleanest part of DQN is also why it became such a good baseline: one architecture, one training recipe, multiple games. But the method was sample-inefficient, sensitive to tuning, and weak on sparse-reward games such as Montezuma's Revenge. It also inherited Q-learning's tendency to overestimate values, a problem that later work attacked directly with Double DQN and related fixes.

**Take-home message:** DQN made deep reinforcement learning feel real. A CNN, replay memory, and a target network were enough to learn useful control policies from pixels, and that combination became the starting point for much of the modern deep-RL toolbox.
