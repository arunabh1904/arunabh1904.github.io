---
title: 'Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model'
date: '2019-11-01T05:00:00.000Z'
section: paper-shorts
postSlug: mastering-atari-go-chess-shogi-muzero
legacyPath: /paper shorts/2019/11/01/mastering-atari-go-chess-shogi-muzero.html
tags:
  - Other
field: Reinforcement Learning
summary: >-
  2019 – Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model
  (MuZero)
---
## 2019 – Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model (MuZero)

**arXiv:** [1911.08265](https://arxiv.org/abs/1911.08265)

**GitHub:** [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) (unofficial re-implementation)

**Project page / DeepMind blog:** [MuZero](https://deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules)

**Journal / Conference:** Nature 2020 (pre-print Nov 2019)

![MuZero Architecture](/assets/images/muzero.png)

**Summary:** MuZero combines search with a learned model, but it does not try to reconstruct the full environment. Instead, it learns three networks: a representation network that maps observations into hidden states, a dynamics network that rolls those hidden states forward, and a prediction network that outputs policy, value, and reward. Those learned summaries feed Monte Carlo Tree Search, letting the agent plan without hand-coded game rules.

The important design choice is what MuZero chooses not to model. It predicts only the quantities needed for planning, not future pixels or full board states. That makes one architecture work across Go, chess, shogi, and 57 Atari games, with MCTS continually producing stronger targets for the networks to imitate.

**Evals / Benchmarks**
- **Atari-57** – mean human-normalised score exceeds 1000%, surpassing prior agents.
- **Go (19×19)** – achieves AlphaZero-level Elo ratings.
- **Chess / Shogi** – matches AlphaZero with fewer training games.

**Critiques & limitations:** MuZero is elegant because it preserves the strength of search without requiring explicit rules. The cost is large. Training used enormous compute, inference requires MCTS, and the learned dynamics remain hard to interpret: the model plans well, but it is not obvious what environment structure it has actually learned.

**Take-home message:** MuZero showed that learned latent models can support serious planning across very different domains. It replaced hand-coded rules with learned prediction, but it paid for that generality with compute and search latency.
