---
title: 'Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model'
date: '2019-11-01T05:00:00.000Z'
section: paper-shorts
postSlug: mastering-atari-go-chess-shogi-muzero
legacyPath: /paper shorts/2019/11/01/mastering-atari-go-chess-shogi-muzero.html
tags:
  - Other
field: Reinforcement Learning
summary: MuZero learned just enough model dynamics to plan with MCTS, without reconstructing full environment observations.
---
## 2019 – Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model (MuZero)

**arXiv:** [1911.08265](https://arxiv.org/abs/1911.08265)

**GitHub:** [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) (unofficial re-implementation)

**Project page / DeepMind blog:** [MuZero](https://deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules)

**Journal / Conference:** Nature 2020 (pre-print Nov 2019)

## Paper Insights

MuZero combines learned dynamics with tree search without requiring known game rules. It learns three functions: representation from observations to latent state, dynamics from latent state/action to next latent state plus reward, and prediction from latent state to policy and value. Planning uses MCTS over the learned latent model, optimizing only the quantities needed for control rather than reconstructing observations. The evidence spans Atari-57 and board games such as Go, chess, and shogi, showing one algorithm can plan in visual and perfect-information domains. The caveat is compute and data intensity: MuZero is powerful but expensive, and the learned model is task-specific. The lasting idea is model-based RL without explicit environment simulators.

![Figure 1: Planning, acting, and training with a learned model from Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model](/assets/images/mastering-atari-go-chess-shogi-muzero-paper-figure.png)
_Figure 1: Planning, acting, and training with a learned model. From the [Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model paper](https://arxiv.org/abs/1911.08265), via arXiv HTML._

**Summary:** MuZero combines search with a learned model, but it does not try to reconstruct the full environment. Instead, it learns three networks: a representation network that maps observations into hidden states, a dynamics network that rolls those hidden states forward, and a prediction network that outputs policy, value, and reward. Those learned summaries feed Monte Carlo Tree Search, letting the agent plan without hand-coded game rules.

The important design choice is what MuZero chooses not to model. It predicts only the quantities needed for planning, not future pixels or full board states. That makes one architecture work across Go, chess, shogi, and 57 Atari games, with MCTS continually producing stronger targets for the networks to imitate.

**Evals / Benchmarks**

| Setting | Signal | Why it matters |
| ------- | ------ | -------------- |
| Atari-57 | Mean human-normalized score exceeds 1000% | Shows the learned latent model works beyond perfect-information board games. |
| Go 19x19 | Achieves AlphaZero-level Elo ratings | Matches a search-based expert system without hand-coded game rules. |
| Chess / Shogi | Matches AlphaZero with fewer training games | Tests whether the same model-based planning recipe transfers across rule systems. |

**Critiques & limitations:** MuZero is elegant because it preserves the strength of search without requiring explicit rules. The cost is large. Training used enormous compute, inference requires MCTS, and the learned dynamics remain hard to interpret: the model plans well, but it is not obvious what environment structure it has actually learned.

**Take-home message:** MuZero showed that learned latent models can support serious planning across very different domains. It replaced hand-coded rules with learned prediction, but it paid for that generality with compute and search latency.
