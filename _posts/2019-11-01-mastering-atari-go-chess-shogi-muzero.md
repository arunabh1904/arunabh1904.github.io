---
layout: content
title: "Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model"
date: 2019-11-01 00:00:00 -0500
categories: ["Paper Shorts"]
field: Reinforcement Learning
---

## 2019 – Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model (MuZero)

**arXiv:** [1911.08265](https://arxiv.org/abs/1911.08265)

**GitHub:** [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) (unofficial re-implementation)

**Project page / DeepMind blog:** [MuZero](https://deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules)

**Journal / Conference:** Nature 2020 (pre-print Nov 2019)

![MuZero Architecture](/assets/images/muzero.png)

**Summary of the paper**
MuZero unifies model-free and model-based reinforcement learning. It learns three networks—representation,
dynamics and prediction—that act as a latent simulator. The dynamics network rolls the hidden state forward
while the prediction head outputs policy, value and reward. These learned summaries drive Monte Carlo Tree
Search so MuZero plans effectively without knowing the environment rules.

**Novel insights**
- Planning-focused latent model predicts only reward, value and policy, avoiding full observation reconstruction.
- Identical architecture masters Go, chess, shogi and 57 Atari games purely from interaction.
- MCTS provides improved targets that bootstrap network training.
- Robust hyper-parameters remain unchanged across domains.

**Evals / Benchmarks**
- **Atari-57** – mean human-normalised score exceeds 1000%, surpassing prior agents.
- **Go (19×19)** – achieves AlphaZero-level Elo ratings.
- **Chess / Shogi** – matches AlphaZero with fewer training games.

**Critiques & limitations**
- **What works well:** Elegant integration of search and learning; removes the need for hand-coded rules.
- **Limitations:** Training requires hundreds of TPUs and inference needs an MCTS, adding latency.
- Learned dynamics remain opaque; true environment structure is unclear.

**Take-home message**
MuZero shows that a learned model can power effective planning across diverse domains without explicit rules,
though at significant computational cost.
