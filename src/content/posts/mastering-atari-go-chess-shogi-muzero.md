---
title: 'Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model'
date: '2019-11-19T00:00:00.000Z'
section: paper-shorts
postSlug: mastering-atari-go-chess-shogi-muzero
legacyPath: /paper shorts/2019/11/01/mastering-atari-go-chess-shogi-muzero.html
tags:
  - Other
field: 'Reinforcement Learning'
summary: "2019 – Mastering Atari, Go, Chess & Shogi by Planning with a Learned Model"
---

## 2019 – Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

**Paper:** [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) · Nature 2020, preprint 2019

## Summary

> MuZero learns a latent dynamics model for tree search by predicting rewards, values, and policies rather than reconstructing future observations. It matches AlphaZero's board-game performance and improves Atari results using the same general algorithm, with domain-specific inputs and separately trained models. The strongest intuition comes from its search experiments: more thinking continues to help in Go, but Atari gains flatten around 100 simulations. A learned model can support useful planning without being a complete simulator, while its errors still limit the value of additional search.

## Core Insights

### The latent state needs to preserve decisions, not every visible detail

A planning model can spend capacity predicting the exact background of a game screen even when that background does not change the best action. MuZero instead trains its state to support three quantities needed by search: immediate reward, the value of continuing, and a prior over promising actions.

Its representation function maps observation history into an initial state, $s^0=h(o_{1:t})$. The dynamics function takes a state and hypothetical action and returns a new state plus reward, $(r^k,s^k)=g(s^{k-1},a^k)$. The prediction function supplies policy and value, $(p^k,v^k)=f(s^k)$. Reward belongs to the dynamics output; policy and value belong to the prediction output.

The source planning panel starts with a real observation at the top. The blue representation arrow creates the root; green dynamics arrows extend selected hypothetical actions; pink predictions evaluate the resulting states. The interior nodes need not be recognizable boards or images. Their meaning is defined by whether these predictions support good choices.

![MuZero source Figure 1A: representation creates the root and learned dynamics expands hypothetical actions](/assets/images/muzero-source-figure-1a-planning.png)
*Fig 1: The real observation initializes the search once. Dynamics advances latent states under hypothetical actions, while policy and value predictions guide search. Cropped from the source planning panel. | source: [MuZero, Figure 1A](https://arxiv.org/abs/1911.08265)*

This reduces what the model must preserve, but also narrows what it has demonstrated. A model trained for one reward and action space is not automatically an interpretable simulator or a reliable predictor under a changed objective. The original dynamics function is deterministic; explicit stochastic transitions are left for future work.

### Search improves the targets that the model later learns

At each real decision, Monte Carlo Tree Search explores the learned latent model. Root visit counts define an improved action distribution. After an action is executed, the environment supplies the next real observation and reward. Completed trajectories enter replay with their search policies and value estimates.

Training samples a trajectory segment and unrolls the model for five steps using the actions actually taken in that segment. Each unrolled prediction is supervised at its corresponding real timestep: observed reward, the recorded search policy, and a value target. Board-game value targets use the final outcome; Atari uses a bootstrapped return containing future real rewards and a later search value.

This creates a policy-improvement loop. Search can choose more carefully than a single network evaluation; training then teaches the policy head to approximate those improved choices. The latent transitions are optimized through their downstream reward, value, and policy losses, rather than through a requirement to reconstruct the next observation or match a separately encoded next-state vector.

### “Without rules” still has an environment interface

Appendix A makes a useful qualification to the headline. MuZero receives legal-action information at the root, where it must act in the real environment. It does not use a rules engine to mask actions at every imagined internal node. The policy is expected to learn which continuations are plausible from experience.

Terminal states receive similar treatment. Search does not invoke an internal game simulator to identify terminal nodes; it may continue expanding them, with absorbing-state training teaching the network the appropriate continuation value. The system still requires an environment that provides observations, rewards, and an action interface. It learns the dynamics used inside search, not the existence of the task from nothing.

The same distinction applies across domains. Board games use 800 simulations per move; Atari uses 50 and repeats a selected action for four emulator frames. The representation and action encoding depend on the domain. These are common algorithmic principles across separate training runs, not one checkpoint playing all games with identical inputs.

### Extra search helps only while the model supplies useful distinctions

The Go experiment increases thinking time far beyond the training search budget and continues to improve similarly to AlphaZero's perfect-model search. Atari behaves differently. The source panel below evaluates a fixed trained network with increasing simulations per move.

![MuZero source Figure 3B: Atari performance against the number of search simulations at evaluation](/assets/images/muzero-source-figure-3b-search-budget.png)
*Fig 2: Atari's aggregate score improves modestly and levels off around 100 simulations. The learned policy already performs well with one simulation; additional search has diminishing value. Cropped from the source ablation. | source: [MuZero, Figure 3B](https://arxiv.org/abs/1911.08265)*

Read the nearly flat right side as a deployment trade-off: more search calls do not buy proportionate improvement. The authors suggest greater model inaccuracy as one explanation. Strong performance with one simulation is also consistent with the policy head internalizing useful search behavior during training. This does not imply that search was unnecessary for producing that policy.

A separate Ms. Pacman ablation replaces the objective and heads with a model-free Q-learning setup while retaining the network size and training amount. It learns more slowly and reaches a lower score. That supports the search-based learning recipe against the tested control, but does not isolate every difference in losses, targets, and inference compute.

### Reanalyzing old experience trades interaction for fresh computation

The large-data Atari run reports 2,041.1% median and 4,999.2% mean human-normalized score with 20B environment frames per game. MuZero Reanalyze uses a different 200M-frame setting and reaches 731.1% median. Comparing the large-data result directly with a 200M-frame baseline would mix interaction budgets.

Reanalyze reruns search over past states using the latest model. Fresh search policies supply 80% of policy targets; a target network supplies fresher bootstrapped values. It also increases samples drawn per state from 0.1 to 2.0, reduces the value-loss weight, and shortens the return horizon. The gain is a combined sample-reuse recipe, not just relabeling one tensor.

Compute remains substantial. The report uses 16 third-generation TPUs for training and 1,000 for self-play per board game, versus eight and 32 respectively per Atari game. Its reported wall times therefore cannot be compared with another method's days of training without accounting for accelerator and actor counts.

## High-Level Takeaways

- MuZero trains latent dynamics for reward, value, and policy prediction, without requiring pixel or true-state reconstruction.
- Search supplies stronger policy targets; a good one-step policy after training can still owe its competence to search during learning.
- Legal actions are supplied at the real root, while internal search uses learned dynamics and policies rather than a full rules engine.
- More search keeps helping in Go but shows diminishing Atari returns, exposing the practical limits of the learned model and search budget.
- Reanalyze improves interaction efficiency through fresh search targets and additional reuse, with significant computation still required.
