---
title: "Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning"
date: '2026-08-16T00:00:00.000Z'
section: paper-shorts
postSlug: unified-pedestrian-path-prediction-using-inverse-reinforcement-learning
legacyPath: /paper shorts/2026/08/16/unified-pedestrian-path-prediction-using-inverse-reinforcement-learning.html
tags:
  - Autonomous Driving
  - Pedestrian Prediction
  - Reinforcement Learning
field: 'Motion Forecasting & Planning'
summary: "2026 – Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning"
---

**arXiv:** [2608.15929](https://arxiv.org/abs/2608.15929)

## Summary

> This study asks whether a strong social trajectory architecture benefits from a decision formulation rather than only supervised coordinate regression. It adapts STGAT to one-time and step-wise decision processes, uses squared distance to expert pedestrian trajectories as the reward signal, and compares supervised learning, REINFORCE, PPO, value baselines, and discount settings. The best reported average minADE/minFDE comes from PPO with a full-state baseline (0.5227/1.0551), while the step-wise supervised formulation at lambda=0.4 reaches 0.5079/1.0822. The paper is a controlled objective study, not a new perception backbone or a learned reward-discovery system.

## Core Insights

### The change is the decision granularity around STGAT

STGAT uses a G-LSTM for temporal interaction, an M-LSTM for spatial interaction, and a D-LSTM decoder. The paper observes eight positions for every pedestrian and predicts the next twelve. In one-time decision making (ODM), a single action contains all twelve future positions. In single decision making (SDM), each next position is an action and the episode lasts twelve steps; the state grows by appending each predicted position.

The distinction changes where the objective places pressure. ODM asks the policy to choose a complete path at once. SDM exposes the sequence of local decisions and applies a discount factor to cumulative error. The paper calls its reward an inverse-reinforcement-learning construction because expert behavior is used to define the reward, but the reward is an explicit squared L2 distance to ground truth rather than a learned discriminator or an inferred latent human cost.

![The STGAT encoder is reused while the batch is decomposed into per-pedestrian decision episodes](/assets/images/unified-pedestrian-path-prediction-paper-figure.webp)
*Fig 1: The formulation keeps a shared scene encoder, then evaluates one trajectory-level or twelve step-wise policy episode per pedestrian using the same MDP template. | source: [Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning, Figure 1](https://arxiv.org/abs/2608.15929)*

### PPO's value baseline matters more than stochasticity alone

The experiment separates the variance of policy gradients from the benefit of a value estimate. On the five ETH/UCY subsets, plain supervised STGAT averages 0.5334 minADE and 1.0843 minFDE. The stochastic REINFORCE-style SL-MSE-SPG formulation averages 0.5598 and 1.1144, and the authors attribute the regression to high-variance updates and a fixed 150-epoch fine-tuning budget.

PPO with a full-state baseline averages 0.5227 minADE and 1.0551 minFDE, compared with 0.5370 and 1.0844 for PPO without a baseline. REINFORCE with a full-state baseline reaches 0.5350 and 1.0840. The value network reuses the STGAT encoder and adds a shallow three-layer dense network for the whole scene; the simplified baseline sees only one pedestrian's observed state. The full state helps REINFORCE substantially on Univ and Hotel, while the two baselines are nearly equivalent for PPO. The mechanism is therefore variance and credit assignment in the decision objective, not a generic claim that every RL method beats supervised learning.

| Formulation | Average minADE | Average minFDE |
| --- | ---: | ---: |
| SL-MSE | 0.5334 | 1.0843 |
| SL-MSE-SPG | 0.5598 | 1.1144 |
| REINFORCE + full-state baseline | 0.5350 | 1.0840 |
| PPO + full-state baseline | **0.5227** | **1.0551** |
| PPO without baseline | 0.5370 | 1.0844 |

### Discounting changes which pedestrian decisions receive learning pressure

The SDM ablation varies lambda over 0.2, 0.4, 0.6, and 0.8. The lambda=0.4 row averages 0.5079 minADE and 1.0822 minFDE, while lambda=0.2 gives 0.5099 and 1.1010, lambda=0.6 gives 0.5126 and 1.0875, and lambda=0.8 gives 0.5178 and 1.0921. The improvement is not simply “more discount is better.” The paper's interpretation is that the cumulative SDM loss concentrates useful gradient on early predicted steps, whose errors propagate into the final position.

The evaluation uses ETH and UCY subsets ETH, Hotel, Zara1, Zara2, and Univ. Predictions are sampled 20 times per pedestrian for minADE/minFDE, and the reported values average five training runs. The first two STGAT pretraining phases use 250 epochs; the third formulation-specific phase uses 150 epochs and is fine-tuned on ETH train/validation. That last choice is material: improvements on ETH need not transfer equally to Hotel, Zara, or Univ, and the authors themselves attribute some variation to fine-tuning and policy-gradient variance.

This study therefore supports a narrower decision. If a trajectory model already represents interactions, an objective that exposes sequential decisions and a well-trained value baseline can improve offline errors. It does not show that the learned policy interacts with pedestrians, discovers a human reward, or improves a vehicle's closed-loop safety behavior.

## High-Level Takeaways

- The main intervention is state/action design around STGAT: twelve future coordinates at once versus twelve sequential decisions.
- PPO's full-state value baseline is the strongest matched RL result; stochastic sampling by itself underperforms the supervised baseline in the reported average.
- The lambda=0.4 SDM result improves minADE most, supporting the intuition that early pedestrian decisions shape later displacement.
- Results use five ETH/UCY subsets, 20 samples per pedestrian, five runs, and ETH fine-tuning; cross-dataset and closed-loop conclusions require care.
- The paper's “IRL” signal is an expert-distance reward, so its evidence concerns optimization and credit assignment rather than recovery of an unknown social cost function.
