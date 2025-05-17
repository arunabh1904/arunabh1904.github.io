---
layout: post
title: "Paper Short: Playing Atari with Deep Reinforcement Learning"
date: 2013-12-01 00:00:00 -0500
categories: ["Paper Shorts"]
---

## 2013 – Playing Atari with Deep Reinforcement Learning

**Abstract (arXiv):** This paper introduced the Deep Q-Network (DQN), the first deep learning model to successfully learn control policies directly from high-dimensional visual input using reinforcement learning. A convolutional neural network was trained with a variant of Q-learning to play Atari 2600 video games from raw pixel inputs, outputting Q-value estimates for possible actions. No game-specific tuning was needed; the same architecture trained on seven Atari games outperformed all previous approaches on six of them and surpassed human expert scores on three games.

**GitHub:** (Unofficial) deepmind/Atari-DQN (reimplementation)

**Project Page:** DeepMind blog – “Playing Atari with Deep RL” (2013)

**Summary:** The DQN uses a CNN to process frames and a Q-learning objective with experience replay and target networks. By learning value estimates for pixel states, it achieved superhuman performance in several classic Atari games using only screen images and game rewards. Key innovations include using a replay memory to stabilize training and an iterative update to align the Q-network with a slowly updating target network. This work demonstrated that deep neural networks can learn complex control policies from unprocessed pixels.

**Novel Insights:**
- End-to-end learning from pixels to game actions via reinforcement learning, without handcrafted features.
- Introduced techniques (experience replay, target network) to stabilize deep RL training, enabling learning of successful policies in a high-dimensional state space.
- Showed that increasing network depth and complexity can significantly improve performance in RL tasks previously dominated by specialized methods.

**Evals / Benchmarks:** DQN achieved higher scores than prior algorithms on six out of seven Atari games tested and beat human expert performance on games like Breakout and Enduro. These results were achieved with a single architecture across games, whereas previous methods required game-specific adjustments. Training required on the order of 10 million frames per game.

**Critiques:** The DQN required long training times and still struggled on games requiring long-term planning (for example Montezuma’s Revenge with sparse rewards). It could be unstable without careful hyperparameter tuning. Later research found that DQN’s replay mechanism could be improved (for instance prioritized replay) and that it overestimates some Q-values. Nonetheless, it was a landmark result in deep RL.

**Additional Info:** ArXiv Dec 2013. Presented at NIPS Deep Learning Workshop 2013 (full paper in Nature 2015).

