---
title: Playing Atari with Deep Reinforcement Learning
date: '2013-12-19T00:00:00.000Z'
section: paper-shorts
postSlug: playing-atari-with-dqn
legacyPath: /paper shorts/2013/12/01/playing-atari-with-dqn.html
tags:
  - Other
field: 'Reinforcement Learning'
summary: "2013 – Playing Atari with Deep Reinforcement Learning"
---

## 2013 – Playing Atari with Deep Reinforcement Learning

**Paper:** [arXiv:1312.5602](https://arxiv.org/abs/1312.5602) · NIPS Deep Learning Workshop, 2013

## Summary

> The 2013 DQN paper shows that a convolutional Q-network trained from pixels and experience replay can outperform earlier methods on six of seven Atari games and a human expert on three. Its generality claim is one architecture and learning recipe trained separately for each game, not one shared set of weights playing all seven. The original network has two convolutional layers; Algorithm 1 does not contain the periodically synchronized separate target network commonly associated with later DQN. Keeping those versions distinct makes the actual contribution clearer: end-to-end visual value learning became practical through replay and a largely shared training setup.

## Core Insights

### Learn features for choosing an action, not for reconstructing a screen

A single Atari frame can show where the ball is without revealing where it is moving. DQN stacks four recent preprocessed frames, giving its Q-network a short visual history. Frames are converted to grayscale, downsampled, and cropped to an 84×84 playing region. This is learned control from pixels, but not unprocessed emulator input or access to its hidden state.

The original network applies 16 filters of size 8×8 with stride four, then 32 filters of size 4×4 with stride two. A 256-unit fully connected hidden layer feeds one linear output per legal action. For a channels-first implementation, the tensor path is $4\times84\times84\rightarrow16\times20\times20\rightarrow32\times9\times9\rightarrow256\rightarrow|\mathcal A|$.

All action values share the expensive visual computation. An alternative network taking both a state and a candidate action would require a separate forward pass for each action. Here, one pass gives every legal action's estimated return, and greedy selection takes the largest. The network has to learn which visual differences change those returns; there is no separate object-labeling or image-reconstruction target.

### Replay changes both sample reuse and the distribution of updates

The replay buffer stores transitions from many episodes and samples them uniformly for minibatch updates. This reuses an interaction more than once, reduces correlations between adjacent updates, and smooths the training distribution across several past behaviors.

The last role is easy to overlook. If a policy begins moving left, its next observations mostly describe the left side. Training only on that stream can immediately reinforce the preference that produced it. Replay supplies a mixture of earlier states, weakening that rapid feedback between the current policy and its next update. It does not make the data independent or remove all distribution shift.

For a nonterminal transition, the bootstrapped target has the familiar form

$$
y=r+\gamma\max_{a'}Q(s',a';\theta),
\qquad
L=(y-Q(s,a;\theta))^2.
$$

The target is treated as fixed while differentiating the current prediction. The background section writes this using previous-iteration parameters, but the 2013 algorithm does not specify a second network held unchanged for a periodic synchronization interval. Importing that later stabilization mechanism would misdescribe this paper's experiment.

Q-learning is off-policy here: a stored transition may come from an older exploratory policy, while its target evaluates a greedy continuation. The buffer contains the most recent million frames, so uniform replay also has limits. Rare decisive transitions may be overwritten or sampled too infrequently; the paper itself identifies unequal learning value across transitions as a reason to investigate better sampling.

### Shared hyperparameters conceal a deliberate change to the learning objective

The experiments use RMSProp, minibatches of 32, and ten million training frames. Exploration decreases linearly from fully random to an epsilon of 0.1 over the first million frames. Positive training rewards are clipped to +1 and negative rewards to −1, while evaluation reports ordinary game scores.

Reward clipping makes update scales more comparable across games, helping one learning rate work across different score systems. It also discards reward magnitude: a small positive event and a large positive event receive the same immediate training signal. That is a trade-off in the learned objective, not simply numerical bookkeeping.

Frame skipping reduces how often the network must act. The last action is repeated for four frames in most games. Space Invaders uses three because four-frame sampling makes blinking lasers invisible. This exception is revealing: the broad reuse claim remains useful, but sensor sampling can erase a relevant event even when the representation learner is expressive enough to recognize it.

### The value function anticipates reward rather than tracking accumulated score

The source's Seaquest example follows a short event sequence. At A, an enemy appears and predicted value rises. At B, the torpedo is about to hit and value peaks. At C, the enemy has disappeared and value returns near its earlier level.

![DQN source Figure 3: predicted value around the appearance and destruction of a Seaquest enemy](/assets/images/playing-atari-with-dqn-source-figure-3.webp)
*Fig 1: Value rises when an enemy creates a reward opportunity, peaks before the hit, and drops after that opportunity is consumed. This source panel plots the remaining expected return, not cumulative game score. | source: [Playing Atari with Deep Reinforcement Learning, Figure 3](https://arxiv.org/abs/1312.5602)*

The drop after the hit is therefore not necessarily evidence that the action was bad. A value estimate describes reward still available from the current state. Consuming an opportunity can increase realized score while reducing future value. The plot is an intuitive qualitative check; it does not establish that all predicted values are calibrated.

Figure 2 separately shows noisy episode rewards alongside smoother average predicted Q-values on a fixed state set. Smooth Q-values are easier to monitor, but can look reassuring even when values are systematically wrong. The authors observed no divergence in these runs; that observation is weaker than a convergence guarantee for nonlinear off-policy learning.

### Compare the same evaluation statistic

Table 1's main rows use average scores under an epsilon-greedy evaluation policy with epsilon 0.05. Its lower rows report the best single episode for comparison with deterministic evolutionary policies. Those are different statistics.

| Game | DQN average score | Human reference score |
| --- | ---: | ---: |
| Breakout | 168 | 31 |
| Enduro | 470 | 368 |
| Pong | 20 | −3 |
| Seaquest | 1,705 | 28,010 |
| Q*bert | 1,952 | 18,900 |

The human reference is the median score after roughly two hours of play per game. DQN exceeds it on Breakout, Enduro, and Pong, while remaining far behind on Seaquest and Q*bert. This establishes meaningful cross-task reuse of a recipe, with substantial remaining long-horizon weaknesses. It does not show multitask transfer through shared weights or human-level sample efficiency.

## High-Level Takeaways

- The 2013 result reuses architecture and training choices across separately trained games; it is not a single multitask policy.
- Four-frame visual input and one output per action make value-based control tractable without hand-engineered object features.
- Replay smooths the update distribution across past behavior as well as reusing samples; uniform finite memory still loses or undersamples important events.
- Reward clipping and frame skipping help standardize training but change reward priorities and observability.
- Distinguish the original two-convolution replay-based paper from later target-network DQN, and compare average scores with averages rather than best episodes.
