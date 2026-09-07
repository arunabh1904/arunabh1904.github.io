---
title: 'VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning'
date: '2025-09-19T00:00:00.000Z'
section: paper-shorts
postSlug: vlac-vision-language-action-critic-for-real-world-rl
legacyPath: /paper shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html
tags:
  - Robotics
  - Reward Models
field: 'Robot Post-Training & Evaluation'
summary: "2025 – VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning"
---

**arXiv:** [2509.15937](https://arxiv.org/abs/2509.15937)

## Summary

> VLAC turns a multimodal model into a process critic. Given a language goal and two observations, it predicts signed progress and completion; prompt control also lets the same autoregressive model emit actions. Training mixes vision-language tasks, more than 4,000 hours of language-annotated manipulation data, and constructed negatives for regressions, stagnation, irrelevant goals, and semantic mismatches.

## Core Insights

### A critic learns relative progress

![VLAC training and deployment overview combining progress-understanding data with action generation and dense rewards for real-world reinforcement learning](/assets/images/vlac-vision-language-action-critic-for-real-world-rl-paper-figure.png)
*Fig 1: Shows the model's dual role: mixed robot and vision-language data teach task progress and actions, then the same network supplies dense progress rewards while acting as a policy in real-world RL. | source: [VLAC, Figure 1](https://arxiv.org/abs/2509.15937)*

![Figure 5 from VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning](/assets/images/vlac-vision-language-action-critic-for-real-world-rl-source-figure-5.webp)
*Fig 2: Example results of VLAC for task progress understanding across entities, scenes, and successful or failed processes. | source: [VLAC, Figure 5](https://arxiv.org/abs/2509.15937)*

![Figure 3 from VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning](/assets/images/vlac-vision-language-action-critic-for-real-world-rl-source-figure-3.webp)
*Fig 3: VLAC forward pass generates structured action tokens, reward tokens, and a value head is attached to estimate state value for PPO updates. | source: [VLAC, Figure 3](https://arxiv.org/abs/2509.15937)*


VLAC is trained on more than 3,000 hours of human data, 1,200 hours of public robot data, and more than 15 hours of self-collected manipulation data; the authors sample 40 million training examples from that mixture. The critic is an 8B model in the RL experiments. On RoboFAC it separates successful from failed videos with VOC-F1 0.89 versus 0.44, which is more diagnostic than a terminal success label because it tests whether the score tracks a process rather than only its endpoint.

### The reward is a model interface

The critic sits inside an asynchronous real-world RL system with graded human support: demonstration replay, return-and-explore, and human-guided exploration. Across four real manipulation tasks, the paper reports improvement from roughly 30% to roughly 90% success within 200 interaction episodes; human intervention improves sample efficiency by about 50% and final success reaches as high as 100%. Figure 3 makes the interface concrete: action tokens and reward tokens share the autoregressive trunk, while a value head supplies the PPO baseline.

Dense progress is more informative than terminal success, but also easier to exploit. A model can reward visual motion, object proximity, or familiar subtask order without understanding contact or irreversible damage. VLAC's negative construction is therefore as important as its scale.

| Critic output | Intended role | Failure risk |
| --- | --- | --- |
| Progress delta | Dense learning signal | Rewards visible motion without causal progress |
| Done probability | Episode termination | Confuses appearance with completion |
| Action tokens | Shared actor interface | Actor and critic errors become correlated |

## High-Level Takeaways

- VLAC informs whether to hand-engineer rewards, learn a task-specific success detector, or train a visual-language process critic. Its atomic unit is a pair of observations plus a goal; progress labels come from ordering and curated negatives. Sharing the actor and critic interface buys transfer while creating correlated-failure risk.
- The strongest evidence is the combination of RoboFAC discrimination (VOC-F1 0.89 on successful videos versus 0.44 on failed ones) and the four-task RL loop. A reward-model audit should compare critic gains with blinded completion, contact safety, and intervention rate, because a denser score can still reward visible motion.
- Time ordering is a useful source of scale but a real semantic assumption: pauses, retries, and necessary backtracking can receive the wrong sign. The image-difference filter reduces static-frame noise; it does not make progress causal.
- VLAC makes the learned critic a reusable post-training interface. An independent critic or privileged contact/geometry ablation would test whether the shared model is improving exploration or merely sharing the policy's visual shortcuts.
- The paper's useful boundary is explicit negative construction: regression, stagnation, irrelevant goals, and semantic mismatch matter as much as adding more demonstrations.
