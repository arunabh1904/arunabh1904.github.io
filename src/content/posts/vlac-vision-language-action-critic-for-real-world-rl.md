---
title: 'VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning'
date: '2025-09-19T09:00:00.000Z'
section: paper-shorts
postSlug: vlac-vision-language-action-critic-for-real-world-rl
legacyPath: /paper shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html
tags:
  - Robotics
  - Reward Models
field: 'Robot Post-Training & Evaluation'
summary: 2025 – VLAC predicts dense visual progress and completion signals inside an asynchronous real-robot RL loop.
---

## 2025 – VLAC: A Vision-Language-Action-Critic Model for Real-World Reinforcement Learning

**arXiv:** [2509.15937](https://arxiv.org/abs/2509.15937)

VLAC turns a multimodal model into a process critic. Given a language goal and two observations, it predicts signed progress and completion; prompt control also lets the same autoregressive model emit actions. Training mixes vision-language tasks, more than 4,000 hours of robot/human trajectories, and constructed negatives for regressions, stagnation, irrelevant goals, and semantic mismatches.

## Paper Insights

The critic sits inside an asynchronous real-world RL system with graded human support: demonstration replay, return-and-explore, and human-guided exploration. Across four real manipulation tasks, the paper reports improvement from roughly 30% to roughly 90% success within 200 interaction episodes; human intervention improves sample efficiency by about 50% and final success reaches as high as 100%.

Dense progress is more informative than terminal success, but also easier to exploit. A model can reward visual motion, object proximity, or familiar subtask order without understanding contact or irreversible damage. VLAC's negative construction is therefore as important as its scale.

| Critic output | Intended role | Failure risk |
| --- | --- | --- |
| Progress delta | Dense learning signal | Rewards visible motion without causal progress |
| Done probability | Episode termination | Confuses appearance with completion |
| Action tokens | Shared actor interface | Actor and critic errors become correlated |

## Decision Lens

VLAC informs whether to hand-engineer rewards, learn a task-specific success detector, or train a general visual-language process critic. Its atomic unit is a pair of temporally related observations plus a goal; progress labels come from ordering and curated negatives. Actor and critic share one model interface, buying transfer while increasing correlated-failure risk.

The real-robot loop establishes promising sample efficiency on four tasks, not universal reward validity. A missing ablation gives the critic structured geometry/contact state and tests whether that changes reward hacking and transfer. At ten times the task diversity, visually similar but physically different progress will dominate. The central claim fails if critic score improves while blinded human success, safety, or intervention rate does not.

**Context:** VLAC makes the learned critic—not the policy—the central reusable model in real-world post-training.

**Limits:** Progress supervision inferred from time can label pauses or necessary backtracking incorrectly.

**Takeaway:** A general critic needs explicit negative cases for regression, stagnation, and goal mismatch; scale alone does not make progress causal.
