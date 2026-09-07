---
title: 'Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning'
date: '2025-06-23T01:57:14.000Z'
section: paper-shorts
postSlug: drive-r1-bridging-reasoning-and-planning-in-vlms-for-autonomous-driving-with-reinforcement-learning
legacyPath: /paper shorts/2025/06/23/drive-r1-bridging-reasoning-and-planning-in-vlms-for-autonomous-driving-with-reinforcement-learning.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning"
---
## 2025 – Drive-R1

**arXiv:** [2506.18234](https://arxiv.org/abs/2506.18234)

## Summary

> Drive-R1 trains a small domain-specific VLM to reason from visual input to a driving plan, then reinforces it with trajectory- and meta-action-based rewards. Its supervised stage contains both long and short chains of thought; the reinforcement stage is intended to favor reasoning paths that improve planning rather than merely sound plausible. The paper reports superior results on nuScenes and DriveLM-nuScenes relative to its compared VLMs. On nuScenes validation it reports 0.31 m average L2 error and 0.09% collision, with evidence limited to logged-scene evaluations.

## Core Insights

### Rewards connect a reasoning trace to its trajectory

The paper starts from two failure hypotheses: VLMs may exploit historical input rather than visual evidence, and their chains of thought may be misaligned with the trajectories they generate. Long and short reasoning traces give the model alternative paths, while the RL reward connects those paths to trajectory and meta-action outcomes. This makes reasoning a policy component rather than an unscored explanation.

The supervised RP-CoT set contains 4,072 nuScenes samples with long and short reasoning traces; their training-set proportions are not reported. Each answer has a reasoning section followed by six waypoints over three seconds. GRPO then samples a group of outputs and combines four signals: trajectory L2, lateral/longitudinal meta-action correctness, repetition penalty, and format validity. The trajectory reward compares the full predicted path with the ground-truth trajectory; the meta-action reward separately checks the lateral and longitudinal decisions. That separation is useful because a fluent trace can still propose the wrong lane or speed.

On nuScenes validation, Drive-R1 reaches average L2 0.31 m and collision 0.09%, versus EMMA’s 0.32 m and an unavailable collision result in the reported table. The reward ablation reduces collision from 0.18% with trajectory/format rewards to 0.10% after repetition and meta-action rewards; 24 rollouts reaches 0.08% before training becomes unstable. The visual-input control is the paper’s most important warning: a trajectory model trained without CoT performs better when images are removed, so the later alignment gains should be read as an attempt to repair a measured shortcut, not as evidence that every chain of thought is grounded.

Read the overview as a curriculum with a specific causal handoff. SFT first teaches the small VLM the output grammar and the long/short reasoning formats; GRPO then compares several sampled answers using driving rewards. The reinforcement stage can select a plan that is geometrically better or less repetitive, but it cannot by itself prove that the accompanying text caused the improvement.

![Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning source figure: The overview of the proposed Drive-R1 , which comprises the supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) phases.](/assets/images/drive-r1-bridging-reasoning-and-planning-in-vlms-for-autonomous-driving-with-reinforcement-learning-paper-figure.webp)
*Fig 1: Drive-R1 builds a long/short reasoning-and-trajectory dataset, performs supervised domain alignment, and then applies GRPO with trajectory and meta-action rewards. The split between SFT and RFT mirrors the paper’s claim that a domain-adapted planner is needed before reinforcement can safely shape reasoning. | source: [Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning, Figure 3](https://arxiv.org/abs/2506.18234)*

The image-removal comparison tests a more basic question than explanation quality: is the planner using its cameras? Better performance without images in this no-CoT model means that history can support a strong prediction while visual input becomes a distraction. This control motivates alignment, but does not by itself show that the later policy has eliminated the shortcut.

![Figure 1 from Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning](/assets/images/drive-r1-bridging-reasoning-and-planning-in-vlms-for-autonomous-driving-with-reinforcement-learning-source-figure-1.webp)
*Fig 2: Inference results with and without visual inputs from the model which is trained to predict trajectory without chain of thoughts. | source: [Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning, Figure 1](https://arxiv.org/abs/2506.18234)*


The image-removal control detects a visual shortcut, but it does not establish how much of the trajectory gain is caused by the reasoning trace or whether reasoning remains useful under counterfactual history edits. An appropriate ablation would compare visual-only, history-only, and jointly conditioned policies with matched trace lengths and rewards, then test whether visual edits alter both the trace and plan consistently.

## High-Level Takeaways

- Drive-R1 uses supervised chains of thought plus trajectory-aware reinforcement learning to connect visual reasoning with a driving plan.
- Its reported nuScenes and DriveLM results support that alignment objective, but the image-removal control shows why current visual evidence cannot be assumed to drive a good trajectory score.
- The key falsification is a counterfactual visual-and-history evaluation: if the plan remains unchanged when the causal visual evidence changes, a planning-aligned trace is still only a plausible narration.
