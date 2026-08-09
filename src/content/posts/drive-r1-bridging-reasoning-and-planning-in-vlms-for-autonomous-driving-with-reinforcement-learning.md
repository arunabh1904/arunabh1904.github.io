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

Drive-R1 trains a small domain-specific VLM to reason from visual input to a driving plan, then reinforces it with trajectory- and meta-action-based rewards. Its supervised stage contains both long and short chains of thought; the reinforcement stage is intended to favor reasoning paths that improve planning rather than merely sound plausible. The paper reports superior results on nuScenes and DriveLM-nuScenes relative to its compared VLMs. The abstract does not provide planning metrics, reward weights, or a closed-loop evaluation.

## Core Insights

The paper starts from two failure hypotheses: VLMs may exploit historical input rather than visual evidence, and their chains of thought may be misaligned with the trajectories they generate. Long and short reasoning traces give the model alternative paths, while the RL reward connects those paths to trajectory and meta-action outcomes. This makes reasoning a policy component rather than an unscored explanation.

The crucial causal question remains open. The abstract does not say how visual shortcuts are detected, what portion of the trajectory reward is attributable to the trace, or whether reasoning remains useful when history is masked or counterfactually edited. An appropriate ablation would compare visual-only, history-only, and jointly conditioned policies with matched trace lengths and rewards, then test whether visual edits alter both the trace and plan consistently.

## High-Level Takeaways

- Drive-R1 uses supervised chains of thought plus trajectory-aware reinforcement learning to connect visual reasoning with a driving plan.
- Its reported nuScenes and DriveLM results support that alignment objective, but the abstract does not establish that the model relies on current visual evidence rather than historical shortcuts.
- The key falsification is a counterfactual visual-and-history evaluation: if the plan remains unchanged when the causal visual evidence changes, a planning-aligned trace is still only a plausible narration.
