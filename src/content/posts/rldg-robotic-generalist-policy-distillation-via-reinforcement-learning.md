---
title: 'RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning'
date: '2024-12-13T09:00:00.000Z'
section: paper-shorts
postSlug: rldg-robotic-generalist-policy-distillation-via-reinforcement-learning
legacyPath: /paper shorts/2024/12/13/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning.html
tags:
  - Robotics
  - Distillation
field: 'Robot Post-Training & Evaluation'
summary: "2024 – RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning"
---

## 2024 – RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning

**arXiv:** [2412.09858](https://arxiv.org/abs/2412.09858)

**Project:** [generalist-distillation.github.io](https://generalist-distillation.github.io/)

RLDG avoids applying unstable RL updates directly to a large generalist policy. It trains smaller task-specific RL specialists, rolls them out to collect higher-quality and broader-coverage trajectories, and distills those trajectories back into the generalist with supervised fine-tuning.

## Paper Insights

On precise insertion and assembly tasks, the paper reports that generalists trained on RL-generated data outperform those trained on human demonstrations by as much as 40%. Analysis attributes the gain to both cleaner action distributions and state coverage: RL specialists visit states and execute corrections that teleoperators may not demonstrate consistently.

The separation also protects general capabilities. The specialist can optimize aggressively under a task reward; the generalist only consumes selected trajectories. The cost is maintaining one RL pipeline per task and deciding which specialist behavior is safe to distill.

| Stage | Model | Objective |
| --- | --- | --- |
| Specialist learning | Task-specific policy | Maximize environment reward |
| Data generation | Converged specialist | Produce high-quality state/action coverage |
| Distillation | Generalist policy | Imitate selected RL trajectories |

## Decision Lens

RLDG informs whether RL should update a generalist directly or serve as a data-generation tool. Its atomic unit is the distilled specialist trajectory. Parameters are not shared during RL; knowledge crosses the boundary through curated experience.

The results establish that specialist-generated data can beat human demonstrations on precise tasks. A missing comparison holds total robot hours constant across direct generalist RL, specialist distillation, and correction SFT. At ten times the task count, training specialists becomes the bottleneck and their styles may conflict. The thesis fails if distilled gains vanish on new tasks or if direct post-training achieves equal improvement without forgetting at lower systems cost.

**Context:** RLDG turns RL into a data flywheel rather than insisting it be the final optimizer.

**Limits:** Task rewards and specialist training remain expensive to engineer and validate.

**Takeaway:** When direct RL is too brittle for a foundation policy, let RL improve the data before it improves the model.
