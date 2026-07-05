---
title: 'EMMA: End-to-End Multimodal Model for Autonomous Driving'
date: '2024-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: emma-end-to-end-multimodal-model-for-autonomous-driving
legacyPath: /paper shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: EMMA represented driving inputs and outputs as language tokens so one multimodal model could handle planning, perception, and road structure tasks.
---
## 2024 - EMMA

**arXiv:** [2410.23262](https://arxiv.org/abs/2410.23262)

**Project:** [Waymo research page](https://waymo.com/research/emma/)

**Plain-language summary:** EMMA is Waymo's end-to-end multimodal driving model. It uses camera data plus non-sensor state such as navigation commands and ego status, then predicts driving outputs including trajectories, objects, and road graph elements through task-specific prompts.

The striking design choice is to represent many non-sensor inputs and outputs as text. That lets the model reuse the structure and world knowledge of a multimodal language model while training across several driving tasks.

**Why it mattered:** EMMA is a strong example of the generalist-model thesis entering autonomous driving: one model, many outputs, shared representations.

**Take-home message:** Language can be a unifying interface for driving tasks, but EMMA also makes the costs obvious: limited temporal context, no full 3D sensor stack, and heavy compute.
