---
title: 'Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: eagle-2-post-training-data-strategies-for-frontier-vision-language-models
legacyPath: /paper shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html
tags:
  - Other
field: Vision-Language Models
summary: Eagle 2 made the post-training data recipe the main contribution, showing how curation can move frontier VLM performance.
---
## 2025 - Eagle 2

**arXiv:** [2501.14818](https://arxiv.org/abs/2501.14818)

**GitHub:** [NVlabs/Eagle](https://github.com/NVlabs/EAGLE)

**Plain-language summary:** Eagle 2 argues that frontier VLM performance depends heavily on post-training data strategy. Instead of presenting only final weights, the paper documents how the team builds and balances instruction data, benchmark-oriented data, visual reasoning tasks, and embodied/driving-style examples.

The main lesson is that strong VLMs are not just pretrained once and then lightly tuned. Their behavior is shaped by a careful curriculum of visual tasks and response styles.

**Why it mattered:** Eagle 2 is valuable because it makes the data engineering visible. That helps turn VLM building from folklore into something closer to an inspectable recipe.

**Take-home message:** Post-training is where a general multimodal model becomes useful. The data mixture is a control surface.
