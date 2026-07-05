---
title: 'GPT-Driver: Learning to Drive with GPT'
date: '2023-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: gpt-driver-learning-to-drive-with-gpt
legacyPath: /paper shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html
tags:
  - Other
field: Autonomous Driving
summary: GPT-Driver reframed motion planning as language modeling over scene tokens and future waypoints.
---
## 2023 - GPT-Driver

**arXiv:** [2310.01415](https://arxiv.org/abs/2310.01415)

**GitHub:** [PointsCoder/GPT-Driver](https://github.com/PointsCoder/GPT-Driver)

**Plain-language summary:** GPT-Driver asks whether a language model can act as a motion planner when the driving scene is serialized into tokens. Instead of directly predicting a trajectory with a specialized planner, the system prompts and fine-tunes GPT-style models to produce future waypoints and rationales.

This is not a deployable AV stack by itself. It is a useful probe: language models can absorb structured scene descriptions and generate plausible plans, but latency, grounding, and closed-loop reliability remain hard.

**Why it mattered:** It opened a line of work where language is not just for explanation after the fact. It becomes an intermediate representation for planning.

**Take-home message:** LLMs can help expose the reasoning behind a plan, but driving needs that reasoning to stay grounded, fast, and controllable.
