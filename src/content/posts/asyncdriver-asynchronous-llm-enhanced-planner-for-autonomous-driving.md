---
title: 'AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving'
date: '2024-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving
legacyPath: /paper shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: AsyncDriver separated slow LLM reasoning from fast motion planning so semantic guidance could enter the loop without blocking control.
---
## 2024 - AsyncDriver

**arXiv:** [2406.14556](https://arxiv.org/abs/2406.14556)

**Plain-language summary:** AsyncDriver addresses a practical problem: LLMs are too slow to sit directly in a high-frequency driving control loop. The system runs a fast planner continuously while a slower LLM produces scene-associated instructions asynchronously.

Those instructions can guide the planner through complex or ambiguous situations without requiring the LLM to produce every control update.

**Why it mattered:** It is a sober architecture. Instead of pretending language models are real-time controllers, AsyncDriver gives them a slower advisory role.

**Take-home message:** In autonomy, reasoning and control do not need to run at the same frequency. VLM/LLM components can be useful if the system boundary respects latency.
