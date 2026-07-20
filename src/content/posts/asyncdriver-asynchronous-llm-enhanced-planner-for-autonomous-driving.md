---
title: 'AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving'
date: '2024-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving
legacyPath: /paper shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2024 – AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving"
---
## 2024 – AsyncDriver

**arXiv:** [2406.14556](https://arxiv.org/abs/2406.14556)

**Summary:** AsyncDriver addresses a practical problem: LLMs are too slow to sit directly in a high-frequency driving control loop. The system runs a fast planner continuously while a slower LLM produces scene-associated instructions asynchronously.

Those instructions can guide the planner through complex or ambiguous situations without requiring the LLM to produce every control update.

## Paper Insights

AsyncDriver uses an LLM as a high-level driving advisor without putting that LLM directly in the synchronous planning loop. The planner keeps generating trajectories while the language model asynchronously contributes scene reasoning or strategic guidance. This design tries to preserve real-time control while still benefiting from language-level interpretation of traffic context. The key tradeoff is staleness: asynchronous advice can reduce latency, but the system must know when guidance no longer matches the current scene. The paper matters as a pattern for using foundation models near safety-critical loops instead of making every control update wait on them.

![Figure 2: Overview of our proposed AsyncDriver framework from AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving](/assets/images/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving-paper-figure.png)
_Figure 2: Overview of our proposed AsyncDriver framework. From the [AsyncDriver: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving paper](https://arxiv.org/abs/2406.14556), via arXiv HTML._

**What to look at:**
- The key design is two clocks: fast planner, slow LLM.
- LLM outputs scene-associated instructions instead of direct controls.
- The latency solution is the contribution.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Runtime design | Asynchronous reasoning loop | Avoids blocking high-frequency planning. |
| Planner role | Fast conventional motion planner | Keeps control responsive. |
| Caveat | LLM advice must be bounded | Bad slow guidance should not override safety. |

## Decision Lens

AsyncDriver informs where a slow semantic model can enter a fast control stack without setting the control-loop latency. The operative units run at two clocks: infrequent LLM guidance summarizes scene-level intent, while a conventional planner produces high-frequency trajectories from current perception.

The asynchronous boundary buys latency but risks stale guidance during rapid scene changes. The decisive ablation varies guidance age and event-trigger policy while matching planner capacity against no-LLM and synchronous-LLM baselines in closed loop. At 10× traffic complexity, semantic updates and planner state can diverge. The architecture would fail if delayed guidance produced no safety or progress gain over a smaller fast planner, or if rare stale-command failures erased the average benefit.

**Context:** It is a sober architecture. Instead of pretending language models are real-time controllers, AsyncDriver gives them a slower advisory role.

**Takeaway:** In autonomy, reasoning and control do not need to run at the same frequency. VLM/LLM components can be useful if the system boundary respects latency.
