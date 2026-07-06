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

## Paper map

GPT-Driver reformulates motion planning as GPT-style sequence generation. It serializes structured scene state into language-model tokens and predicts future waypoints plus a rationale. This gives the model an interpretable interface: the generated plan can be paired with an explanation of the driving decision. The evidence focuses on open-loop planning quality. The caveat is that open-loop waypoint prediction does not prove closed-loop safety, and LLM latency remains a deployment problem. The paper is useful as an early example of adapting pretrained language models to structured planning rather than raw perception.

![Figure 1: Overview of GPT-Driver from GPT-Driver: Learning to Drive with GPT](/assets/images/gpt-driver-learning-to-drive-with-gpt-paper-figure.png)
_Figure 1: Overview of GPT-Driver. From the [GPT-Driver: Learning to Drive with GPT paper](https://arxiv.org/abs/2310.01415), via arXiv HTML._

**What to look at:**
- Driving scene state is serialized into language tokens.
- The model predicts future waypoints and rationales, not low-level control directly.
- Open-loop planning metrics are useful but do not prove closed-loop safety.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Input | Structured scene tokens | Makes the driving problem legible to GPT-style models. |
| Output | Future waypoints plus rationale | Adds interpretability to motion planning. |
| Caveat | Open-loop and LLM latency | Needs closed-loop validation before real deployment. |

**Why it mattered:** It opened a line of work where language is not just for explanation after the fact. It becomes an intermediate representation for planning.

**Take-home message:** LLMs can help expose the reasoning behind a plan, but driving needs that reasoning to stay grounded, fast, and controllable.
