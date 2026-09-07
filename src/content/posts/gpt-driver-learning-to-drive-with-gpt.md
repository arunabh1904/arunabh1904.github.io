---
title: 'GPT-Driver: Learning to Drive with GPT'
date: '2023-10-02T00:00:00.000Z'
section: paper-shorts
postSlug: gpt-driver-learning-to-drive-with-gpt
legacyPath: /paper shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2023 – GPT-Driver: Learning to Drive with GPT"
---
## 2023 – GPT-Driver

**arXiv:** [2310.01415](https://arxiv.org/abs/2310.01415)

**GitHub:** [PointsCoder/GPT-Driver](https://github.com/PointsCoder/GPT-Driver)

## Summary

> GPT-Driver asks whether a language model can act as a motion planner when the driving scene is serialized into tokens. Instead of directly predicting a trajectory with a specialized planner, the system prompts and fine-tunes GPT-style models to produce future waypoints and rationales. This is not a deployable AV stack by itself. It is a useful probe: language models can absorb structured scene descriptions and generate plausible plans, but latency, grounding, and closed-loop reliability remain hard.

## Core Insights

### Structured scene facts become a coordinate sequence

GPT-Driver reformulates motion planning as GPT-style sequence generation. It serializes structured scene state into language-model tokens and predicts future waypoints plus a rationale. This gives the model an interpretable interface: the generated plan can be paired with an explanation of the driving decision. The evidence focuses on open-loop planning quality. The caveat is that open-loop waypoint prediction does not prove closed-loop safety, and LLM latency remains a deployment problem. The paper is useful as an early example of adapting pretrained language models to structured planning rather than raw perception.

The prompting pipeline turns each detected object, predicted path, ego state, and map fact into a sentence. GPT-3.5 first identifies the critical objects, considers their predicted motion against a hypothetical ego path, chooses a high-level action, and then emits six waypoints over three seconds. The output is converted back to numeric coordinates for evaluation. This is more than a natural-language wrapper: tokenization changes the learning problem from regressing values such as 23.17 to selecting a sequence of familiar subword tokens, while the rationale exposes which objects the model considered relevant.

On nuScenes, GPT-Driver reaches average L2 0.44 m and collision 0.17% under the ST-P3 metric grouping; under the UniAD grouping it reaches 0.84 m and 0.44%. The few-shot curve is the stronger result: at 10% of the training scenarios, GPT-Driver reaches 1.20 m L2 and 0.95% collision, versus UniAD’s 1.80 m and 1.31%. Fine-tuning is essential—few-shot in-context prompting alone reaches 3.17 m L2 and 5.30% collision—so the advantage is adaptation of the language model to the coordinate grammar, not generic prompting.

The experiment is best understood as a planner-interface test rather than end-to-end visual driving. GPT-Driver receives detections, predicted object motion, ego state, history, and a mission goal that have already been structured by upstream systems. Its contribution is to serialize those facts, reason about critical objects, and emit six waypoints; any failure in perception or prediction is inherited before the language model gets to plan.

The overview places perception before the language interface. Its inputs are already structured scene facts, so the model is being tested as a planner rather than asked to discover objects directly in pixels.

![Figure 1: Overview of GPT-Driver from GPT-Driver: Learning to Drive with GPT](/assets/images/gpt-driver-learning-to-drive-with-gpt-paper-figure.png)
*Fig 1: GPT-Driver converts structured scene observations into language tokens, prompts a language model to reason about the scene, and decodes the response into a future trajectory. | source: [GPT-Driver: Learning to Drive with GPT paper, Figure 1](https://arxiv.org/abs/2310.01415)*

Read the prompt from the reusable instruction block through the scene-specific facts to the six-coordinate answer. The rationale exposes which objects the model mentions, but faithfulness requires the numeric path to respond to those objects as well. Correct prose and correct geometry remain distinct outputs to check.

![Figure 2 from GPT-Driver: Learning to Drive with GPT](/assets/images/gpt-driver-learning-to-drive-with-gpt-source-figure-2.webp)
*Fig 2: The prompt combines a reusable planning instruction with serialized perception, predicted motion, ego history, and the requested six-point trajectory. The layout makes the interface legible: structured scene facts enter as text, then the model returns both a decision rationale and coordinates. | source: [GPT-Driver: Learning to Drive with GPT, Figure 2](https://arxiv.org/abs/2310.01415)*


## High-Level Takeaways

- GPT-Driver informs whether motion planning can be reframed as conditional language modeling over structured scene tokens and waypoint outputs. The atomic prediction is a discretized coordinate or waypoint token, with textual scene context and chain-of-thought-style supervision preceding the trajectory.
- The formulation gains access to pretrained sequence modeling, but coordinate serialization, numerical precision, and rationale faithfulness become hidden design choices.
- GPT-Driver uses language as an intermediate planning representation rather than an explanation added after the decision.
- LLMs can help expose the reasoning behind a plan, but driving needs that reasoning to stay grounded, fast, and controllable.
