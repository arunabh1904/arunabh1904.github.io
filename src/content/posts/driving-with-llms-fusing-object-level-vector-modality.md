---
title: 'Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving'
date: '2023-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: driving-with-llms-fusing-object-level-vector-modality
legacyPath: /paper shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: Driving with LLMs fed object-level vector scene state into language models to make driving decisions more explainable.
---
## 2023 - Driving with LLMs

**arXiv:** [2310.01957](https://arxiv.org/abs/2310.01957)

**GitHub:** [wayveai/driving-with-llms](https://github.com/wayveai/driving-with-llms)

**Summary:** This paper studies a middle ground between raw image VLMs and classical planning. It converts the scene into object-level vectors, fuses those structured tokens into an LLM, and asks the model to reason about driving actions and explanations.

The bet is that language models may reason better when perception has already converted pixels into meaningful objects, positions, and relationships.

## Paper Insights

Driving with LLMs feeds structured object-level vectors into a language model instead of raw pixels. A Vector-Former converts detected agents, lanes, and scene state into tokens that the LLM can use for QA and action generation. This makes the driving state more explicit and the model's outputs easier to explain. The method depends heavily on upstream perception: missing or incorrect objects become misleading language-model input. The paper matters as an interface design for combining modular autonomy state with LLM reasoning.

![Figure from Driving with LLMs: object-level vector modality feeds a Vector-Former and LLM control loop](/assets/images/driving-with-llms-fusing-object-level-vector-modality-paper-figure.png)
_Source figure from the [Driving with LLMs paper](https://arxiv.org/abs/2310.01957), via arXiv HTML._

**What to look at:**
- Object-level vectors are the multimodal interface, not raw pixels.
- The language model reasons over structured actors and relations.
- The main value is explainable decisions from a cleaner scene representation.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Input | Object vectors | Structured state reduces visual ambiguity. |
| Task | Driving QA and action generation | Tests scene interpretation and decisions. |
| Tradeoff | Depends on upstream perception | Bad object state still misleads the LLM. |

## Decision Lens

Driving with LLMs informs whether an LLM should consume raw visual tokens or a compact object-level vector description of the scene. Its atomic input is an agent or map vector serialized into the language interface; the representation preserves metric relations while the language model supplies reasoning and explanation.

Vectorization buys interpretability and shorter context but makes upstream detection the information bottleneck. The missing comparison matches token budget across object vectors, BEV features, image tokens, and oracle objects while measuring both explanation faithfulness and planning. At 10× agents, serialization order and context length dominate. The claim would fail if a non-language vector planner matched decisions and explanations derived from post-hoc state summaries at lower latency.

**Context:** The paper made object-centric driving language models a serious baseline. It also clarified a recurring theme in autonomy: sometimes the right multimodal interface is not raw pixels, but structured state.

**Takeaway:** For safety-critical planning, language can be useful if it is grounded in the right representation. Object vectors give the LLM a cleaner substrate than raw visual impressions.
