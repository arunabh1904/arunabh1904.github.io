---
title: 'Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving'
date: '2023-10-01T04:00:00.000Z'
section: paper-shorts
postSlug: driving-with-llms-fusing-object-level-vector-modality
legacyPath: /paper shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html
tags:
  - Other
field: Autonomous Driving
summary: Driving with LLMs fed object-level vector scene state into language models to make driving decisions more explainable.
---
## 2023 - Driving with LLMs

**arXiv:** [2310.01957](https://arxiv.org/abs/2310.01957)

**GitHub:** [wayveai/driving-with-llms](https://github.com/wayveai/driving-with-llms)

**Plain-language summary:** This paper studies a middle ground between raw image VLMs and classical planning. It converts the scene into object-level vectors, fuses those structured tokens into an LLM, and asks the model to reason about driving actions and explanations.

The bet is that language models may reason better when perception has already converted pixels into meaningful objects, positions, and relationships.

![Driving VLM loop schematic](/assets/images/driving-vlm-loop-schematic.svg)

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

**Why it mattered:** The paper made object-centric driving language models a serious baseline. It also clarified a recurring theme in autonomy: sometimes the right multimodal interface is not raw pixels, but structured state.

**Take-home message:** For safety-critical planning, language can be useful if it is grounded in the right representation. Object vectors give the LLM a cleaner substrate than raw visual impressions.
