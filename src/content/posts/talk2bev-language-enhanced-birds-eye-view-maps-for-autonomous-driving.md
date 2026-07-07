---
title: "Talk2BEV: Language-enhanced Bird's-eye View Maps for Autonomous Driving"
date: '2023-10-03T04:00:00.000Z'
section: paper-shorts
postSlug: talk2bev-language-enhanced-birds-eye-view-maps-for-autonomous-driving
legacyPath: /paper shorts/2023/10/03/talk2bev-language-enhanced-birds-eye-view-maps-for-autonomous-driving.html
tags:
  - Other
field: Autonomous Driving
summary: Talk2BEV grounds language queries in BEV maps by augmenting objects with image-language features, enabling open-ended scene, spatial, intent, and decision questions.
---
## 2023 - Talk2BEV

**arXiv:** [2310.02251](https://arxiv.org/abs/2310.02251)

**Project:** [Talk2BEV](https://llmbev.github.io/talk2bev/)

**Code:** [llmbev/talk2bev](https://github.com/llmbev/talk2bev)

**Plain-language summary:** Talk2BEV connects language reasoning to bird's-eye-view maps. It builds a BEV map from sensor data, augments objects with aligned vision-language features, and lets a large vision-language model answer scene-level and object-level driving questions.

The core idea is grounding. Language is useful only if the model can bind words like "pedestrian on the right" or "vehicle in front" to the spatial layout a planner uses.

## Paper Insights

Talk2BEV turns BEV maps into a language-addressable representation. Objects in the BEV map are linked with image-language features from large vision-language models, so questions can refer to categories, locations, intents, and driving decisions. The paper also introduces Talk2BEV-Bench, with 1,000 human-annotated nuScenes BEV scenarios and more than 20,000 question-answer pairs.

The contribution is not a low-level driving controller. It is an interface layer that tests whether language models can reason over a spatial driving scene. The caveat is that good answers over a benchmark do not by themselves prove closed-loop planning reliability.

![Figure 2 from Talk2BEV showing BEV map generation, language-enhanced object features, and LVLM question answering](/assets/images/talk2bev-language-enhanced-birds-eye-view-maps-for-autonomous-driving-paper-figure.png)
_Figure 2 shows how Talk2BEV turns generated BEV maps into language-enhanced maps that can answer object-level and scene-level questions. From the [Talk2BEV paper](https://arxiv.org/abs/2310.02251), via ar5iv._

**What to look at:**
- BEV gives the language model an explicit spatial substrate.
- Object-level image-language features make queries grounded rather than purely textual.
- The benchmark tests spatial reasoning, intent prediction, and decision-oriented questions.

**Evals / Benchmarks / Artifacts:**

| Artifact | Detail | Why it matters |
| -------- | ------ | -------------- |
| Representation | Language-enhanced BEV map | Connects semantic language to planner-friendly space. |
| Dataset | 1,000 BEV scenarios | Gives the paper a human-annotated evaluation set. |
| QA volume | More than 20,000 questions and responses | Covers varied scene, object, intent, and decision queries. |
| Source data | nuScenes | Keeps the benchmark tied to a standard driving dataset. |

**Why it mattered:** Talk2BEV is an early clean example of language-grounded scene reasoning over BEV rather than only camera images.

**Take-home message:** Driving language models need spatial grounding, and BEV maps are one natural place to attach it.
