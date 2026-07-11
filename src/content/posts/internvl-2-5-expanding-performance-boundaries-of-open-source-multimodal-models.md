---
title: 'InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models'
date: '2024-12-01T05:00:00.000Z'
section: paper-shorts
postSlug: internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models
legacyPath: /paper shorts/2024/12/01/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models.html
tags:
  - Other
field: Vision-Language Models
summary: InternVL 2.5 scaled open multimodal models with better data, training strategy, and test-time reasoning.
---
## 2024 - InternVL 2.5

**arXiv:** [2412.05271](https://arxiv.org/abs/2412.05271)

**Project:** [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)

**Summary:** InternVL 2.5 is a scaling and training study for open multimodal LLMs. It keeps the broad InternVL architecture but improves data quality, training choices, augmentation, loss balancing, and test-time reasoning.

The paper is useful because it studies several axes together: vision encoder size, language model size, dataset size, and chain-of-thought style inference. The story is not "just scale everything"; it is that scaling only pays off when the data and training recipe stay balanced.

## Paper Insights

InternVL 2.5 is an open MLLM scaling and training study. It keeps the InternVL architecture family but improves data quality, model scale, test-time settings, and coverage across images, documents, video, multilingual tasks, grounding, and hallucination benchmarks. The paper compares against strong open and commercial systems, arguing that open models can approach frontier performance when data and inference strategy improve together. The caveat is that broad benchmark averages can hide reliability gaps. The takeaway is that open multimodal progress depends on data, scale, and test-time configuration as a combined system.

![Figure 1: Performance of various MLLMs on the OpenCompass leaderboard from InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models](/assets/images/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models-paper-figure.png)
_Figure 1: Performance of various MLLMs on the OpenCompass leaderboard. From the [InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models paper](https://arxiv.org/abs/2412.05271), via arXiv HTML._

**What to look at:**
- Progressive scaling across vision encoder, LLM, data size, and inference settings.
- Training details such as augmentation and loss balancing matter as much as model scale.
- Test-time reasoning can improve difficult multimodal benchmarks but may change latency/cost.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Scale | 1B to 78B family | Studies where open VLM scaling pays off. |
| Training | Data quality and balancing | Reduces failures that pure scale does not fix. |
| Evaluation | MMMU and hallucination-style tests | Looks beyond simple captioning/VQA. |

**Context:** InternVL 2.5 showed that open models could compete with leading closed systems on difficult multimodal benchmarks while exposing more of the training recipe.

**Takeaway:** Open VLMs started becoming systems engineering projects: data mixture, encoder choice, LLM scale, and inference strategy all interact.
