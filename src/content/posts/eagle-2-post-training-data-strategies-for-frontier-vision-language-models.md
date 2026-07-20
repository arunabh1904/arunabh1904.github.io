---
title: 'Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models'
date: '2025-01-01T05:00:00.000Z'
section: paper-shorts
postSlug: eagle-2-post-training-data-strategies-for-frontier-vision-language-models
legacyPath: /paper shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2025 – Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models"
---
## 2025 – Eagle 2

**arXiv:** [2501.14818](https://arxiv.org/abs/2501.14818)

**GitHub:** [NVlabs/Eagle](https://github.com/NVlabs/EAGLE)

**Summary:** Eagle 2 argues that frontier VLM performance depends heavily on post-training data strategy. Instead of presenting only final weights, the paper documents how the team builds and balances instruction data, benchmark-oriented data, visual reasoning tasks, and embodied/driving-style examples.

The main lesson is that strong VLMs are not just pretrained once and then lightly tuned. Their behavior is shaped by a careful curriculum of visual tasks and response styles.

## Paper Insights

Eagle 2 is a data-strategy paper for post-training VLMs. Instead of only releasing a final model, it studies how instruction data, ordering, filtering, and staged tuning affect frontier multimodal performance. The contribution is a transparent recipe and ablation trail showing which data choices move OCR, grounding, visual reasoning, and instruction-following behavior. The evidence comes from step-by-step ablations and broad benchmarks. The caveat is reproducibility: post-training data quality and filtering details are hard to copy exactly, and benchmark leakage must be watched closely.

![Figure 1: Overview of Eagle2-9B’s result across different multimodal benchmarks, in comparison to state-of-the-art open-source and commercial frontier models from Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models](/assets/images/eagle-2-post-training-data-strategies-for-frontier-vision-language-models-paper-figure.png)
_Figure 1: Overview of Eagle2-9B’s result across different multimodal benchmarks, in comparison to state-of-the-art open-source and commercial frontier models. From the [Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models paper](https://arxiv.org/abs/2501.14818), via arXiv HTML._

**What to look at:**
- Post-training data strategy is the primary contribution.
- The paper is useful because it exposes data mixture choices normally hidden in model releases.
- Benchmarks should be read as evidence for the recipe, not only for the final checkpoint.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Main lever | Post-training data mixture | Turns a general VLM into a benchmark-strong assistant. |
| Transparency | Recipe-focused release | Documents data strategy instead of only final weights. |
| Scale signal | Eagle2-9B competitive with larger models | Suggests curation can buy efficiency. |

## Decision Lens

Eagle 2 informs a post-training allocation decision: whether another model-scale increase is worth more than improving the composition, filtering, and ordering of multimodal instruction data. The fundamental unit is the supervised image–instruction–response example, and the curriculum controls which capabilities are introduced and reinforced across stages. The shared model is conventional; the paper's real mechanism is distribution design across OCR, grounding, reasoning, and instruction following.

The ablation trail shows that curation can make a 9B model competitive with larger systems, but benchmark gains do not by themselves establish that the recipe generalizes beyond the evaluation distribution. The missing test is a contamination-audited, held-out capability suite combined with a cost-matched comparison to simply adding parameters or raw examples. At ten times the data scale, quality filters may select increasingly homogeneous synthetic patterns and amplify benchmark style. The claim is falsified if the carefully staged mixture loses its advantage on fresh tasks after controlling for data provenance and total annotation cost.

**Context:** Eagle 2 is valuable because it makes the data engineering visible. That helps turn VLM building from folklore into something closer to an inspectable recipe.

**Takeaway:** Post-training is where a general multimodal model becomes useful. The data mixture is a control surface.
