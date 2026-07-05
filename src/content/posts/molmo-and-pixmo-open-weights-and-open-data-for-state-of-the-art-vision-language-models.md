---
title: 'Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models'
date: '2024-09-01T04:00:00.000Z'
section: paper-shorts
postSlug: molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models
legacyPath: /paper shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html
tags:
  - Other
field: Vision-Language Models
summary: Molmo argued that high-quality open multimodal data can matter more than sheer data volume.
---
## 2024 - Molmo and PixMo

**arXiv:** [2409.17146](https://arxiv.org/abs/2409.17146)

**Project:** [Allen AI Molmo](https://allenai.org/blog/molmo)

**Plain-language summary:** Molmo is a family of open multimodal models trained with PixMo, a carefully built set of image-text datasets. The core bet is data quality: detailed human descriptions, pointing supervision, and open data can close much of the gap to proprietary systems without relying only on massive scraped corpora.

This matters because many VLMs are hard to inspect. Molmo makes the data story more visible, which makes the model easier to study and reuse.

![Figure 1: Datasets in PixMo (left) and the capabilities they enable in Molmo (right) from Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](/assets/images/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models-paper-figure.png)
_Figure 1: Datasets in PixMo (left) and the capabilities they enable in Molmo (right). From the [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models paper](https://arxiv.org/abs/2409.17146), via arXiv HTML._

**What to look at:**
- PixMo data quality is the main mechanism: detailed human captions and pointing supervision.
- Open weights plus open data make the model easier to audit than closed VLMs.
- The interesting comparison is data quality versus raw data quantity.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Data | PixMo collection | Rich captions and pointing supervision improve grounding. |
| Openness | Open weights and data | Makes the training story inspectable. |
| Signal | Small models compete strongly | Suggests annotation quality can substitute for some scale. |

**Why it mattered:** The paper strengthened the case that VLM progress is not just architecture scale. Annotation style, spatial grounding, and openness can move the frontier too.

**Take-home message:** For multimodal models, the caption is part of the architecture. Better supervision changes what the model can see.
