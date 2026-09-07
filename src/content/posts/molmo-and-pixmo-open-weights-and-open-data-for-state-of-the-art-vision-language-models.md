---
title: 'Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models'
date: '2024-09-25T00:00:00.000Z'
section: paper-shorts
postSlug: molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models
legacyPath: /paper shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models"
---
## 2024 – Molmo and PixMo

**arXiv:** [2409.17146](https://arxiv.org/abs/2409.17146)

**Project:** [Allen AI Molmo](https://allenai.org/blog/molmo)

## Summary

> Molmo is a useful open-model experiment because it opens the supervision as well as the weights. PixMo uses long human descriptions, pointing annotations, and synthetic task data to make each image teach more than an image–caption match. On the paper's 11-dataset average, Molmo-72B reaches 81.2, compared with 79.4 for Qwen2-VL-72B and 78.5 for GPT-4o; its human-evaluation Elo is 1077. Those comparisons do not erase the cost of collecting the data, but they make the data-to-capability story inspectable.

## Core Insights

### Supervision is a capability interface

PixMo is not one caption corpus. The authors combine three human-annotated sets—dense captions, instruction following, and pointing—with four synthetic sets for clocks, documents, counting, and related skills. The human caption set contains 712,000 images and about 1.3 million transcripts: annotators speak for 60–90 seconds, yielding captions that average 196 words. Pointing adds more than 2.3 million grounding annotations. A point answers “where?” in a way a sentence-level caption cannot, so the same image can supervise recognition, localization, and a later answer that must refer to a particular object.

![Figure 1: PixMo datasets and the capabilities they support](/assets/images/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models-paper-figure.png)
*Fig 1: This source Figure 1 maps three human-annotated PixMo datasets and four synthetic datasets to the Molmo capabilities they are intended to teach; the important design choice is the mix of language and spatial supervision. | source: [Molmo and PixMo, Figure 1](https://arxiv.org/abs/2409.17146)*

The open-data claim is therefore stronger than “we released a large dataset.” The paper releases enough of the ingredients to inspect how a model acquires a behavior. It also exposes a limit: human speech, transcription, point validation, and synthetic generation are an annotation pipeline with its own coverage and cultural biases. Openness makes those choices easier to study; it does not make them free or neutral.

### Crops and the connector make that supervision usable

The visual path is a standard ViT plus connector plus language model, but two implementation choices determine whether detailed supervision survives. Molmo encodes a low-resolution full image and overlapping high-resolution crops. The connector combines the third-to-last and tenth-to-last ViT layers, attention-pools each 2×2 patch window, and maps the result through an MLP. Earlier features preserve local evidence while later features supply stronger semantics, so the connector is not forced to choose one representation for both jobs.

The crop ablation makes the intuition concrete. On the paper's 11-task average, a single crop scores 62.8, non-overlapping multi-crops 75.7, and overlapping multi-crops 76.9. At test time the model sees 36 crops even though it was trained with 12, a deliberate stress test of whether the model learned a crop composition rule rather than memorized one layout. Overlap preserves context at crop boundaries; without it, a small object can become a fragment with no surrounding relation.

### The evidence separates data design from raw scale

| Comparison | Reported result | What it isolates |
| --- | ---: | --- |
| Molmo-72B, 11-dataset average | 81.2 | Best model in the reported Molmo family |
| Molmo-7B-D, 11-dataset average | 77.3 | Strong performance without the 72B language backbone |
| PixMo-Cap size sweep, 712k images | 76.9 | More high-quality caption data still helps, with diminishing returns |
| Academic data only | 72.5 | The PixMo mixture contributes beyond the open academic baseline |

The scale curve rises from 74.9 with no PixMo-Cap data to 75.5 at 89,000 images, 76.3 at 178,000, and 76.9 at 712,000. Removing documents lowers the reported average to 75.8; removing pointing lowers it to 76.2. These are useful component clues, though the paper does not turn them into a cost-normalized causal comparison between human annotation, weak web captions, and synthetic data.

Pointing also has an ordering effect. On the two counting benchmarks, training on pointing before counting reaches 89.4 and 86.3, while counting before pointing reaches 81.5 and 77.6. The result suggests that spatially grounded examples can teach a representation on which later count queries are easier to express. It is a training-schedule result, not proof that pointing is universally the best first task.

## High-Level Takeaways

- Molmo makes annotation design part of the model interface: long descriptions say what is present, while points say where the evidence is.
- The best reported model reaches 81.2 on the paper's 11-dataset average, but the paper does not report a human-and-compute cost curve against matched weakly labeled data.
- Overlapping crops and multi-level visual features are practical mechanisms for turning detailed supervision into tokens the language model can use.
- Pointing helps counting most when it comes first, a paper-specific clue that supervision order changes which capabilities become easy to learn.
