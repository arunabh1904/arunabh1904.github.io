---
title: 'Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs'
date: '2024-06-24T00:00:00.000Z'
section: paper-shorts
postSlug: cambrian-1-vision-centric-exploration-of-multimodal-llms
legacyPath: /paper shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs"
---

## 2024 – Cambrian-1

**arXiv:** [2406.16860](https://arxiv.org/abs/2406.16860)

## Summary

> Cambrian-1 treats the visual side of a multimodal LLM as an experimental object. It compares 23 vision backbones, studies instruction-tuning choices, introduces the 2,638-example CV-Bench for spatial and object-centric questions, and proposes the Spatial Vision Aggregator (SVA) for combining high-resolution features. The resulting 8B, 13B, and 34B models use 576 visual tokens and report average benchmark scores of 73.1, 73.7, and 76.8 respectively. The paper's deeper claim is about evaluation: many popular VLM benchmarks can be answered with limited visual evidence, so a stronger language model can conceal a weak visual representation.

## Core Insights

Cambrian-1 begins by asking whether a benchmark actually needs the image. The authors train models with 23 different vision backbones and compare each model with vision enabled, vision disabled, and random guessing. In the reported analysis, SQA-I3, MMMU, MathVista, and AI2D show less than a 5% gap between vision-enabled and vision-disabled scores. TextVQA and GQA have a nearly 40% gap between random guessing and the vision-disabled score, which the authors interpret as evidence of language bias. MMVP and MME Perception fall below random guessing without vision, making them more useful tests of grounding.

![Cambrian-1 benchmark analysis: vision-enabled versus vision-disabled performance and benchmark clusters](/assets/images/cambrian-1-vision-centric-exploration-of-multimodal-llms-source-figure-3.webp)
*Fig 1: The left panel sorts benchmarks by the gap between vision-enabled and vision-disabled models; the right panel clusters them into General, Knowledge, Chart & OCR, and Vision-Centric groups. | source: [Cambrian-1, Figure 3](https://arxiv.org/abs/2406.16860)*

That analysis motivates CV-Bench. It repurposes classic 2D and 3D vision annotations as natural-language questions: 650 spatial-relationship examples, 788 object-count examples, 600 depth-order examples, and 600 relative-distance examples, for 2,638 manually inspected samples in total. The test asks for evidence such as which object is closer to the camera or how many instances appear in an image. It is a useful boundary around the headline scores because the answer cannot be recovered from a language-only prior as easily as a generic knowledge question.

SVA addresses the next bottleneck: several encoders can preserve complementary information, but concatenating every feature map makes the language context too long. A learnable $L \times L$ query grid cross-attends to spatially corresponding regions of multiple encoder feature maps. The connector can be inserted at several language-model layers, so the model repeatedly accesses uncompressed visual features instead of compressing everything once at the input.

![Figure 8 from Cambrian-1: Spatial Vision Aggregator connects multiple vision encoders to the LLM](/assets/images/cambrian-1-paper-figure-8-sva.png)
*Fig 2: SVA uses spatially aligned cross-attention to aggregate multiple vision encoders into a compact latent-token grid, with optional aggregation blocks inserted through the LLM. | source: [Cambrian-1, Figure 8](https://arxiv.org/abs/2406.16860)*

The connector ablation gives the intuition a measurable anchor. With four vision encoders and a Vicuna-1.5-7B backbone, SVA reaches 68.5 on General, 49.7 on Knowledge, 55.5 on OCR & Chart, and 53.2 on Vision-Centric benchmarks, versus 67.2, 48.9, 50.1, and 52.6 for concatenation. The gain is largest where high-resolution feature aggregation matters. A global resampler is a weaker comparison because it lacks SVA's spatial subregions and multi-layer access.

The data study is separate from the final-model recipe. In the controlled 23-backbone ablations, the authors pretrain the connector on 1.2M adapter examples and fine-tune on a 737K instruction mix; Figure 5 measures how that recipe changes encoder rankings. The final Cambrian-1 models instead use 2.5M adapter data—1.2M ShareGPT4V captioning examples plus 1.2M MiniGemini captioning examples—and instruction-tune on Cambrian-7M. Keeping those stages separate matters: the 1.2M + 737K numbers describe the ablation, not the released model.

![Cambrian-1 training-recipe comparison across visual encoder families](/assets/images/cambrian-1-vision-centric-exploration-of-multimodal-llms-source-figure-5.png)
*Fig 3: The source comparison shows the distribution of benchmark scores under different instruction-tuning recipes for language-supervised, self-supervised, and other visual encoders. | source: [Cambrian-1, Figure 5](https://arxiv.org/abs/2406.16860)*

The final table reports Cambrian-1-8B at 73.1 average, ahead of LLaVA-NeXT-8B at 72.5, while using 576 visual tokens instead of LLaVA-NeXT's 2,880. At 34B, Cambrian-1 reaches 76.8 versus 76.0 for LLaVA-NeXT-34B. These comparisons support the vision-centric recipe, but they do not isolate SVA from the four-encoder ensemble, data curation, instruction recipe, and model scale.

| Decision | Cambrian-1's answer | Boundary |
| --- | --- | --- |
| Evaluation | Compare vision-on, vision-off, and random baselines; add CV-Bench | Benchmark language priors still affect any VQA protocol. |
| Representation | Test many supervised and self-supervised vision backbones | Backbone quality depends on tuning and token budget. |
| Connector | Spatially aligned cross-attention with repeated aggregation | More encoder features increase memory and serving complexity. |
| Data | Balance adapter and instruction sources | Mixture thresholds can change the apparent backbone ranking. |

## High-Level Takeaways

- Cambrian-1's strongest contribution is a measurement discipline: first ask whether the benchmark needs visual evidence, then compare visual representations under a matched recipe.
- CV-Bench turns depth, distance, position, and counting into a 2,638-example grounding surface that complements language-heavy VQA.
- SVA's gain is specifically tied to spatially aligned aggregation and repeated access to high-resolution features; it is not evidence that every VLM needs four encoders.
- The 576-token result is attractive for context cost, but the full system still pays for four vision towers and their preprocessing.
- A clean deployment comparison should match activated vision FLOPs, visual-token count, instruction data, and latency before attributing the gain to the connector.
