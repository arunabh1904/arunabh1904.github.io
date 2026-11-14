---
title: 'What a Vision Encoder Actually Buys'
date: '2026-11-14T16:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: what-a-vision-encoder-actually-buys
legacyPath: /blog/2026/11/14/what-a-vision-encoder-actually-buys.html
tags:
  - Vision-Language Models
  - Pretraining
topics:
  - multimodal
  - learning
summary: A controlled study separates pretrained visual knowledge from dedicated vision capacity, showing why early fusion can still need specialized image processing.
---
# What a Vision Encoder Actually Buys

[Chris Hoang's thread](https://x.com/choang333/status/2121335484236386574) raises a question that “encoder-free” leaves surprisingly vague: which part of the encoder are we removing? A typical vision-language model inherits visual features from pretraining, reserves parameters for images, and runs those parameters before the language backbone. Deleting the encoder changes all three at once.

[*Unbundling Multimodal Vision Encoding*](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html), by Ellis Brown, Chris Hoang, and colleagues, separates these choices. Its useful result is that dedicated vision capacity can remain valuable even when separate visual pretraining and a separate encoder do not. Moving toward one backbone need not mean making every parameter serve every modality.

## How the encoder became the default

[CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) made reusable visual features practical by aligning images and text at scale in 2021. The original [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) then connected a frozen CLIP encoder to a language model through a learned projection. This was an economical way to assemble a visual assistant from existing components. It established that pretrained features worked; it did not isolate why they worked.

I covered this progression in my earlier [VLM post, including LLaVA's projector and training stages](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html#from-q-formers-to-mlp-projectors). The [spatial-adapter guide](/blog/2026/10/06/adding-spatial-grids-and-object-tokens-to-vision-language-models.html#what-the-connector-literature-provides) looks more closely at what those connectors preserve or compress. Here the question moves upstream: which visual processing should produce the features in the first place?

[Fuyu-8B](https://www.adept.ai/blog/fuyu-8b/) took another route in 2023, projecting image patches directly into a decoder-only transformer. But comparing released models cannot settle the encoder question: their data, training budgets, and backbones also differ. The new study makes those controls explicit, including the upstream compute spent pretraining an encoder. A reusable checkpoint is cheap to download, but its training is still part of the scientific comparison.

## Capacity has an address

The authors train components on common data and compare recipes at matched lifetime training compute. Encoder pretraining counts toward that budget; instruction tuning does not. Equal compute permits different token counts. A scratch-trained encoder matches the tested pretrained encoders, while removing dedicated vision capacity hurts visual question answering. The table shows what happens when 0.4 billion parameters are allocated differently around a 1.7-billion-parameter backbone. Avg. VQA averages four category scores across 16 visual question-answering benchmarks. [Paper, Sections 2–5](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html#what-the-comparison-holds-fixed)

| Allocation of the extra capacity | Avg. VQA |
| --- | ---: |
| No extra capacity: encoder-free baseline | 39.0 |
| Larger shared backbone: wider, deeper, or both | 38.6–38.8 |
| Separate scratch-trained vision encoder | 44.4 |
| Extra feed-forward branch, first eight layers, all tokens | 42.3 |
| Extra feed-forward branch, first eight layers, vision tokens only | **45.0** |
| Extra feed-forward branch, last eight layers, vision tokens only | 35.1 |

*Selected results from [Tables 2 and 3](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html#matched-compute-capacity), at $2.6\times10^{21}$ lifetime FLOPs, evaluated after three epochs of Cambrian-7M instruction tuning at 448 × 448 resolution. Every added-capacity row uses 0.4 billion extra parameters; the baseline has no such addition.*

Adding shared parameters leaves visual performance near the encoder-free baseline. Reserving an early branch for vision matches the separate encoder; moving that branch to the end loses the gain. The result makes parameter count an incomplete description of capacity. Which tokens can use the parameters, and when they can use them, changes their value.

The authors call the internal design **in-fusion**. Every token uses the shared backbone; image tokens also use vision-only branches whose outputs are added to the shared outputs. The feed-forward-only variant compares placement at a fixed parameter budget. The fuller design also adds vision-only attention projections while retaining joint attention. This is specialized processing inside a common model, as the [authors' explanation](https://unbundling.github.io/#placement) makes clear.

That distinction survives a scaling comparison. As I discussed in the [earlier scaling-laws post](/blog/2026/08/19/how-to-read-scaling-laws-for-language-models.html#what-moved-after-chinchilla), a fitted compute advantage depends on the experimental regime. The original figure plots visual question answering and held-out caption loss against lifetime compute.

![Source scaling plots: scratch encoders and in-fusion models track closely, while encoder-free models require more compute for comparable visual scores and caption loss.](/assets/images/unbundling-vision-encoding-source-figure-7.png)
*Figure 7 from [Unbundling Multimodal Vision Encoding](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html#scaling-and-serving), reproduced unchanged. Blue circles represent scratch encoders, pink triangles in-fusion models, and orange squares encoder-free models. Points shown use 64 token-equivalents per non-embedding backbone parameter; fitted curves use the wider training sweep. The roughly threefold VQA compute penalty applies within this study, not as a universal scaling constant.*

[Open the source figure at full size](/assets/images/unbundling-vision-encoding-source-figure-7.png)

In-fusion tracks the separate encoder as compute grows. A plausible interpretation is that raw image patches benefit from specialized processing before the shared layers must use them. The placement ablation supports that interpretation, but it does not establish that language gradients interfere with vision or that every modality needs its own branch.

## What changes in practice

Placement also changes the execution path. A separate encoder finishes before the backbone consumes its output. In the authors' serving setup, full in-fusion reduces time to first token from 210 to 182 milliseconds: 13% at matched vision capacity. That measurement uses a 1.7B backbone, one H200, 16 concurrent requests, and one 448-pixel image plus 256 text tokens per request. It is a result for that workload, not a general serving guarantee. [Paper, Table 36](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html#scaling-and-serving)

The boundary on the pretraining claim matters just as much. The study pretrains encoders on the same image pool used for multimodal training. It leaves open the benefit of external encoders trained on additional data, along with higher-resolution pretraining, sparse backbones, and video. [Paper, Discussion](/paper%20shorts/2026/11/12/unbundling-multimodal-vision-encoding.html#high-level-takeaways)

For a new architecture, this suggests a more useful experiment than choosing an “encoder-based” or “encoder-free” label: hold compute fixed, then vary vision-only routing and early placement separately. For an existing system, removing a working pretrained encoder still needs a task-specific justification. The deeper lesson is about modularity: a shared backbone can preserve specialized computation without preserving every historical boundary between components.
