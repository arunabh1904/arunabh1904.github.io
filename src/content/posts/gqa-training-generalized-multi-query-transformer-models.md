---
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
date: '2023-05-22T00:00:00.000Z'
section: paper-shorts
postSlug: gqa-training-generalized-multi-query-transformer-models
legacyPath: /paper shorts/2023/05/22/gqa-training-generalized-multi-query-transformer-models.html
tags:
  - Attention
  - Inference
field: "Language Models"
summary: "2023 \u2013 GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
---

## 2023 – GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Paper:** [Ainslie et al., arXiv:2305.13245v3](https://arxiv.org/abs/2305.13245v3)

## Summary

> GQA shares keys and values within groups of query heads, recovering most of multi-head attention's measured quality while approaching multi-query inference speed. Converting and uptraining T5-XXL for 5% of its original training steps gives an average benchmark score of 47.1 versus 47.2 for multi-head attention, with reported time per sample falling from 1.51 to 0.28 seconds. These are encoder-decoder experiments on TPUs, not a universal decoder-only speedup.

## Core Insights

The expensive object is the cached history. In ordinary [multi-head attention](/paper%20shorts/2017/06/01/attention-is-all-you-need.html), every query head has its own key and value projections. Multi-query attention keeps the query heads but shares one key/value pair across them. GQA chooses an intermediate number of groups: each group has its own cached keys and values, shared by its query heads.

The source diagram shows what stays separate as the cache shrinks.

![Source Figure 2 compares multi-head, grouped-query, and multi-query key/value sharing.](/assets/images/attention-variants-mha-gqa-mqa.png)
*Fig 1: Query heads remain separate across all three designs. Grouping changes how many key/value representations each token contributes to the cache, preserving more retrieval capacity than a single shared pair. | source: [GQA, Figure 2](https://arxiv.org/html/2305.13245v3#S2.F2)*

[Open the figure at full size](/assets/images/attention-variants-mha-gqa-mqa.png)

GQA therefore preserves token-addressable history. It does not summarize all previous tokens into a fixed recurrent state. For an unchanged head dimension and sequence length, reducing key/value heads reduces cache storage proportionally; realized latency also depends on the rest of the serving system.

### Conversion is followed by adaptation

The authors mean-pool the pretrained key and value projection matrices within each new group, then continue the original pretraining recipe. Mean pooling beats selecting one old head or initializing new heads randomly. Five percent uptraining provides most of the measured recovery, with diminishing returns at ten percent. The five-percent run still costs approximately 600 TPUv3 chip-days.

The experiment modifies T5.1.1 decoder self-attention and cross-attention, leaving encoder self-attention unchanged. Evaluation fine-tunes on five summarization datasets, WMT English–German translation, and TriviaQA. The aggregate combines their task metrics, so it is a comparison within this suite rather than a percentage of general capability. Timing uses eight TPUv4 chips with model-specific parallelization and the largest fitting batch up to 32 per chip.

Eight groups give the reported compromise: 47.1 average score at 0.28 seconds, versus MQA's 46.6 at 0.24 seconds. Increasing groups toward full multi-head attention increases latency more substantially. The appendix also reports instability for MQA trained from scratch and during some long-input fine-tuning; the uptrained GQA runs appeared stable.

The missing control is a comparable XXL GQA model trained from scratch. This study supports converting an existing checkpoint, but does not establish whether conversion is better than spending the same total compute on a native GQA model. For a deployment decision, my next test would match quality, batch size, context length, and serving kernels before treating cache reduction as an end-to-end speed claim.

## High-Level Takeaways

- GQA compresses the number of cached heads while retaining a separate key/value record for every token.
- Mean-pooling plus continued pretraining makes architectural conversion practical; changing the checkpoint layout alone is not the complete recipe.
- The eight-group T5 result nearly preserves the measured MHA quality while approaching MQA timing in the reported TPU setup.
- Decoder-only transfer and a matched native-GQA training comparison remain outside the paper's evidence.
