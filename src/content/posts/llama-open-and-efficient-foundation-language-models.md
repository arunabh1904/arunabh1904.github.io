---
title: 'LLaMA: Open and Efficient Foundation Language Models'
date: '2023-02-27T09:00:00.000Z'
section: paper-shorts
postSlug: llama-open-and-efficient-foundation-language-models
legacyPath: /paper shorts/2023/02/27/llama-open-and-efficient-foundation-language-models.html
tags:
  - Language Models
  - Pretraining
  - Scaling Laws
field: "Language Models"
summary: "2023 – LLaMA: Open and Efficient Foundation Language Models"
---

## 2023 – LLaMA: Open and Efficient Foundation Language Models

**Paper:** [Touvron et al., arXiv:2302.13971v1](https://arxiv.org/abs/2302.13971v1) · **Code:** [LLaMA repository](https://github.com/facebookresearch/llama)

## Summary

> LLaMA moves part of the cost of language modeling from serving to pretraining. A smaller model can take more training tokens to reach a target score but remain cheaper per generated token. The original family tests this idea with four dense models trained on 1.0–1.4 trillion tokens from publicly available sources. The benchmarks support the recipe for several tasks, though they do not isolate the effect of data quantity or establish a measured serving advantage.

## Core Insights

### Optimize for the model you will serve

[Chinchilla](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) asks how to allocate a fixed training budget between parameters and tokens. LLaMA starts with a different constraint: how small can the deployed model be for a given level of performance? Additional training can make a smaller model more useful without increasing its parameter count. This trade pays off only if the model meets the target and is used enough to justify the extra training.

Table 2 gives the training allocation. The 6.7B and 13B models see 1.0T tokens; the 32.5B and 65.2B models see 1.4T. All use batches of 4M tokens. The horizontal axis in the figure counts tokens, so matching points on that axis does not match compute across model sizes.

![Training loss curves for four LLaMA models plotted against billions of training tokens.](/assets/images/llama-source-figure-1.svg)
*Fig 1: The two smaller models stop at 1.0T tokens, while the larger ones extend to 1.4T. | source: [Touvron et al., Figure 1](https://arxiv.org/html/2302.13971v1#S2.F1)*

[Open the figure at full size](/assets/images/llama-source-figure-1.svg) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The decoder remains dense and causal. Pre-layer RMSNorm normalizes the input to each Transformer sublayer. SwiGLU uses a nonlinear gate on a parallel feed-forward projection. [RoPE](/paper%20shorts/2021/04/20/roformer-enhanced-transformer-with-rotary-position-embedding.html) rotates queries and keys at each layer in place of an absolute-position embedding. These are inherited components assembled into one recipe, not individually controlled interventions. The implementation avoids storing the complete attention matrix during training and overlaps some GPU communication with computation. The reported 65B run used 2,048 80 GB A100s for about 21 days.

### Interpret the benchmark gains in context

English Common Crawl and C4 account for 82% of sampled tokens. The rest come from GitHub, Wikipedia, books, arXiv and Stack Exchange. Most sources get roughly one pass in the 1.4T run; Wikipedia and books get roughly two. Public availability describes the inputs selected by the authors. It does not guarantee each item can be redistributed, nor does it supply a bit-for-bit training corpus.

In Table 3, LLaMA-13B beats the GPT-3 175B comparator on five of seven shared zero-shot commonsense benchmarks. It scores 74.8 versus 68.8 on ARC-Easy but trails on PIQA, 80.1 versus 81.0, and OpenBookQA, 56.4 versus 57.6. On Natural Questions in Table 4, its exact-match score is 20.1 zero-shot and 31.9 with 64 examples, versus GPT-3's 14.6 and 29.9. Those are published comparisons; they do not hold the data, tokenizer and architecture fixed.

For a deployment decision, I would measure generation throughput and memory at the relevant prompt length, batch size and precision, then compare costs at matched task accuracy. Parameter count is a useful first estimate. It does not supply those measurements, and a better likelihood-ranking score does not make an original base checkpoint an instruction-following assistant.

## High-Level Takeaways

- Extra pretraining can help a smaller model meet a target without increasing its parameter count.
- The results reflect a combined data, architecture and training recipe.
- LLaMA-13B beats larger published comparators on several tests, with exceptions that matter for model selection.
- The paper motivates a lifecycle-cost comparison but does not provide a matched serving benchmark.
