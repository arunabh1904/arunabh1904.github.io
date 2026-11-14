---
title: 'InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition'
date: '2026-05-04T00:00:00.000Z'
section: paper-shorts
postSlug: infolaw-information-scaling-laws-for-large-language-models-with-quality-weighted-mixture-data-and-repetition
legacyPath: /paper shorts/2026/05/04/infolaw-information-scaling-laws-for-large-language-models-with-quality-weighted-mixture-data-and-repetition.html
tags: [Scaling Laws, Data Mixtures, Data Efficiency]
field: 'Language Models'
summary: '2026 – InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition'
---

## 2026 – InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition

**Paper:** [Liu et al., arXiv:2605.02364v1](https://arxiv.org/abs/2605.02364v1)

## Summary

> Upweighting high-quality text also repeats it more often. InfoLaw predicts the resulting tradeoff with a fitted measure of accumulated information that accounts for quality buckets, unique tokens, exposure count, and model scale. Its useful result is predicting and searching data mixtures beyond the fitted runs. The “information” quantity is an empirical modeling device, not an independently measured amount of knowledge.

## Core Insights

### Mixture weights also set repetition rates

The authors deduplicate an English Common Crawl corpus to 3.7T tokens and score documents using combined FineWeb-Edu and DCLM classifiers. They divide the ranked corpus into six quality buckets, with the top bucket containing 5% of the source. LayerMix chooses the fraction of consumed tokens drawn from each bucket; the lowest-quality 20% is excluded in the tested presets.

Let $K$ be consumed tokens, $S$ the available source tokens, $w_d$ a bucket's sampling weight, and $B_d$ its source share. Its unique-token count is $M_d=\min(w_dK,B_dS)$ and average exposure count is $R_d=w_dK/M_d$. If the top 5% receives 80% of training tokens with $K=S$, it is repeated about 16 times. This couples a quality choice to a repetition cost even when the packed training set is traversed only once.

### A fitted proxy combines quality and diminishing returns

InfoLaw sums bucket contributions of the form

$$
I=\sum_d f_d M_d\log K\left(1-e^{-\lambda(N)R_d/\log K}\right),
$$

then fits evaluation loss as a power law in $I$. Here $f_d$ is fitted quality density and $\lambda(N)$ controls exposure saturation. The paper defines $N$ as non-embedding FLOPs per token, so it should not be substituted casually with parameter count. Quality density decreases exponentially with bucket index; the fitted saturation parameter grows logarithmically with model scale.

The $\log K$ normalization is an empirical choice. Appendix B discusses failed alternatives without providing their plots. The resulting curve is useful evidence of predictive fit, but it does not derive a general law of information acquisition.

![InfoLaw and conventional scaling fits diverge when extrapolating quality-weighted repeated data.](/assets/images/infolaw-source-figure-1.png)
*Fig 1: Solid curves cover the fitting range and dashed curves extrapolate. In both quality mixtures, the conventional compute-only fit predicts overly optimistic loss at larger scale; the repetition-aware fit tracks the held-out measurements more closely. | source: [Paper, Figure 1](https://arxiv.org/html/2605.02364v1#S1.F1)*

[Open the source figure at full size](/assets/images/infolaw-source-figure-1.png).

### Test ranking as well as small prediction errors

The fit uses 27 runs: nine model sizes from 252M to 1.2B, each trained on three mixtures under the same overtraining regime. Tests change mixture weights, model scale, or both, extending to approximately 7B models and 425B tokens. The authors report mean and maximum loss prediction errors of 0.15% and 0.96% across unseen points. These are errors in their benchmark evaluation-loss target, not accuracy gains or errors on arbitrary downstream tasks.

For recipe selection, a more direct test is ranking nearby alternatives. On 25 held-out mixtures for a 1.2B model, predicted and measured loss have Pearson correlation 0.76. A search over 100,000 candidate weight vectors selects a 2.5B recipe that beats the compared presets. That supports a useful shortlist for training experiments, not a guarantee of a global optimum.

The fitted recommendations shift toward broader data as model size or consumed-token budget increases. This extends [Data Mixing Laws](/paper%20shorts/2024/03/25/data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance.html) by explicitly tracking repeated exposure. It addresses a different question from whole-corpus repetition: a small curated subset can become saturated while other buckets still contain unseen text.

The boundaries matter. Bucket definitions and quality classifiers are fixed, with no search over their number or boundaries. A RefinedWeb check refits quality and per-size saturation parameters on three model scales; it does not demonstrate zero-shot transfer of the entire fitted law. Higher-overtraining experiments show shifted curves whose theoretical explanation remains open. Treat the model as a calibrated recipe-selection tool for a specified data pipeline.

## High-Level Takeaways

- A quality-mixture change also changes how often each subset is repeated.
- InfoLaw's information quantity is fitted to outcomes, rather than directly measured knowledge.
- Recipe-ranking accuracy matters even when aggregate prediction errors look small.
- Recalibrate when the corpus, quality classifier, bucket structure, or training regime changes.
