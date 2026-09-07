---
title: 'How Much Do Language Models Memorize?'
date: '2025-05-30T00:00:00.000Z'
section: paper-shorts
postSlug: how-much-do-language-models-memorize
legacyPath: /paper shorts/2025/05/30/how-much-do-language-models-memorize.html
tags:
  - Other
field: 'Language Models'
summary: "2025 – How Much Do Language Models Memorize?"
---

## 2025 – How Much Do Language Models Memorize?

**Paper:** [arXiv:2505.24832](https://arxiv.org/abs/2505.24832)

## Summary

> The paper measures sample-specific information through how much a trained model helps compress its training data beyond a reference model. Uniform random sequences isolate memorization because they contain no reusable linguistic structure; the measured capacity grows roughly with parameter count, at about 3.6 bits per parameter in the studied GPT-style models. On text, the result depends on what the reference already explains. The deeper contribution is separating storage, generalization, extraction, and membership inference, which can move differently as the dataset grows.

## Core Insights

### Reproducing a string does not explain why the model knows it

A familiar phrase can be predictable from general language structure, while a random identifier requires information about that particular sample. An extraction test treats both as reproduced strings. The paper instead asks how much information the target model contributes beyond a reference that approximates the data-generating distribution.

Its conceptual definition uses conditional information and Kolmogorov complexity. Exact Kolmogorov complexity is uncomputable, so the experiments use likelihood-based code lengths as a practical approximation. A model that assigns probability $p(x)$ to a sequence gives an idealized arithmetic-coding length near $-\log_2 p(x)$ bits.

For the compression scheme used here, access to both reference and target means choosing the shorter of their two codes. The sample-specific gain is therefore approximated by

$$
\widehat{m}_U(x)=\max\left(0,\log_2 p_{target}(x)-\log_2 p_{reference}(x)\right).
$$

This is a gain in coding efficiency, not a count of tokens that can be recalled verbatim. The reference matters: if it already predicts a sample well, less of the target's performance is credited to unintended memorization.

### Uniform random sequences make the reference exact

The synthetic experiments usually use 64-token sequences drawn uniformly from a vocabulary of 2,048 symbols. Each token has 11 bits of entropy, so a sequence contains 704 bits before any sample-specific information is available. There is no grammar or shared semantic structure that would let the model predict an unseen random sequence better.

For illustration, assigning an average of ten bits per observed token would compress such a training sequence to 640 bits, a gain of 64 bits. That would demonstrate partial information about the sequence without requiring perfect free-running reproduction of all 64 tokens.

The capacity experiment sums these gains over training samples and varies the dataset size. Small enough datasets can be largely memorized. Once more samples are added, the total information stored stops growing proportionately.

![Source Figure 1 showing memorization plateaus for GPT-style models trained on uniform random data](/assets/images/how-much-do-language-models-memorize-paper-figure.png)
*Fig 1: The dashed line grows with the dataset's information content. Each model eventually falls below it and approaches a plateau, distinguishing more available data from more information retained in the weights. | source: [Memorization paper, Figure 1](https://arxiv.org/abs/2505.24832)*

The horizontal axis counts datapoints; the vertical axis counts memorized bits. Following one colored curve to the right holds the architecture fixed while changing data size. Comparing plateau heights across colors then estimates how capacity changes with model size.

### The bits-per-parameter estimate is an empirical lower bound

The direct synthetic sweep uses GPT-2-style models of roughly 100K to 20M parameters, trained from scratch for one million steps, typically with five seeds per model–dataset setting. The broader membership-scaling experiments extend to 1.5B parameters. Those are different pieces of evidence; the largest membership model is not itself the basis of every capacity estimate.

Across widths and depths, the main fit gives roughly 3.6 bits per parameter. Table 1 reports averages of 3.51 in bfloat16 and 3.83 in float32. Doubling the numerical storage precision therefore does not double the measured information acquired through this training procedure.

These quantities are not file sizes or a proof that weights can be quantized losslessly to 3.6 bits. Optimization may miss a higher-capacity solution, some large datasets are not fully converged, and the chosen compressor may fail to recover information that another procedure could use. The paper explicitly treats the measured capacity as a lower bound, conditional on architecture, training, and measurement.

### Text makes the distinction between memorization and generalization reference-dependent

For natural language, the experiments use FineWeb sequences of 64 tokens, with additional deduplication after truncation. That extra step matters because different full documents can yield duplicate short sequences. The paper considers a same-sized reference trained on the full dataset and a stronger oracle reference selected for low held-out loss.

With the stronger reference, unintended memorization first rises and then falls as the target's training dataset grows. The target becomes less specialized to individual examples and more useful on unseen text. This is not information literally being emptied from the weights; it is a change in what the target explains *beyond that reference*.

The double-descent analysis relates dataset information to estimated model capacity. Near the capacity boundary, held-out performance worsens before improving as data grows further. The authors interpret this as pressure to reuse shared structure once storing samples individually is no longer sufficient. That is an explanation supported by the controlled trend, rather than a universal theorem that all language-model generalization begins at one sharp storage threshold.

### Extraction and membership inference answer separate questions

The prefix-completion experiment gives the model 8, 16, or 32 tokens and asks it to reproduce the remainder of a 64-token sequence. Training extraction falls sharply with dataset size, but eventually becomes comparable to held-out extraction instead of necessarily reaching zero.

![Source Figure 10 comparing extraction of training and held-out sequences at different prefix lengths](/assets/images/how-much-do-language-models-memorize-source-figure-10.webp)
*Fig 2: Solid training curves approach dashed held-out curves at large dataset sizes. Nonzero extraction alone therefore cannot establish training membership; longer prefixes also change how much information the prompt already supplies. | source: [Memorization paper, Figure 10](https://arxiv.org/abs/2505.24832)*

Read the separation between solid and dashed curves of the same color, not only their absolute height. A long prefix can make a continuation predictable for both training and test examples. Under this controlled, deduplicated setup, matching extraction rates supports the authors' generalization account; it does not prove every extractable string in an arbitrary deployed model is harmless or unseen.

Membership inference instead thresholds model loss to decide whether a sequence belonged to training. It can succeed even when exact extraction fails: the paper reports settings with a membership score of 0.97 and extraction rate zero. A small likelihood advantage can reveal membership without being large enough to reproduce an entire suffix.

The fitted membership curves weaken as the dataset-to-capacity ratio grows, approaching the reported chance baseline for average examples. This is an empirical claim about the tested loss-based attack and sample distribution. It does not rule out stronger attacks, duplicated rare strings, or unusually memorized outliers.

## High-Level Takeaways

- Memorization is measured relative to information a reference model already explains.
- Uniform random data isolates sample-specific storage; likelihood gains can reveal partial storage without exact recall.
- Roughly 3.6 bits per parameter is a measured lower bound for the studied setup, not a universal quantization limit.
- Text memorization estimates change with the reference and the dataset-to-capacity regime.
- Extraction and membership inference remain distinct: either can be misleading when treated as a complete measure of memorization.
