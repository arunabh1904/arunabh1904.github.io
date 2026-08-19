---
title: 'Scaling Native Multimodal Pre-Training From Scratch'
date: '2026-07-24T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-native-multimodal-pre-training-from-scratch
legacyPath: /paper shorts/2026/07/24/scaling-native-multimodal-pre-training-from-scratch.html
tags:
  - Multimodal Pre-Training
  - Scaling Laws
  - Data Mixtures
field: 'Multimodal Scaling & Data Mixtures'
topics:
  - multimodal
  - learning
summary: '2026 – Scaling Native Multimodal Pre-Training From Scratch'
---

## 2026 – Scaling Native Multimodal Pre-Training From Scratch

**arXiv:** [2607.22043](https://arxiv.org/abs/2607.22043)

## Summary

> Native multimodal pre-training shares one Transformer across text and image patches from the start, but the two objectives need not want the same compute allocation. This paper fits separate IsoFLOP frontiers for language loss and multimodal loss across six model sizes and three image-text mixture ratios. Language allocation is nearly invariant to the mixture once text compute is held fixed; multimodal allocation is not. As the multimodal ratio increases, the loss-minimizing recipe shifts toward more training tokens and slower parameter growth.

## Core Insights

That asymmetry is the paper's main result. It means a single aggregate scaling law can conceal a real resource conflict: the text objective may tolerate the added modality without changing its preferred parameter-token balance, while the multimodal objective becomes increasingly data-hungry. The fitted joint Pareto frontier turns that conflict into a planning tool, but its exponents are measured only up to 3B active parameters and should not be treated as a universal recipe for frontier-scale runs.

The experiments train auxiliary-loss-free MoE decoder-only Transformers with 71M, 128M, 340M, 590M, 874M, and 3B activated non-embedding parameters. Images are converted directly to continuous $32\times32$ patch embeddings by one projection layer rather than a separate vision encoder. The available corpus contains 250B text tokens and 75B multimodal tokens; mixture ratio $r$ ranges from 0.1 to 0.3 for multimodal runs, with $r=0$ as the text-only control.

For each compute budget, the authors vary model size and token count under $C\approx6ND$, fit a parabola to the IsoFLOP profile, and use its minimum to estimate $N_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$. A second estimator takes the lower envelope of every training curve. Agreement between these methods is useful because either one alone can confuse a noisy local minimum with a scaling law.

| Objective | Mixture ratio $r$ | Fitted parameter exponent $a$ in $N_{\mathrm{opt}}\propto C^a$ | Token exponent $b=1-a$ |
| --- | ---: | ---: | ---: |
| Language | 0.0 | 0.697 | 0.303 |
| Language | 0.1 | 0.684 | 0.316 |
| Language | 0.2 | 0.667 | 0.333 |
| Language | 0.3 | 0.663 | 0.337 |
| Multimodal | 0.1 | 0.709 | 0.291 |
| Multimodal | 0.2 | 0.679 | 0.321 |
| Multimodal | 0.3 | 0.643 | 0.357 |

The small drift in language exponents is not reproduced monotonically by the independent envelope estimator, so the authors interpret language allocation as composition-invariant within fitting error. Both estimators, however, show the same decline for the multimodal parameter exponent. At $r=0.3$, a compute increase should therefore buy relatively more tokens than at $r=0.1$ if minimizing the multimodal loss is the target.

![Joint language-multimodal Pareto frontier and fitted parameter-token allocations across mixture ratios](/assets/images/native-multimodal-pareto-frontier.png)
_Sweeping the mixture ratio traces different language–multimodal trade-offs at fixed total compute; the fitted allocation changes with the target mixture. Cropped from Figure 7 of the [paper](https://arxiv.org/abs/2607.22043)._

The downstream evidence adds two narrower findings. Holding the text budget at 250B tokens, adding up to 75B multimodal tokens changes the average over 16 text benchmarks by less than one point at every model scale. On the text-only abstract-spatial portion of SpatialEval, multimodal models outperform text-only controls and the gap widens with scale. Multimodal in-context learning also appears only after enough capacity and data: the 3-shot gain is near zero at 71M parameters but reaches 2.43 points at 3B, with most of the benefit concentrated in spatial reasoning rather than OCR or recognition.

## High-Level Takeaways

- This paper informs the allocation of a native multimodal pre-training budget across active parameters, total tokens, and modality mixture. Its atomic units are text tokens and continuous image-patch tokens optimized by shared parameters. The evidence argues against choosing model size from a text-only Chinchilla curve and then filling the remaining budget with images. The multimodal objective's preferred allocation changes with the amount of multimodal data.
- The expensive decision is committing a large run to exponents estimated from smaller models, one architecture family, one image-text corpus, and training loss as the proxy objective. The study provides fitted scaling behavior within its measured range, not evidence that downstream utility or a different tokenizer obeys the same frontier. At 10× scale, data quality and patch-token redundancy may dominate before the fitted compute curve does.
- Before a frontier run, the decisive experiment is a prospective holdout sweep: train several configurations just beyond the fitted compute range, vary mixture ratio and patch compression independently, and select on both held-out multimodal loss and task utility. The claim should be rejected if the predicted allocation fails out of range, if a stronger image encoder changes the exponents, or if equal-loss configurations produce materially different downstream behavior.
- The work extends compute-optimal language-model analysis to a shared native multimodal system by fitting the language and multimodal objectives separately before reconciling them on a Pareto frontier.
- Models stop at 3B active parameters; the visual data comes from one image-text family; images use one direct patch-embedding scheme; and the scaling fits optimize smoothed training loss rather than a held-out multimodal likelihood or downstream score.
- In native multimodal training, the language objective keeps roughly the same allocation law, but the multimodal objective asks for progressively more data as its mixture share grows.
