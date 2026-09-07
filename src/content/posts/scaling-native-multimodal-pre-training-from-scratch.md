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

**arXiv:** [2607.22043](https://arxiv.org/abs/2607.22043)

## Summary

> This study fits compute-optimal frontiers for a decoder-only native multimodal model trained from scratch on text and continuous image patches. Language and multimodal objectives share parameters but prefer different allocations: language scaling is nearly invariant to the multimodal ratio, while multimodal scaling becomes more token-hungry as that ratio rises. The resulting Pareto frontier connects model size, text tokens, multimodal tokens, and data composition under one compute budget.

## Core Insights

### Two objectives share a model but not a frontier

The experiments use auxiliary-loss-free MoE transformers with 71M, 128M, 340M, 590M, 874M, and 3B active non-embedding parameters. A single projection turns 32×32 image patches into continuous embeddings; there is no separate vision encoder. The corpus contains 250B text tokens and 75B multimodal tokens. Multimodal runs use $r=D_{\mathrm{mm}}/D_{\mathrm{text}}\in\{0.1,0.2,0.3\}$, with $r=0$ as the text-only control. Thus $r=0.3$ means 75B multimodal tokens per 250B text tokens, or about 23% of the combined tokens.

Total compute is approximated by $C=6ND$, but the separate objective fits use $C_{\mathrm{text}}=6ND_{\mathrm{text}}$ and $C_{\mathrm{mm}}=6ND_{\mathrm{mm}}$. For each objective-specific budget, the authors vary model size and its corresponding token count and fit a parabola to the loss against model size. The minimum estimates $N_{\mathrm{opt}}$, then $D_{\mathrm{opt}}=C/(6N_{\mathrm{opt}})$. A lower envelope over complete training curves supplies an independent check. The use of two estimators matters: a single under-sampled IsoFLOP curve could mistake an irregular checkpoint for the true compute optimum.

![IsoFLOP profiles for the language objective across multimodal ratios](/assets/images/scaling-native-multimodal-pre-training-from-scratch-source-figure-1.webp)
*Fig 1: At fixed text compute, each multimodal ratio produces a parabolic loss profile over model size; the stars mark the fitted compute-optimal model sizes. | source: [Scaling Native Multimodal Pre-Training From Scratch, Figure 1](https://arxiv.org/abs/2607.22043)*

### Language allocation stays stable while multimodal allocation moves

Using text compute and text token counts, the language fits are $N_{\mathrm{opt}}\propto C^a$ with $a=0.697,0.684,0.667,0.663$ for $r=0,0.1,0.2,0.3$, and $D_{\mathrm{opt}}\propto C^b$ with the complementary exponents 0.303, 0.316, 0.333, and 0.337. The envelope cross-check does not reproduce a monotonic decline, so the paper treats the small drift as fitting noise rather than evidence that multimodal data changes the language allocation law.

The multimodal objective uses multimodal compute and token counts. Its IsoFLOP exponents are $a=0.709,0.679,0.643$ for $r=0.1,0.2,0.3$, with token exponents $b=0.291,0.321,0.357$. The envelope estimator follows the same downward trend in $a$. Increasing the multimodal share therefore shifts compute toward data: at $r=0.3$, buying capacity without enough multimodal tokens is a worse use of the budget than it is at $r=0.1$.

![Training-curve envelopes for the language objective](/assets/images/scaling-native-multimodal-pre-training-from-scratch-source-figure-2.webp)
*Fig 2: The lower envelope of language training curves yields a compute law and an independent allocation estimate; the fitted frontier is nearly the same across mixture ratios. | source: [Scaling Native Multimodal Pre-Training From Scratch, Figure 2](https://arxiv.org/abs/2607.22043)*

### The downstream effect is transfer, not a free accuracy gain

With the text budget fixed at 250B, adding up to 75B multimodal tokens changes the average over 16 text benchmarks by less than one percentage point at every model scale. That is a preservation result, not evidence that vision improves every language task. The sharper transfer appears on the text-only abstract spatial-reasoning subtasks of SpatialEval: multimodal runs consistently beat text-only controls, and the gap widens toward 3B parameters.

The same model family acquires multimodal in-context learning without parameter updates. At 71M parameters, one- and three-shot templates offer essentially no average gain; at 874M the three-shot gain reaches about 1.80 points, and at 3B it reaches 2.43 points. The gains are concentrated in spatial reasoning, while OCR and recognition categories can plateau or decline with extra shots. This pattern ties the scaling result to a mechanism: multimodal pretraining appears to teach reusable relational structure, but it does not uniformly improve visual recognition.

The authors combine the two objectives in a joint Pareto analysis. At total compute, $r=0.1$ yields approximately $N_{\mathrm{opt}}\propto C_{\mathrm{total}}^{0.69}$, while $r=0.3$ yields approximately $C_{\mathrm{total}}^{0.66}$ for parameters and $C_{\mathrm{total}}^{0.34}$ for tokens. These are planning curves within one model family, not a replacement for a held-out downstream sweep.

## High-Level Takeaways

- Native multimodal pretraining creates separate language and multimodal allocation laws inside shared parameters.
- More multimodal data makes the multimodal objective more token-hungry, so text-only compute ratios are unsafe defaults.
- Spatial transfer and few-shot gains emerge with scale, while average text ability stays within about one point.
- The frontier is measured only through 3B active parameters, one patch-embedding design, one corpus family, and smoothed training loss; larger runs need prospective validation.
