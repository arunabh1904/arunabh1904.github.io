---
title: 'Scaling Laws for Vision-Language Fusion'
date: '2026-11-17T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-vision-language-fusion
legacyPath: /paper shorts/2026/11/17/scaling-laws-for-vision-language-fusion.html
tags:
  - Vision-Language Models
  - Pretraining
  - Scaling Laws
field: "Multimodal Scaling & Data Mixtures"
summary: "2026 – Scaling Laws for Vision-Language Fusion"
---

## 2026 – Scaling Laws for Vision-Language Fusion

**Paper:** [Liew, Suzuki and Tokui, arXiv:2611.22351v1](https://arxiv.org/abs/2611.22351v1) · **Full text:** [arXiv HTML](https://arxiv.org/html/2611.22351v1)

## Summary

> If you have a fixed supply of text and image-caption tokens, when should the captions enter training? In these experiments, mixing captions from the start preserves language performance. Starting later and concentrating the captions improves caption validation loss, but the advantage shrinks with model size and does not consistently improve visual question answering.

## Core Insights

### Token totals and the mixture shift answer different questions

[Sections 4.1–4.3](https://arxiv.org/html/2611.22351v1#S4) fit validation loss as a power law in an effective token count, plus a shift that develops as the model trains on its new mixture. The effective count weights out-of-domain tokens by their estimated contribution to progress. For interleaved image-text validation, the fitted text-token exchange rate is 1.34; for caption validation, it is 0.013 (Table 1). In this setup, longer text-only pretraining moves caption loss much less than it moves interleaved-document loss. The numbers describe these datasets and runs, not a general value for a text token.

The shift has a different role: it moves the loss curve as the model adapts to the new mixture. Its size depends on the multimodal token fraction after the switch, and its effect builds over continued training. The fit uses a fractional competition term rather than assuming a linear penalty per added caption token. Total effective tokens can stay fixed while this shift changes.

![Two training timelines for equal text and caption budgets: mixed training from the start, or a text-only phase followed by captions.](/assets/images/vision-language-fusion-schedules.svg)
*Fig 1: The same tokens can be spread across training or concentrated after a text-only phase. Original schematic, not a loss plot. | source: [Liew et al., Figure 1 and Section 6.2](https://arxiv.org/html/2611.22351v1#S6.SS2)*

[Open the figure at full size](/assets/images/vision-language-fusion-schedules.svg)

Under fixed totals, the base effective count for caption loss is the number of caption tokens plus the number of text tokens multiplied by the caption exchange rate. Moving the switch later does not change that sum; it raises the caption fraction after the switch. In the fit, the shift then favors caption loss and hurts text loss. A pure-caption final stage lands outside the fitted text-loss regime: the authors restrict the caption-fusion text fit to multimodal fractions up to 0.82. The measured deterioration at 100% is evidence of an extreme failure mode, not a calibrated prediction of its size.

### Test the loss trade-off on the task you actually need

[Table 3](https://arxiv.org/html/2611.22351v1#S6.T3) holds the budget at 56B tokens and compares schedules on 300M and 1B models. At 300M, moving from early fusion to the pure-caption final stage lowers caption loss by 0.159 nats; at 1B the gain is 0.095 nats. The text-loss penalty remains large at both sizes. Yet at 1B, the same shift makes MME worse (0.7064 to 0.6839) and GQA slightly better (0.4134 to 0.4209). Caption loss and visual question answering are not interchangeable objectives.

To measure base checkpoints without fine-tuning for instruction following, the authors recast multiple-choice answers as candidate statements about the image and rank them by byte-normalized log likelihood. They also compare with and without the image prefix. This distinguishes gains that depend on the image from improvements due to answer-format preferences, although it does not establish the ranking after instruction tuning.

Early fusion is the first schedule I would try for a model that needs to retain language ability. I would compare it with a later switch that still replays text and measure both target capability and training cost. The paper's schedule comparison matches tokens, not compute; it uses a Gemma-3-style backbone with a pretrained SigLIP2 encoder. Neither the fitted rates nor the downstream ordering should be assumed to transfer to a different architecture or data mixture.

The earlier [native multimodal scaling study](/paper%20shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html) compares early and late fusion as architectures. This paper holds an early-fusion design fixed and moves the same image-token allocation through training. The two results inform different budget decisions: which model to build and when to expose it to image tokens.

## High-Level Takeaways

- Fixing the text and caption token totals isolates a schedule-dependent mixture shift in the fitted law.
- Caption and interleaved data have very different exchange rates with text in these experiments.
- The late-stage caption-loss advantage shrinks with scale and does not reliably translate to downstream vision gains.
- Test the final task and language retention before choosing a fusion schedule.
