---
title: 'Wiener Representation Filtering for VLM Hallucination Suppression'
date: '2026-08-08T14:51:17.000Z'
section: paper-shorts
postSlug: wiener-representation-filtering-for-vlm-hallucination-suppression
legacyPath: /paper shorts/2026/08/08/wiener-representation-filtering-for-vlm-hallucination-suppression.html
tags: [Other]
field: 'Vision-Language Models'
summary: '2026 – Wiener filtering suppresses hallucination-associated directions in a VLM without changing inference cost'
---

## 2026 – Wiener Representation Filtering for VLM Hallucination Suppression

**arXiv:** [2608.08167](https://arxiv.org/abs/2608.08167)

## Summary

> This paper makes hallucination mitigation an offline weight edit: estimate which deep language-model directions are associated with hallucinated responses, attenuate them with a covariance-derived Wiener filter, and absorb the result into the existing feed-forward projections. The reported CHAIR gains come with unchanged architecture, parameter count, and inference-time cost, but depend on calibration data that pairs truthful and hallucinatory generations.

## Core Insights

### A spectral filter rather than a decoding loop

The method treats a hidden state as a truthful component plus a hallucination-associated distortion. From paired samples, it estimates their second-order statistics, solves a generalized eigendecomposition, and applies a Wiener-style gain that weakens modes with a high distortion-to-signal ratio. The edit is made only to down-projection matrices in selected deep feed-forward blocks. It is calibrated once with forward passes, then folded into the weights; it adds neither a second decoding pass nor an inference module.

![Generalized eigenvalue spectrum used to identify representation directions associated with hallucination distortion](/assets/images/wiener-representation-filtering-paper-figure.webp)
*The concentrated spectrum motivates direction-dependent attenuation: a small set of modes carries much more estimated distortion relative to truthful signal than the rest. source: [Wiener Representation Filtering](https://arxiv.org/abs/2608.08167)*

![Figure 2 from Wiener Representation Filtering for VLM Hallucination Suppression](/assets/images/wiener-representation-filtering-for-vlm-hallucination-suppression-source-figure-2.webp)
*Figure 2 A qualitative case study comparing model-generated descriptions for a complex meme. We visualize the outputs from the Baseline and our method, color-coding text segments as Hallucinations and Truth. source: [Wiener Representation Filtering for VLM Hallucination Suppression](https://arxiv.org/abs/2608.08167)*

![Figure 3 from Wiener Representation Filtering for VLM Hallucination Suppression](/assets/images/wiener-representation-filtering-for-vlm-hallucination-suppression-source-figure-3.webp)
*Figure 3 Dolan–Moré performance profiles on 10 MME subsets for mPLUG-Owl2 (left) and LLaVA-1.5 (right). source: [Wiener Representation Filtering for VLM Hallucination Suppression](https://arxiv.org/abs/2608.08167)*


That placement is the central empirical choice. On MiniGPT-4, the paper reports sentence-level CHAIR of 23.0 for the baseline, 14.0 when editing layers 14–24, and 13.0 for layers 24–32. Mean subtraction and uniform shrinkage do not reproduce that result, supporting the narrower claim that the useful signal is direction-dependent rather than a global bias or a generic reduction in activation magnitude.

### The evidence is broad, but the calibration contract matters

Across LLaVA-1.5, MiniGPT-4, and mPLUG-Owl2 on CHAIR, the reported edit reaches sentence-level hallucination rates of 14.93, 16.87, and 15.40, respectively; the corresponding instance-level rates are 4.70, 6.13, and 6.07. The paper also reports POPE, MME, video reasoning, and diffusion-language-model experiments. Those are useful transfer checks, but they do not remove the operational dependency on representative calibration pairs, model-specific layer ranges, and a held-out tuning set.

| Decision | Reported choice | Consequence |
| --- | --- | --- |
| Intervention | Deep FFN down projections | Reuses the original runtime graph. |
| Signal | Paired truthful and hallucinatory hidden states | Requires a calibration set that reflects the target failure. |
| Filter | Covariance-derived, mode-wise attenuation | Preserves some semantic directions that uniform shrinkage removes. |
| Evaluation | CHAIR, POPE, MME and supplementary transfers | Measures several grounding settings, not every safety-critical hallucination mode. |

## High-Level Takeaways

- The paper shifts a common hallucination trade-off from decoding-time control to a one-time representation edit. Its atomic object is a hidden-state direction, not a token or a retrieved fact.
- The strongest controlled result is the comparison with mean subtraction, uniform shrinkage, and shallower layer ranges. It supports spectral, deep-layer editing on the tested models; it does not establish a universal hallucination subspace.
- A deployment decision should hold calibration size, image domain, and latency fixed against decoding-based controls, then test counterfactual images and rare objects. The method's case weakens if the same edit suppresses correct visual details under distribution shift.
- The unreported scaling question is recalibration: a changing model, data domain, or failure taxonomy may change the covariance estimate and the directions worth preserving.
- Hallucination reduction can be a weight-space filtering problem, but the filter is only as trustworthy as the contrast data used to estimate it.
