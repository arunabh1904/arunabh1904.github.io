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

> Wiener Representation Filtering turns object hallucination into an offline weight-editing problem. From paired truthful and hallucinatory generations of the same image, it estimates which hidden-state directions have unusually high distortion relative to truthful signal, attenuates those modes with a Wiener-shaped filter, and folds the edit into selected feed-forward projections. The architecture and inference graph stay unchanged. The reported gains are substantial on CHAIR and POPE, but the edit is model-specific and depends on calibration pairs that represent the deployment failure.

## Core Insights

### Hallucination is modeled as paired residual geometry

For an image with a truthful caption representation x− and a hallucinatory representation x+, the paper constructs a residual d = x+ − x−. It treats the truthful activation as signal with covariance ΣT and the residual as hallucination-associated distortion with covariance ΣH. The additive model is a calibration construction induced by the paired samples; it is not a claim that a transformer literally stores independent “truth” and “hallucination” neurons.

Under an approximately zero signal–residual cross-covariance assumption, the linear MMSE operator is A* = ΣT(ΣT + ΣH)−1. For a stable, interpretable edit, the authors eigendecompose ΣH = QΛQᵀ, measure the truthful variance τj² = qjᵀΣTqj in each distortion mode, and apply

$$
\tilde\gamma_j = \left(1 + \frac{\lambda_j}{\tau_j^2}\right)^{-\alpha}.
$$

The sharpness α changes how strongly the ordering is expressed. All modes remain available and gains are bounded between zero and one, unlike hard projection that deletes an entire subspace. With approximately 3,000 paired calibration samples per model, the filter is absorbed into the FFN output projection W̃out = Fα Wout, so there is no second decoding pass or new inference module.

![Figure 4: Hallucination covariance spectra for three VLMs](/assets/images/wiener-representation-filtering-paper-figure.webp)
*Fig 1: This source Figure 4 shows a few large eigenvalue spikes followed by a long decaying tail in LLaVA-7B, MiniGPT-4, and mPLUG-Owl2; the spectrum motivates continuous, direction-dependent attenuation. | source: [Wiener Representation Filtering, Figure 4](https://arxiv.org/abs/2608.08167)*

### Deep layers and anisotropy do the empirical work

The controlled ablation is more informative than the headline average. On MiniGPT-4, baseline CHAIR sentence/object rates are 23.0/8.4. Subtracting only the mean residual gives 22.0/8.2, and uniform shrinkage gives 24.0/8.5. Editing layers 14–24 reaches 14.0/5.0; editing the deeper 24–32 range reaches 13.0/4.9 with BLEU 0.162. This pattern says the useful correction is structured and appears late in the language backbone: a global offset or isotropic contraction cannot reproduce it.

The deployment configuration is selected on 100 held-out MSCOCO images and then fixed. LLaVA-1.5 uses layers 20–32 with α = 60, mPLUG-Owl2 layers 20–32 with α = 20, and MiniGPT-4 layers 24–32 with α = 10. Generation uses beam size 3 for CHAIR and greedy decoding for MME, with a 64-token limit for CHAIR/POPE and 128 for MME. These details matter because a weight edit can otherwise be credited for changes caused by decoding or layer selection.

### Grounding improves across tests, with a clear calibration contract

| Backbone | CHAIRS ↓ | CHAIRI ↓ | BLEU ↑ |
| --- | ---: | ---: | ---: |
| LLaVA-1.5, Wiener | 14.93 ± 2.61 | 4.70 ± 0.66 | 0.151 ± 0.005 |
| MiniGPT-4, Wiener | 16.87 ± 2.81 | 6.13 ± 1.33 | 0.157 ± 0.002 |
| mPLUG-Owl2, Wiener | 15.40 ± 1.93 | 6.07 ± 0.60 | 0.142 ± 0.001 |

These CHAIR results average 500 MSCOCO validation images across three seeds. The paper also reports POPE object-presence scores and Dolan–Moré profiles on ten relevant MME subsets, where the edit dominates competitors over a broad performance-ratio range. A qualitative example makes the desired behavior tangible: the filtered model removes nonexistent clouds and a striped pole from a meme description while keeping the scene's actual content.

![Figure 2: Qualitative hallucination reduction](/assets/images/wiener-representation-filtering-for-vlm-hallucination-suppression-source-figure-2.webp)
*Fig 2: This source Figure 2 compares a baseline and Wiener-filtered description of a complex meme, marking hallucinated and truthful text to show which claims disappear. | source: [Wiener Representation Filtering, Figure 2](https://arxiv.org/abs/2608.08167)*

![Figure 3: Dolan–Moré MME profiles](/assets/images/wiener-representation-filtering-for-vlm-hallucination-suppression-source-figure-3.webp)
*Fig 3: This source Figure 3 plots the fraction of ten MME subsets within a performance ratio of the best method for mPLUG-Owl2 and LLaVA-1.5; it tests whether the edit helps beyond CHAIR-style captions. | source: [Wiener Representation Filtering, Figure 3](https://arxiv.org/abs/2608.08167)*

The boundary is the calibration contract. A new model, domain, or hallucination taxonomy can change ΣH, the best layers, and the sharpness. The paper's evidence supports a training-free post-hoc edit for the evaluated backbones; it does not establish one universal hallucination subspace or guarantee that suppressing a frequent false positive will preserve rare but correct details.

## High-Level Takeaways

- The method preserves inference cost by editing FFN weights offline, while using covariance geometry to decide which directions to attenuate.
- Mean subtraction and uniform shrinkage fail to match deep, anisotropic Wiener editing on the controlled MiniGPT-4 ablation.
- The strongest CHAIR results are 14.93/4.70 for LLaVA-1.5, 16.87/6.13 for MiniGPT-4, and 15.40/6.07 for mPLUG-Owl2 at sentence/instance level.
- The practical risk is recalibration: the filter is only as representative as the truthful–hallucinatory pairs used to estimate it.
