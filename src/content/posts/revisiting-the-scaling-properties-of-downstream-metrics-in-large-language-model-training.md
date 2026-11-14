---
title: "Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training"
date: '2025-12-09T00:00:00.000Z'
section: paper-shorts
postSlug: revisiting-the-scaling-properties-of-downstream-metrics-in-large-language-model-training
legacyPath: /paper shorts/2025/12/09/revisiting-the-scaling-properties-of-downstream-metrics-in-large-language-model-training.html
tags:
  - Scaling Laws
  - Evaluation
  - Language Models
field: "Language Models"
summary: "2025 – Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training"
---

## 2025 – Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training

**Paper:** [Krajewski et al., arXiv:2512.08894v1](https://arxiv.org/abs/2512.08894v1) · **Code and results:** [apple/ml-scaling-downstream-metrics](https://github.com/apple/ml-scaling-downstream-metrics)

## Summary

> Downstream scores can be forecast directly from training compute when the recipe, data mixture, benchmark, and fit range are held fixed. Across 130 experiments and twelve tasks, the paper models the negative logarithm of chance-normalized accuracy with a power law in training FLOPs, extends it across token-to-parameter ratios, and derives a pass@k form for code. Its direct fit extrapolates better than the evaluated two-stage loss-to-accuracy baselines, but the claim remains conditional: near-chance points are excluded from fitting, larger budgets are held out for testing, and post-training changes are untested.

## Core Insights

### Directly fit the quantity you plan to decide with

The authors train decoder-only Transformers across 48 budgets and five token-to-parameter ratios, reaching models up to 17B parameters and 350B tokens. The main mixture is 75% DCLM, 15% filtered Stack v2, and 10% OpenMathReasoning; a separate C4-only sweep tests whether the pattern survives a different corpus. The model recipe is fixed: RoPE, SwiGLU, a 150k-token tokenizer, 4,096-token sequences, a batch size scaled with $D^{0.5}$, and a proxy-tuned maximum learning rate of $5\times10^{-3}$. These controls make the fitted curve interpretable as a recipe-specific map from compute to evaluation, not a law over arbitrary model families.

For a benchmark score $Q$ at training budget $C$, the main fit is

$$
-\log Q = \frac{A}{C^{\alpha}}, \qquad A>0,\;\alpha>0.
$$

Multiple-choice scores are first normalized above a measured random baseline, estimated from ten small models below $10^{17}$ FLOPs. The transformed curve can represent the observed S-shape more naturally than a strictly concave power law in raw accuracy. The paper's Figure 1 panel (a) shows the ARC-Easy points and fitted curve; the comparable panels cover ARC-Challenge, SciQ, PIQA, HellaSwag, WebQS, LAMBADA, GSM8K, and HumanEval.

![Cropped source Figure 1 panel (a) fits normalized ARC-Easy accuracy as a function of training FLOPs.](/assets/images/downstream-metrics-scaling-source-figure-1.png)
*Fig 1: The ARC-Easy score follows the direct $-\log Q = A/C^\alpha$ fit across the reported training-budget range; points are final checkpoints and the line is the fitted law. This is a source crop of panel (a), not a new composite. | source: [Krajewski et al., Figure 1](https://arxiv.org/html/2512.08894v1#S1.F1)*

[Open the source panel at full size](/assets/images/downstream-metrics-scaling-source-figure-1.png)

### The extrapolation test is stricter than the in-sample fit

For the held-out comparison, the authors fit on runs between $3\times10^{18}$ and $6\times10^{21}$ FLOPs and predict models up to 6.7 times more expensive. They discard scores less than five percentage points above the random baseline because those points have higher variance. In Table 3, the direct power law reaches mean absolute error 0.0203 and mean relative error 4.72% across twelve benchmarks, versus 0.0254 and 4.96% for the broken neural scaling law (BNSL). The result is a small average advantage, not evidence that the simple form wins on every benchmark; Table 3 includes cases where BNSL is better.

The two-stage comparison asks a different question. It first predicts a proxy such as negative log-likelihood from compute and then maps that proxy to accuracy. Proxy-to-accuracy fits can have $R^2>0.95$ on the training points, yet the composed forecast is worse: Table 4 reports mean MRE 1.963% for the direct power law, 2.713% for BNSL, 6.667% for a two-stage linear map, and 6.351% for a two-stage logistic map. The table's RMSE and $R^2$ are training-fit diagnostics, while the MRE and MAE comparison is the extrapolation result; they should not be read as one metric hierarchy.

This directly sharpens the earlier [overtraining and downstream-scaling note](/paper%20shorts/2024/03/13/language-models-scale-reliably-with-overtraining-and-on-downstream-tasks.html). That paper shows that a loss forecast and an aggregate downstream forecast require different fitting investments. Here, the authors avoid one source of compounded error by fitting the downstream target itself, while retaining the same lesson about recipe scope. The [reality check on downstream scaling](/paper%20shorts/2025/07/01/scaling-laws-are-unreliable-for-downstream-tasks-a-reality-check.html) supplies the necessary counterweight: a direct curve can be reliable in a controlled ladder even though changing the validation corpus or harness can change the qualitative behavior.

### More tokens and more samples are separate axes

Across token-to-parameter ratios, the paper adapts the Chinchilla-style form to

$$
-\log Q = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
$$

with a holdout containing runs above $6\times10^{21}$ FLOPs or the largest ratio, 160. The fit reports training MAE 0.0087 and validation MAE 0.0182 (4.05% and 6.42% relative error). The model assumes a perfect asymptotic score; Appendix L adds an irreducible term and finds plausible ceilings below one for PIQA, HellaSwag, and LAMBADA, while warning that poorly covered benchmarks can produce implausible ceilings.

Inference compute is modeled separately for code. If a single sample succeeds with probability $q$, independent sampling gives $\operatorname{pass@}k=1-(1-q)^k$; Appendix D derives bounds before introducing a fitted log-linear form. Renaming its signed coefficients to distinguish them from the positive exponents above gives

$$
\log(-\log Q(C,k)) = a_0 + a_C\log C + a_k\log k + a_{Ck}\log C\log k.
$$

HumanEval pass@k validation uses fits below $6\times10^{21}$ FLOPs and $k\le 32$, then predicts the held-out region with MAE 0.0284 and 7.94% relative error. This is a generative coverage estimate: it presumes independent attempts and a correctness test. It does not measure the cost or reliability of selecting a correct sample in a deployed system, a distinction also made in the earlier [generative-evaluation scaling note](/paper%20shorts/2025/09/28/pretraining-scaling-laws-for-generative-evaluations-of-language-models.html).

### The boundary is visible in the appendices

The C4-only ablation preserves smooth scaling for general QA but returns code and math tasks toward chance, because the mixture no longer contains those domains. The authors also report that reliable extrapolation depends on a task-specific FLOPs threshold; Appendix C sweeps the train/validation split and defines success as MRE below 10%. MoE, retrieval, multimodal models, alternative optimizers, instruction tuning, and preference optimization are not tested. Prediction intervals and the train-versus-inference compute frontier remain open.

## High-Level Takeaways

- A direct law can be the right engineering baseline when the target is downstream accuracy and the training ladder is held fixed.
- Normalizing multiple-choice scores by a measured random baseline is part of the model, not a cosmetic preprocessing step.
- Direct extrapolation beats the evaluated two-stage maps on average, even when those maps fit the observed points more closely.
- Token-to-parameter allocation and repeated sampling add separate compute axes; neither result implies a universal model of total deployment cost.
- Treat the fitted range, data mixture, benchmark composition, and pretraining-only scope as part of the claim; changing them is the experiment that could reverse it.
