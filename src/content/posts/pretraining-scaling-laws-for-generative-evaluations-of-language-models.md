---
title: 'Pretraining Scaling Laws for Generative Evaluations of Language Models'
date: '2025-09-28T00:00:00.000Z'
section: paper-shorts
postSlug: pretraining-scaling-laws-for-generative-evaluations-of-language-models
legacyPath: /paper shorts/2025/09/28/pretraining-scaling-laws-for-generative-evaluations-of-language-models.html
tags: [Scaling Laws, Evaluation, Reasoning]
field: 'Language Models'
summary: '2025 – Pretraining Scaling Laws for Generative Evaluations of Language Models'
---

## 2025 – Pretraining Scaling Laws for Generative Evaluations of Language Models

**Paper:** [Schaeffer et al., arXiv:2509.24012v2](https://arxiv.org/abs/2509.24012v2)

## Summary

> A model's generative scaling curve depends on how many answers it is allowed to sample. This study fits and backtests pass@k using training compute, parameters plus tokens, or gold-reference likelihood. A tight fit across existing models does not guarantee a good forecast of the largest model, and stable fitted parameters do not automatically produce lower prediction error.

## Core Insights

### Sampling budget is part of the evaluation target

Pass@k measures the probability that at least one of $k$ generated answers is correct. For independent draws with single-answer success probability $p$, that probability is $1-(1-p)^k$. It can rise sharply even when the underlying single-answer probability changes smoothly. The paper uses the standard finite-sample estimator and averages over problems; it does not assume that every problem has the same $p$.

This extends the measurement concern in the earlier [emergent-abilities analysis](/paper%20shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html): the scoring procedure changes the apparent scaling behavior. Pass@k also assumes access to a correctness check. Finding a correct answer somewhere among many samples does not establish that a deployed system can select it without that check.

The experiments use Pythia checkpoints spanning 14M–12B parameters and up to 300B pretraining tokens, with temperature-only sampling at 1.0. Each benchmark is restricted to 128 randomly selected GSM8K or MATH problems. Many thousands of generations per problem improve resolution, but they do not turn those 128 problems into a broad sample of tasks.

### Backtest the forecast, not just the fit

The compute law models negative log pass rate as

$$
-\log\operatorname{pass@}k=E_0(k)+C_0(k)C^{-\alpha(k)}.
$$

All three fitted quantities depend on the answer budget. For GSM8K, increasing $k$ lowers the fitted residual floor and steepens the compute exponent. That floor is a property of the fitted family, benchmark, and sampling setup; it is not an irreducible limit on mathematical reasoning.

![Source Figure 3 shows forecast errors and fitted parameters changing as more expensive checkpoints enter the fit.](/assets/images/generative-evaluation-scaling-source-figure-3.png)
*Fig 1: The horizontal axis is the largest fitting checkpoint's compute relative to the target. Fits based only on much cheaper models are unstable; including checkpoints closer to the target improves the forecast. Colors represent different numbers of attempts. | source: [Paper, Figure 3](https://arxiv.org/html/2509.24012v2#S2.F3)*

[Open the source figure at full size](/assets/images/generative-evaluation-scaling-source-figure-3.png).

The target is Pythia-12B at 300B tokens, approximately $2.16\times10^{22}$ training FLOPs. In the GSM8K backtests, useful compute-based forecasts generally require fitting checkpoints within roughly 1.5–2.5 orders of magnitude of that target. Separating parameters and tokens tightens the in-range fit but gives little consistent improvement in forecasting the largest checkpoint.

### Likelihood is a useful proxy with a different information requirement

Gold-reference likelihood produces more stable fitted parameters across roughly five orders of magnitude in the GSM8K analysis. Its forecast errors remain broadly comparable to the other laws: somewhat better at small $k$, somewhat worse at large $k$. Parameter stability and predictive accuracy are separate diagnostics.

This proxy also requires measuring the model's likelihood of a reference solution. It can reduce expensive generative evaluation once a checkpoint exists; using it to forecast a not-yet-trained model still requires predicting that likelihood. A particular gold solution need not represent every valid reasoning path, and the paper leaves robustness under strong optimization pressure open.

The benchmark changes the conclusion. On MATH, the fitted residual floor remains substantial even at 10,000 attempts, and the gold-likelihood exponent stabilizes across a shorter compute range. The strongest GSM8K stability result should therefore not be generalized to all reasoning tasks.

The theoretical result is similarly conditional. If negative log pass rate is additive in parameter and token power laws, minimizing it under $C\propto ND$ yields the compute-only law, with exponent $\alpha=\beta\gamma/(\beta+\gamma)$ for fixed $k$. It describes the best allocation for that fitted target. It does not jointly optimize pretraining and the cost of generating or selecting $k$ answers.

## High-Level Takeaways

- State the sampling budget whenever comparing generative scaling curves.
- Validate a forecast by withholding the expensive checkpoint, rather than relying on fit quality.
- Gold-reference likelihood offers a useful evaluation proxy, with benchmark-dependent limits.
- Pass@k coverage, answer selection, and total inference cost require separate measurements.
