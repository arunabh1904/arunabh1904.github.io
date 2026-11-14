---
title: "Test-Time Scaling Makes Overtraining Compute-Optimal"
date: '2026-04-01T00:00:00.000Z'
section: paper-shorts
postSlug: test-time-scaling-makes-overtraining-compute-optimal
legacyPath: /paper shorts/2026/04/01/test-time-scaling-makes-overtraining-compute-optimal.html
tags:
  - Inference
  - Scaling Laws
field: "Language Models"
summary: "2026 – Test-Time Scaling Makes Overtraining Compute-Optimal"
---

## 2026 – Test-Time Scaling Makes Overtraining Compute-Optimal

**Paper:** [Roberts et al., arXiv:2604.01411v1](https://arxiv.org/abs/2604.01411v1)

## Summary

> Train-to-Test scaling recommends smaller, longer-trained models when a fixed inference budget can buy more attempts from a cheaper model. Two fitted approaches agree on this direction across eight tasks and a sub-billion-parameter checkpoint suite, with additional overtrained checkpoints testing the prediction. The evaluation derives repeated-sampling performance from correct-answer likelihoods; it does not demonstrate a deployed selector recovering correct answers from generated candidate sets.

## Core Insights

[Generative-evaluation scaling](/paper%20shorts/2025/09/28/pretraining-scaling-laws-for-generative-evaluations-of-language-models.html) asks how pretraining predicts pass@$k$. Train-to-Test changes the decision: choose the model size, training duration, and number of attempts together. If a smaller model costs less per attempt, its lower single-attempt quality may be offset by a larger sampling budget. That possibility changes which model should be pretrained in the first place.

The paper imposes separate training and inference constraints:

$$
6ND\le C_{\mathrm{train}},\qquad 2Nk\le C_{\mathrm{inf}}.
$$

The inference expression is normalized per token served; it omits differing answer lengths, attention costs, and verification. Under this approximation, $k=C_{\mathrm{inf}}/(2N)$, so halving model size doubles the affordable number of attempts. “Overtraining” means allocating more tokens per parameter than the training-loss optimum under the same training budget.

### Difficulty variation changes the sampling forecast

For a problem with success probability $p_i$, oracle pass@$k$ is $1-(1-p_i)^k$. Applying this formula to the benchmark's mean probability overestimates mean pass@$k$: it hides the concentration of success on easy questions. The second modeling approach therefore fits a Beta distribution over per-question probabilities and predicts the average of the nonlinear sampling transformation.

The first approach instead fits an additive power law to expected negative log pass@$k$, with separate terms for model size, training tokens, and attempts. These objectives are related but not identical: averaging negative logs differs from taking the negative log of average success. Agreement between their recommendations is evidence about the direction of the allocation shift, not proof that their forecasts are interchangeable.

### The new checkpoints test a region the baseline suite barely covers

The study extends the [Porian scaling suite](/paper%20shorts/2024/06/27/resolving-discrepancies-in-compute-optimal-scaling.html) using the same RefinedWeb training setup. The full grid spans 5M–901M parameters and training budgets from $1.25\times10^{16}$ to $2.56\times10^{19}$ FLOPs. The source figure identifies the added high-token-ratio runs.

![Source Figure 8 shows model size against training compute, with orange cells marking newly trained overtrained checkpoints and cell values reporting tokens per parameter.](/assets/images/train-to-test-scaling-source-figure-8.png)
*Fig 1: Orange checkpoints extend the evaluation into smaller models trained for far more tokens per parameter. They test the predicted direction beyond the original fitting population; the grid remains entirely below one billion parameters. | source: [Train-to-Test scaling, Figure 8](https://arxiv.org/html/2604.01411v1#A3.F8)*

[Open the figure at full size](/assets/images/train-to-test-scaling-source-figure-8.png)

New runs use AdamW, 2,048-token sequences, a global batch of 256, and a cosine schedule matched to each run's token budget. Hyperparameters remain fixed across model sizes. The held-out extrapolation fits the original 85 checkpoints and evaluates additional overtrained runs. Both approaches overestimate performance somewhat; reported relative errors are 2.8% for the loss-based approach and 8.4% for the accuracy-based approach.

The tasks comprise LAMBADA, ARC-Easy, SciQ, OpenBookQA, and four synthetic sets of 1,000 short-answer questions each. The synthetic sets cover knowledge, arithmetic, causal reasoning, and spatial reasoning, generated with GPT-5 and Claude Opus 4.6. Appendix E specifies measuring likelihood on the correct-answer span. Consequently, the paper's pass@$k$ tables should be read as likelihood-derived sampling estimates, not conventional multiple-choice accuracies or observed success after answer selection.

At $C_{\mathrm{train}}=2.56\times10^{19}$ and $C_{\mathrm{inf}}=2\times10^9$, the reported LAMBADA estimate is 49.9% for a 37M overtrained checkpoint versus 27.3% for the 455M training-optimal comparison. Several science-task values remain very low despite relative gains. Both absolute performance and the scoring construction matter when interpreting these improvements.

### Post-training preserves the direction within a narrower test

The authors fine-tune on ARC-Easy, SciQ, and OpenBookQA for six epochs. One variant trains on question and answer tokens; the other masks the question and trains only the completion. Smaller overtrained models remain favored after these stages, although the recommended overtraining is less extreme. Equal epochs across model sizes do not imply matched post-training FLOPs, and these supervised experiments do not establish the same result after reasoning RL.

The useful conclusion is to include intended repeated sampling in pretraining design. My next test would measure actual generated solutions, selector accuracy, full sequence costs, and post-training compute at larger scale. Forecasts extending to $10^{25}$ training FLOPs lie far outside the observed checkpoint grid and should remain forecasts until that test is made.

## High-Level Takeaways

- The best pretraining allocation can change when cheaper inference permits more attempts per problem.
- A distribution over per-question success is necessary; transforming average success alone gives an optimistic sampling forecast.
- Additional overtrained checkpoints support the allocation shift within a sub-billion-parameter regime, while frontier-scale predictions remain extrapolations.
- Likelihood-derived oracle pass@$k$, fixed per-token cost, and supervised post-training define the present evidence boundary.
