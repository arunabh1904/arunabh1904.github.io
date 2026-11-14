---
title: 'Prescriptive Scaling Laws for Data Constrained Training'
date: '2026-05-02T00:00:00.000Z'
section: paper-shorts
postSlug: prescriptive-scaling-laws-for-data-constrained-training
legacyPath: /paper shorts/2026/05/02/prescriptive-scaling-laws-for-data-constrained-training.html
tags: [Scaling Laws, Data Efficiency, Regularization]
field: 'Language Models'
summary: '2026 – Prescriptive Scaling Laws for Data Constrained Training'
---

## 2026 – Prescriptive Scaling Laws for Data Constrained Training

**Paper:** [Lovelace et al., arXiv:2605.01640v2](https://arxiv.org/abs/2605.01640v2) · [Code](https://github.com/justinlovelace/data-constrained-scaling). This note covers the November 2026 revision.

## Summary

> Repeated tokens can continue to help optimization while also increasing overfitting. Lovelace et al. model these effects separately and test the resulting training recommendations. In their data-constrained experiments, adding model capacity and reducing epochs can outperform further repetition. Stronger weight decay shifts that tradeoff, but its benefit depends on how tightly data constrains the run.

## Core Insights

### Saturating benefit does not describe increasing damage

[Scaling Data-Constrained Language Models](/paper%20shorts/2023/05/25/scaling-data-constrained-language-models.html) replaces repeated tokens with an effective-data quantity whose marginal value decays. That can represent diminishing improvement, but not a validation loss that rises under excessive repetition. This paper adds an explicit penalty to a Chinchilla-style baseline, fitted first on single-epoch runs.

Its simplest form is

$$
L(N,U,R)=E+\frac{A}{N^\alpha}+\frac{B}{[U(1+R)]^\beta}+P R\frac{N}{U},
$$

where $N$ is total parameters, $U$ is unique tokens, and $R$ counts additional epochs after the first. Repeated tokens still reduce the data term; the last term charges for repetition relative to available data and model capacity. Two- and four-parameter penalties allow nonlinear capacity and repetition effects. The richer form supplies the paper's allocation recommendations; the one-parameter form makes the regularization comparison easier to interpret.

The main sweep trains Llama-2-style models from 15M to 1B parameters on 50M–6B unique FineWeb tokens for up to 16 epochs. The 512-token context, tokenizer, optimizer, and schedule define the regime. An additive penalty fits both this sweep and the earlier paper's public runs better than the tested saturating formulations. Refitting on FineWeb-Edu preserves that ranking, providing a second corpus check rather than proof of universal coefficients.

### Test the recommended configuration, not just the curve

![Source Figure 5 compares observed loss contours with compute-optimal model and token allocations.](/assets/images/prescriptive-data-constrained-scaling-source-figure-5.png)
*Fig 1: The left panel shows the measured loss landscape at 100M unique tokens. On the right, the additive-penalty frontier bends toward larger models and fewer repeated tokens under a fixed compute constraint. | source: [Paper, Figure 5](https://arxiv.org/html/2605.01640v2#S5.F5)*

[Open the source figure at full size](/assets/images/prescriptive-data-constrained-scaling-source-figure-5.png).

At 500M unique tokens and a nominal $2\times10^{19}$-FLOP budget, the additive law recommends a 2.2B model for three epochs. Its measured perplexity is 17.73, versus 19.34 for the effective-parameter law's 950M model trained for seven epochs and 20.44 for the Chinchilla baseline's 700M model trained for ten. The selected configurations also improve average bits per byte over 19 OLMES tasks. Bits per byte is a graded likelihood metric, not average task accuracy.

These held-out runs are stronger evidence than a high fit score alone: the fitted law chooses a training decision and the authors execute it. They also extend beyond the 1B fitting sweep, while remaining far below frontier model scale.

### Regularization changes the useful repetition budget

Increasing weight decay from 0.1 to 1.0 reduces the fitted one-parameter overfitting coefficient by approximately 70%. This is not a 70% reduction in total validation loss. Strong decay incurs a single-epoch loss penalty, then becomes useful when repeated exposure is sufficiently costly. Additional experiments at decay values 0.5 and 3.0 reinforce that the best setting depends on compute and unique data.

Appendix A also diagnoses a historical disagreement. Refitting the earlier study's single-epoch baseline on its own runs changes the subsequent allocation advice toward larger models and fewer epochs. An apparent repetition effect can partly compensate for a baseline transferred between incompatible training setups. Fit the baseline for the actual tokenizer, corpus, and training recipe before interpreting the new penalty.

The remaining boundary is substantial: the law does not model double descent or establish behavior under extreme repetition. Bootstrap uncertainty in the penalty holds the single-epoch fit fixed, so it is conditional on that baseline. Each tested weight-decay setting receives a separate fit; the paper does not establish a continuous law for arbitrary regularization values.

## High-Level Takeaways

- Diminishing returns and worsening generalization require different terms in a scaling model.
- Validate a scaling law by training its recommended configuration under the intended budget.
- Stronger weight decay trades a single-epoch cost for greater repetition tolerance.
- A mismatched baseline can change the allocation policy inferred from a repetition study.
