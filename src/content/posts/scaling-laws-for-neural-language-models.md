---
title: 'Scaling Laws for Neural Language Models'
date: '2020-01-23T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-neural-language-models
legacyPath: /paper shorts/2020/01/23/scaling-laws-for-neural-language-models.html
tags: [Language Models, Scaling Laws, Pretraining]
field: 'Language Models'
summary: '2020 – Scaling Laws for Neural Language Models'
---

## 2020 – Scaling Laws for Neural Language Models

**arXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)

## Summary

> Kaplan et al. show that decoder-only Transformer loss follows smooth power laws in non-embedding parameters, training tokens, and compute over several orders of magnitude. The laws make a strong local claim: while data and capacity are not yet bottlenecks, a small family can forecast a larger run. Under their training setup, compute should favor a much larger model and relatively little data—a prescription that later compute-optimal work revises.

## Core Insights

### Three one-resource slices share a shape

The experiments train decoder-only Transformers primarily on WebText2, with a 50,257-token BPE vocabulary and a 1,024-token context. They vary non-embedding parameters $N$, data tokens $D$, and compute $C$, while holding the other resources far enough from their bottleneck that a clean fit is visible. The resulting held-out losses are approximately

$$
L(N)\propto N^{-0.076},\qquad L(D)\propto D^{-0.095},\qquad L(C)\propto C^{-0.050}.
$$

The precise coefficients are less portable than the shape. A change in tokenizer, data mixture, optimizer, or architecture changes the intercept and may change the usable range. This is the same empirical pattern seen across modalities in [Hestness et al.](/paper%20shorts/2017/12/01/deep-learning-scaling-is-predictable-empirically.html), now measured in a tightly controlled language-model family.

![Kaplan et al. Figure 1: loss against compute, data, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)
*Fig 1: The three fitted slices are straight on log-scaled axes, summarizing one WebText2 training regime rather than a universal law. | source: [Scaling Laws for Neural Language Models, Figure 1](https://arxiv.org/abs/2001.08361)*

### The compute frontier is not the convergence frontier

The paper defines training compute approximately as $6ND$ FLOPs, ignoring subleading terms for the model sizes studied. Its compute allocation result is counterintuitive: the best loss at a fixed budget comes from increasing parameters quickly and data slowly. Roughly, $N\propto C^{0.73}$ and $D\propto C^{0.27}$, or about 5.5 times more parameters and 1.8 times more tokens for a tenfold compute increase. These models are intentionally stopped well before full convergence; the frontier is about loss per training FLOP, not the cheapest way to finish a fixed model.

The overfitting experiments explain why data cannot be held fixed indefinitely. The penalty depends strongly on a ratio close to $N^{0.74}/D$: a larger model needs more data, even though the compute-optimal frontier still favors parameter growth in the measured range. Transfer to other text distributions mostly adds a roughly constant loss offset, which supports using the law for nearby distributions but not for arbitrary domain shifts.

### Read the appendices as experimental conditions

Training curves are predictable early enough to extrapolate, and the optimal batch size tracks the gradient-noise scale. Shape changes at fixed $N$ matter less than total non-embedding parameter count in the tested family, but embeddings, context length, learning-rate schedule, and optimizer details are not irrelevant outside that regime. The paper's learning-rate schedules also rely on warmup and a sufficiently long decay; a poor schedule can be mistaken for a broken scaling law.

The prescription should therefore be treated as a map of this setup. [Chinchilla](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) later asks the same fixed-FLOP question with a larger sweep and obtains a near-balanced growth of parameters and tokens. The disagreement is not a contradiction about whether scale helps; it is evidence that the allocation law is more sensitive to data, schedule, and fit design than the headline exponents suggest.

## High-Level Takeaways

- Language-model loss follows clean power laws in parameters, data, and compute when the other resources are not binding.
- Kaplan's compute frontier favors parameters over data and stops short of convergence; it is not a general token-budget rule.
- The useful output is an extrapolation inside a matched training family, not a universal exponent detached from tokenizer and schedule.
- Overfitting couples model size to data even when the compute frontier appears parameter-heavy.
- Read [the Chinchilla note](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) as the later re-estimation of the allocation problem, not as a rejection of the underlying smooth scaling behavior.
