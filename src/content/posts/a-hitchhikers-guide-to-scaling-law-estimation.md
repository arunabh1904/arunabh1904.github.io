---
title: "A Hitchhiker's Guide to Scaling Law Estimation"
date: '2024-10-15T00:00:00.000Z'
section: paper-shorts
postSlug: a-hitchhikers-guide-to-scaling-law-estimation
legacyPath: /paper shorts/2024/10/15/a-hitchhikers-guide-to-scaling-law-estimation.html
tags: [Language Models, Scaling Laws, Experiment Design]
field: 'Training Systems & Reliability'
summary: '2024 – A Hitchhiker’s Guide to Scaling Law Estimation'
---

## 2024 – A Hitchhiker’s Guide to Scaling Law Estimation

**arXiv:** [2410.11840](https://arxiv.org/abs/2410.11840) · **Conference:** ICML 2025

## Summary

> Choshen et al. treat a scaling law as an estimator that deserves its own experimental design. Their dataset covers 485 pretrained models, more than 40 families, 1.9M training steps, and over 1,000 fitted laws. The practical result is not a new exponent: intermediate checkpoints, several diverse models, and a target-near size usually beat a single polished endpoint, while seed noise can make a larger run actively misleading.

## Core Insights

### Reliability has a floor

The paper fits the Chinchilla-style law

$$
\widehat L(f)=e^E+\frac{e^A}{\#\mathrm{params}(f)^\alpha}+\frac{e^B}{\#\mathrm{toks}(f)^\beta}
$$

to a training family and evaluates the mean absolute relative error (ARE) on held-out checkpoints. Across the literature, seed variation can reach about 4% in loss; the authors therefore treat 4% as roughly the smallest effect a scaling law needs to resolve reliably. A fit with a lower numerical residual is not automatically useful if the experimental change itself is smaller than run-to-run noise.

![Source Figure 4: error when fitting scaling laws to more of the training trajectory](/assets/images/hitchhikers-guide-scaling-law-estimation-source-figure-4.png)
*Fig 1: Error is often lower when the fit uses checkpoints from most of the trajectory, after discarding the uninformative beginning. This is source Figure 4. | source: [A Hitchhiker's Guide to Scaling Law Estimation, Figure 4](https://arxiv.org/abs/2410.11840)*

### Intermediate checkpoints are data, not contamination

Most prior scaling studies keep only the final loss from each run because the learning-rate schedule depends on the intended training horizon. Here, fitting intermediate checkpoints generally predicts final loss better than fitting final endpoints alone. The first roughly 10% of training is often unstable or even rises, so removing those checkpoints helps; after that, a model can often be predicted from about one-third of its trajectory.

This is a useful change in budget arithmetic. One run supplies many observations along the token axis, provided the schedule is interpretable. It does not make the observations independent, and it does not erase schedule mismatch. The paper's recommendation is empirical: validate the checkpoint policy for the family rather than assuming every early point is a clean scaling datum.

### More models can beat bigger models

Models near the target size usually extrapolate better, but the effect is not monotonic. In Pythia, a single unusually noisy model can worsen a fit even after larger models are added. Across GPT-3, Gopher, OPT, and Pythia, adding several diverse sizes generally reduces ARE; five models are a reasonable safe minimum, not a theorem. The authors suggest varying sizes and conditions rather than spending the entire budget on multiple seeds of one point.

The meta-analysis also tests transfer across families. Fixing the parameter-scaling terms from another family and fitting the remaining parameters can predict a new family surprisingly well—for example, under 1% error for a 7B OLMO target from 1B checkpoints—but errors grow when extrapolating far beyond the source range, as in OPT to 175B. The fit can borrow a shape; it cannot borrow an architecture's intercept for free.

The principal-component analysis finds that three components explain 99.49% of the variation in five fitted parameters, with tight relationships between each coefficient and exponent pair. This may enable simpler laws, but the result is observational and fails for some encoder-decoder and repeated-data families. Treat the lower-dimensional structure as a modeling clue, not a replacement for held-out validation.

## High-Level Takeaways

- A scaling law has an empirical error floor set by seed variance and the size of the modeling change being tested.
- Use intermediate checkpoints after the unstable beginning when the training schedule makes them meaningful.
- Five diverse models are a reasonable starting point; one large or noisy model is not sufficient evidence.
- Target-near models help, but model count and seed diversity can be a better use of a fixed budget.
- Cross-family parameter borrowing can work locally, yet distant extrapolation remains a separate empirical claim.
