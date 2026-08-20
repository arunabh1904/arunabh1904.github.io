---
title: 'How to Read Scaling Laws for Language Models'
date: '2026-08-19T12:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: how-to-read-scaling-laws-for-language-models
legacyPath: /blog/2026/08/19/how-to-read-scaling-laws-for-language-models.html
tags:
  - Language Models
  - Scaling Laws
  - Pretraining
  - Inference
summary: Kaplan, Chinchilla, data constraints, inference economics, and why every scaling law is a local map of one experimental regime.
---

# How to Read Scaling Laws for Language Models

Kaplan and Chinchilla appear in almost every discussion of a model release: sometimes as an argument for a bigger model, sometimes as a fixed token-to-parameter ratio, and sometimes as a vague claim that progress is inevitable. None of those versions survives a close reading.

A scaling law does not say that a model becomes intelligent if we make it larger. It makes a narrower and more useful claim: within a measured training regime, a chosen metric changes predictably as we spend more parameters, data, or compute. That can turn a frontier run from a heroic bet into a resource-allocation problem. A good small sweep tells us whether the next unit of budget should buy a larger model, more tokens, a different data mixture, a better architecture, or more inference-time search.

The phrase *within a measured regime* does most of the work. A fitted curve inherits its architecture, tokenizer, data construction, optimizer, schedule, context length, compute accounting, and evaluation distribution. Change enough of those and we are not moving along the same curve anymore.

I separate four claims that are too often all called a “scaling law.”

1. A **descriptive** law says measured runs follow a smooth relationship.
2. A **predictive** law says that relationship forecasts a held-out, more expensive run.
3. A **prescriptive** law optimizes the fitted relationship under a resource constraint.
4. A **system-optimal** law remains useful after serving demand, latency, memory, post-training, and test-time compute enter the objective.

[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) established striking descriptive regularity. [Chinchilla](https://arxiv.org/abs/2203.15556) turned a fitted loss surface into a pretraining allocation. Neither result, by itself, solves the system-optimal problem. That distinction is the spine of this post.

## What a scaling law measures

For an autoregressive language model, each next token receives a probability. On a held-out token sequence $x_1, \ldots, x_T$, average negative log-likelihood is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is cross-entropy, measured in *nats* when $\log$ is natural log. A model that assigns the observed next token more probability has lower loss. It is dense, stable, and inexpensive to fit. It is not a certificate that the resulting system is better at everything.

Three quantities are routinely collapsed into one. **Entropy $H(p)$** is uncertainty in a specified source distribution. **Cross-entropy $H(p,q_\theta)$** is the expected negative log-probability assigned by model $q_\theta$. **Perplexity** is $\exp(H(p,q_\theta))$ when loss is in nats. Their relationship is exact:

$$
H(p,q_\theta) = H(p) + D_{\mathrm{KL}}(p\,\Vert\,q_\theta).
$$

The entropy term is not a universal property of “English,” and a validation loss is not an estimate of it. Tokenizer, corpus, and conditioning define the unit. A perplexity comparison across tokenizers is therefore not automatically meaningful; bits per byte can sometimes be a better normalized quantity.

Perplexity is a presentation transform, not another training objective. Whether a fit uses raw loss, log loss, or reducible loss is a modeling choice that should be selected by held-out prediction. The rule is simple: name the target, its units, and the test used to validate the fit.

## From a curve to an allocation

Kaplan measured regular loss trends with non-embedding parameters $N$, training tokens $D$, and compute $C$. The original Figure 1 is useful precisely because it is three curves, not one model-size curve.

![Kaplan et al. 2020 Figure 1: validation loss as a function of compute, dataset size, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)

*Original Figure 1 from [Kaplan et al., “Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361), rendered without alteration from the authors’ arXiv source.*

A useful local model is

$$
L(N,D)=E + A N^{-\alpha}+B D^{-\beta},
\qquad C=\kappa ND.
$$

This is a schematic, not a claim that every experiment has this exact form. $E$ is an asymptotic floor for the stated setup. The remaining terms say finite capacity and finite data can each limit loss. Dense-transformer training cost couples them, approximately, through $ND$.

Under that constraint, minimizing the surface gives

$$
N_{\mathrm{opt}}(C)=G\left(\frac{C}{\kappa}\right)^{\frac{\beta}{\alpha+\beta}},
\qquad
D_{\mathrm{opt}}(C)=G^{-1}\left(\frac{C}{\kappa}\right)^{\frac{\alpha}{\alpha+\beta}},
$$

where $G=\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}$. At the optimum, $\alpha A N^{-\alpha}=\beta B D^{-\beta}$.

The exponents determine how allocation moves with compute. The coefficients determine where the frontier sits in absolute terms. This is why an exponent alone is not a recipe. A prescriptive claim also needs a measured response surface, preferably an IsoFLOP valley with points on both sides of the minimum.

## Chinchilla changed the frontier, not the question

Chinchilla revisited allocation at fixed training compute. Across more than 400 models from 70M to over 16B parameters and 5B to 500B tokens, Hoffmann et al. found that optimal model size and training tokens scale approximately equally with compute.

![Chinchilla paper Figure 1: compute-optimal parameter counts against training FLOPs, including the Kaplan prediction and named language models](/assets/images/chinchilla-compute-frontier-paper-figure.png)

*Original Figure 1 from [Hoffmann et al., “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556), rendered without alteration from the authors’ arXiv source. The solid curves are Chinchilla fits; the dashed line is the Kaplan prediction.*

Chinchilla used the same training-compute budget as Gopher, with 70B rather than 280B parameters and four times as much training data. It outperformed Gopher and several contemporaries on the paper's downstream suite. The lesson is not that 70B is a magic size. A model can be undertrained *relative to a validation-loss objective at fixed pretraining compute*.

The familiar shorthand of roughly twenty tokens per parameter needs the same qualification. Near-equal fitted exponents make $D/N$ approximately constant in that regime. Its numerical value is a coefficient-level property of those experiments, not a universal exponent or a law of nature.

> **Deep insight.** Chinchilla's approximate token-to-parameter ratio is not an independent law. It follows from a particular fitted loss surface: near-equal data and parameter exponents make the ratio nearly constant, while the coefficients set its value. Change the training regime and the ratio is a hypothesis to remeasure, not a recipe to repeat.

The Kaplan-versus-Chinchilla story is not simply “the old recipe was wrong.” [Porian et al. (2024)](https://arxiv.org/abs/2406.19146) reproduced the Kaplan-style result and traced much of the discrepancy to last-layer compute accounting, warmup duration, and scale-dependent optimizer tuning. A scaling law predicts the optimization frontier represented by its experiments. If the proxy runs are not comparably tuned, it can extrapolate the wrong frontier very cleanly.

That gives a practical standard: hold out expensive scale, report extrapolation distance, repeat seeds at anchor points, propagate fitting uncertainty into the allocation, and compare plausible functional forms. [Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891) is a useful reminder that curvature and regime changes are empirical possibilities.

## Tokens are not effective data

The symbol $D$ hides the modern choice that matters most: what those token positions contain. Two streams with the same length can differ in duplication, novelty, quality, contamination, target-domain coverage, and mixture weights. A more honest conceptual object is

$$
D_{\mathrm{eff}} = f(\text{unique data},\text{mixture},\text{quality},\text{repetition},N).
$$

This is not a universally measured unit. It is a warning that consumed tokens are a budget proxy, not useful predictive information itself.

[Muennighoff et al.](https://arxiv.org/abs/2305.16264) found little loss degradation from up to four passes over data at fixed compute, followed by diminishing returns as repetition increased. [DoReMi](https://arxiv.org/abs/2305.10429) and [Data Mixing Laws](https://arxiv.org/abs/2403.16952) make the same point from another direction: corpus mixture is an optimization variable, not appendix detail.

At the boundaries, the additive Chinchilla-style surface is a testable approximation. Recent coupled model-data work, including [Skaling](https://arxiv.org/abs/2608.07222), reports systematic error from independent data and parameter terms in data-scarce or heavily overtrained settings. These are emerging results, not a settled replacement. They are enough to justify checking residuals rather than assuming separability.

## The objective now spans a lifecycle

Chinchilla optimizes pretraining loss under a pretraining-compute budget. A deployed model lives under a larger objective:

$$
C_{\mathrm{total}} = C_{\mathrm{pre}} + C_{\mathrm{post}} + R\,\mathbb{E}_{x}[C_{\mathrm{infer}}(x)],
$$

where $R$ is inference demand. Latency, memory, bandwidth, and reliability are often hard constraints.

[Sardana et al. (2024)](https://arxiv.org/abs/2401.00448) show the consequence: with substantial serving demand, a smaller model trained for much longer can be preferable because additional training buys a permanently smaller inference footprint. “Overtrained” is incomplete without an objective.

Test-time compute adds another axis rather than one scalar knob. Sequential reasoning, sampling complete candidates, voting, verification, search over partial trajectories, and prompt-adaptive allocation have different failure modes. [Snell et al. (2024)](https://arxiv.org/abs/2408.03314) found that adaptive allocation by problem difficulty could substantially outperform fixed best-of-$N$, and that test-time compute could beat a much larger FLOP-matched model when the smaller model already had nontrivial success probability.

The useful mental model is a compute vector $(C_{\mathrm{pre}}, C_{\mathrm{post}}, C_{\mathrm{test}})$ plus a policy for assigning test-time compute across requests. Post-training, verifier quality, and environment interactions add more coordinates. I would model their interactions separately before compressing them into one exponent.

## Smooth loss can coexist with jagged capability

Average validation loss can improve smoothly while a benchmark appears flat and then jumps. Some jumps are metric artifacts: [Schaeffer et al. (2023)](https://arxiv.org/abs/2304.15004) showed how nonlinear and discontinuous metrics can create sharp-looking transitions from smoothly changing model outputs. Probability can move steadily toward the right answer while exact-match accuracy remains zero until the top-ranked answer flips.

But loss does not determine every capability. Under a controlled corpus, tokenizer, and architecture, [Du et al. (2024)](https://arxiv.org/abs/2403.15796) found models with similar pretraining loss could show similar downstream performance, while reporting task-specific thresholds. Conversely, [Lourie et al. (2025)](https://arxiv.org/abs/2507.00885) found reliably predictable downstream scaling in only 39% of examined cases.

These results are compatible. Aggregate loss averages away rare features, formatting constraints, long-horizon reliability, and evaluation protocol. A task can amplify a small per-step change: if success requires $m$ reliable steps, a per-step success probability $p$ becomes roughly $p^m$.

Use loss to choose an efficient pretraining candidate. Use a capability vector to accept a system. When they diverge, look for the missing distribution, metric, inference policy, or post-training stage instead of forcing one scalar to explain the product.

## What I would run before a frontier model

A useful scaling study is an experimental design, not a regression script.

1. **State the decision first.** Model size, token count, data mixture, serving footprint, post-training budget, and test-time strategy require different targets.
2. **Fix the transfer regime.** Hold architecture, tokenizer, context construction, data pipeline, optimizer, schedule semantics, and evaluation fixed unless one is the deliberate axis.
3. **Sweep across the valley.** For several compute budgets, run enough allocations to place points on both sides of the loss minimum.
4. **Tune the proxy frontier.** A family of under-tuned runs can yield a beautiful law for under-tuned models.
5. **Measure uncertainty and hold out scale.** Report intervals for $N_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$, and validate a genuinely more expensive run.
6. **Validate the acceptance metric.** Re-evaluate target capability, safety properties, latency, and inference protocol at the proposed recipe.

The best scaling study ends by changing the next experiment.

## What I think survives

- **Scaling laws are local.** Their usefulness comes from controlled regularity, not universality.
- **A curve is not yet a prescription.** Allocation requires a constraint and a measured frontier.
- **Exponents are not the whole recipe.** Coefficients set absolute allocation, and interactions can matter at the edges.
- **Tokens are not information.** Novelty, quality, mixture, and repetition belong in the data model.
- **Parameters are not a system.** Active compute, architecture, memory, and latency change the deployment optimum.
- **Compute is a vector.** Pretraining, post-training, serving, sampling, search, and verification compete for lifecycle budget.
- **Smooth loss does not imply smooth product behavior.** The metric, task, and inference protocol determine what the user sees.

The law is not the strategy. It is a local map of where marginal return currently lives.

The strategy is deciding which map describes the system we are actually building, and knowing when to redraw it.

## A reading map

- [Hestness et al. (2017), “Deep Learning Scaling is Predictable, Empirically”](https://arxiv.org/abs/1712.00409)
- [Kaplan et al. (2020), “Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361)
- [Hernandez et al. (2021), “Scaling Laws for Transfer”](https://arxiv.org/abs/2102.01293)
- [Hoffmann et al. (2022), “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556)
- [Caballero et al. (2023), “Broken Neural Scaling Laws”](https://arxiv.org/abs/2210.14891)
- [Besiroglu et al. (2024), “Chinchilla Scaling: A Replication Attempt”](https://arxiv.org/abs/2404.10102)
- [Porian et al. (2024), “Resolving Discrepancies in Compute-Optimal Scaling of Language Models”](https://arxiv.org/abs/2406.19146)
- [Muennighoff et al. (2025), “Scaling Data-Constrained Language Models”](https://arxiv.org/abs/2305.16264)
- [Sardana et al. (2024), “Beyond Chinchilla-Optimal”](https://arxiv.org/abs/2401.00448)
- [Snell et al. (2024), “Scaling LLM Test-Time Compute Optimally”](https://arxiv.org/abs/2408.03314)
- [Du et al. (2024), “Understanding Emergent Abilities of Language Models from the Loss Perspective”](https://arxiv.org/abs/2403.15796)
- [Lourie et al. (2025), “Scaling Laws Are Unreliable for Downstream Tasks”](https://arxiv.org/abs/2507.00885)
