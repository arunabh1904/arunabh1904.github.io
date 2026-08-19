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
summary: What a language-model scaling curve measures, why loss becomes perplexity, and how Kaplan and Chinchilla turn a fixed compute budget into a training recipe.
---

# How to Read Scaling Laws for Language Models

A scaling law does not say that a model will become intelligent if we make it larger. It says something more useful and more limited: inside a measured training regime, a chosen error metric changes predictably as we spend more parameters, data, or compute. That turns a large training run from a single bet into a resource-allocation problem. If we can fit the curve cheaply, we can ask whether the next dollar should buy a larger model, more tokens, a different data mixture, or a better evaluation.

The phrase has been diluted into "bigger models work." The original language-model result was sharper. [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) measured validation cross-entropy over enormous ranges of model size, dataset size, and training compute, and found straight-ish lines on log-log axes while the other resources were not the bottleneck. [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556), the Chinchilla paper, then asked the operational question that follows: under a fixed FLOP budget, which combination of parameters and tokens actually minimizes that loss?

This post builds the metric first, then the two papers, then the practical reading of a curve. The final section separates current commentary from the paper evidence; the closing insights are my synthesis, not claims reported by either paper.

## What a scaling law measures

For an autoregressive language model, each next token has a probability assigned by the model. On a held-out token sequence $x_1, \ldots, x_T$, the average negative log-probability is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is the cross-entropy measured in *nats* when $\log$ is natural log. A model that consistently gives the observed next token more probability has lower loss. It is a proper scoring rule: confidently assigning probability to the wrong token hurts more than admitting uncertainty. That makes validation loss a much cleaner object for small-scale extrapolation than a hand-picked benchmark score, even though it is not a complete description of capability.

Three quantities are often collapsed into one. Keeping them separate prevents most confusion around the word *perplexity*:

| Quantity | What it is | What lower means |
| --- | --- | --- |
| Entropy $H(p)$ | The irreducible uncertainty of the true token distribution $p$ under a particular tokenizer and data distribution | The source itself is more predictable; it is not a model score |
| Cross-entropy $H(p, q_\theta)$ | The expected negative log-probability assigned by model $q_\theta$ to data from $p$ | The model predicts the held-out distribution better |
| Perplexity $\operatorname{PPL}$ | $\exp(H(p,q_\theta))$ when cross-entropy is in nats | A multiplicative reduction in the model's effective branching factor |

The relationship is exact:

$$
H(p,q_\theta) = H(p) + D_{\mathrm{KL}}(p\,\Vert\,q_\theta).
$$

Entropy is the floor for that tokenization and distribution; the KL divergence is the model's excess loss above it. We almost never know the true $H(p)$ for natural language, so a measured cross-entropy is not "how much entropy the model has." It is a score for one model on one held-out sample. It changes if the tokenizer, document mixture, de-duplication policy, or evaluation corpus changes.

Perplexity is simply the exponentiated cross-entropy. It is often intuitive because $\operatorname{PPL}=20$ reads as an effective choice among roughly twenty equally likely continuations, but it can conceal additive changes in the quantity being optimized. A decrease from $2.0$ to $1.9$ nats has the same loss difference as $3.0$ to $2.9$ nats; after exponentiation, the first reduces perplexity by about $9.5\%$ and the second by about $9.5\%$. Compare and fit loss in log space. Report perplexity when its multiplicative interpretation helps a reader.

The metric establishes the object. A scaling law is the fitted relationship between that object and a resource, not a property of language in the abstract.

## Kaplan et al.: three curves, not one parameter curve

Kaplan et al. varied non-embedding parameters $N$, training-data tokens $D$, and compute $C$. The original Figure 1 below is worth reading slowly. Each right-hand panel holds the other constraints loose enough that the plotted resource is the dominant limitation. The approximately straight trend on log-log axes is the empirical signature of a power law, not a claim that loss falls linearly with the resource.

![Kaplan et al. 2020 Figure 1: validation loss as a function of compute, dataset size, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)

*Original Figure 1 from [Kaplan et al., “Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361), rendered without alteration from the authors’ arXiv source. The three panels show the measured loss curves against training compute, dataset size, and non-embedding parameters.*

The compact mental model is a loss surface with three kinds of limitation:

$$
L(N,D) \approx L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

This is a useful schematic, not the exact Kaplan fitting equation. $L_\infty$ represents an irreducible floor for the stated data distribution and setup. The parameter term says capacity can be the limiting factor; the data term says coverage and repetition can be the limiting factor. A training-compute constraint couples the two because dense-transformer training cost is approximately proportional to $ND$ up to architecture and systems constants. If either $N$ or $D$ stays fixed, the other eventually delivers diminishing returns because its term is no longer dominant.

Kaplan's reported compute-optimal prescription had a surprising consequence: for a fixed compute budget, it favored a very large model trained for relatively few tokens, well short of convergence. In their fitted regime, the optimal parameter count grew much faster with compute than the token count. That conclusion was conditional on the curve, the WebText2-based setup, the optimizer, and the assumptions used to turn updates into compute. It was not a permanent rule that data is less valuable than parameters.

That distinction is why Figure 1 should change an experiment plan, not end one. A straight line tells us that an interpolation is plausible over the measured range. It does not tell us that the same exponent survives a new tokenizer, a long-context architecture, cleaner data, a multimodal objective, or a scale jump far outside the sweep.

## Chinchilla: the compute-optimal recipe changed

Chinchilla revisited the allocation rather than disputing the premise. Hoffmann et al. trained more than 400 language models from 70M to over 16B parameters and varied training length over 5B to 500B tokens. Their three fitting approaches all placed the compute-optimal solution close to equal exponents: approximately $N_{\mathrm{opt}} \propto C^{0.5}$ and $D_{\mathrm{opt}} \propto C^{0.5}$. In plain language, each additional scale step should increase both parameters and training tokens, rather than primarily parameters.

![Chinchilla paper Figure 1: compute-optimal parameter counts against training FLOPs, including the Kaplan prediction and named language models](/assets/images/chinchilla-compute-frontier-paper-figure.png)

*Original Figure 1 from [Hoffmann et al., “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556), rendered without alteration from the authors’ arXiv source. The solid curves are three Chinchilla fitting approaches; the dashed line is the Kaplan prediction. The plot is a recipe comparison at fixed training FLOPs, not a benchmark leaderboard.*

The figure makes the revision concrete. At a given compute budget, the Chinchilla curves select fewer parameters than the Kaplan projection and reserve the released FLOPs for a longer training run. The flagship comparison held training compute roughly fixed: Chinchilla used 70B parameters and four times the training data of 280B-parameter Gopher. The paper reports that Chinchilla outperformed Gopher and several contemporaries across its downstream evaluation suite. The scientific point is not that 70B is a magic size; it is that *undertraining is an allocation error*. A smaller deployed model can also reduce inference cost, so a compute-optimal pretraining recipe and a serving-optimal choice may happen to point in the same direction.

| Fixed question | Kaplan's fitted answer | Chinchilla's fitted answer | What changed operationally |
| --- | --- | --- | --- |
| How should $N$ grow with training compute? | Roughly $C^{0.73}$ | Roughly $C^{0.5}$ | Spend less marginal compute on parameters |
| How should $D$ grow with training compute? | Roughly $C^{0.27}$ | Roughly $C^{0.5}$ | Train on substantially more tokens |
| What does the frontier optimize? | Validation loss at fixed compute | Validation loss at fixed compute | The objective stayed; the estimated frontier moved |

Neither table entry is a universal token-to-parameter ratio. The popular Chinchilla shorthand of about twenty training tokens per parameter describes a point on one dense-language-model frontier, not a law of nature. For a different data distribution, tokenizer, architecture, context length, recurrent reuse, mixture, or objective, refit the proxy curve before spending the full budget.

## How to read a scaling plot without being fooled

An empirical scaling result earns a decision only when the fitted quantity, controlled variables, and extrapolation are visible. I use the following questions before accepting an exponent as a training recipe.

| Question | Why it changes the decision |
| --- | --- |
| What is on the vertical axis? | Cross-entropy, perplexity, benchmark accuracy, reward, and human preference do not have interchangeable slopes. |
| What was held fixed? | A model-size curve may silently change batch size, data quality, sequence length, optimizer, or architecture. |
| Is the x-axis compute actually comparable? | Training FLOPs omit data preparation, communication, failed runs, and sometimes inference; deployment economics may reverse a pretraining-only recommendation. |
| Does the fit include a visible optimum? | A compute-allocation claim needs a valley or frontier, not merely two monotone curves. |
| How far is the extrapolation? | A good interpolation over one or two orders of magnitude can still be a poor bet five orders further out. |
| Which capability is missing from the loss? | Held-out NLL can improve while safety, tool use, reasoning reliability, latency, or a rare domain remains the product bottleneck. |

The last question matters most today. Loss is a scalar average over tokens. A product is not. A data mixture can lower average NLL by improving common web patterns while leaving a codebase, a multilingual low-resource language, long-horizon agent behavior, or calibration unchanged. A capability evaluation is noisier and easier to game than NLL, but it tells us whether the capability vector that matters moved.

That is also where entropy becomes practical rather than philosophical. Adding a trillion near-duplicate tokens may increase the counter $D$ while contributing little new conditional information. Adding a smaller source with rare syntax, expert reasoning traces, or an underrepresented language can be much more valuable per token, but only if it improves the target distribution rather than merely shifting it. Token count is a budget proxy. The real resource is useful predictive signal under the objective we care about.

## What the recent commentary adds

The supplied threads are useful as prompts for the *decision layer* above the equations. [Jie Tang’s note](https://x.com/jietang/status/2089941544581403107) pushes against treating parameter count as a standalone description of a model. [Ishaan’s long-form thread](https://x.com/auto_grad_/status/2089970913408380932) frames the broader capability-extraction question. [Zixuan Li’s post](https://x.com/ZixuanLi_/status/2089950717347774919) reduces the allocation problem to its sharpest operational form: spend the next unit of compute where the system currently loses the most capability.

I agree with that framing, with one guardrail. We cannot identify the bottleneck from a headline model size or a single validation-loss curve. Build a small response surface first: sweep a few model sizes, token budgets, and data mixtures; keep the architecture and evaluation fixed; fit uncertainty, not only a line; then check whether the candidate recipe improves the capability that motivated the run. If an engineering constraint such as latency, data access, or evaluation reliability is the active bottleneck, the next unit of compute may have lower return than work on that constraint.

## Deep insights

**A scaling law is an instrument, not a destiny.** Its durable contribution is experimental compression: a carefully designed small sweep can tell us which expensive run is worth attempting. The moment the training distribution, architecture, loss, or deployment constraint changes enough to alter the bottleneck, the old exponent becomes a hypothesis to test rather than a rule to apply.

**Loss and capability should be modeled as different ledgers.** Cross-entropy is the best common currency for pretraining progress because it is dense, stable, and cheap to measure. Capability is the reason to train because a real system fails in specific places. Use loss to locate broad learning efficiency; use a capability vector to decide whether the frontier is commercially or scientifically relevant. Conflating those ledgers produces models that look better on a curve while remaining weak where it matters.

**Data scale has two axes: count and conditional novelty.** Chinchilla corrected an allocation that gave too little training exposure for its setting. It did not make raw token count sufficient. At frontier scale, the decisive question increasingly becomes which new conditional structure the next token adds after de-duplication, contamination control, and mixture weighting. That is why data curation, tokenization, and evaluation design can move a scaling frontier as much as another parameter multiplier.

**The most valuable scaling result is a reversible decision.** A clean proxy study says, before the full run, what allocation it recommends, how uncertain that recommendation is, and what later measurement would prove it wrong. That makes a scaling law useful even when its extrapolated exponent fails. The failure identifies a regime change; the experiment has then found the next bottleneck instead of merely missing a forecast.
