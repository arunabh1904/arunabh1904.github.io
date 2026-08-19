---
title: 'Some Thoughts on Scaling Laws for Language Models'
date: '2026-08-19T12:00:00.000Z'
section: blog
postSlug: how-to-read-scaling-laws-for-language-models
legacyPath: /blog/2026/08/19/how-to-read-scaling-laws-for-language-models.html
tags:
  - Language Models
  - Scaling Laws
  - Pretraining
summary: A close reading of cross-entropy, perplexity, Kaplan, and Chinchilla—and why a scaling curve should change an experiment plan, not end the discussion.
---

# Some Thoughts on Scaling Laws for Language Models

This post is long overdue. I have had scaling laws in my backlog for a long time—not because Kaplan and Chinchilla are obscure, but because I kept postponing the point where I had to form an opinion about what they mean now. They appear in almost every discussion of a model release: sometimes as an argument for a bigger model, sometimes as a fixed token-to-parameter ratio, and sometimes as a vague claim that progress is inevitable. None of those versions survives a close reading.

A scaling law does not say that a model becomes intelligent if we make it larger. It makes a more useful, narrower claim: inside a measured training regime, a chosen error metric changes predictably as we spend more parameters, data, or compute. That turns a large pretraining run from one heroic bet into a resource-allocation problem. If a cheap sweep can fit the curve, it can tell us whether the next dollar should buy a larger model, more tokens, a different mixture, or a better evaluation.

The original language-model result was much sharper than "bigger models work." [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) measured validation cross-entropy across model size, dataset size, and training compute, then showed the smooth trend that appears when the other resources are not yet binding. [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556), the Chinchilla paper, asked the question I think people should ask whenever they invoke a scaling law: with a fixed FLOP budget, what combination of parameters and tokens actually minimizes the loss? I want to build that answer from the metric upward, then return to what the curve can and cannot decide for us.

## What a scaling law measures

For an autoregressive language model, each next token has a probability assigned by the model. On a held-out token sequence $x_1, \ldots, x_T$, the average negative log-probability is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is cross-entropy, measured in *nats* when $\log$ is natural log. A model that consistently gives the observed next token more probability has lower loss. It is a proper scoring rule: confidently assigning probability to the wrong token hurts more than admitting uncertainty. I like validation loss as the first object for a scaling study because it is dense, stable, and cheap; I do not like the move from "loss improved" to "the system is now better at everything." Those are different claims.

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

That distinction establishes the object of study. A scaling law is a fitted relationship between a metric and a resource. It is not a property of language in the abstract, and it is definitely not a license to forget what changed in the data or the evaluation.

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

This is where I think the paper has often been over-read. Figure 1 should change an experiment plan, not end one. A straight line says that an interpolation is plausible over the measured range. It does not say that the same exponent survives a new tokenizer, a long-context architecture, cleaner data, a multimodal objective, or a scale jump far outside the sweep.

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

This is the corrective I wish accompanied every invocation of Chinchilla. The popular shorthand of about twenty training tokens per parameter describes a point on one dense-language-model frontier; it is not a law of nature. Change the data distribution, tokenizer, architecture, context length, recurrent reuse, mixture, or objective, and the proxy curve deserves to be refit before anyone spends the full budget.

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

The last question matters most to me now. Loss is a scalar average over tokens. A product is not. A data mixture can lower average NLL by improving common web patterns while leaving a codebase, a multilingual low-resource language, long-horizon agent behavior, or calibration unchanged. A capability evaluation is noisier and easier to game than NLL, but it is also where we find out whether the capability vector that matters actually moved.

That is also where entropy becomes practical rather than philosophical. Adding a trillion near-duplicate tokens may increase the counter $D$ while contributing little new conditional information. A smaller source with rare syntax, expert reasoning traces, or an underrepresented language can be far more valuable per token, but only if it improves the target distribution rather than merely shifting it. Token count is a budget proxy. The real resource is useful predictive signal under the objective we actually care about.

## What the recent commentary adds

The supplied threads are useful as prompts for the *decision layer* above the equations. [Jie Tang’s note](https://x.com/jietang/status/2089941544581403107) pushes against treating parameter count as a standalone description of a model. [Ishaan’s long-form thread](https://x.com/auto_grad_/status/2089970913408380932) frames the broader capability-extraction question. [Zixuan Li’s post](https://x.com/ZixuanLi_/status/2089950717347774919) reduces the allocation problem to its sharpest operational form: spend the next unit of compute where the system currently loses the most capability.

I agree with that framing, with one guardrail. We cannot identify the bottleneck from a headline parameter count or a single validation-loss curve. I would first build a small response surface: sweep a few model sizes, token budgets, and data mixtures; keep the architecture and evaluation fixed; fit uncertainty, not only a line; then check whether the candidate recipe improves the capability that motivated the run. If latency, data access, or evaluation reliability is the active bottleneck, the next unit of compute may have lower return than work on that constraint.

## Deep insights

**A scaling law is an instrument, not a destiny.** Its durable contribution is experimental compression: a carefully designed small sweep can tell us which expensive run is worth attempting. The moment the training distribution, architecture, loss, or deployment constraint changes enough to alter the bottleneck, the old exponent becomes a hypothesis to test rather than a rule to apply.

**Loss and capability should be modeled as different ledgers.** Cross-entropy is the best common currency for pretraining progress because it is dense, stable, and cheap to measure. Capability is the reason to train because a real system fails in specific places. Use loss to locate broad learning efficiency; use a capability vector to decide whether the frontier is commercially or scientifically relevant. Conflating those ledgers produces models that look better on a curve while remaining weak where it matters.

**Data scale has two axes: count and conditional novelty.** Chinchilla corrected an allocation that gave too little training exposure for its setting. It did not make raw token count sufficient. At frontier scale, the decisive question increasingly becomes which new conditional structure the next token adds after de-duplication, contamination control, and mixture weighting. That is why data curation, tokenization, and evaluation design can move a scaling frontier as much as another parameter multiplier.

**The most valuable scaling result is a reversible decision.** A clean proxy study says, before the full run, what allocation it recommends, how uncertain that recommendation is, and what later measurement would prove it wrong. That makes a scaling law useful even when its extrapolated exponent fails. The failure identifies a regime change; the experiment has then found the next bottleneck instead of merely missing a forecast.
