---
title: 'Some Thoughts on Scaling Laws for Language Models'
date: '2026-08-19T12:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: how-to-read-scaling-laws-for-language-models
legacyPath: /blog/2026/08/19/how-to-read-scaling-laws-for-language-models.html
tags:
  - Language Models
  - Scaling Laws
  - Pretraining
summary: A close reading of cross-entropy, perplexity, Kaplan, and Chinchilla—and why a scaling curve should change an experiment plan, not end the discussion.
---

# Some Thoughts on Scaling Laws for Language Models

This post is long overdue. I have had it in my backlog for a long time, not because Kaplan and Chinchilla are obscure, but because I wanted to understand what their results actually mean now. They appear in almost every discussion of a model release: sometimes as an argument for a bigger model, sometimes as a fixed token-to-parameter ratio, and sometimes as a vague claim that progress is inevitable. None of those versions survives a close reading.

A scaling law does not say that a model becomes intelligent if we make it larger. It makes a more useful, narrower claim: inside a measured training regime, a chosen error metric changes predictably as we spend more parameters, data, or compute. That turns a large pretraining run from one heroic bet into a resource-allocation problem. If a cheap sweep can fit the curve, it can tell us whether the next dollar should buy a larger model, more tokens, a different mixture, or a better evaluation.

The original language-model result was much sharper than "bigger models work." [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) measured validation cross-entropy across model size, dataset size, and training compute, then showed the smooth trend that appears when the other resources are not yet binding. [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556), the Chinchilla paper, asked the question that matters whenever someone cites a scaling law: with a fixed FLOP budget, what combination of parameters and tokens actually minimizes loss? The answer starts with the metric, not the model-size headline.

## What a scaling law measures

For an autoregressive language model, each next token has a probability assigned by the model. On a held-out token sequence $x_1, \ldots, x_T$, the average negative log-probability is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is cross-entropy, measured in *nats* when $\log$ is natural log. A model that consistently gives the observed next token more probability has lower loss. It is a proper scoring rule: confidently assigning probability to the wrong token hurts more than admitting uncertainty. I like validation loss as the first object for a scaling study because it is dense, stable, and cheap; I do not like the move from "loss improved" to "the system is now better at everything." Those are different claims.

Three quantities are routinely collapsed into one. They should not be.

**Entropy $H(p)$** is the irreducible uncertainty of the true token distribution $p$, under one tokenizer and one data distribution. It belongs to the source, not to a particular model.

**Cross-entropy $H(p, q_\theta)$** is the expected negative log-probability assigned by model $q_\theta$ to data from $p$. Lower cross-entropy means the model predicts the held-out distribution better.

**Perplexity $\operatorname{PPL}$** is $\exp(H(p,q_\theta))$ when cross-entropy is measured in nats. It turns an additive loss into an effective branching factor, which is useful for intuition but easy to misuse in comparison.

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

The disagreement is exact. Kaplan's fitted frontier scales parameters roughly as $C^{0.73}$ and tokens as $C^{0.27}$; Chinchilla estimates both close to $C^{0.5}$. The objective did not change—both optimize validation loss at fixed training compute. The estimated frontier changed. That is why Chinchilla is a useful warning against memorizing a recipe from an earlier fit, not proof that there is now one correct recipe forever.

This is the corrective I wish accompanied every invocation of Chinchilla. The popular shorthand of about twenty training tokens per parameter describes a point on one dense-language-model frontier; it is not a law of nature. Change the data distribution, tokenizer, architecture, context length, recurrent reuse, mixture, or objective, and the proxy curve deserves to be refit before anyone spends the full budget.

## How I read a scaling plot

I do not take an exponent as a training recipe until I can answer six questions.

1. **What is on the vertical axis?** Cross-entropy, perplexity, benchmark accuracy, reward, and human preference do not have interchangeable slopes.
2. **What changed besides the x-axis?** A model-size curve can quietly change batch size, data quality, sequence length, optimizer, or architecture.
3. **Is compute really comparable?** Training FLOPs omit data preparation, communication, failed runs, and often inference. A pretraining-optimal choice can be a poor deployed system.
4. **Is there a measured optimum?** An allocation claim needs an IsoFLOP valley or an equivalent frontier, not two separate monotone curves.
5. **How far is the extrapolation?** A fit can be excellent across one or two orders of magnitude and still reverse five orders later.
6. **What capability is absent from the loss?** Held-out NLL can improve while safety, tool use, reasoning reliability, latency, or a rare domain remains the actual bottleneck.

The last question matters most to me now. Loss is a scalar average over tokens. A product is not. A data mixture can lower average NLL by improving common web patterns while leaving a codebase, a multilingual low-resource language, long-horizon agent behavior, or calibration unchanged. A capability evaluation is noisier and easier to game than NLL, but it is also where we find out whether the capability vector that matters actually moved.

That is also where entropy becomes practical rather than philosophical. Adding a trillion near-duplicate tokens may increase the counter $D$ while contributing little new conditional information. A smaller source with rare syntax, expert reasoning traces, or an underrepresented language can be far more valuable per token, but only if it improves the target distribution rather than merely shifting it. Token count is a budget proxy. The real resource is useful predictive signal under the objective we actually care about.

## Some thoughts

The supplied threads are useful because they pull the discussion back to the decision layer. [Jie Tang’s note](https://x.com/jietang/status/2089941544581403107) pushes against treating parameter count as a standalone description of a model. [Ishaan’s long-form thread](https://x.com/auto_grad_/status/2089970913408380932) asks what actually governs capability extraction. [Zixuan Li’s post](https://x.com/ZixuanLi_/status/2089950717347774919) compresses the allocation problem into one question: where does the next unit of compute remove the most important current failure?

I agree with that framing, with one guardrail. We cannot identify the bottleneck from a headline parameter count or a single validation-loss curve. I would first build a small response surface: sweep a few model sizes, token budgets, and data mixtures; keep the architecture and evaluation fixed; fit uncertainty, not only a line; then check whether the candidate recipe improves the capability that motivated the run. If latency, data access, or evaluation reliability is the active bottleneck, the next unit of compute may have lower return than work on that constraint.

## Deep insights

**The frontier is a two-dimensional constrained optimization problem.** A curve for $L(N)$ does not prescribe how to spend a fixed budget because $N$ and $D$ are coupled by training compute. The allocation question needs an IsoFLOP sweep: vary model size while holding total FLOPs fixed, train each candidate long enough to expose the loss valley, then fit how the valley moves. Without that valley, an exponent is descriptive rather than prescriptive.

**Perplexity is a presentation transform, not a second metric.** Optimization occurs in cross-entropy or NLL; perplexity is its exponential. Fit and compare the additive loss, especially when curves are close. Entropy supplies the irreducible term in $H(p,q_\theta)=H(p)+D_{\mathrm{KL}}(p\Vert q_\theta)$, but changing the tokenizer or the held-out distribution changes the unit of measurement itself. A perplexity comparison across different tokenizations is not automatically meaningful.

**Token count is not effective data.** $D$ records how many token positions the optimizer saw. It does not record how much new conditional structure those positions supplied after de-duplication, filtering, contamination, overlap, and mixture weighting. Two corpora with the same $D$ can produce different loss surfaces. A modern scaling study should therefore report data construction and mixture weights as part of the fitted regime, not as appendix detail.

**Use loss to choose a pretraining frontier; use a capability vector to accept a system.** NLL is the right dense signal for deciding whether one pretraining recipe is more sample- or compute-efficient. It cannot decide whether the resulting system is acceptable for a target domain, latency budget, safety envelope, or tool-use task. Keep those evaluations separate, and treat a divergence between loss and capability as evidence that the scalar training objective is no longer the bottleneck you care about.
