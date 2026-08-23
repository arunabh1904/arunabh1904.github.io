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

[Jie Tang's post about GLM-5.3](https://x.com/jietang/status/2089941544581403107) finally pulled this out of my backlog. Z.ai kept the GLM-5.2 base fixed, then spent another month scaling long-horizon environments and reinforcement-learning post-training. Tang's point was not that parameter scaling had ended. It was that scaling has several dials, and the dial with the largest return can change.

That post begs for a refresher. If the base model stayed fixed, what exactly scaled? What did Kaplan and Chinchilla actually establish about where the next unit of compute should go? And which parts of their conclusions were empirical observations rather than permanent laws?

Understanding those distinctions matters more now because a modern model budget no longer ends at pretraining. Parameters, data mixture, post-training environments, reinforcement-learning compute, serving cost, and test-time search can all be the binding constraint. Treating *scaling* as shorthand for *make the base model larger* hides the decision we now have to make.

Scaling laws still appear in almost every model-release discussion. They justify larger models, fixed token-to-parameter ratios, and vague claims that progress is inevitable. None of those versions survives a close reading.

A scaling law does not say that a model becomes intelligent if we make it larger. It makes a narrower claim. Inside a measured regime, a chosen error metric changes predictably as we spend more parameters, data, or compute.

I still find that regularity remarkable. Neural-network training is messy, yet the expensive end of a training family can sometimes be forecast from much cheaper runs. In [Dario Amodei's discussion with Lex Fridman](https://lexfridman.com/dario-amodei-transcript/), he describes the confidence behind scaling as inductive: the pattern has repeated across domains even though its theoretical explanation remains incomplete.

That regularity turns a large pretraining run from one heroic bet into a resource-allocation problem. A cheap sweep can tell us whether the next dollar should buy a larger model, more tokens, a different mixture, or a better evaluation.

The original result was much sharper than “bigger models work.” [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) varied model size, dataset size, and training compute. Validation cross-entropy followed a smooth trend when the other resources were not yet binding.

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556), the Chinchilla paper, asked the next question. With a fixed FLOP budget, what combination of parameters and tokens minimizes loss? The answer starts with the metric, not the model-size headline.

## What a scaling law measures

For an autoregressive language model, each next token has a probability assigned by the model. On a held-out token sequence $x_1, \ldots, x_T$, the average negative log-probability is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is cross-entropy, measured in *nats* when $\log$ is the natural logarithm. A model that gives the observed next token more probability has lower loss. It is a proper scoring rule: a confident error hurts more than admitted uncertainty.

I like validation loss as the first object in a scaling study. It is dense, stable, and cheap. I do not like the move from “loss improved” to “the system is now better at everything.” Those are different claims.

Three quantities are routinely collapsed into one. They should not be.

**Entropy $H(p)$** is the irreducible uncertainty of the true token distribution $p$, under one tokenizer and one data distribution. It belongs to the source, not to a particular model.

**Cross-entropy $H(p, q_\theta)$** is the expected negative log-probability assigned by model $q_\theta$ to data from $p$. Lower cross-entropy means the model predicts the held-out distribution better.

**Perplexity $\operatorname{PPL}$** is $\exp(H(p,q_\theta))$ when cross-entropy is measured in nats. It turns an additive loss into an effective branching factor, which is useful for intuition but easy to misuse in comparison.

The relationship is exact:

$$
H(p,q_\theta) = H(p) + D_{\mathrm{KL}}(p\,\Vert\,q_\theta).
$$

Entropy is the floor for that tokenization and distribution; the KL divergence is the model's excess loss above it. We almost never know the true $H(p)$ for natural language, so a measured cross-entropy is not "how much entropy the model has." It is a score for one model on one held-out sample. It changes if the tokenizer, document mixture, de-duplication policy, or evaluation corpus changes.

Perplexity is exponentiated cross-entropy. A value of $20$ can be read as an effective choice among roughly twenty equally likely continuations. The intuition is useful, but exponentiation can hide the additive quantity being optimized.

A drop from $2.0$ to $1.9$ nats has the same loss difference as a drop from $3.0$ to $2.9$ nats. Both reduce perplexity by about $9.5\%$. Compare and fit loss in log space. Report perplexity when the multiplicative view helps.

That distinction establishes the object of study. A scaling law is a fitted relationship between a metric and a resource. It is not a property of language in the abstract, and it is definitely not a license to forget what changed in the data or the evaluation.

## Kaplan et al.: three curves, not one parameter curve

Kaplan et al. varied non-embedding parameters $N$, training-data tokens $D$, and compute $C$. The original Figure 1 below is worth reading slowly.

Each right-hand panel keeps the other constraints loose enough for one resource to dominate. The nearly straight line on log-log axes is the signature of a power law. It does not say that loss falls linearly with the resource.

![Kaplan et al. 2020 Figure 1: validation loss as a function of compute, dataset size, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)

*Original Figure 1 from [Kaplan et al., “Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361), rendered without alteration from the authors’ arXiv source. The three panels show the measured loss curves against training compute, dataset size, and non-embedding parameters.*

The compact mental model is a loss surface with three kinds of limitation:

$$
L(N,D) \approx L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

This is a useful schematic, not Kaplan's exact fitting equation. $L_\infty$ is the irreducible floor for the stated setup. The parameter term captures limited capacity. The data term captures limited coverage and repetition.

Training compute couples the terms because dense-Transformer cost is roughly proportional to $ND$, up to architecture and system constants. Hold either $N$ or $D$ fixed and the other eventually delivers diminishing returns. Its term is no longer the bottleneck.

Kaplan's compute-optimal prescription had a surprising consequence. At fixed compute, it favored a very large model trained on relatively few tokens, well short of convergence. In that fitted regime, optimal parameter count grew much faster than token count.

The result depended on the curve, WebText2, the optimizer, and the accounting that turned updates into compute. It was not a permanent rule that data is less valuable than parameters.

This is where I think the paper has been over-read. Figure 1 should change an experiment plan, not end one. A straight line makes interpolation plausible within the measured range. It does not promise the same exponent after a new tokenizer, architecture, dataset, objective, or large jump in scale.

## Chinchilla: the compute-optimal recipe changed

Chinchilla revisited the allocation rather than disputing the premise. Hoffmann et al. trained more than 400 models from 70M to over 16B parameters. Training length ranged from 5B to 500B tokens.

All three fitting approaches placed the compute-optimal solution near equal exponents: $N_{\mathrm{opt}} \propto C^{0.5}$ and $D_{\mathrm{opt}} \propto C^{0.5}$. Each scale step should fund both parameters and tokens, not primarily parameters.

![Chinchilla paper Figure 1: compute-optimal parameter counts against training FLOPs, including the Kaplan prediction and named language models](/assets/images/chinchilla-compute-frontier-paper-figure.png)

*Original Figure 1 from [Hoffmann et al., “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556), rendered without alteration from the authors’ arXiv source. The solid curves are three Chinchilla fitting approaches; the dashed line is the Kaplan prediction. The plot is a recipe comparison at fixed training FLOPs, not a benchmark leaderboard.*

The figure makes the revision concrete. At a given budget, the Chinchilla curves select fewer parameters than Kaplan and spend the released FLOPs on more training.

The flagship comparison held training compute roughly fixed. Chinchilla used 70B parameters and four times the training data of 280B-parameter Gopher. The paper reports better results across its downstream suite. The point is not that 70B is magic. *Undertraining is an allocation error.*

A smaller deployed model can also reduce inference cost. In that case, the pretraining optimum and serving optimum happen to point in the same direction.

The disagreement is exact. Kaplan scales parameters roughly as $C^{0.73}$ and tokens as $C^{0.27}$; Chinchilla puts both near $C^{0.5}$. Both optimize validation loss at fixed training compute. The estimated frontier changed.

Chinchilla is therefore a warning against memorizing an earlier fit. It is not proof that one recipe is now correct forever.

The popular shorthand of about twenty training tokens per parameter describes one dense-model frontier. It is not a law of nature. Change the data, tokenizer, architecture, context length, recurrent reuse, mixture, or objective. Then refit before spending the full budget.

## What moved after Chinchilla

Later work did not erase Chinchilla. It showed how local the result was. [Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102) found problems in one fitting procedure, but their corrected estimate agreed with Chinchilla's other methods.

[Porian et al. (2024)](https://arxiv.org/abs/2406.19146) traced much of the Kaplan–Chinchilla gap to three choices: last-layer FLOP accounting, warmup length, and scale-dependent optimizer tuning. Small experimental choices had moved the estimated frontier.

The coordinates also became less clean. A token count records positions consumed, not unique information. [Muennighoff et al.](https://arxiv.org/abs/2305.16264) found that limited repetition cost little before returns diminished.

[DoReMi](https://arxiv.org/abs/2305.10429) and [Data Mixing Laws](https://arxiv.org/abs/2403.16952) made domain weights part of the scaling problem. Two runs with the same $D$ can sit on different surfaces because quality, mixture, contamination, or repetition changed.

Parameter count is no more portable. A dense model, a recurrent model, and a routed mixture-of-experts model do not turn parameters into active compute in the same way. [Clark et al.](https://arxiv.org/abs/2202.01169) showed that routed-model performance depends on both parameter count and computational requirement. Total parameters describe storage. Active parameters, memory traffic, communication, and latency describe a different system.

The objective widens again after pretraining. A smaller model trained longer can cost more once and save compute on every request. [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) formalizes that lifecycle trade-off.

Test-time search adds another coordinate. A smaller model with selective search can beat a larger single-pass model on some tasks. The measurement must include the search policy, verifier, latency, and failures.

Tang's GLM-5.3 note makes the same point from post-training. Z.ai kept the GLM-5.2 base fixed, then spent another month on long-horizon environments and reinforcement-learning post-training. [The release report](https://z.ai/blog/glm-5.3) attributes the gains to that stage. The controlled intervention is informative. It is not yet a post-training scaling law: one before-and-after release gives no response surface, held-out forecast, or compute-optimal valley.

The modern scaling problem is therefore not one curve. It is an allocation across pretraining data, model capacity, post-training environments, serving cost, and test-time effort. The right coordinate is the one that removes the current bottleneck under a measured constraint.

## How I read a scaling plot

I do not take an exponent as a training recipe until I can answer six questions.

1. **What is on the vertical axis?** Cross-entropy, perplexity, benchmark accuracy, reward, and human preference do not have interchangeable slopes.
2. **What changed besides the x-axis?** A model-size curve can quietly change batch size, data quality, sequence length, optimizer, or architecture.
3. **Is compute really comparable?** Training FLOPs omit data preparation, communication, failed runs, and often inference. A pretraining-optimal choice can be a poor deployed system.
4. **Is there a measured optimum?** An allocation claim needs an IsoFLOP valley or an equivalent frontier, not two separate monotone curves.
5. **How far is the extrapolation?** A fit can be excellent across one or two orders of magnitude and still reverse five orders later.
6. **What capability is absent from the loss?** Held-out NLL can improve while safety, tool use, reasoning reliability, latency, or a rare domain remains the actual bottleneck.

The last question matters most to me now. Loss is a scalar average over tokens. A product is not.

A mixture can lower average NLL by improving common web patterns while leaving code, a low-resource language, long-horizon agents, or calibration unchanged. Capability evaluations are noisier than NLL and easier to game. They are also where we learn whether the capability that matters actually moved.

That is where entropy becomes practical. A trillion near-duplicate tokens can increase $D$ while adding little conditional information. A smaller source may carry rare syntax, expert reasoning, or an underrepresented language. It is more valuable only if it improves the target distribution rather than shifting it.

Token count is a budget proxy. The real resource is useful predictive signal under the objective we care about.

## Where the next unit of compute goes

The supplied threads pull the discussion back to the decision layer. [Ishaan's long-form thread](https://x.com/auto_grad_/status/2089970913408380932) asks what governs capability extraction. [Zixuan Li's post](https://x.com/ZixuanLi_/status/2089950717347774919) compresses the allocation problem into one question: where does the next unit of compute remove the most important current failure?

I agree, with one guardrail. A headline parameter count or one loss curve does not identify the bottleneck.

I would build a small response surface first. Sweep a few model sizes, token budgets, and mixtures. Keep architecture and evaluation fixed. Fit uncertainty, not only a line. Then test whether the candidate recipe improves the capability that motivated the run.

If latency, data access, or evaluation reliability is the bottleneck, more compute may have lower return than work on that constraint.

## What survives

**The frontier is a two-dimensional constrained optimization problem.** A curve for $L(N)$ does not prescribe how to spend a fixed budget because $N$ and $D$ are coupled by training compute. The allocation question needs an IsoFLOP sweep: vary model size while holding total FLOPs fixed, train each candidate long enough to expose the loss valley, then fit how the valley moves. Without that valley, an exponent is descriptive rather than prescriptive.

**Perplexity is a presentation transform, not a second metric.** Optimization occurs in cross-entropy or NLL; perplexity is its exponential. Fit and compare the additive loss, especially when curves are close. Entropy supplies the irreducible term in $H(p,q_\theta)=H(p)+D_{\mathrm{KL}}(p\Vert q_\theta)$, but changing the tokenizer or the held-out distribution changes the unit of measurement itself. A perplexity comparison across different tokenizations is not automatically meaningful.

**Token count is not effective data.** $D$ records how many token positions the optimizer saw. It does not record how much new conditional structure those positions supplied after de-duplication, filtering, contamination, overlap, and mixture weighting. Two corpora with the same $D$ can produce different loss surfaces. A modern scaling study should therefore report data construction and mixture weights as part of the fitted regime, not as appendix detail.

**Use loss to choose a pretraining frontier; use a capability vector to accept a system.** NLL is the right dense signal for deciding whether one pretraining recipe is more sample- or compute-efficient. It cannot decide whether the resulting system is acceptable for a target domain, latency budget, safety envelope, or tool-use task. Keep those evaluations separate, and treat a divergence between loss and capability as evidence that the scalar training objective is no longer the bottleneck you care about.

> **The law is not the strategy.** It is a local map of where marginal return currently lives. The strategy is choosing which map describes the system we are building and knowing when to redraw it.

## A reading map

These are the papers that most changed my model of the field.

### Foundations

- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409), Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), Kaplan et al., 2020.
- [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293), Hernandez et al., 2021.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al., 2022.
- [Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169), Clark et al., 2022.

### Estimation and functional form

- [Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891), Caballero et al., 2022/2023.
- [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/abs/2404.10102), Besiroglu et al., 2024.
- [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146), Porian et al., 2024.
- [A Hitchhiker's Guide to Scaling Law Estimation](https://arxiv.org/abs/2410.11840), Choshen et al., 2024/2025.
- [Gemstones: A Model Suite for Multi-Faceted Scaling Laws](https://arxiv.org/abs/2502.06857), McLeish et al., 2025.
- [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859), Lourie et al., 2026.
- [Skaling: Chinchilla's Exponents Meet Kaplan's Coupling](https://arxiv.org/abs/2608.07222), Videau et al., 2026.

### Data, mixtures, and repetition

- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), Muennighoff et al., 2023/2025.
- [DoReMi](https://arxiv.org/abs/2305.10429), Xie et al., 2023.
- [Data Mixing Laws](https://arxiv.org/abs/2403.16952), Ye et al., 2024/2025.
- [Prescriptive Scaling Laws for Data Constrained Training](https://arxiv.org/abs/2605.01640), Lovelace et al., 2026.
- [InfoLaw](https://arxiv.org/abs/2605.02364), Liu et al., 2026.
- [Data-Constrained Language Model Pretraining](https://arxiv.org/abs/2606.06888), Xu et al., 2026.

### Inference and lifecycle optimization

- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448), Sardana et al., 2024.
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787), Brown et al., 2024.
- [Inference Scaling Laws](https://arxiv.org/abs/2408.00724), Wu et al., 2024.
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314), Snell et al., 2024.
- [Scaling Inference-Efficient Language Models](https://arxiv.org/abs/2501.18107), Bian et al., 2025.
- [Test-Time Scaling Makes Overtraining Compute-Optimal](https://arxiv.org/abs/2604.01411), Roberts et al., 2026.
- [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001), Hariri et al., 2026.

### Capabilities and downstream prediction

- [Predictability and Surprise in Large Generative Models](https://arxiv.org/abs/2202.07785), Ganguli et al., 2022.
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682), Wei et al., 2022.
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004), Schaeffer et al., 2023.
- [Understanding Emergent Abilities of Language Models from the Loss Perspective](https://arxiv.org/abs/2403.15796), Du et al., 2024.
- [Language Models Scale Reliably with Over-Training and on Downstream Tasks](https://arxiv.org/abs/2403.08540), Gadre et al., 2024.
- [Scaling Laws Are Unreliable for Downstream Tasks](https://arxiv.org/abs/2507.00885), Lourie et al., 2025.
- [Revisiting the Scaling Properties of Downstream Metrics](https://arxiv.org/abs/2512.08894), Krajewski et al., 2025.
- [Pretraining Scaling Laws for Generative Evaluations](https://arxiv.org/abs/2509.24012), Schaeffer et al., 2025.

### Post-training

- [Scaling Behaviors of LLM Reinforcement Learning Post-Training](https://arxiv.org/abs/2509.25300), Tan et al., 2025/2026.
- [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097), Shen et al., 2026.
