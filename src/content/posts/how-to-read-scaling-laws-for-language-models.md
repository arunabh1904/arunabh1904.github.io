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
summary: Kaplan, Chinchilla, data constraints, inference economics, test-time compute, and why every scaling law is a local map of one experimental regime.
---
# How to Read Scaling Laws for Language Models

Scaling laws for language models are ubiquitous. They appear in almost every model-release discussion: as an argument for a larger model, as a fixed token-to-parameter ratio, or as a vague claim that progress is inevitable. The phrase is familiar. The reasoning underneath it often is not.

The latest round of commentary followed [a post by Jie Tang](https://x.com/jietang/status/2089941544581403107), Z.ai's co-founder and chief scientist, about GLM-5.3. Z.ai kept the same base model as GLM-5.2, spent another month scaling long-horizon environments and reinforcement-learning post-training, and reported large capability gains. Tang's conclusion was not that parameter scaling had ended. It was that scaling has several dials, and the dial with the most return can change from one release to the next.

That is the useful question hiding under the online debate: where should the next unit of compute go?

So let me take a careful stab at scaling laws from that angle. This is meaty, but the core idea is simple. A scaling law is an **empirical regularity**: within a measured regime, a chosen metric changes predictably as we vary model size, data, or compute. It is a trend found in experiments, not a physical law that guarantees intelligence. In [Dario Amodei's discussion with Lex Fridman](https://lexfridman.com/dario-amodei-transcript/), he describes the confidence behind scaling as inductive: the pattern has repeated across domains, while its theoretical explanation remains incomplete.

That narrower claim is still enormously useful. If a cheap sweep predicts an expensive run, training stops being one heroic bet and becomes a resource-allocation problem. The sweep can tell us whether the next dollar should buy a larger model, more tokens, a different data mixture, a better architecture, a longer post-training run, or more inference-time search.

The important phrase is **within a measured regime**. A fitted curve inherits its architecture, tokenizer, data construction, optimizer, learning-rate schedule, context length, compute accounting, and evaluation distribution. Change enough of those and we are not moving along the same curve. We are testing whether a new curve transfers.

## From a trend line to a system decision

Before reading Kaplan or Chinchilla, we need to separate four questions. They are here because each question requires stronger evidence than the previous one, and because arguments about scaling often slide from the first to the fourth without noticing.

1. A **descriptive** result says the measured runs lie on a smooth trend. It summarizes experiments we already performed.
2. A **predictive** result says that trend forecasts a held-out, more expensive run. It earns the right to guide the next experiment.
3. A **prescriptive** result says that optimizing the fitted surface under a resource constraint gives a better allocation. It tells us how much budget should go to model size and how much to data.
4. A **system-optimal** result says that allocation still makes sense after serving demand, latency, memory, post-training, and test-time compute enter the objective. It guides the product, not only the pretraining run.

A straight line on log-log axes proves only a descriptive relationship. It may interpolate beautifully and extrapolate badly. Even a predictive loss curve does not identify the best model-data split until we impose a constraint and measure both sides of the optimum. And a compute-optimal pretraining recipe can be a poor deployed system when inference is paid billions of times.

This ladder is the spine of the post. Kaplan established unusually clean descriptive and predictive regularity. Chinchilla used fitted regularity to revise a pretraining allocation. Tang's GLM-5.3 argument asks us to widen the allocation problem again: the most valuable scaling axis may now sit in post-training rather than parameter count. None of these results, by itself, solves the full system-optimal problem.

## Entropy, cross-entropy, and perplexity

Start with one prediction. Given the tokens so far, an autoregressive language model assigns a probability to every possible next token. If the observed next token receives probability $0.5$, its negative log-likelihood is $-\log(0.5)\approx0.69$ nats. At probability $0.1$, the penalty is about $2.30$ nats. At probability $0.01$, it is about $4.61$ nats. The logarithm turns a product of probabilities across a sequence into a sum, while the negative sign makes better predictions produce smaller numbers.

On a held-out sequence $x_1, \ldots, x_T$, we average that penalty over tokens:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This sample average is the validation negative log-likelihood. With enough representative held-out data, it estimates **cross-entropy**. We use the natural logarithm here, so the unit is a *nat*; base-two logarithms would give bits. The score is proper: a model cannot improve its expected score by reporting probabilities it does not believe. Confident mistakes cost much more than modest uncertainty.

Now separate the three terms that people often blur together.

**Entropy $H(p)$ asks how uncertain the source itself is.** Imagine an ideal conditional distribution $p(x_t\mid x_{<t})$ that describes how the held-out text is generated. If the next token were always determined by the context, the entropy would be zero. If several continuations remained plausible, entropy would be positive. This uncertainty belongs to the specified source, tokenizer, and conditioning setup. It is not a property of English in the abstract, and it is not a property of one trained model.

**Cross-entropy $H(p,q_\theta)$ asks how well our model $q_\theta$ predicts that source.** We sample tokens from $p$ but score them using the probabilities from $q_\theta$. Cross-entropy includes both uncertainty that no model can remove under the stated setup and error caused by the model's mismatch with the source.

That split is exact:

The relationship is exact:

$$
H(p,q_\theta) = H(p) + D_{\mathrm{KL}}(p\,\Vert\,q_\theta).
$$

The entropy term is the floor for that source and representation. The Kullback-Leibler divergence $D_{\mathrm{KL}}(p\,\Vert\,q_\theta)$ is the model's avoidable excess loss; it is zero only when the model distribution matches the source distribution. For natural language we do not know the true $p$, so we do not know the exact entropy floor. A measured validation loss is therefore not “the entropy of the model.” It is an empirical estimate of one model's cross-entropy on one held-out distribution.

**Perplexity repackages the same cross-entropy.** When loss is measured in nats,

$$
\operatorname{PPL}=\exp(H(p,q_\theta)).
$$

If cross-entropy is $\log 10\approx2.30$ nats, perplexity is $10$. This is sometimes described as an effective branching factor: the model behaves, on average, as though it were choosing among ten equally likely continuations. Real token distributions are highly uneven, so that picture is intuition, not a literal count of ten candidate words.

Perplexity contains no information that loss does not. If two models have losses $L_1$ and $L_2$, then

$$
\frac{\operatorname{PPL}_2}{\operatorname{PPL}_1}
= \exp(L_2-L_1).
$$

A loss reduction of $0.1$ nats multiplies perplexity by $e^{-0.1}\approx0.905$, a reduction of about $9.5\%$. Reporting perplexity can make a log score easier to read, but it does not create a second piece of evidence.

That score changes when the document mixture, de-duplication policy, contamination, context construction, or evaluation corpus changes. Its per-token value also changes with the tokenizer because the denominator changes. Perplexity from two tokenizers is therefore not directly comparable. When cross-tokenizer comparison matters, a representation-normalized quantity such as bits per byte is often more informative:

$$
\operatorname{BPB}
= \frac{T\,\mathcal{L}_{\mathrm{NLL}}}{B\log 2},
$$

where $T$ is the number of tokens and $B$ is the number of source bytes.

I use validation loss as the first metric in a scaling study because it is dense, stable, and comparatively cheap: every token contributes a score. Treat NLL as the primitive measured quantity. Whether a statistical model should fit raw loss, log loss, or loss above an estimated floor is a separate modeling decision that must win on held-out runs.

The boundary is equally important. Lower cross-entropy means better prediction of the measured token distribution. It does not certify factuality, reasoning, instruction following, safety, or long-horizon task completion. A scaling law begins with a precise metric-resource relationship. The rest of the post asks how far that relationship can support a decision.

## Before Kaplan: power laws were already showing up

The empirical story did not begin in 2020. [Hestness et al. (2017)](https://arxiv.org/abs/1712.00409) measured power-law generalization curves across language modeling, machine translation, vision, and speech. Their result was already suggesting something important: deep-learning progress could be smooth enough to forecast, even when the mechanism behind the exponent was not understood.

Kaplan's contribution was to turn that observation into a particularly clean language-model resource model. It connected model size, dataset size, and training compute over a wide range, then used those relationships to derive a compute allocation.

That last step is what made the work operational.

## Kaplan: loss became predictable across three resources

The simple version of [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) is this: language-model loss changed smoothly enough with model size, data, and compute that the expensive end of a training family could be forecast from cheaper runs. The paper did not study “size” as one undifferentiated knob. It varied non-embedding parameters $N$, training-data tokens $D$, and training compute $C$, then asked which resource limited each run.

Kaplan's original Figure 1 is worth reading slowly. Each right-hand panel holds the other constraints loose enough that the plotted resource is the dominant limitation. The approximately straight trend on log-log axes is the empirical signature of a power law. It does not mean that loss falls linearly with the resource. It means a relationship such as $L(X)\approx L_\infty + AX^{-\alpha}$ becomes a line after taking logarithms of the reducible part.

![Kaplan et al. 2020 Figure 1: validation loss as a function of compute, dataset size, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)

*Original Figure 1 from [Kaplan et al., “Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361), rendered without alteration from the authors' arXiv source. The three panels show measured loss curves against training compute, dataset size, and non-embedding parameters.*

The compact mental model is that a run can be limited by at least three things:

- insufficient model capacity,
- insufficient data or optimization steps,
- insufficient total compute to increase both.

If model size stays fixed, more data eventually gives diminishing returns because capacity becomes the bottleneck. If data stays fixed, a larger model eventually gives diminishing returns because the data becomes the bottleneck. A compute constraint couples the two.

Kaplan's fitted compute-optimal prescription had a surprising consequence. At fixed training compute, it favored very large models trained on relatively modest amounts of data and stopped well before convergence. In their fitted regime, optimal parameter count grew roughly as $C^{0.73}$ while optimal token count grew roughly as $C^{0.27}$.

That was not a permanent result that data is less valuable than parameters. It was the optimum of one measured surface under one compute model.

This is where the paper is often over-read. A straight line says a local interpolation or extrapolation may be plausible. It does not say that the same exponent survives a new optimizer, tokenizer, long-context architecture, cleaner data distribution, multimodal objective, or a jump far outside the sweep.

A scaling curve should change an experiment plan, not end the discussion.

## Chinchilla: a smaller model trained on more data

The one-sentence version of Chinchilla is easy to remember. At the same pretraining-compute budget, Hoffmann et al. found that many large language models were too large and had seen too few tokens. Their demonstration model used 70B parameters rather than Gopher's 280B and trained on roughly four times as much data, then outperformed Gopher across the paper's evaluation suite.

The deeper result is not “70B beats 280B” or “always use twenty tokens per parameter.” Chinchilla fitted a two-dimensional loss surface, imposed a compute constraint, and solved for the model-data allocation along that constraint. We need the algebra because it shows exactly which parts of the recipe come from exponents, which come from coefficients, and which disappear when the objective changes.

### How a loss surface becomes an allocation

The cleanest way to see the allocation problem is the later separable loss model used by Chinchilla:

$$
\hat{L}(N,D)
= E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

Here, $E$ is the asymptotic floor for the stated setup. The $N$ term captures model limitation. The $D$ term captures finite-data and finite-optimization limitation.

For a dense Transformer, training compute is often approximated as

$$
C \approx \kappa ND,
$$

with $\kappa \approx 6$ under the common forward-plus-backward FLOP approximation. The exact constant and even the adequacy of the $ND$ model depend on architecture and accounting.

Minimizing $\hat{L}(N,D)$ subject to that compute constraint gives

$$
N_{\mathrm{opt}}(C)
= G\left(\frac{C}{\kappa}\right)^{\frac{\beta}{\alpha+\beta}},
$$

$$
D_{\mathrm{opt}}(C)
= G^{-1}\left(\frac{C}{\kappa}\right)^{\frac{\alpha}{\alpha+\beta}},
$$

where

$$
G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}.
$$

The first deep point is that the loss exponents determine how the optimal allocation *moves* as compute grows:

$$
a = \frac{\beta}{\alpha+\beta},
\qquad
b = \frac{\alpha}{\alpha+\beta}.
$$

The second deep point is that the coefficients determine where the frontier actually sits. At the optimum,

$$
\alpha A N^{-\alpha}
= \beta B D^{-\beta}.
$$

The marginal return from spending compute on capacity balances the marginal return from spending it on training tokens. Exponents alone do not give the full recipe. The coefficients, parameter definition, FLOP model, and fitted regime matter.

This also clarifies the famous tokens-per-parameter rule. If $\alpha \approx \beta$, then $N$ and $D$ both grow approximately as $C^{1/2}$, so $D/N$ is approximately constant along the fitted frontier. The numerical ratio is set mostly by the fitted coefficients through $G$.

So “about twenty tokens per parameter” is not a universal exponent and not merely a magic point. It is a coefficient-level property of the Chinchilla experimental regime that looked roughly constant because the fitted exponents were close to equal.

Change the coefficients, introduce interactions between $N$ and $D$, or optimize a different system objective, and the ratio moves.

### What the Chinchilla sweep changed

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556), the Chinchilla paper, revisited the allocation rather than disputing the premise. The authors trained more than 400 models from 70M to over 16B parameters and varied training length from 5B to 500B tokens. Their three approaches placed the compute-optimal solution near equal scaling in parameters and data:

$$
N_{\mathrm{opt}} \propto C^{0.5},
\qquad
D_{\mathrm{opt}} \propto C^{0.5}.
$$

The exact estimates varied by method. Their direct IsoFLOP approach produced exponents near $0.49$ and $0.51$; the parametric surface produced approximately $0.46$ and $0.54$. The common conclusion was the important part: each increase in compute should fund both a larger model and substantially more training tokens, rather than primarily more parameters.

![Chinchilla paper Figure 1: compute-optimal parameter counts against training FLOPs, including the Kaplan prediction and named language models](/assets/images/chinchilla-compute-frontier-paper-figure.png)

*Original Figure 1 from [Hoffmann et al., “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556), rendered without alteration from the authors' arXiv source. The solid curves are three Chinchilla fitting approaches; the dashed line is the Kaplan prediction. The plot compares training recipes at fixed FLOPs, not benchmark scores at fixed model size.*

The flagship comparison made the revision concrete. At roughly comparable training compute, Chinchilla used 70B parameters and four times as much training data as the 280B-parameter Gopher model. It then outperformed Gopher across the paper's evaluation suite.

The scientific point was not that 70B is a magic size. It was that many large models of the period were undertrained relative to the measured train-compute optimum.

That qualification matters. Undertraining is an allocation error **relative to a training-loss objective at fixed pretraining compute**. A deliberately overtrained smaller model can be the better choice once inference demand, memory, or latency enters the objective. Chinchilla answered a pretraining question. It did not settle the lifecycle optimum of a deployed model.

The disagreement between Kaplan and Chinchilla is therefore precise. The objective was essentially the same, validation loss at fixed training compute. The estimated frontier changed. That is why Chinchilla is a warning against memorizing a recipe from an earlier fit, not proof that one correct recipe had finally been discovered.

## What later replications changed

The post-Chinchilla literature makes the methodological lesson even stronger.

[Besiroglu et al. (2024)](https://arxiv.org/abs/2404.10102) found that Chinchilla's third fitting procedure, as reported, was inconsistent with the first two methods, fit the reconstructed data poorly, and produced implausibly narrow confidence intervals. Their rederived estimate was nevertheless compatible with the first two Chinchilla approaches.

[Porian et al. (2024)](https://arxiv.org/abs/2406.19146) reproduced the Kaplan-style result and traced much of the Kaplan-Chinchilla discrepancy to three experimental details: last-layer computational cost, warmup duration, and scale-dependent optimizer tuning. Correcting those factors brought the result into close agreement with Chinchilla.

This is a more useful conclusion than “Kaplan was wrong.” Small accounting and optimization choices can rotate a fitted frontier enough to change a billion-dollar training prescription.

The estimation literature then made the same point from several directions:

- [A Hitchhiker's Guide to Scaling Law Estimation](https://arxiv.org/abs/2410.11840) found that intermediate checkpoints can improve fits, nearby scales are generally more predictive, and seed variation can make several smaller runs more useful than one larger proxy.
- [Gemstones](https://arxiv.org/abs/2502.06857) used thousands of checkpoints across model shapes and showed that architecture, learning-rate choices, cooldown, parameter definitions, and model selection can materially change the derived prescription.
- [Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891) showed why one unbroken power law is not expressive enough for every regime. Smooth changes in slope, saturation, delayed improvement, and non-monotonic behavior need richer forms.
- A very recent 2026 study, [Small-Scale Experiments: Are We There Yet?](https://arxiv.org/abs/2608.11859), argues that scaling behavior can exist at very small scales but only on a well-tuned frontier. Small models are more hyperparameter-sensitive, so a careless small-scale sweep can measure mistuning rather than scaling.

The operational lesson is simple:

> A scaling law predicts the frontier represented by its experiments. If the proxy runs are not themselves on a comparable optimization frontier, the law faithfully extrapolates the wrong thing.

A high in-sample $R^2$ is not enough. I want a held-out scale, a held-out token-to-parameter extreme, seed variation, an extrapolation distance, and at least one plausible alternative functional form.

## Three ways the original map becomes incomplete

Kaplan and Chinchilla give us a map from model size, tokens, and pretraining compute to validation loss. The next part widens that map in three different directions. First, the **resource coordinates** can be too crude: a token count hides quality and repetition, while a parameter count hides active compute and architecture. Second, the **objective** can be too narrow: a pretraining optimum ignores post-training, test-time compute, and repeated serving cost. Third, the **output metric** can be too narrow: smooth average loss can hide jagged changes in the capability a user sees.

These are separate sections because they repair different parts of the optimization problem. Better data accounting does not fix a deployment objective. A lifecycle objective does not make loss a complete capability metric. We need all three corrections before a scaling curve can guide a system decision.

## A token count does not measure useful data

The variable $D$ is usually described as “data,” but it records consumed token positions. It does not directly record unique information, quality, coverage, contamination, domain balance, or repetition.

Two corpora with the same token count can produce different loss surfaces. The same corpus can also have different value for two targets. Code may be unusually valuable for a coding model and irrelevant to a different deployment distribution. “Data quality” is therefore not one universal scalar. It is quality relative to an objective and evaluation distribution.

This is the biggest conceptual limitation of a two-variable $L(N,D)$ view. Modern data pipelines intentionally make tokens non-exchangeable.

### Repetition changes the value of a token

[Muennighoff et al. (2023, revised 2025)](https://arxiv.org/abs/2305.16264) studied data-constrained training across hundreds of runs. In their setup, repeating data for up to roughly four epochs caused negligible loss change compared with using unique data at fixed compute. Beyond that regime, the marginal value of repetition decayed and eventually approached zero.

That should not become a new universal “four epochs are free” rule. It says that repeated tokens have a regime-dependent discount, not zero value on first reuse and not full value forever.

Recent 2026 work pushes this further. [Prescriptive Scaling Laws for Data Constrained Training](https://arxiv.org/abs/2605.01640) models an explicit overfitting penalty and finds regimes where further repetition becomes counterproductive, making extra capacity a better use of compute. [Data-Constrained Language Model Pretraining](https://arxiv.org/abs/2606.06888) proposes a coupled loss form for repeated-data settings and reports that the additive Chinchilla form fits those experiments poorly.

The important shift is from

$$
D = \text{tokens seen}
$$

toward something closer to

$$
D_{\mathrm{eff}}
= f(\text{unique data},\text{mixture},\text{quality},\text{repetition},N).
$$

The dependence on $N$ matters. A larger model may extract more from a source, memorize it sooner, or expose a different bottleneck. Effective data need not be separable from model size.

### Mixture weights are part of the scaling problem

[DoReMi](https://arxiv.org/abs/2305.10429) showed that a small proxy model could optimize domain weights and transfer the resulting mixture to a much larger model. [Data Mixing Laws](https://arxiv.org/abs/2403.16952) and later work made mixture proportions explicit prediction variables rather than hidden data-pipeline choices.

Recent approaches such as [InfoLaw](https://arxiv.org/abs/2605.02364) attempt to model consumed tokens, mixture weights, quality, and repetition together. These are still young formulations, but they point in the right direction: the data axis is a vector that we compressed into $D$ because the first generation of laws needed a tractable coordinate system.

The practical consequence is not “quality beats quantity.” That slogan is almost as empty as “bigger is better.” The consequence is that data construction belongs in the fitted regime. A modern scaling report should expose domain weights, filters, de-duplication, repetition, contamination policy, and target distribution with the same seriousness as model size.

The real resource is useful predictive signal under the objective we care about. Token count is only its easiest proxy.

## The additive Chinchilla surface is now being questioned

The Chinchilla form

$$
E + A N^{-\alpha} + B D^{-\beta}
$$

assumes that the reducible effects of model size and data are additive. In particular, it forces the cross-partial interaction between $N$ and $D$ to zero.

That is a useful approximation in the interior of a well-behaved grid. It is not guaranteed at the corners.

Two very recent 2026 preprints make this issue explicit. The SoftQ formulation in [Data-Constrained Language Model Pretraining](https://arxiv.org/abs/2606.06888) couples model and data limits in repeated-data regimes. [Skaling](https://arxiv.org/abs/2608.07222) introduces an explicit coupling between model capacity and data and reports that the additive Chinchilla form develops systematic errors at data-scarce and heavily overtrained boundaries.

These papers are too recent to treat as the new settled law. Their deeper point is already useful: **separability is an assumption, not a fact**.

If the target run lies near the center of the proxy grid, a simple additive form may be sufficient and easier to estimate. If the target lies at an extreme token-to-parameter ratio, data scarcity, or heavy overtraining regime, interaction terms deserve to be tested explicitly.

A more expressive law is not automatically better. It can overfit a small sweep. The right comparison is held-out extrapolation error, especially at the boundary the final recipe will occupy.

## Parameter count is not a portable measure of capacity

Dense Transformer scaling made parameter count look like a universal axis. Modern architectures make that interpretation harder.

For a routed or mixture-of-experts model, total parameters and active computation are separate variables. [Clark et al. (2022)](https://arxiv.org/abs/2202.01169) showed that routed language-model performance depends on both parameter count and computational requirement. A trillion total parameters with a small active subset is not equivalent to a trillion-parameter dense model, either statistically or operationally.

Model shape matters too. Width, depth, head configuration, attention pattern, recurrence, state, retrieval, vocabulary, and context mechanism can change what one parameter or one FLOP buys. The same nominal parameter count can imply different memory traffic, communication, kernel efficiency, and latency.

This makes three distinctions important:

- **total parameters**, which influence stored capacity and memory,
- **active parameters or FLOPs per token**, which influence computation,
- **wall-clock and systems cost**, which depend on hardware utilization and communication.

They are correlated. They are not interchangeable.

A 2025 study on [inference-efficient scaling](https://arxiv.org/abs/2501.18107) reported up to a 3.5x latency difference among similarly sized architectures in its experimental setup. That is exactly why a FLOP-optimal architecture can fail a latency objective.

The same caution applies to the common $C \approx 6ND$ approximation. It is useful for a dense family with stable shape. It becomes increasingly lossy when embeddings, long-context attention, sparse routing, recurrence, or architecture shape changes across the sweep.

A parameter count is a model description. It is not a system description.

## The optimum moves when the model must be served

The previous sections repaired the resource axes. Now the objective itself has to expand. Chinchilla optimizes validation loss under a pretraining FLOP budget. A deployed model has a lifecycle budget.

A minimal system objective looks more like

$$
C_{\mathrm{total}}
= C_{\mathrm{train}}
+ C_{\mathrm{post}}
+ R\,\mathbb{E}_{x}[C_{\mathrm{infer}}(x)],
$$

where $R$ is the expected number of requests over the model's lifetime. Memory and latency may enter as hard constraints rather than additive costs.

When $R$ is small, a training-compute optimum may be a reasonable approximation. When $R$ is large, every extra active parameter is paid repeatedly. Spending more once on training a smaller model can be cheaper than serving a larger model forever.

[Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) formalized this trade-off and found that sufficiently large inference demand favors models that are smaller and trained longer than the Chinchilla training optimum. The paper also explored token-to-parameter ratios far outside the region used to fit the original law, which exposed how sensitive the result is to training data coverage.

This reframes “overtraining.” Relative to a training-only objective, a model may be trained past the loss-optimal stopping point for its parameter count. Relative to a lifecycle objective, the same decision may be optimal because it buys a smaller serving footprint.

There is no contradiction. There are two objective functions.

### Test-time compute creates another frontier

The original scaling-law picture treats inference as one forward generation from a fixed model. Modern reasoning systems can spend variable compute after the prompt arrives.

That compute can take several forms:

- extending one sequential reasoning trajectory,
- sampling many complete candidates and voting or verifying,
- searching over partial trajectories,
- calling tools, environments, or external verifiers,
- adapting the budget to estimated prompt difficulty.

These are not equivalent uses of a scalar “test-time FLOP” budget. A recent taxonomy, [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001), separates single-trajectory, leaf-level, and prefix-level regimes precisely because their statistics, accounting, and failure modes differ.

The empirical literature has made three things clear.

First, allocation matters. [Snell et al. (2024)](https://arxiv.org/abs/2408.03314) found that prompt-adaptive compute allocation could be substantially more efficient than a fixed best-of-$N$ strategy, and that a smaller model with test-time compute could beat a much larger model on some FLOP-matched problems. [Wu et al. (2024)](https://arxiv.org/abs/2408.00724) similarly found Pareto-optimal combinations of smaller models and stronger search procedures.

Second, generation and selection are separate scaling problems. [Large Language Monkeys](https://arxiv.org/abs/2407.21787) found that solution coverage continued improving across large sampling budgets, especially on verifiable tasks. Without a strong verifier, however, voting and reward-model selection plateaued. More candidates only help the end-to-end system if the system can identify the good one.

Third, test-time scaling changes the pretraining optimum. [Test-Time Scaling Makes Overtraining Compute-Optimal](https://arxiv.org/abs/2604.01411) jointly models model size, training tokens, and inference samples. In its studied tasks, including test-time cost shifted the preferred pretraining recipe deep into the overtrained regime.

The useful mental model is no longer one scalar compute budget. It is a compute vector:

$$
(C_{\mathrm{pre}}, C_{\mathrm{post}}, C_{\mathrm{test}}),
$$

plus a policy for allocating $C_{\mathrm{test}}$ across requests.

Chinchilla asks how to train a one-shot predictor under fixed pretraining FLOPs. A modern system asks how to allocate compute across model creation, adaptation, and each interaction with the world.

### Post-training is becoming its own scaling regime

Pretraining is no longer the only expensive learning stage. Supervised fine-tuning, preference optimization, reinforcement learning, distillation, and verifier training can materially change the final capability frontier.

The science here is less mature than pretraining scaling, and the objectives are more heterogeneous. Reward can saturate, be exploited, or fail to transfer. A token of verifiable mathematics is not equivalent to a token of conversational preference data. The base model also changes the return to post-training compute.

Early studies are beginning to measure these interactions. [Scaling Behaviors of LLM Reinforcement Learning Post-Training](https://arxiv.org/abs/2509.25300) reports predictive relationships among model scale, RL data, and compute in mathematical-reasoning experiments. [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097) uses controlled chess and math settings to connect pretraining loss and tokens to later RL returns.

I would not fold these results into a pretraining exponent. I would model the stages separately and then fit their interaction.

A stronger base model can increase the slope of post-training improvement. Better post-training can also change which pretraining errors matter. Once the acceptance metric is post-trained capability, the system optimum may spend less on reducing average web loss and more on the data, verifiers, or environment interactions that shape the final policy.

### What Jie Tang's GLM-5.3 argument shows

[Jie Tang's post](https://x.com/jietang/status/2089941544581403107) is easiest to understand as a concrete lifecycle-allocation argument. Tang is Z.ai's co-founder and chief scientist, not its CEO. He wrote that GLM-5.3 kept the same base, architecture, total parameters, and activated parameters as GLM-5.2. Z.ai then spent one month scaling long-horizon environments and reinforcement-learning post-training. In other words, the team tried to hold the pretraining resource coordinates roughly fixed while increasing $C_{\mathrm{post}}$ and changing the post-training environment distribution.

[Z.ai's GLM-5.3 release report](https://z.ai/blog/glm-5.3) makes the intervention more concrete. The company says every gain over GLM-5.2 came from post-training: more executable environments, more varied long-horizon tasks, and more compute spent training on them. Z.ai reports a jump from $4.6$ to $28.3$ on Terminal-Bench 3.0 and from $46.2$ to $66.9$ on DeepSWE v1.1, alongside larger gains on its security evaluations. These are vendor-reported results, and some evaluations are private, so they should not be treated as independent confirmation. The controlled comparison is still informative because it shows how much capability remained latent in a fixed base model under a better post-training pipeline.

Tang's “many dials” metaphor maps cleanly to the framework in this post. Parameter count, pretraining tokens, active compute per forward pass, post-training environments, reinforcement-learning compute, and test-time effort are different coordinates. They do not have to scale in lockstep. A team should turn the coordinate with the largest measured marginal return under the current bottleneck.

But GLM-5.3 is not yet a post-training equivalent of Chinchilla. One before-and-after release does not provide a fitted response surface, a held-out forecast, an exponent, or a compute-optimal valley. It is strong descriptive evidence for this intervention, not a general predictive or prescriptive law. The next scientific question is exactly the one Tang's post creates: across several bases and compute budgets, how do environment diversity, rollout length, verifier quality, and reinforcement-learning compute trade off, and where do their returns flatten?

That is why the post belongs in this article. It does not overturn Kaplan or Chinchilla. It demonstrates the modern extension of their resource-allocation logic: once one stage becomes well supplied, the next unit of compute should move to the stage with more slack.

## Smooth loss does not guarantee smooth capability

The resource coordinates and lifecycle objective can now be correct while the acceptance metric is still wrong. This is the third and final widening of the map.

Average validation loss can improve smoothly while a benchmark appears flat and then jumps. [Predictability and Surprise in Large Generative Models](https://arxiv.org/abs/2202.07785) framed this as predictable aggregate loss coexisting with unpredictable specific capabilities and outputs. [Wei et al. (2022)](https://arxiv.org/abs/2206.07682) catalogued abilities that appeared absent in smaller models and present in larger ones.

Some apparent discontinuities are measurement artifacts. [Schaeffer et al. (2023)](https://arxiv.org/abs/2304.15004) showed that nonlinear or discontinuous metrics can manufacture sharp-looking transitions from smoothly changing model outputs. Exact match is an obvious example. A model can move probability steadily toward the right solution while accuracy remains zero until the top-ranked answer flips.

But “all emergence is a metric illusion” is also too strong. [Du et al. (2024)](https://arxiv.org/abs/2403.15796) found that, under a fixed corpus, tokenizer, and architecture, models with similar pretraining loss had similar downstream behavior, and some tasks changed sharply below a loss threshold even under continuous metrics.

The later downstream-scaling literature is appropriately mixed:

- [Gadre et al. (2024)](https://arxiv.org/abs/2403.08540) found reliable relationships between perplexity and average downstream error within a controlled model and data family.
- [Lourie et al. (2025)](https://arxiv.org/abs/2507.00885) reanalyzed individual tasks and found smooth predictable scaling in only 39% of the examined cases, with inverse, noisy, non-monotonic, and breakthrough patterns elsewhere.
- [Krajewski et al. (2025)](https://arxiv.org/abs/2512.08894) found that direct laws from training budget to transformed task accuracy could extrapolate well across several tasks under fixed token-to-parameter ratios.
- Work on [generative evaluations](https://arxiv.org/abs/2509.24012) shows that sampling protocol and pass@$k$ become part of the scaling law itself.

These results are not mutually exclusive. Predictability depends on the coordinate system.

Aggregate metrics average away task-specific thresholds and noise. Fixed data and architecture remove distributional confounders. Individual tasks can depend on rare features, multi-step reliability, prompt format, sampling, or a minimum competence threshold. A long-horizon task can also amplify a small per-step error: if success requires $m$ reliable steps, a per-step success probability $p$ becomes roughly $p^m$. Smooth changes in $p$ can produce a sharp product-level transition.

The right conclusion is not that loss is useless or that every capability is emergent. It is this:

> A pretraining scaling law predicts a statistical substrate. It does not automatically predict every coordinate of the capability vector built on top of it.

Use loss to compare pretraining efficiency. Use target capability evaluations to accept a system. When they diverge, investigate the missing distribution, metric, inference protocol, or post-training stage rather than forcing one scalar to explain the whole product.

## What I would run before spending the frontier budget

A useful scaling study is an experimental design, not a regression script. I would move through the following steps in order: define the decision, make the proxy runs comparable, measure the constrained frontier, challenge the fit, and only then turn the result into a recipe. Skipping an early step cannot be repaired by a more sophisticated regression at the end.

### 1. State the decision first

Write the decision in one sentence before launching a run. Are we choosing model size and token count at fixed pretraining FLOPs, minimizing lifecycle cost at a target quality, selecting a data mixture, or deciding how much test-time search to allow? Each decision needs a different vertical axis and constraint. A fit cannot be judged without the decision it is supposed to support.

### 2. Fix the regime you want to transfer

Hold architecture family, tokenizer, context construction, data pipeline, optimizer family, schedule semantics, evaluation, and compute accounting fixed unless one of them is an explicit axis. This creates a family in which a small run and the proposed large run mean the same thing. If architecture or data quality is the question, vary it deliberately and add enough matched controls to distinguish its effect from scale. Record every exception.

### 3. Sweep across the valley, not only along one ray

For each of several compute budgets, train several model sizes for corresponding token counts so that total FLOPs stay approximately fixed. Loss should first improve as allocation approaches the best split and then worsen after passing it. Points on both sides identify the bottom of that IsoFLOP valley. Two monotone curves that never bracket the minimum cannot support a prescriptive allocation.

### 4. Tune the proxy frontier

Use validated hyperparameter transfer or tune learning rate, batch size, warmup, weight decay, and schedule at representative scales. Small models can be more sensitive than large ones. A family of under-tuned proxies produces a clean law for under-tuned models.

### 5. Save intermediate checkpoints, with schedule awareness

Intermediate checkpoints add many useful token budgets, but a checkpoint halfway through a long cosine schedule is not automatically equivalent to a run designed to end there. Model the schedule or use training protocols whose checkpoints have comparable semantics.

### 6. Measure variance where it matters

Run multiple seeds at a few anchor configurations, especially near the inferred valley. The prescription can be more uncertain than the loss curve makes it look.

### 7. Fit more than one plausible surface

At minimum, compare a simple separable form with a local envelope or a coupled alternative when the target is near a boundary. Consider a broken law if residuals show curvature. More parameters are justified only by held-out prediction.

### 8. Hold out the expensive direction

Do not only report interpolation. Hold out the largest model, the longest training run, or the most extreme token-to-parameter ratio. Report the ratio between the target scale and the largest proxy scale.

### 9. Propagate uncertainty into the recipe

Bootstrap runs, seeds, and fitting choices. Report an interval for $N_{\mathrm{opt}}$ and $D_{\mathrm{opt}}$, not only point estimates. If several recipes are statistically indistinguishable, choose using latency, memory, data availability, and engineering risk.

### 10. Validate the capability that motivated the run

The loss frontier chooses an efficient pretraining candidate. It does not certify the system. Re-evaluate the target capability vector, calibration, safety, latency, and inference protocol at the chosen recipe.

The best scaling study ends by changing the next experiment.

## How I read a scaling plot in 2026

I do not take an exponent as a recipe until I can answer these questions:

1. **What decision is the law intended to support?** Description, forecasting, allocation, and deployment require different evidence.
2. **What is on the vertical axis?** NLL, perplexity, accuracy, pass@$k$, reward, and human preference do not have interchangeable slopes.
3. **What exactly is on each resource axis?** Non-embedding or total parameters? Unique or consumed tokens? Theoretical FLOPs or measured cost? One sample or a search procedure?
4. **What changed besides the plotted variable?** Architecture, data quality, sequence length, optimizer, schedule, and evaluation can rotate the curve.
5. **Is there a measured constrained optimum?** A prescriptive claim needs an IsoFLOP valley or an equivalent response surface.
6. **Were the proxy models comparably tuned?** Otherwise the law may measure optimization error.
7. **How was the functional form chosen?** Inspect residuals and compare separable, coupled, and broken alternatives where relevant.
8. **How far is the extrapolation?** State the scale-up factor and whether the target sits inside the grid or beyond a boundary.
9. **Did the law predict a held-out run?** In-sample fit is not the operational test.
10. **Where is uncertainty reported?** Seeds, checkpoints, fitting procedure, and data construction all contribute.
11. **Does the objective include deployment and test-time compute?** A pretraining optimum can be a poor lifecycle optimum.
12. **What capability is absent from the scalar metric?** Loss can improve while the actual bottleneck remains unchanged.

The last question matters most to me. Loss is a scalar average over tokens. A product is not.

## What I think survives

Okay, that was a lot. Let us compress it.

A scaling law is a local empirical map. It says that, for a controlled family of runs, a measured output changed regularly with a measured resource. The map becomes useful when it predicts a held-out run. It becomes a recipe only after we add a constraint and measure the frontier on both sides of its optimum.

Kaplan showed how predictable the map could be. Chinchilla showed that the same broad question could yield a very different allocation after the experimental frontier was measured more carefully. The exponents describe how the preferred model-data split moves as compute grows; the coefficients and accounting choices determine where that split sits. Neither paper supplied a timeless token-to-parameter ratio.

The modern corrections change three different objects. Tokens are not interchangeable units of information, and parameters are not interchangeable units of capacity or cost. Pretraining compute is not the whole lifecycle objective once post-training, serving, and test-time search matter. Validation loss is not the whole acceptance metric once a product depends on rare skills, long-horizon reliability, calibration, and safety.

The practical habit is therefore consistent across Kaplan, Chinchilla, and Jie Tang's post-training argument. Measure the current bottleneck. Run the smallest sweep that can locate marginal return. Test the prediction outside the fitted points. Then accept the system on the capability and cost vector that motivated the experiment in the first place.

The law is not the strategy. It is a local map of where marginal return currently lives.

The strategy is deciding which map describes the system we are actually building, and knowing when to redraw it.

## A reading map

The scaling-law literature is now too large for one post to enumerate. These are the papers that most changed my mental model.

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
- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429), Xie et al., 2023.
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
