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

A scaling law fits how a measured error changes with parameters, data, or compute. It can forecast larger runs from smaller experiments, but its prescription depends on what the experiment holds fixed. Pretraining FLOPs, available data, serving cost, and test-time computation lead to different optima.

The original result was much sharper than “bigger models work.” [Kaplan et al. (2020)](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html) varied model size, dataset size, and training compute. Validation cross-entropy followed a smooth trend when the other resources were not yet binding.

[Hoffmann et al. (2022)](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html), the Chinchilla paper, asked the next question. With a fixed FLOP budget, what combination of parameters and tokens minimizes loss? The answer starts with the metric, not the model-size headline.

## What a scaling law measures

For an autoregressive language model, each next token has a probability assigned by the model. On a held-out token sequence $x_1, \ldots, x_T$, the average negative log-probability is

$$
\mathcal{L}_{\mathrm{NLL}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}).
$$

This is cross-entropy, measured in *nats* when the log operator $\log$ denotes the natural logarithm. A model that gives the observed next token more probability has lower loss. It is a proper scoring rule: a confident error hurts more than admitted uncertainty.

Three quantities describe different parts of that measurement:

**Entropy $H(p)$** is the irreducible uncertainty of the true token distribution $p$, under one tokenizer and one data distribution. It belongs to the source, not to a particular model.

**Cross-entropy $H(p, q_\theta)$** is the expected negative log-probability assigned by model distribution $q_\theta$ to data from the true distribution $p$. Lower cross-entropy means the model predicts the held-out distribution better.

**Perplexity $\operatorname{PPL}$** is exponentiated cross-entropy $\exp(H(p,q_\theta))$ when cross-entropy is measured in nats. It turns an additive loss into an effective branching factor: a value of 20 can be read as an effective choice among roughly twenty equally likely continuations. This is useful intuition, but exponentiation can hide the additive quantity being optimized.

The relationship is exact:

$$
H(p,q_\theta) = H(p) + D_{\mathrm{KL}}(p\,\Vert\,q_\theta).
$$

Entropy is the floor for that tokenization and distribution; the KL divergence is the model's excess loss above it. We almost never know the true entropy $H(p)$ for natural language, so a measured cross-entropy is not "how much entropy the model has." It is a score for one model on one held-out sample. It changes if the tokenizer, document mixture, de-duplication policy, or evaluation corpus changes.

A drop from 2.0 to 1.9 nats has the same loss difference as a drop from 3.0 to 2.9 nats. Both reduce perplexity by about 9.5%. Cross-entropy loss is already a mean negative log-probability. Whether a scaling fit takes another logarithm of that loss depends on the fitted model. Report perplexity when the multiplicative view helps.

## Kaplan et al.: three curves, not one parameter curve

Kaplan et al. varied non-embedding parameters $N$, training-data tokens $D$, and compute $C$. The original Figure 1 below is worth reading slowly.

Read the three panels as separate one-resource slices: compute on the left, dataset size in the middle, and parameter count on the right. The nearly straight lines on these log-scaled axes are the signature of a power law; they do not say that loss falls linearly with the resource.

![Kaplan et al. 2020 Figure 1: validation loss as a function of compute, dataset size, and non-embedding parameters](/assets/images/kaplan-2020-simple-power-laws-paper-figure.png)
*Figure 1: Kaplan et al. measure validation loss against compute, dataset size, and non-embedding parameter count. The fitted lines summarize one experimental regime; they are not a universal law. | source: [Kaplan et al., Figure 1](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html)*

The compact mental model is a loss surface with three kinds of limitation:

$$
L(N,D) \approx L_\infty + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

This is a useful schematic, not Kaplan's exact fitting equation. The irreducible floor $L_\infty$ belongs to the stated setup. The parameter term captures limited capacity. The data term captures limited coverage and repetition.

Training compute couples the terms because dense-Transformer cost is roughly proportional to the product of parameter count and token count $ND$, up to architecture and system constants. Hold either parameter count $N$ or token count $D$ fixed and the other eventually delivers diminishing returns. Its term is no longer the bottleneck.

Kaplan's compute-optimal prescription had a surprising consequence. At fixed compute, it favored a very large model trained on relatively few tokens, well short of convergence. In that fitted regime, optimal parameter count grew much faster than token count.

The result depended on that curve, WebText2, the optimizer, and the compute accounting; it was not a permanent rule that data is less valuable than parameters.

## Chinchilla: the compute-optimal recipe changed

Chinchilla revisited the allocation rather than disputing the premise. Hoffmann et al. trained more than 400 models from 70M to over 16B parameters. Training length ranged from 5B to 500B tokens.

All three fitting approaches placed the compute-optimal solution near equal exponents. Optimal parameter count follows approximately square-root scaling $N_{\mathrm{opt}} \propto C^{0.5}$, and optimal token count follows the same scaling $D_{\mathrm{opt}} \propto C^{0.5}$. Each scale step should fund both parameters and tokens, not primarily parameters.

![Chinchilla paper Figure 1: compute-optimal parameter counts against training FLOPs, including the Kaplan prediction and named language models](/assets/images/chinchilla-compute-frontier-paper-figure.png)
*The solid curves are three Chinchilla fitting approaches; the dashed line is the Kaplan prediction. The plot is a recipe comparison at fixed training FLOPs, not a benchmark leaderboard. source: [Hoffmann et al., “Training Compute-Optimal Large Language Models”](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html)*

The figure makes the revision concrete. At a given budget, the Chinchilla curves select fewer parameters than Kaplan and spend the released FLOPs on more training.

The flagship comparison held training compute roughly fixed. Chinchilla used 70B parameters and four times the training data of 280B-parameter Gopher, then outperformed it across the reported downstream suite. The experiment tests a different allocation of the same training budget.

A smaller deployed model can also reduce inference cost. In that case, the pretraining optimum and serving optimum happen to point in the same direction.

The disagreement is exact. Kaplan estimates compute exponents of roughly 0.73 for parameters and 0.27 for tokens; Chinchilla puts both near 0.5. Both optimize validation loss at fixed training compute. The estimated frontier changed.

The shorthand of about twenty tokens per parameter describes the fitted dense-model frontier. It depends on the data, tokenizer, architecture, context length, and objective used to estimate that frontier.

## What moved after Chinchilla

Later work did not erase Chinchilla. It showed how local the result was. [Besiroglu et al. (2024)](/paper%20shorts/2024/04/15/chinchilla-scaling-a-replication-attempt.html) found problems in one fitting procedure, but their corrected estimate agreed with Chinchilla's other methods.

[Porian et al. (2024)](/paper%20shorts/2024/06/27/resolving-discrepancies-in-compute-optimal-scaling.html) traced much of the Kaplan–Chinchilla gap to three choices: last-layer FLOP accounting, warmup length, and scale-dependent optimizer tuning. Small experimental choices had moved the estimated frontier.

[Scaling Data-Constrained Language Models](/paper%20shorts/2023/05/25/scaling-data-constrained-language-models.html) asks what happens when unique data runs out. Its experiments find little loss penalty from up to four epochs of repeated data at fixed compute, followed by diminishing returns from further repetition. The fitted law therefore discounts repeated tokens rather than treating every consumed position as new information.

[DoReMi](/paper%20shorts/2023/05/17/doremi-optimizing-data-mixtures-speeds-up-language-model-pretraining.html) changes the mixture rather than the total token count. A 280M proxy learns domain weights through group distributionally robust optimization; those weights then guide an 8B training run. On The Pile, the resulting model improves average few-shot accuracy by 6.5 percentage points and reaches the default-mixture baseline with 2.6 times fewer steps. The experiment tests whether a cheap mixture search transfers to a larger model.

[Data Mixing Laws](/paper%20shorts/2024/03/25/data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance.html) takes a predictive approach: fit losses across sampled mixtures, then estimate untested combinations. Its 1B model trained on 100B RedPajama tokens matches a default-mixture run trained for 48% more steps. DoReMi optimizes weights through a proxy; Data Mixing Laws fits a response surface over them. Both make data composition an explicit variable in the allocation problem.

Parameter count is no more portable. A dense model, a recurrent model, and a routed mixture-of-experts model do not turn parameters into active compute in the same way. [Clark et al.](/paper%20shorts/2022/02/02/unified-scaling-laws-for-routed-language-models.html) showed that routed-model performance depends on both parameter count and computational requirement. Total parameters describe storage. Active parameters, memory traffic, communication, and latency describe a different system.

The objective widens again after pretraining. A smaller model trained longer can cost more once and save compute on every request. [Beyond Chinchilla-Optimal](/paper%20shorts/2023/12/31/beyond-chinchilla-optimal-accounting-for-inference-in-language-model-scaling-laws.html) formalizes that lifecycle trade-off.

[Snell et al.](/paper%20shorts/2024/08/06/scaling-llm-test-time-compute-optimally.html) move the budget from training to inference. They compare verifier-guided search with adaptive response revision and find that the best allocation depends on prompt difficulty. Their adaptive strategy is more than four times as efficient as best-of-N in the reported setting. A smaller model can also beat a 14-times-larger model under matched FLOPs when the smaller model already has non-trivial success on the problem. Search helps recover accessible solutions; it does not guarantee that extra sampling solves problems outside the model's competence.

The GLM-5.3 release provides a post-training example. Z.ai kept the GLM-5.2 base fixed, then spent another month on long-horizon environments and reinforcement-learning post-training. [The release report](https://z.ai/blog/glm-5.3) attributes the gains to that stage. The controlled intervention is informative. It is not yet a post-training scaling law: one before-and-after release gives no response surface, held-out forecast, or compute-optimal valley.

The modern scaling problem is therefore an allocation across pretraining, post-training, serving, and test-time effort.

## From a loss frontier to a deployment budget

The literature has widened the optimization problem in stages. Kaplan and Chinchilla allocate training compute between model size and token count. Data-constrained laws change the value assigned to repeated tokens. Mixture methods change which tokens are sampled. Inference-aware laws add the number and cost of future requests, while test-time scaling chooses how much computation to spend on each problem.

These extensions require different experiments. An IsoFLOP sweep identifies a training frontier. A mixture sweep tests data allocation. A serving study includes the workload and request count. A test-time study includes the search policy and verifier. A result from one of these settings cannot supply the optimum for the others without those extra measurements.

The deployment discussion in [Ishaan's thread](https://x.com/auto_grad_/status/2089970913408380932) and [Zixuan Li's post](https://x.com/ZixuanLi_/status/2089950717347774919) asks where the next unit of compute should go. The reviewed work turns that into a measurable question: fit the relevant allocation surface, forecast held-out runs, and test whether lower loss transfers to the capability and latency required by the application.

## A reading map

### Foundations

- [Deep Learning Scaling is Predictable, Empirically](/paper%20shorts/2017/12/01/deep-learning-scaling-is-predictable-empirically.html), Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html), Kaplan et al., 2020.
- [Scaling Laws for Transfer](/paper%20shorts/2021/02/02/scaling-laws-for-transfer.html), Hernandez et al., 2021.
- [Training Compute-Optimal Large Language Models](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html), Hoffmann et al., 2022.
- [Unified Scaling Laws for Routed Language Models](/paper%20shorts/2022/02/02/unified-scaling-laws-for-routed-language-models.html), Clark et al., 2022.

### Estimation and functional form

- [Broken Neural Scaling Laws](/paper%20shorts/2022/10/26/broken-neural-scaling-laws.html), Caballero et al., 2022/2023.
- [Chinchilla Scaling: A Replication Attempt](/paper%20shorts/2024/04/15/chinchilla-scaling-a-replication-attempt.html), Besiroglu et al., 2024.
- [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](/paper%20shorts/2024/06/27/resolving-discrepancies-in-compute-optimal-scaling.html), Porian et al., 2024.
- [A Hitchhiker's Guide to Scaling Law Estimation](/paper%20shorts/2024/10/15/a-hitchhikers-guide-to-scaling-law-estimation.html), Choshen et al., 2024/2025.
- [Gemstones: A Model Suite for Multi-Faceted Scaling Laws](/paper%20shorts/2025/02/07/gemstones-a-model-suite-for-multi-faceted-scaling-laws.html), McLeish et al., 2025.
- [Small-Scale Experiments: Are We There Yet?](/paper%20shorts/2026/08/12/small-scale-experiments-are-we-there-yet.html), Lourie et al., 2026.
- [Skaling: Chinchilla's Exponents Meet Kaplan's Coupling](/paper%20shorts/2026/08/07/skaling-chinchillas-exponents-meet-kaplans-coupling.html), Videau et al., 2026.

### Data, mixtures, and repetition

- [Scaling Data-Constrained Language Models](/paper%20shorts/2023/05/25/scaling-data-constrained-language-models.html), Muennighoff et al., 2023/2025.
- [DoReMi](/paper%20shorts/2023/05/17/doremi-optimizing-data-mixtures-speeds-up-language-model-pretraining.html), Xie et al., 2023.
- [Data Mixing Laws](/paper%20shorts/2024/03/25/data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance.html), Ye et al., 2024/2025.
- [Prescriptive Scaling Laws for Data Constrained Training](/paper%20shorts/2026/05/02/prescriptive-scaling-laws-for-data-constrained-training.html), Lovelace et al., 2026.
- [InfoLaw](/paper%20shorts/2026/05/04/infolaw-information-scaling-laws-for-large-language-models-with-quality-weighted-mixture-data-and-repetition.html), Liu et al., 2026.
- [Data-Constrained Language Model Pretraining](/paper%20shorts/2026/06/05/data-constrained-language-model-pretraining-improved-regularization-and-scaling-laws.html), Xu et al., 2026.

### Inference and lifecycle optimization

- [Beyond Chinchilla-Optimal](/paper%20shorts/2023/12/31/beyond-chinchilla-optimal-accounting-for-inference-in-language-model-scaling-laws.html), Sardana et al., 2024.
- [Large Language Monkeys](/paper%20shorts/2024/07/31/large-language-monkeys-scaling-inference-compute-with-repeated-sampling.html), Brown et al., 2024.
- [Inference Scaling Laws](/paper%20shorts/2024/08/01/inference-scaling-laws-compute-optimal-llm-problem-solving.html), Wu et al., 2024.
- [Scaling LLM Test-Time Compute Optimally](/paper%20shorts/2024/08/06/scaling-llm-test-time-compute-optimally.html), Snell et al., 2024.
- [Scaling Inference-Efficient Language Models](/paper%20shorts/2025/01/30/scaling-inference-efficient-language-models.html), Bian et al., 2025.
- [Test-Time Scaling Makes Overtraining Compute-Optimal](/paper%20shorts/2026/04/01/test-time-scaling-makes-overtraining-compute-optimal.html), Roberts et al., 2026.
- [Test-Time Scaling in Reasoning LLMs](/paper%20shorts/2026/08/04/test-time-scaling-in-reasoning-llms.html), Hariri et al., 2026.

### Capabilities and downstream prediction

- [Predictability and Surprise in Large Generative Models](/paper%20shorts/2022/02/15/predictability-and-surprise-in-large-generative-models.html), Ganguli et al., 2022.
- [Emergent Abilities of Large Language Models](/paper%20shorts/2022/06/15/emergent-abilities-of-large-language-models.html), Wei et al., 2022.
- [Are Emergent Abilities of Large Language Models a Mirage?](/paper%20shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html), Schaeffer et al., 2023.
- [Understanding Emergent Abilities of Language Models from the Loss Perspective](/paper%20shorts/2024/03/23/understanding-emergent-abilities-from-the-loss-perspective.html), Du et al., 2024.
- [Language Models Scale Reliably with Over-Training and on Downstream Tasks](/paper%20shorts/2024/03/13/language-models-scale-reliably-with-overtraining-and-on-downstream-tasks.html), Gadre et al., 2024.
- [Scaling Laws Are Unreliable for Downstream Tasks](/paper%20shorts/2025/07/01/scaling-laws-are-unreliable-for-downstream-tasks-a-reality-check.html), Lourie et al., 2025.
- [Revisiting the Scaling Properties of Downstream Metrics](/paper%20shorts/2025/12/09/revisiting-the-scaling-properties-of-downstream-metrics-in-large-language-model-training.html), Krajewski et al., 2025.
- [Pretraining Scaling Laws for Generative Evaluations](/paper%20shorts/2025/09/28/pretraining-scaling-laws-for-generative-evaluations-of-language-models.html), Schaeffer et al., 2025.

### Post-training

- [Scaling Behaviors of LLM Reinforcement Learning Post-Training](/paper%20shorts/2025/09/29/scaling-behaviors-of-llm-reinforcement-learning-post-training-an-empirical-study-in-mathematical-reasoning.html), Tan et al., 2025/2026.
- [Understanding Reasoning from Pretraining to Post-Training](/paper%20shorts/2026/07/17/understanding-reasoning-from-pretraining-to-post-training.html), Shen et al., 2026.
