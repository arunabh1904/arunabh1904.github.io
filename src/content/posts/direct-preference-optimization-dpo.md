---
title: 'Direct Preference Optimization'
date: '2023-05-29T00:00:00.000Z'
section: paper-shorts
postSlug: direct-preference-optimization-dpo
legacyPath: /paper shorts/2023/05/01/direct-preference-optimization-dpo.html
tags:
  - Preference Optimization
  - Post-Training
field: 'Alignment & Post-Training'
topics:
  - language-systems
  - learning
summary: '2023 – Direct Preference Optimization: Your Language Model Is Secretly a Reward Model'
---

## 2023 – Direct Preference Optimization: Your Language Model Is Secretly a Reward Model

**arXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

**Code:** [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)

**Conference:** NeurIPS 2023 (spotlight)

## Summary

> Direct Preference Optimization (DPO) rewrites a KL-constrained RLHF objective as a binary classification loss over preferred and dispreferred completions. In the usual pipeline, a reward model is fitted to preference pairs and a policy is then trained with online reinforcement learning. DPO uses the policy's log-probability ratio to a frozen reference model as an implicit reward, so the policy can be trained directly from an offline preference dataset without an explicit reward model, value model, or PPO loop. On controlled sentiment, Reddit TL;DR summarization, and single-turn dialogue, the paper finds a stronger reward–KL trade-off and results comparable to or better than its PPO baselines, with models up to 6B parameters. The derivation depends on the Bradley–Terry preference model and on a reference policy that covers the compared responses; offline data still bounds what the method can learn.

## Core Insights

### DPO removes a loop by changing the representation of reward

![Figure 1 from Direct Preference Optimization](/assets/images/direct-preference-optimization-dpo-source-figure-1.webp)
*Fig 1: RLHF fits a reward model and uses reinforcement learning to update a policy, whereas DPO fits the final language model directly from preference pairs with maximum likelihood. | source: [Direct Preference Optimization, Figure 1](https://arxiv.org/abs/2305.18290)*

The left side of the figure contains two moving models: a reward model scores sampled completions, and an LM policy is repeatedly updated from those scores. The right side keeps the preference pairs and moves the likelihood objective directly onto the final LM. DPO still needs preference data, but it removes reward-model training as a separate artifact and removes online sampling from the policy-optimization loop.

### The closed-form bridge from KL control to a classification loss

RLHF commonly optimizes a reward while keeping the policy near a reference model, usually the SFT checkpoint:

$$
\max_{\pi_\theta}\;\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(y\mid x)}
\left[r_\phi(x,y)-\beta D_{\mathrm{KL}}\left(\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\right)\right].
$$

The KL term limits the policy's ability to exploit reward-model mistakes, while also keeping generation near the distribution on which the reward model was trained. For a fixed reward $r$, the exact optimum has the Gibbs form

$$
\pi_r(y\mid x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right),
$$

where $Z(x)$ is a prompt-dependent normalizer. Rearranging gives a reward in terms of its optimal policy:

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

Under a Bradley–Terry preference model, only the difference between two rewards matters, so the unknown $\log Z(x)$ cancels. Replacing the unknown optimal policy with a trainable policy yields the DPO loss for preferred $y_w$ and rejected $y_l$:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=-\mathbb{E}_{(x,y_w,y_l)\sim D}
\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

The reference-relative margin is the key. Maximizing $\pi_\theta(y_w\mid x)$ alone can increase both responses and collapse the model into a narrow mode. DPO instead asks whether the policy raises the chosen response's log probability relative to its reference more than it raises the rejected response's probability.

### The update is selective because its weight is an implicit reward error

DPO's policy defines an implicit reward

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

The gradient increases the likelihood of $y_w$ and decreases the likelihood of $y_l$, but weights the pair by how strongly the current implicit reward ranks the rejected answer above the chosen one. Easy pairs therefore contribute less after the model orders them correctly; pairs that the model still misorders receive more pressure. The weighting is not a cosmetic detail: the paper's ablation finds that a naive unweighted probability-ratio objective can make the language model degenerate.

### DPO is offline, and the reference distribution still matters

The practical pipeline is two steps. First, use existing or newly collected preference pairs $D=\{(x,y_w,y_l)\}$; when the pairs were sampled from an SFT model, initialize $\pi_{\mathrm{ref}}$ with that SFT model. Second, minimize the loss above with ordinary language-model training. When no SFT checkpoint is available, the paper first fits the reference to the preferred completions so that its support is closer to the data-generating distribution.

That choice makes DPO simple, but it also fixes the method's exposure. The model does not sample new failure states during optimization, and the loss assumes the chosen and rejected responses are comparable under the same prompt. Preference-model assumptions still enter through the Bradley–Terry derivation, even though no standalone reward model is saved.

### The source experiments test both optimization and task behavior

![Figure 2: IMDb sentiment reward versus KL to the reference policy](/assets/images/direct-preference-optimization-dpo-source-figure-2.webp)
*Fig 2: Reward–KL frontier on controlled IMDb sentiment generation; DPO reaches the highest expected reward across the tested divergence range, including comparisons with PPO using ground-truth rewards. | source: [Direct Preference Optimization, Figure 2 (left panel)](https://arxiv.org/abs/2305.18290)*

The plotted horizontal axis is sequence-level KL from the reference policy and the vertical axis is the true sentiment reward. The yellow DPO points form the upper frontier: for a similar amount of deviation from the reference, DPO reaches higher sentiment reward than the PPO and pseudo-supervised alternatives. This is the right intuition for the derivation. DPO and PPO target the same KL-constrained objective, but DPO moves directly along the policy family instead of estimating a reward and then trusting an actor–critic optimizer to find the frontier.

| Experiment | Reported result | Measurement boundary |
| --- | --- | --- |
| Controlled IMDb sentiment | DPO produces the most efficient reward–KL frontier; the sweep contains 22 runs | Ground-truth sentiment classifier, not human preference quality |
| Reddit TL;DR summarization | About 61% win rate at temperature 0, versus PPO's 57% at its best temperature | GPT-4 win rate against test-set reference summaries |
| Anthropic Helpful and Harmless dialogue | DPO is the only computationally efficient method that improves over the chosen completions; it is similar to or better than best-of-128 sampling at its best temperature | GPT-4 win rate against preferred test completions; dataset has 170k dialogues |
| CNN/DailyMail distribution shift | DPO wins 0.36 at temperature 0 and 0.31 at 0.25, versus PPO's 0.26 and 0.23 | GPT-4 win rate against ground-truth news summaries |

The TL;DR result is not just a best point. DPO remains more stable as sampling temperature changes, while PPO can fall toward the base GPT-J model at high temperatures. In the human comparison, DPO samples at temperature 0.25 are preferred 58% of the time over PPO samples at temperature 0. The dialogue experiment makes the compute trade-off visible: best-of-128 can search many candidates at inference, while DPO pays its training cost once.

The paper also checks whether GPT-4 is a reasonable evaluator for the summarization results. In the DPO-versus-greedy-PPO comparison, humans prefer DPO 58% of the time, while the concise GPT-4 prompt prefers it 54%; agreement between that GPT-4 prompt and humans is 67%, close to the 65% human–human agreement reported for the same comparison. The judge is useful evidence, but its prompt is part of the measurement.

### Decision test and boundary

Choose DPO when the available signal is a trustworthy, offline set of matched preferences and the team values a single supervised training loop over active exploration. Compare it with reward-model-plus-PPO using the same base model, reference, preference pairs, and evaluation temperatures; report both reward/KL behavior and task-level human or judge comparisons. Use an online method when new states, reward hacking, or active data collection are central. The source study reaches models up to 6B parameters and offers initial, not definitive, evidence for out-of-distribution generalization. Its GPT-4 win rates are prompt-sensitive, its preference model is an assumption, and the paper leaves reward over-optimization and scaling to much larger models as open questions.

## High-Level Takeaways

- DPO preserves the KL-constrained RLHF target while turning reward fitting and policy optimization into one reference-relative classification loss.
- The log-probability ratio to the reference is the implicit reward; the chosen-versus-rejected margin keeps the update from becoming plain maximum likelihood on preferred text.
- On the paper's tasks, DPO reaches a stronger reward–KL frontier and about 61% TL;DR win rate at temperature 0 versus PPO's 57% best point.
- Offline simplicity comes with offline coverage: DPO cannot discover failures that are absent from its preference pairs, and the reference model shapes the support of the update.
- The reported gains use models up to 6B parameters and GPT-4 or human comparisons with stated prompts; they do not establish universal preference or safety improvement.
