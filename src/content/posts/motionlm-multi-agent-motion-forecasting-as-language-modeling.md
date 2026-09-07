---
title: 'MotionLM: Multi-Agent Motion Forecasting as Language Modeling'
date: '2023-09-28T00:00:00.000Z'
section: paper-shorts
postSlug: motionlm-multi-agent-motion-forecasting-as-language-modeling
legacyPath: /paper shorts/2023/09/28/motionlm-multi-agent-motion-forecasting-as-language-modeling.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2023 – MotionLM: Multi-Agent Motion Forecasting as Language Modeling"
---

**arXiv:** [2309.16534](https://arxiv.org/abs/2309.16534)

## Summary

> MotionLM recasts multi-agent forecasting as next-token prediction. It quantizes each agent's future displacement into discrete motion tokens, flattens the agent-time sequence, and trains a causal Transformer with the same likelihood objective used by a language model. Because the sequence is joint, the model can generate several agents' futures in one rollout rather than fitting independent trajectories and trying to reconcile them afterward. The useful distinction is between joint generation and simultaneous prediction: at each future time, the agents' tokens are conditionally independent given the previous timesteps, but every later timestep sees the whole emerging scene. That causal ordering lets a vehicle react to a pedestrian's earlier predicted motion while keeping sampling parallel across agents at the current step.

## Core Insights

### Tokenize displacement, then make interaction an ordering problem

Each trajectory is normalized in the agent frame and converted into 2 Hz displacement actions. The paper uses 13 bins per coordinate, giving 169 Cartesian-product motion tokens. A Verlet wrapper lets a zero action repeat the previous displacement index, which reduces the effective vocabulary for stationary or smoothly moving agents. An 8-second future therefore becomes 16 tokens per agent. The representation turns continuous regression into classification, while making quantization, horizon, and token ordering explicit modeling choices.

![MotionLM's scene encoder, causal decoder, and rollout aggregation pipeline](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-paper-figure.png)
*Fig 1: Scene features condition an autoregressive decoder, which rolls out motion tokens before NMS and k-means reduce many samples to the benchmark's modes. | source: [MotionLM, Figure 2](https://arxiv.org/abs/2309.16534)*

The decoder embeds the token, timestep, and agent position, then applies self-attention over the flattened sequence. A staircase mask exposes all agents' histories up to the previous timestep and blocks future tokens. The factorization is
`p(A_1, ..., A_T | S) = product_t p(A_t | A_<t, S)`,
with `p(A_t | A_<t, S)` factorized across agents. This is a compact way to express an interactive rollout: reaction is causal across time, rather than an unsupported same-timestep dependency.

### Causal structure changes what “conditional” means

The authors compare marginal prediction, temporally causal conditional prediction, and acausal conditioning. On the reported WOMD validation slice, minADE/minFDE/MR/soft-mAP are 0.6069/1.2236/0.1406/0.3951 for marginal, 0.5997/1.2034/0.1377/0.4096 for temporally causal conditional, and 0.5899/1.1804/0.1338/0.4274 for acausal conditioning. The acausal numbers look better because the query agent's full future is exposed. That setting can still be useful when conditioning on a candidate plan, but it does not represent the causal reaction of another agent to that plan. In the paper's supplementary example, conditioning on a trailing vehicle braking makes the acausal model spuriously brake the lead vehicle, whereas the temporally causal model keeps the lead vehicle moving; conditioning on the lead vehicle braking correctly makes the trailing vehicle stop. The causal mask is therefore an intervention interface, not only a regularizer.

![MotionLM's causal Bayesian-network view of joint rollouts](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-source-figure-4.webp)
*Fig 2: A temporally causal rollout lets later agent reactions depend on earlier joint actions; the acausal graph has access to information unavailable at prediction time. | source: [MotionLM, Figure 4](https://arxiv.org/abs/2309.16534)*

### More interaction and more rollouts buy different kinds of quality

On the WOMD interactive test set, MotionLM reports minADE 0.8911, minFDE 2.0067, miss rate 0.4115, and mAP 0.2178, compared with JFP's 0.8817/1.9905/0.4233/0.2050. Its joint prediction overlap is 0.02607, close to JFP's 0.02671 and below Scene Transformer's 0.04336. In the interaction-frequency ablation, ensemble mAP rises from 0.2007 at 0.125 Hz to 0.2150 at 2 Hz; the single-replica result rises from 0.1558 to 0.1687. The model therefore benefits when agents can “see” one another repeatedly during the 8-second rollout.

The sampling curve is separate from the architecture curve. With one to 512 rollouts per replica, ensemble mAP increases from 0.1524 to 0.2150, while the single-replica result increases from 0.0578 to 0.1687. The final reported interactive result uses 512 rollouts per replica and six retained modes after NMS plus k-means; the appendix notes that 32 rollouts already surpass the previous top entry.

## High-Level Takeaways

- MotionLM is a strong test of whether interactive futures can be represented as a discrete sequence: a token is one agent's quantized displacement at one time, and the causal order carries the interaction.
- The model's useful inductive bias is temporal causality. Acausal conditioning can be useful when a candidate future is explicitly supplied, but it should not be read as a causal reaction model; the paper's lead/trailing-car example shows the direction-of-influence failure.
- Interaction frequency and sample count are distinct budgets. Repeated cross-agent attention improves the joint metric, while more rollouts improve mode coverage after aggregation.
- Quantization makes training simple and scalable, but vocabulary resolution, token horizon, and rollout latency become part of the planner's operating envelope.
