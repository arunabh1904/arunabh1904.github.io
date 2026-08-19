---
title: 'AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models'
date: '2026-03-10T00:00:00.000Z'
section: paper-shorts
postSlug: ar-vla-true-autoregressive-action-expert-for-vision-language-action-models
legacyPath: /paper shorts/2026/07/24/ar-vla-true-autoregressive-action-expert-for-vision-language-action-models.html
tags:
  - VLA
  - Autoregressive Models
  - Efficient Inference
field: 'Vision-Language-Action & Robotics'
topics:
  - embodied
  - learning
  - multimodal
summary: '2026 – AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models'
---

## 2026 – AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models

**arXiv:** [2603.10126](https://arxiv.org/abs/2603.10126)  
**Project, code, and videos:** [arvla.insait.ai](https://arvla.insait.ai/)

## Summary

> Most continuous-action VLAs are temporally reactive even when their action decoder is called autoregressive: they regenerate an action chunk from the latest observation, then discard the decoder state. AR-VLA instead keeps an action expert alive across control steps. It predicts one continuous action vector at a time from a rolling proprioceptive history while conditioning on the most recent vision-language features, so slow perception can refresh without resetting fast motor memory.

## Core Insights

That distinction matters in the paper’s matched BridgeV2 experiment. With the same PaliGemma-3B backbone and roughly 300M-parameter action module, AR-VLA reaches 61.5% average success in SIMPLER, versus 49.0% for the reproduced FAST-token head and 51.0% for the reproduced flow-matching head. Its advantage is not universal: Diffusion Policy remains better on PushT, and ACT is better on human-demonstration ALOHA insertion.

![AR-VLA framework re-anchoring vision-language keys into a rolling hybrid cache used by an autoregressive action expert](/assets/images/ar-vla-true-autoregressive-action-expert-for-vision-language-action-models-paper-figure.png)
_Figure 2 shows the persistent-controller mechanism: refreshed visual-language keys are timestamped onto the action timeline, while the action expert retains rolling kinematic history and predicts one future action at a time. Source: [AR-VLA](https://arxiv.org/abs/2603.10126)._

The action expert is a causal Transformer over continuous robot states and actions. A linear layer maps each action vector to one token, and a deterministic regression head maps the next hidden state back to an action. Its hybrid key-value cache has two update rules: a token-wise FIFO retains recent proprioception and executed actions, while a single visual-language block is replaced whenever the backbone produces a new observation embedding.

Dynamic Temporal Re-anchoring makes those streams temporally comparable. Visual-language keys receive the action-timeline index at which their image was captured; action tokens keep their ordinary sequential indices. Rotary position encoding then makes attention depend on relative visual staleness rather than an absolute deployment timestep. A frame captured five steps ago therefore presents the same positional relationship at step 25 and step 500, narrowing the gap between short training windows and long rollouts.

Training separates motion modeling from visual grounding. Phase one learns next-action prediction from action trajectories alone. Phase two attaches the VLM and predicts future actions from an anchored visual prefix plus history. Independent random masks corrupt future-history tokens during training so the expert cannot ignore vision and merely extrapolate its own motion.

| Evidence | AR-VLA | Comparison | Qualification |
| --- | ---: | ---: | --- |
| SIMPLER average success | 61.5% | CogACT: 52.1%; matched FAST: 49.0%; matched flow matching: 51.0% | Four WidowX tasks; BridgeV2 training |
| Real WidowX average success | 89% | Reported as best among tested policies | Challenging zero-shot tasks after all policies pass an easy in-distribution control |
| PushT success | 60.4% | Diffusion Policy: 65.2%; ACT: 52.0% | AR is competitive, not best |
| ALOHA cube, human demonstrations | 67.3% | ACT: 50.0%; Diffusion Policy: 10.0% | Specialist setting |
| ALOHA insertion, human demonstrations | 6.7% | ACT: 20.0%; Diffusion Policy: 1.7% | Reveals task-dependent failure |
| Effective latency per action | 46.25 ms | OpenVLA: 321.72 ms; flow matching: 84.26 ms | Model-side comparison, not full robot-loop latency |

The ablations support the memory mechanism more directly than the headline benchmarks. Removing action-only pretraining drops SIMPLER success from 61.5% to 37.5% at the standard training budget; doubling the no-pretraining budget recovers only to 54.2%. With no historical dropout, validation action error is low but task success is zero, showing that pure next-action fit can produce a history shortcut. Replacing temporal anchoring with static visual positions yields 3.1% success, while a 20-step history reaches 61.5%; longer 40-step history slips to 59.4%.

## High-Level Takeaways

- AR-VLA informs a system-level choice: should temporal continuity live inside a persistent controller, or be approximated by repeatedly asking a snapshot-conditioned model for action chunks? The evidence favors persistent motor state when perception and control run at different frequencies. The hybrid cache gives a concrete interface—refresh semantic context as a block, retain kinematics as a stream—and the pretraining result suggests that action-only trajectories can improve the controller before expensive vision-language alignment.
- The strongest causal test would compare persistent autoregression with a recurrent or cached flow-matching controller at equal backbone, parameters, data, control rate, and wall-clock latency. Current comparisons change both the action objective and the memory structure. The specialist rows already show that autoregression is not automatically superior: chunked methods win on PushT and one insertion setting.
- At ten times the rollout length, cache policy and error accumulation become the likely bottlenecks. The action expert conditions on its own executed history, so small out-of-distribution mistakes can compound. Visual input is still a sequence of replaceable snapshots rather than a persistent visual memory. The decisive next test is a partially observable, long-horizon physical benchmark with delayed perception, controlled cache lengths, recovery after induced errors, and several seeds.
- AR-VLA turns the VLA action head into a stateful high-frequency process rather than a stateless chunk generator.
- The strongest controlled results use a small set of simulated and tabletop tasks. Comparisons do not isolate memory from objective under a fully matched recurrent baseline, and the paper notes error accumulation, possible damage to VLM priors without insulation, and snapshot-based visual processing.
- Preserve action history across control steps and timestamp slow visual context; do not assume that calling a within-chunk decoder “autoregressive” gives a robot temporal memory.
