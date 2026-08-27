---
title: 'Understanding Reasoning from Pretraining to Post-Training'
date: '2026-07-17T00:00:00.000Z'
section: paper-shorts
postSlug: understanding-reasoning-from-pretraining-to-post-training
legacyPath: /paper shorts/2026/07/17/understanding-reasoning-from-pretraining-to-post-training.html
tags:
  - Reinforcement Learning
  - Scaling Laws
  - Reasoning
field: 'Alignment & Post-Training'
topics:
  - learning
  - language-systems
summary: '2026 – Understanding Reasoning from Pretraining to Post-Training'
---

## 2026 – Understanding Reasoning from Pretraining to Post-Training

**arXiv:** [2607.16097](https://arxiv.org/abs/2607.16097)

**Code:** [pavelslab-nyu/pre2post-chess](https://github.com/pavelslab-nyu/pre2post-chess)

**Models and data:** [Pre2Post Chess collection](https://huggingface.co/collections/pavelslab-nyu/pre2post-chess)

## Summary

> Pretraining and reinforcement learning are usually scaled in separate experiments, even though every reasoning policy presented to RL inherits a particular prior. This paper builds a controlled chess analogue of the full language-model pipeline—pretraining on human games, supervised fine-tuning on synthetic search traces, then RL on verifiable puzzles—and asks how the first stage predicts returns in the last.

## Core Insights

Across 36 pretraining-to-RL runs, two different pretraining properties predict two different parts of the local RL curve. Lower held-out pretraining loss predicts the pass@1 level reached at a fixed RL compute, while more pretraining tokens predict a steeper improvement per decade of RL compute. This is useful evidence for allocating compute, but it is a fitted local relationship on an intermediate-difficulty chess benchmark, not a universal scaling law.

The study pretrains decoder-only Transformers from 5M to 1B parameters on decontaminated Lichess games. A proposal model then generates candidate continuations for chess positions; the continuations are merged into a serialized search tree for supervised fine-tuning. RL operates on 156,000 puzzles with a binary verifiable reward: the model must choose the unique correct move sequence while the environment supplies opponent moves.

![Controlled chess pipeline from human-game pretraining through synthetic reasoning SFT and verifiable-reward RL](/assets/images/pretraining-posttraining-overview.png)
*Fig 1: Chess exposes the whole pretraining–SFT–RL pipeline while keeping actions and correctness inspectable. | source: [paper](https://arxiv.org/abs/2607.16097)*

![Figure 3 from Understanding Reasoning from Pretraining to Post-Training](/assets/images/understanding-reasoning-from-pretraining-to-post-training-source-figure-3.webp)
*Fig 2: Pretraining properties predict local RL scaling behavior. (a), the fitted post-RL performance (pass@1 metric) at a reference RL compute level, versus pretraining validation loss. Each curve corresponds to a different reference compute level; the fit tightens as RL compute increases ( to ). (b), the local slope measuring performance gain per decade of RL compute, versus pretraining tokens. | source: [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097)*

![Figure 5 from Understanding Reasoning from Pretraining to Post-Training](/assets/images/understanding-reasoning-from-pretraining-to-post-training-source-figure-5.webp)
*Fig 3: RL reshapes the move policy in qualitatively different ways across puzzle difficulty. Each panel shows the proportion of puzzle states assigned to one policy-update category (Table 14 ) across difficulty bins B1-B5, at RL training steps until 750. | source: [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097)*


For the joint analysis, the authors use 20M, 50M, 200M, and 680M checkpoints from pretraining compute sweeps and fit each run’s pass@1 reward as a linear function of log RL compute. The reference reward at $10^{20}$ RL FLOPs is strongly ordered by pretraining loss: Spearman correlation tightens from -0.93 at $10^{16}$ to -0.99 at $10^{20}$ reference FLOPs. The fitted slope correlates with log pretraining tokens at Pearson $r=0.84$; a joint token-and-model-size fit reaches $R^2=0.84$.

_Left: lower pretraining loss predicts higher fitted reward at fixed RL compute. Middle and right: more tokens, with a smaller model-size correction, predict the local RL slope. Figure 3 in the source: [paper](https://arxiv.org/abs/2607.16097)_
_Left: lower pretraining loss predicts higher fitted reward at fixed RL compute. Middle and right: more tokens, with a smaller model-size correction, predict the local RL slope. source: Figure 3 in the [paper](https://arxiv.org/abs/2607.16097)._

Combining these regressions with a Chinchilla-style pretraining-loss model produces a simulated compute frontier. Within the fitted range, the estimated optimal RL share rises from about 20% at 50M parameters to 28% at 680M, while the selected pretraining token counts remain close to Chinchilla allocation. This is an extrapolated recipe derived from the same local fit; it should guide a prospective sweep, not substitute for one.

The mechanism analysis adds an important counterweight to the aggregate curve. RL is not a single global sharpening temperature. On easy puzzles, it mostly amplifies a correct move already near the top of the SFT distribution. On harder puzzles, it sometimes promotes a correct move whose initial probability is below 0.05 into the top three, but it increasingly reinforces the leading wrong move as well. That mixture helps explain why pass@1 can improve without a consistent pass@$k$ gain.

_With increasing difficulty, correct-mode amplification declines and wrong-mode amplification rises; genuine tail discovery occurs, but remains relatively rare. Figure 5 in the source: [paper](https://arxiv.org/abs/2607.16097)_
_With increasing difficulty, correct-mode amplification declines and wrong-mode amplification rises; genuine tail discovery occurs, but remains relatively rare. source: Figure 5 in the [paper](https://arxiv.org/abs/2607.16097)._

| Finding | Evidence | Boundary |
| --- | --- | --- |
| Pretraining quality predicts post-RL level | Reference reward vs pretraining loss reaches Spearman $\rho=-0.99$ at the largest reference compute | The fitted level is extrapolated for runs that stop before the reference point. |
| Data exposure predicts RL slope | Log tokens vs slope has Pearson $r=0.84$; joint fit $R^2=0.84$ | Easy benchmarks saturate and compress the measured slope. |
| RL does more than sharpen | Some hard-puzzle correct moves rise from below 0.05 probability into the top three | Wrong-mode amplification also grows with difficulty. |
| Pattern appears in math | Fourteen 1B OLMo-2 checkpoints spanning 10B–200B tokens show the same ordering | One model size and one training trajectory make this qualitative transfer evidence. |

The math case study uses a fixed 1B OLMo-2 architecture, checkpoints from 10B to 200B math-heavy pretraining tokens, one epoch of NuminaMath-CoT SFT, and RL on a 24,900-problem mixture. Lower pretraining loss again orders the fitted post-RL level, and longer pretraining correlates with a steeper local slope. Because model size, corpus, and checkpoint ancestry are not independently varied, this supports plausibility beyond chess rather than validating the full compute-allocation law for language models.

## High-Level Takeaways

- This paper informs whether a fixed reasoning-training budget should buy a stronger pretrained prior or more RL. The result rejects a one-number answer. Pretraining loss predicts the level from which RL can operate, data exposure predicts the observed rate of improvement, and the estimated optimal RL fraction rises with total compute inside the studied regime.
- The expensive next decision should therefore be made with a small joint sweep, not a pretraining-only scaling curve. Train several checkpoints that vary tokens and model size independently, run matched RL budgets on a non-saturating target benchmark, and fit the interaction before committing the full run. The paper’s conclusion would weaken if independently seeded language-model checkpoints with matched loss but different token histories showed the same RL slope, or if the relationship vanished under leave-one-model-family-out prediction.
- At 10× scale, the principal bottlenecks are experimental coverage and reward diversity. A single pass@1 curve can hide saturation, reward hacking, and wrong-mode reinforcement. Compute allocation should be evaluated against pass@$k$, calibration, held-out task families, and failure severity—not only the metric optimized by RL.
- The work joins two previously separate scaling questions: what pretraining buys and how quickly a pretrained policy converts verifiable experience into downstream performance.
- Chess supplies exact actions and cheap verifiers but is far smaller and more structured than natural-language reasoning. The law is local to the measured compute range, the frontier is model-based extrapolation, and the math extension follows one 1B pretraining trajectory.
- Pretraining determines more than an RL starting point: its loss predicts the attainable local level, while its data exposure predicts how quickly RL improves—but only inside a measured, non-saturated regime.
