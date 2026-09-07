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

> Pretraining and reinforcement learning are usually scaled in separate experiments, even though every policy presented to RL inherits a particular prior. This paper builds a controlled chess analogue of the language-model pipeline—pretraining on human games, supervised fine-tuning on synthetic search traces, then RL on verifiable puzzles—and asks which pretraining properties predict the level and slope of the later RL curve.

## Core Insights

### Chess makes the pretraining-to-RL dependency measurable

The testbed keeps the action space and verifier explicit. Decoder-only Transformers from 5M to 1B parameters are pretrained on 54B tokens of 2022 Lichess Blitz and Rapid games. The data splits are disjoint at the board-position level, which is the paper’s contamination control. A proposal model samples continuations from a puzzle position; common prefixes are merged into a search tree and serialized for SFT. RL then uses GRPO on 156,000 quality-filtered puzzles with a binary reward: the full line must match the unique correct solution.

The evaluation contains 1,480 tactical puzzles divided into difficulty bins B1–B5. The aggregate pass@k analysis uses B1–B4 because B5 is rarely solved; B5 remains useful for inspecting failure mechanisms. This design asks a clean question: does the prior learned from human games change how efficiently a policy converts verifiable RL experience into correct moves?

![Pretraining, synthetic-trace SFT, and verifiable-RL pipeline](/assets/images/understanding-reasoning-from-pretraining-to-post-training-source-figure-1.png)
*Fig 1: Human-game pretraining, synthetic-trace SFT, and verifiable RL are connected to the paper’s scaling and policy analyses. | source: [Understanding Reasoning from Pretraining to Post-Training, Figure 1](https://arxiv.org/abs/2607.16097)*

### RL compute has two inputs from pretraining: level and slope

The authors fit each local RL curve as

$$
R_{N,T}(C)=R^{\mathrm{ref}}_{N,T}
+B_{N,T}\left(\log_{10}C-\log_{10}C_{\mathrm{ref}}\right),
$$

where $R^{\mathrm{ref}}_{N,T}$ is the fitted pass@1 reward at a reference RL compute and $B_{N,T}$ is the gain per decade of RL compute. The fit uses 36 pretraining-to-RL runs and focuses on 20M, 50M, 200M, and 680M models for the joint analysis. B1 and B2 saturate too quickly, so the scaling analysis uses the intermediate B3–B4 bins.

The source Figure 3 separates the two correlations instead of collapsing them into a single “reasoning scale”:

![Pretraining loss predicts the fitted RL level, while token exposure predicts the local RL slope](/assets/images/understanding-reasoning-from-pretraining-to-post-training-source-figure-3.webp)
*Fig 2: Pretraining loss predicts fitted RL level, token exposure predicts local RL slope, and the joint fit is shown at right. | source: [Understanding Reasoning from Pretraining to Post-Training, Figure 3](https://arxiv.org/abs/2607.16097)*

The fitted joint law uses an exponential function of pretraining loss for the reference level and a log-linear function of model size and token count for the slope. In plain terms, lower loss gives RL a better starting level, while more token exposure is associated with faster local improvement. That interpretation is more informative than saying that “bigger models reason better,” because size alone is not the strongest variable in the slope fit.

### The compute frontier is an extrapolation from a local law

The authors combine the fitted relationship with a Chinchilla-style pretraining-loss surface and evaluate 400 candidate pretraining/RL splits over total budgets from $10^{17}$ to $10^{21}$ FLOPs. Within that modeled range, the estimated optimal RL fraction rises from about 20% at 50M parameters to about 28% at 680M, while the pretraining token allocation stays close to the Chinchilla allocation.

The important word is estimated. The frontier is generated from local linear-in-log-$C$ fits, not from running every candidate split. It is a useful way to choose a prospective sweep; it is not an independently validated universal allocation rule. Its accuracy will degrade once pass@1 saturates, once a verifier changes the learning dynamics, or once language-model transfer changes the relation between loss and recoverable reasoning.

### RL reshapes probability mass in several ways

A single curve does not reveal what improved. The paper classifies move-probability changes into ground-truth amplification, tail discovery, and wrong-mode amplification. Ground-truth amplification means the correct move was already among the leading candidates and becomes more likely. Tail discovery promotes a correct move whose initial probability was below 0.05 into the top three. Wrong-mode amplification increases the leading wrong move while the correct move remains outside the top three.

The difficulty breakdown makes the tradeoff concrete:

![Policy-update categories across chess puzzle difficulty bins](/assets/images/understanding-reasoning-from-pretraining-to-post-training-source-figure-5.webp)
*Fig 3: RL update categories across puzzle difficulty: ground-truth amplification, tail discovery, and wrong-mode amplification. | source: [Understanding Reasoning from Pretraining to Post-Training, Figure 5](https://arxiv.org/abs/2607.16097)*

This explains why pass@1 can keep improving while pass@16 is mixed or degrades for larger models under the fixed RL budgets. RL can make one answer more decisive without making the candidate distribution broadly better. It also widens search and branching in the traces, while maximum reasoning depth stays roughly flat; the model becomes better at proposing and committing to useful branches, not simply at writing longer continuations.

### The math result is a transfer check, not a second scaling law

The paper repeats the comparison on one 1B OLMo-2 trajectory: 14 checkpoints from 10B to 200B math-heavy pretraining tokens, one epoch of NuminaMath-CoT SFT, and RL on a 24,900-problem GSM8K/MATH/DeepScaler mixture. Lower pretraining loss again predicts a higher fitted post-RL level, and more tokens correlate with a steeper local slope. Because model size and checkpoint ancestry are not independently varied in this extension, it supports qualitative transfer beyond chess rather than validating the full compute-allocation model for language.

## High-Level Takeaways

- Treat pretraining loss and token exposure as different predictors. In this testbed, loss orders the level from which RL starts, while tokens order the local rate at which RL improves.
- Use a small joint sweep before committing a large reasoning-training budget. Vary model size and token count independently, run matched RL budgets on a non-saturating difficulty band, and fit the interaction instead of extrapolating from pretraining alone.
- Inspect pass@1 together with pass@k and policy-shift categories. A higher top answer can come from useful tail discovery or from concentrating probability on a wrong mode.
- The 20% to 28% RL-share result is a modeled frontier inside the measured regime. It is most useful as a sweep proposal, with the chess verifier, B3–B4 focus, finite model sizes, and one-trajectory math transfer treated as explicit limits.
