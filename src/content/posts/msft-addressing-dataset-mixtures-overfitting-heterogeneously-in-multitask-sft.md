---
title: 'mSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT'
date: '2026-03-23T04:00:00.000Z'
section: paper-shorts
postSlug: msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft
legacyPath: >-
  /paper
  shorts/2026/03/23/msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft.html
tags:
  - Other
field: Alignment
summary: mSFT adapts multitask fine-tuning mixtures by tracking which datasets overfit at different rates.
---
## 2026 – mSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT

**arXiv:** [2603.21606](https://arxiv.org/abs/2603.21606)

**Hugging Face:** [Paper page](https://huggingface.co/papers/2603.21606)

**GitHub:** [reiss-koh/msft](https://github.com/reiss-koh/msft)

**Conference:** Preprint

**Summary:** mSFT targets a practical failure mode in multi-task supervised fine-tuning: different datasets learn and overfit at different speeds. A uniform compute budget treats every sub-dataset as if it has the same training dynamics, so easier or faster-learning tasks can start overfitting while harder tasks are still under-trained.

The proposed algorithm makes the mixture overfitting-aware. It trains on the active mixture, identifies the sub-dataset that overfits earliest, reverts to that sub-dataset's best checkpoint, removes it from the active set, and continues training on the remaining tasks. The result is a staged SFT procedure that spends compute where it is still useful instead of forcing every dataset through the same number of updates.

**Why it mattered:** Data mixture tuning is usually treated as a static weighting problem. mSFT reframes it as a training-dynamics problem: the right mixture can change over time because tasks saturate at different rates. That is especially relevant for post-training, where datasets often differ in size, difficulty, quality, and target behavior.

**Evals / Benchmarks:**

| Setting | Result |
| ------- | ------ |
| Baselines | Outperforms 4 baselines |
| Benchmarks | Evaluated across 10 benchmarks |
| Base models | Tested across 6 base models |
| Robustness | Gains hold across dataset sizes, task granularities, and compute budgets |
| Low-compute regime | Can improve performance while reducing training FLOPs |

The implementation also exposes a fairly simple workflow: example mixtures live under `data/`, the default config targets `Qwen/Qwen2.5-3B`, and the reference setup assumes 4 RTX 3090 GPUs with an effective batch size of 64.

**Critiques & limitations:** The paper's strength is that the algorithm is simple and addresses a real post-training nuisance. The main tradeoff is operational complexity. mSFT needs periodic evaluation, overfitting detection, checkpoint management, and staged dataset removal, which makes the training loop less straightforward than ordinary SFT. The method also depends on having evaluation signals that can reliably say when a sub-dataset has peaked.

**Take-home message:** mSFT argues that multi-task SFT should not spend compute uniformly across heterogeneous datasets. If each task overfits on its own schedule, the training loop should notice and adapt.
