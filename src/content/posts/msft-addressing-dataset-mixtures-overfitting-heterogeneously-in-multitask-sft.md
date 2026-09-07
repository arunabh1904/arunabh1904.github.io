---
title: 'mSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT'
date: '2026-03-23T00:00:00.000Z'
section: paper-shorts
postSlug: msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft
legacyPath: >-
  /paper
  shorts/2026/03/23/msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft.html
tags:
  - Other
field: 'Alignment & Post-Training'
summary: "2026 – mSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT"
---
## 2026 – mSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT

**arXiv:** [2603.21606](https://arxiv.org/abs/2603.21606)

**Hugging Face:** [Paper page](https://huggingface.co/papers/2603.21606)

**GitHub:** [reiss-koh/msft](https://github.com/reiss-koh/msft)

**Conference:** Preprint

## Summary

> MSFT studies multi-task supervised fine-tuning when sub-datasets learn and overfit at different rates. A fixed mixture can keep updating an easy task after it has peaked while harder tasks are still under-trained. MSFT repeatedly rolls out training on the active mixture, identifies the earliest overfitting sub-dataset, rolls back to its best checkpoint, excludes it, and continues. The paper reports gains across six base models, ten benchmarks, dataset sizes, and task granularities.

## Core Insights

### One global stopping point is already a mixture decision

Let $D_i$ be a sub-dataset and let

$$
c_i^*=\arg\max_c \operatorname{Metric}(\theta_c;D_i^{\mathrm{test}})
$$

be the compute at which its held-out metric peaks. Standard SFT sets every task’s compute to the same $c_{\mathrm{global}}$. If $c_{\mathrm{global}}>c_i^*$, task $i$ is overfit; if it is below $c_i^*$, a slower task is under-trained.

The Qwen3 8B experiment makes this mismatch concrete. It uses ten sub-datasets and shows that their absolute peak-epoch offsets from the overall mixture range from 0 for CommonsenseQA to 5 for HellaSwag; the average absolute difference between each task’s peak and the mixture’s peak is 1.93 epochs:

![Test-accuracy curves and peak-epoch offsets for ten sub-datasets in Qwen3 8B SFT](/assets/images/msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft-source-figure-2.png)
*Fig 1: Test-set training curves and absolute peak-epoch offsets for ten Qwen3 8B sub-datasets. | source: [MSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT, Figure 2](https://arxiv.org/abs/2603.21606)*

This is more than a case for choosing a better global epoch. The update at each step is a weighted sum of sub-dataset gradients. Once one task begins supplying over-specialized gradients, continuing to sample it changes the trajectory seen by every other task.

### A precomputed exclusion schedule becomes stale

The obvious fix is to run one full-mixture sweep, record each $c_i^*$, then exclude tasks at those times in a second run. The paper calls this single-rollout searched SFT. It fails because the schedule changes the gradients that produced the original peaks.

The authors test this with ten equal-weighted sub-datasets of 1,800 examples each. After the first task overfits, they branch the run: one branch keeps the full mixture and the other removes that task. The remaining tasks’ optimal compute shifts. Even removing one tenth of the data changes later peak locations; across the reported model families and scales, the mean absolute shift is 0.91 epochs:

![Optimal-compute shifts after excluding one sub-dataset](/assets/images/msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft-source-figure-3.webp)
*Fig 2: Changes in each remaining benchmark’s optimal compute after one sub-dataset is removed; this is panel (a) of the source figure. | source: [MSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT, Figure 3](https://arxiv.org/abs/2603.21606)*

The sign matters. Some tasks need more compute after the early task is removed; others peak earlier. A static table of stopping points cannot capture this interaction.

### MSFT aligns the search with the training trajectory

MSFT keeps an exclusion set $E$ and starts from the base model. It rolls out the current active mixture $D\setminus E$ for a compute budget $C$, recorded in fractional-epoch increments. It evaluates every active sub-dataset, finds the one with the smallest peak compute, adds it to $E$, and rolls back to that task’s peak checkpoint. The next rollout starts from that checkpoint with the remaining active mixture. If no task overfits inside the current budget, the algorithm advances to the end of the rollout without exclusion.

This roll-out/roll-back loop is the key design choice. Every later peak is searched under the gradient mixture that will actually be used after earlier exclusions. It also explains the operational cost: evaluation must be tracked per sub-dataset, intermediate checkpoints must be retained, and the training job must be able to resume from the selected rollback point.

### The gains are broad, while the compute story depends on the budget

Across six base models—OLMo 2 1B, Qwen2.5 0.5B/1.5B/3B/7B, and Qwen3 8B—and ten benchmarks, Table 2 reports average accuracy of 63.7 for MSFT, 62.5 for IES, 62.1 for DynamixSFT, and 61.9 for standard SFT. In the ablation against the two single-rollout variants, MSFT also reaches 63.7 versus 63.4 for SRO SFT and 62.1 for Soft SRO SFT.

The budget sweep shows the most concrete systems tradeoff:

![Accuracy and FLOPs changes for MSFT across compute budgets](/assets/images/msft-addressing-dataset-mixtures-overfitting-heterogeneously-in-multitask-sft-source-figure-6.png)
*Fig 3: Accuracy, rollout overhead, and FLOPs changes across MSFT compute budgets. | source: [MSFT: Addressing Dataset Mixtures Overfitting Heterogeneously in Multi-task SFT, Figure 6](https://arxiv.org/abs/2603.21606)*

The same analysis reports robustness across 9K, 18K, and 27K dataset mixtures with 5, 10, and 15 tasks, and a +5.4% average improvement over SFT in that study. The method is therefore making a claim about a control loop, not just about one hand-tuned mix. Its reliance on task metrics remains the main practical boundary. Section 4.1 evaluates each benchmark test set every quarter epoch and reports the best checkpoint, so these results include selection on the reported benchmarks. A deployment study should use separate validation data for exclusion and rollback, then evaluate once on an untouched test set. A noisy or misaligned selection metric can exclude the wrong dataset.

## High-Level Takeaways

- The relevant control variable in multi-task SFT is the active mixture over time, not only its initial sampling weights. Different datasets can peak at different compute levels even when they share a model and optimizer.
- A one-shot schedule is insufficient because excluding a dataset changes the gradient trajectory and moves the remaining peaks. MSFT’s value comes from re-searching after every rollback.
- The reported 63.7 average accuracy and the $C=1$ efficiency result are tied to the paper’s ten-task benchmark suite, test-set checkpoint-selection protocol, and FLOPs accounting. They support the mechanism without proving that every mixture benefits.
- Use MSFT when per-dataset evaluation is reliable and checkpoint storage is available. If a task has no useful held-out metric, the algorithm has no principled signal for when to exclude it.
