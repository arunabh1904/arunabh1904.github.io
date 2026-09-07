---
title: 'VisualPRM: An Effective Process Reward Model for Multimodal Reasoning'
date: '2025-03-13T00:00:00.000Z'
section: paper-shorts
postSlug: visualprm-process-reward-model-for-multimodal-reasoning
legacyPath: /paper shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html
tags:
  - Multimodal Reasoning
  - Reward Models
field: 'Alignment & Post-Training'
summary: "2025 – VisualPRM: An Effective Process Reward Model for Multimodal Reasoning"
---

## 2025 – VisualPRM: An Effective Process Reward Model for Multimodal Reasoning

**arXiv:** [2503.10291](https://arxiv.org/abs/2503.10291)

## Summary

> VisualPRM is an 8B process reward model that judges intermediate steps in multimodal reasoning. It uses about 400,000 automatically supervised process examples, introduces a human-labeled VisualProcessBench for critic evaluation, and improves Best-of-$N$ selection across seven multimodal benchmarks. The paper’s key contribution is to evaluate the critic as a step-level model, rather than treating final-answer accuracy as enough evidence that the critic works.

## Core Insights

### Automatic process supervision turns a final answer into step labels

A final correctness label cannot say where a visual solution went wrong. VisualPRM400K addresses this with Monte Carlo process supervision. For a solution step $s_i$, the authors sample continuations and estimate its expected accuracy,

$$
mc_i=\frac{\text{number of correct sampled completions}}
{\text{number of sampled completions}}.
$$

A step is labeled correct when $mc_i>0$. The image, question, previous steps, and current step are formatted as a multi-turn conversation; the model predicts the correctness of every step. This is a noisy but scalable recoverability label: at least one sampled continuation reached the correct answer. It is not a direct check of local logical truth. An incorrect step can be repaired later, while a finite sample can miss a valid continuation.

The data and evaluation sets answer different questions, which the source figure makes explicit:

![Examples from VisualPRM400K automatic supervision and VisualProcessBench human step labels](/assets/images/visualprm-process-reward-model-for-multimodal-reasoning-source-figure-2.webp)
*Fig 1: VisualPRM400K attaches Monte Carlo expected-accuracy labels to generated solution steps, while VisualProcessBench contains human correctness judgments for evaluating critics. | source: [VisualPRM: An Effective Process Reward Model for Multimodal Reasoning, Figure 2](https://arxiv.org/abs/2503.10291)*

VisualProcessBench contains 2,866 samples and 26,950 human step labels drawn from MMMU, MathVista, MathVision, MathVerse’s Vision-Only split, DynaMath, and WeMath. The benchmark includes incorrect steps throughout a solution, not only the first error, and reports macro F1 over correct and incorrect step judgments, with an overall micro average across data sources. That makes it a test of critic discrimination rather than a proxy for final answer accuracy.

### Value and advantage labels ask different process questions

The paper compares two labels. A value-based process reward model uses positive continuation success as its correctness target. An advantage-based model scores how the step changes expected accuracy relative to the preceding step. The distinction matters: a step may be correct in isolation while failing to improve the solution, or it may be a neutral transition that preserves a good trajectory.

The source schematic below compares the two process-modeling choices:

![Value-based and advantage-based process reward modeling](/assets/images/visualprm-process-reward-model-for-multimodal-reasoning-source-figure-3.png)
*Fig 2: Value-based and advantage-based process reward modeling compare expected correctness with step-to-step improvement. | source: [VisualPRM: An Effective Process Reward Model for Multimodal Reasoning, Figure 3](https://arxiv.org/abs/2503.10291)*

At inference, VisualPRM reads all steps and produces a score for each in one forward pass by using a “+” placeholder and the probability of generating it. For Best-of-$N$, a policy samples candidate solutions at temperature 0.7 and the critic aggregates their step scores to select one response. The main experiments use $N=8$ and evaluate MMMU, MathVista, MathVision, MathVerse-VO, DynaMath, WeMath, and LogicVista.

### The critic improves selection even for large policy models

The source Figure 1 compares pass@1 with Best-of-8 under an open-source critic and VisualPRM:

![Best-of-8 multimodal reasoning results with different policy and critic models](/assets/images/visualprm-process-reward-model-for-multimodal-reasoning-source-figure-1.webp)
*Fig 3: Best-of-8 results with an open-source critic and VisualPRM across policy-model families and scales. | source: [VisualPRM: An Effective Process Reward Model for Multimodal Reasoning, Figure 1](https://arxiv.org/abs/2503.10291)*

The reported Best-of-8 gains are 8.0 points for MiniCPM-V2.6, 3.7 for QwenVL2.5-7B, 8.4 for InternVL2.5-8B, 8.9 for InternVL2.5-26B, 6.3 for InternVL2.5-38B, and 5.9 for InternVL2.5-78B. The critic therefore adds value through ranking, not by making the policy generate a better single sample.

The human benchmark explains why the model is useful as a critic. Random guessing scores 50.0 overall on VisualProcessBench; VisualPRM reaches 62.0. InternVL2.5-8B reaches 76.8 F1 on positive steps but only 19.2 on negative steps, which shows the failure mode of a generally capable MLLM used as a process judge: it tends to call most steps correct. VisualPRM’s advantage is discrimination between a plausible step and an actually erroneous one.

### Ablations say aggregation matters as much as the critic

For InternVL2.5-8B, VisualPRM beats self-consistency by 2.4 points and an outcome reward model by 1.5 points at Best-of-8. At $N=128$, those gaps widen to 3.1 and 4.3 points. The outcome critic improves at first but is not consistently monotonic as $N$ grows.

The modeling ablations explain the result. Value-based PRMs outperform advantage-based PRMs in both Best-of-$N$ and VisualProcessBench. Averaging step scores beats selecting the maximum, because a single high-scored early step can hide a later error; averaging acts like an ensemble over the whole solution. Supervising all steps also slightly beats stopping supervision at the first incorrect step. The automatic labels are imperfect, so retaining later steps gives the model more signal about where solutions diverge.

## High-Level Takeaways

- Evaluate a process critic twice: first on localized human step labels, then on whether it improves Best-of-$N$ selection. A good final-answer score alone cannot distinguish a useful critic from a lucky sampler.
- Process reward is not one scalar design. Value labels, advantage labels, score aggregation, and supervision cutoff change what the critic prefers; the paper’s best combination is value-based, all-step supervision with average aggregation.
- More test-time samples are useful only when the critic can rank them. VisualPRM keeps improving from $N=8$ to $N=128$, while the outcome critic saturates or declines in the same comparison.
- The source labels are generated from multimodal reasoning traces and human judgments, so the result establishes step-level selection for these benchmarks. A physical process would need labels for temporal state, contact, and irreversible consequences before this critic pattern could be trusted.
