---
title: "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving"
date: '2024-08-01T00:00:00.000Z'
section: paper-shorts
postSlug: inference-scaling-laws-compute-optimal-llm-problem-solving
legacyPath: /paper shorts/2024/08/01/inference-scaling-laws-compute-optimal-llm-problem-solving.html
tags:
  - Inference
  - Scaling Laws
field: "Language Models"
summary: "2024 – Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving"
---

## 2024 – Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving

**Paper:** [Wu et al., arXiv:2408.00724v3](https://arxiv.org/abs/2408.00724v3) · **Project:** [Inference scaling](https://thu-wyz.github.io/inference-scaling/)

## Summary

> Rebase allocates solution-tree expansion according to a process reward model, directing generation toward promising partial answers without spending separate rollouts on value estimation. On MATH500, Llemma-7B with 32 Rebase candidates reaches 46.8% accuracy at a reported $1.48\times10^{14}$ FLOPs, versus 45.5% with 256 independently sampled candidates at $10.0\times10^{14}$ FLOPs. This is a comparison of trained generators and a shared 34B reward model on math tasks, not a universal preference for small models or tree search.

## Core Insights

[Large Language Monkeys](/paper%20shorts/2024/07/31/large-language-monkeys-scaling-inference-compute-with-repeated-sampling.html) exposes the gap between generating and selecting a correct answer. This paper changes the generation distribution itself: use intermediate reward scores to decide where the next reasoning steps should be sampled. The resource-allocation question becomes which model, inference algorithm, and number of generated tokens minimize error at a fixed inference budget.

The source diagram shows where Rebase spends its budget. A higher-scoring partial solution receives more children; a low-scoring branch can receive none. The score is a learned estimate, so the depicted arithmetic example illustrates the intended behavior rather than a correctness guarantee.

![Source Figure 3's tree-expansion panel shows Rebase assigning more continuations to higher-reward reasoning prefixes.](/assets/images/inference-scaling-laws-source-figure-3a.png)
*Fig 1: Rebase distributes a fixed expansion budget across partial solutions using their reward scores. This source panel isolates the branching decision; the accompanying paper panel explains how a process reward model scores each prefix. | source: [Inference Scaling Laws, Figure 3](https://arxiv.org/html/2408.00724v3#S3.F3)*

[Open the figure at full size](/assets/images/inference-scaling-laws-source-figure-3a.png)

At each depth, Rebase subtracts completed answers from the remaining candidate budget. It divides that budget across unfinished prefixes in proportion to $\exp(R_j/T_b)$, rounds the allocations, and samples the next steps. The balancing temperature controls concentration; experiments use $T_b=0.1$. Completed answers are then combined by best-of-$N$ or voting. Unlike independent sampling, earlier reward judgments affect which complete answers can ever reach the selector.

### A strong verifier is part of the system

The main generators include Pythia models, Llemma-7B and 34B, and Mistral-7B. Pythia and Llemma receive full supervised fine-tuning on step-formatted MetaMath; Pythia uses its GSM8K portion. The shared reward model is Llemma-34B fine-tuned for two epochs on Math-Shepherd, with a scalar head scoring each step. A “7B result” therefore does not describe the memory footprint of the complete system.

Evaluation uses GSM8K and MATH500, generation temperature 1.0, a 1,024-token output limit, and a 256-token step limit. The controlled comparison in Table 1 uses weighted voting for both independent sampling and Rebase:

| Generator | Independent sampling: candidates / accuracy | Rebase: candidates / accuracy |
|---|---:|---:|
| Mistral-7B | 256 / 42.8% | 32 / 45.0% |
| Llemma-7B | 256 / 45.5% | 32 / 46.8% |
| Llemma-34B | 64 / 46.7% | 32 / 49.2% |

The reported inference FLOPs fall in all three comparisons. These estimates are not measurements of end-to-end latency or dollars; deployment accounting must include reward-model execution, memory, batching, and sequential search dependencies. Training costs are outside this inference-only objective.

### Changing the sampler changes the voting ceiling

With independent samples from a fixed policy, majority voting converges toward the most probable final answer, whether or not that answer is correct. The paper's proof assumes bounded answer length and a positive probability gap between the leading answers. Weighted voting changes the relevant probability mass through reward scores; it still has a ceiling determined by the fixed generator and reward model. Rebase can shift that ceiling by changing which reasoning paths are sampled.

The ablations show larger Rebase gains on MATH levels 3–5 than on levels 1–2, while GSM8K saturates earlier. The tested MCTS configurations have worse cost–accuracy trade-offs; this bounds the result to those configurations rather than disproving MCTS as a family. Appendix experiments extend the comparison to Llama-3-8B-Instruct and MBPP, but do not establish reliability on open-ended tasks without checkable answers.

My deployment test would hold the complete generator–verifier system and wall-clock budget fixed, then compare Rebase against independent sampling. A reward model that misranks unfamiliar prefixes could spend more compute exploring the wrong branch; the paper's favorable math results do not remove that dependency.

## High-Level Takeaways

- Rebase spends inference compute on reward-guided continuation instead of treating every independent candidate equally.
- The measured benefit combines a small generator with a much larger process reward model; both belong in the deployment budget.
- Voting saturation is a property of the answer distribution and selector, so changing the search policy can change the ceiling.
- The preferred model size depends on the task, inference budget, and search recipe; the fitted Pythia relationship is not a hardware-independent law.
