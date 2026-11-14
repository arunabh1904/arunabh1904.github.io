---
title: "Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation"
date: '2026-05-20T00:00:00.000Z'
section: paper-shorts
postSlug: advantage-collapse-in-group-relative-policy-optimization
legacyPath: /paper shorts/2026/05/20/advantage-collapse-in-group-relative-policy-optimization.html
tags:
  - Reinforcement Learning
  - GRPO
field: "Alignment & Post-Training"
summary: "2026 – Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation"
---

## 2026 – Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation

**Paper:** [He et al., arXiv:2605.21125v2](https://arxiv.org/abs/2605.21125v2) · **Code and data:** [Advantage-Collapse-Rate](https://github.com/hexixiang/Advantage-Collapse-Rate)

## Summary

> AVSPO changes GRPO's reward normalization so that all-correct and all-incorrect groups can contribute updates without generating additional responses. On Qwen2.5-Math-1.5B, the paper reports MATH-500 accuracy rising from 58.6% to 67.2% after 500 training steps, alongside a lower advantage-collapse rate. The intervention introduces a deliberately biased update; creating nonzero advantages is not the same as discovering a correct solution or proving that every update improves the policy.

## Core Insights

GRPO compares rewards within a group of responses to the same prompt. If every reward is identical, subtracting the group mean produces zero advantages. A group of eight correct answers and a group of eight wrong answers therefore both lose their reward-driven policy-gradient contribution. This statement concerns the group-relative term; other regularizers could still produce gradients. The paper's experiments set the reference KL penalty to zero.

The Advantage Collapse Rate, ACR, measures the fraction of prompt groups whose reward standard deviation falls below a small threshold. It is a useful utilization diagnostic, not a complete quality metric. All-correct collapse can reflect a solved task, and lower collapse obtained through noisier sampling need not improve accuracy.

### Virtual rewards shift the baseline without adding trajectories

When batch-level ACR exceeds an adaptive threshold, AVSPO augments collapsed groups with scalar reward values. For all-correct groups it adds values below one, producing positive advantages for the real responses. For all-wrong groups it adds small positive anchors, producing negative advantages for the real responses. Virtual rewards affect only the mean and standard deviation; there is no virtual response whose tokens enter the policy loss.

The source diagram locates this intervention between reward computation and the policy update. Its “Reward Model” box represents the reward interface; the reported math experiments actually use deterministic final-answer parsing and symbolic equivalence checks.

![Source Figure 2 shows ACR monitoring and virtual reward injection before recomputing advantages for real sampled responses.](/assets/images/advantage-collapse-source-figure-2.png)
*Fig 1: AVSPO modifies reward statistics after rollout generation. Virtual scalar rewards create contrast for homogeneous groups, while only real response tokens contribute to the policy objective; the diagram's reward interface is deterministic verification in the math experiments. | source: [Advantage Collapse, Figure 2](https://arxiv.org/html/2605.21125v2#S0.F2)*

[Open the figure at full size](/assets/images/advantage-collapse-source-figure-2.png)

The number of virtual rewards is bounded by the real group size and increases with observed collapse. The default uses a square-root dependence on ACR and a 0.1 anchor for all-wrong groups. This suppresses sampled failures rather than supplying an unseen successful action. Appendix B derives conditional update directions and bias bounds under simplified on-policy assumptions; its convergence statement requires smoothness, bounded gradient noise, and control of the introduced bias. It is not an unconditional guarantee for clipped neural-network training.

### The experiment isolates a small, selected training regime

Training uses 500 MATH training-split problems selected for a 30–50% success rate under Qwen2.5-Math-1.5B. This model-relative “Level 3” is distinct from the benchmark's human difficulty labels. Runs use eight responses per prompt, temperature 1.0, a 1,024-token limit, and 500 optimizer steps on eight A800 GPUs. The evaluation MATH-500 examples come from the test split.

The six tested Qwen2.5 checkpoints span 0.5B–14B, including base, instruction-tuned, and math-specialized models. Table 1 combines six math benchmarks and MMLU-Pro; its mean mixes the paper's greedy pass@1 evaluations with the stated avg@32 protocol for the small AMC and AIME sets. The cleanest uncertainty evidence is the five-seed comparison on four representative models and two benchmarks. For Math-1.5B, MATH-500 is 58.6 ± 1.4 versus 67.2 ± 1.2, and GSM8K is 49.8 ± 1.8 versus 59.3 ± 1.5.

The gain over GRPO does not imply improvement over the starting checkpoint on every task: Math-1.5B's base GSM8K score is 80.2%, above both trained variants. Preserving that regression matters when deciding whether the training distribution is appropriate.

### Augmentation and filtering spend the rollout budget differently

[DAPO](/paper%20shorts/2025/03/18/dapo-an-open-source-llm-reinforcement-learning-system-at-scale.html) resamples to obtain useful reward contrast; AVSPO reuses homogeneous groups by changing their normalization. In the paper's controlled Math-7B comparison, AVSPO reaches 74.1% MATH accuracy versus 68.3% for the tested DAPO configuration, with reported rollout costs of 1.0× and 1.8×. This comparison uses the selected 500-problem regime and does not establish that augmentation dominates the full DAPO recipe at larger scale.

The scalar calculation adds no LLM calls. Appendix timing reports approximately three milliseconds of additional work per step, while generation dominates runtime. A code-generation extension reports MBPP pass@1 improving from 83.7% to 86.7% over three seeds. Soft rewards, noisy verification, and broader training distributions remain untested boundaries.

I would monitor all-correct and all-wrong collapse separately, then compare accuracy and retained capabilities at matched rollout cost. Reducing ACR alone is insufficient: the diagnostic counts missing contrast, while the deployment decision concerns what the resulting updates actually improve.

## High-Level Takeaways

- Homogeneous reward groups remove GRPO's relative reward signal, but all-correct and all-wrong groups represent different training conditions.
- AVSPO changes the baseline with virtual scalar rewards; it neither generates extra solutions nor labels a new successful trajectory.
- The reported gains are supported by selected multi-seed comparisons, with important regressions relative to some starting checkpoints.
- Treat collapse rate as one diagnostic alongside reward, held-out accuracy, and matched rollout cost, rather than a target to minimize by itself.
