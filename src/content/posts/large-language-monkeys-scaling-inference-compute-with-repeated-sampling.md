---
title: "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
date: '2024-07-31T00:00:00.000Z'
section: paper-shorts
postSlug: large-language-monkeys-scaling-inference-compute-with-repeated-sampling
legacyPath: /paper shorts/2024/07/31/large-language-monkeys-scaling-inference-compute-with-repeated-sampling.html
tags:
  - Inference
  - Scaling Laws
field: "Language Models"
summary: "2024 – Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
---

## 2024 – Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

**Paper:** [Brown et al., arXiv:2407.21787v3](https://arxiv.org/abs/2407.21787v3) · **Code:** [ScalingIntelligence/large_language_monkeys](https://github.com/ScalingIntelligence/large_language_monkeys)

## Summary

> Repeated sampling reveals correct answers that a model rarely produces, but finding those answers is a separate scaling problem. On a 128-problem MATH subset, Llama-3-8B-Instruct's coverage rises from 82.9% with 100 samples to 98.44% with 10,000; the best improvement among the tested answer selectors is only 40.50% to 41.41%. The result supports investing in verification alongside generation, with much stronger benefits when an executable checker is available.

## Core Insights

A sample budget changes what the model can produce without changing its weights. For a problem whose independent attempts succeed with probability $p$, the probability of generating at least one correct answer in $k$ attempts is $1-(1-p)^k$. That is coverage, or oracle pass@$k$. A deployed system must also identify the correct attempt. Majority voting instead favors the most frequent answer, which can remain wrong as the sample collection grows.

The source figure makes this distinction visible at the individual-problem level. Each bar is one problem, and its height measures how often the model answered correctly across 10,000 attempts. A red bar above zero means correct answers exist in the collection but majority voting still chooses incorrectly.

![Source Figure 8 shows per-problem correct-sample fractions for Llama-3-8B and 70B on GSM8K and MATH, colored by majority-vote correctness.](/assets/images/large-language-monkeys-source-figure-8.png)
*Fig 1: Rare correct answers contribute to coverage without necessarily changing the most common answer. The long red region on MATH exposes the selection problem that additional sampling alone leaves unresolved. | source: [Large Language Monkeys, Figure 8](https://arxiv.org/html/2407.21787v3#S4.F8)*

[Open the figure at full size](/assets/images/large-language-monkeys-source-figure-8.png)

### What the experiments actually sample

The main Llama experiments generate 10,000 independent attempts per problem: 128 problems each from GSM8K and MATH, 130 formalized MATH problems in MiniF2F, and 140 CodeContests problems without image descriptions. Math uses temperature 0.6 and a 512-token output cap; Lean uses temperature 0.5 and 200 tokens. CodeContests uses temperature 0.6, top-$p$ 0.95, and 1,024 tokens. These settings delimit the measured distribution of attempts.

SWE-bench Lite uses a different unit: one complete multi-turn Moatless Tools trajectory with DeepSeek-Coder-V2-Instruct. Across 250 independent trajectories per issue, reported coverage rises from 15.9% to 56%. The authors tune temperature on 50 test-set problems before choosing 1.6, so this is not a fully untouched test protocol. A successful test-suite outcome also remains conditional on the suite's adequacy.

The paper fits coverage with an exponentiated power law, $c(k)\approx\exp(a k^b)$, across the sampled range. Some formal-proof curves fit poorly. This is an empirical description of fixed models and sampling recipes, not evidence that every task obeys the same extrapolation law. Pythia models obtain zero CodeContests coverage even at 10,000 samples.

### The checker changes the economic conclusion

At matched estimated inference FLOPs, Llama-3-8B-Instruct has better coverage than 70B on the tested math and formal-proof tasks; CodeContests usually favors 70B. Smaller models plus more attempts therefore need a task-specific comparison. FLOPs also omit serving utilization, latency constraints, and the cost of verification.

For math answer selection, the authors test majority voting, ArmoRM best-of-$N$, and reward-weighted voting on the same generated samples. All plateau while oracle coverage keeps improving. Manual review of 105 correct-answer GSM8K chains finds over 90% valid reasoning, suggesting the gap is not explained entirely by lucky final-answer guesses. That audit is small and does not establish equivalent reasoning quality across every task.

Executable checks have their own errors. The authors identify flaky tests in 34 of 300 SWE-bench Lite issues, repeat those suites 11 times, and also report the 266-issue subset without them. In CodeContests, 35 of 122 test problems with Python reference solutions have at least one supplied correct solution that fails the tests. More sampling can expose these inconsistencies as well as genuine solutions.

The connection to [metric-driven emergence](/paper%20shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html) is precise: the same underlying answer distribution supports different capability claims depending on the scoring and selection rule. For a practical system, I would compare selected-answer accuracy per dollar, including checker costs and false positives, before buying a larger sampling budget.

## High-Level Takeaways

- Coverage measures whether a correct attempt exists; selected-answer accuracy measures whether the system can recover it.
- Sampling can make a smaller model competitive, but the preferred model size reverses across tasks even under the paper's FLOP accounting.
- A proof checker, a unit-test suite, and a learned reward model provide different correctness guarantees and failure modes.
- The next useful scaling experiment must improve verification precision alongside the supply of rare correct answers.
