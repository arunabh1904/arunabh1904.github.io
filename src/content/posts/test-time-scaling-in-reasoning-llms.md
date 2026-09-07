---
title: 'Test-Time Scaling in Reasoning LLMs'
date: '2026-08-04T17:57:20.000Z'
section: paper-shorts
postSlug: test-time-scaling-in-reasoning-llms
legacyPath: /paper shorts/2026/08/04/test-time-scaling-in-reasoning-llms.html
tags:
  - Language Models
  - Reasoning
  - Inference
  - Evaluation
field: 'Language Models'
summary: '2026 – Test-Time Scaling in Reasoning LLMs'
---

## 2026 – Test-Time Scaling in Reasoning LLMs

**arXiv:** [2608.04001](https://arxiv.org/abs/2608.04001)

## Summary

> Test-time scaling is not one technique or one budget axis. This paper separates single-trajectory deliberation, sampling complete responses, and search over unfinished prefixes, then treats the generator, evidence signals, reducer, stopping rule, and cost accounting as one inference system. Its 80-response mathematics banks expose the practical reason: correct answers become available much faster than reference-free reducers learn to select them, leaving a 5.38–14.52 percentage-point selection gap at the largest sampled budget.

## Core Insights

### The branching point defines the inference regime

The paper models autoregressive generation as an implicit prefix tree. A sequential method spends more compute on one active path through longer reasoning, critique, or revision. A leaf-level method samples complete paths independently and applies a terminal reducer such as plurality, best-of-$N$, or verifier-constrained selection. A prefix-level method scores unfinished states and reallocates compute before the responses finish.

This taxonomy changes what a fair comparison must hold fixed. Two methods with the same number of generated tokens can pay very different costs for verifier calls, pairwise comparisons, control decisions, discarded rollouts, or synchronization. The paper therefore decomposes total cost as

$$
C_{\mathrm{total}}
= C_{\mathrm{generation}}
+ C_{\mathrm{signal}}
+ C_{\mathrm{control}}
+ C_{\mathrm{decision}}.
$$

A shared-bank experiment can isolate a reducer because every method receives the same completed candidates. An end-to-end experiment can support a system claim because each method runs its own generation, evaluation, and stopping policy. Confusing those two designs turns a post-generation diagnostic into an unsupported deployment claim.





### Candidate discovery is not submitted-answer accuracy

The decisive experiment uses 80-response banks for 186 problems from five 2025–2026 mathematics competitions. Pass@$k$ measures whether at least one correct response exists; it is an oracle ceiling, not an executable selection policy. At $k=80$, the paper reports Pass@80 values of 94.62% for Qwen3.6-35B-A3B and 72.58%, 91.94%, and 93.55% for gpt-oss-20b at low, medium, and high reasoning effort. Reference-free pointwise selection reaches 86.56%, 58.06%, 75.81%, and 81.72%.

![Source Figure 6 comparing the candidate ceiling with plurality and reference-free pointwise selection](/assets/images/test-time-scaling-reducer-gap.png)
*Fig 1: The dashed ceiling asks whether a correct candidate exists. The other curves measure submitted answers under specific reducers; their separation is a selection failure, not a shortage of generated candidates. | source: [Test-Time Scaling, Figure 6](https://arxiv.org/abs/2608.04001)*

The arrows compare the ceiling with the *better* of plurality and pointwise selection, giving the 5.38–14.52-point range in the summary. Comparing the ceiling with pointwise selection alone gives larger gaps, 8.06–16.13 points. Those are two different comparisons, not interchangeable estimates of one reducer's error.

The gap is the central systems result. More sampling compute can create useful candidates without improving the submitted answer at the same rate. In another 120-question block, selecting the response with the highest mean token log-probability falls from 75.56% to 65.83% as the bank grows. A larger bank amplifies a misaligned selector.

The figure's uncertainty is also conditional. Shading bootstraps prompts from the observed banks; intermediate reducer points replay nested subsets, while the one- and 80-response endpoints are exact for those banks. It does not capture every change that would arise from generating new banks, changing checkpoint versions, or using another verifier.

### Occasional success and reliable success can look similar under Pass@k

Under an independent-attempt abstraction with per-question success probability $p$, at least one success in $k$ tries has probability $1-(1-p)^k$. Even a model correct only one time in ten approaches certainty of finding *some* correct answer after 80 attempts. That says little about which answer it will submit or how consistently individual attempts succeed.

The paper's discovery–stability profile makes that distinction explicit. Low thresholds ask whether a prompt produces an occasional success. High thresholds ask whether success repeats. Reporting only Pass@$k$ rewards discovery while hiding whether a reducer can recover the answer and whether the model succeeds reliably. For a finite stored bank, the paper uses without-replacement subset probabilities rather than treating observed successes as fresh independent trials.

The SuperGPQA experiment makes the distinction tangible. For gpt-oss at high effort on 3,600 questions, mean response accuracy is 45.03%, Pass@80 is 81.94%, and the fraction correct on all 80 responses is 9.47%. There are 650 questions with no successful response and 341 with no failures. A single aggregate accuracy would hide these very different question-level regimes.

### A strong diagnostic verifier may have information unavailable at deployment

CompassVerifier receives the reference answer along with the question and candidate. Its direct correctness score achieves 0.983 trace-level ROC AUC on the competition responses. That is useful for studying outcomes, but the reference is unavailable when the system is solving a new problem.

The reference-free pointwise verifier sees one problem and one candidate, scoring understanding, reasoning validity, and support for the conclusion. Its overall AUC is 0.871, with 0.744 on Qwen-generated responses and 0.876 on gpt-oss responses. The difference suggests selector quality depends on the candidate distribution; it is not a fixed property that transfers unchanged between generators.

AUC also measures ranking across traces, not whether the highest-ranked response within every question's bank is correct. Neither the reference-assisted score nor an aggregate ranking statistic can substitute for evaluating the final reducer on the actual candidate groups.

### Reproducibility belongs to the protocol

Exact replay needs the serialized candidate bank, prompt templates, decoding settings, token-level signals, parsers, verifier versions, reducer code, and stopping logic. Distributional reproducibility is weaker: it asks whether a new run from the documented protocol produces compatible results with uncertainty over prompts and candidate draws.

Budget caps can alter the effort comparison. In the gpt-oss high-effort competition bank, 2,399 of 14,880 responses—16.12%—reach the shared 81,920-token cap; none do at low effort. Longer responses and truncation are therefore part of the observed protocol, not incidental details outside the comparison.

The released banks are unusually rich, but the empirical conclusions remain conditional on the studied checkpoints, benchmarks, 80-sample cap, and verifier designs. Figure 6 also counts generated candidates on its horizontal axis while excluding verifier computation. It diagnoses selection quality; it is not a latency- or dollar-matched deployment comparison.

## High-Level Takeaways

- An inference budget includes generation, evidence acquisition, control, and final selection.
- Pass@k is a candidate-availability ceiling; deployable accuracy depends on a reducer finding the correct answer.
- Discovery–stability profiles distinguish rare successes from consistently solved questions.
- Reference-assisted verification and trace-level AUC do not establish deployable selection quality.
- Shared banks isolate reducer differences, while end-to-end cost claims require fresh execution with verifier costs and stopping rules included.
