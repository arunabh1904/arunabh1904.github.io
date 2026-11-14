---
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'
date: '2025-01-22T00:00:00.000Z'
section: paper-shorts
postSlug: deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning
legacyPath: /paper shorts/2025/01/22/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning.html
tags: [Reinforcement Learning, Reasoning, Distillation]
field: 'Language Models'
summary: '2025 – DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'
---

## 2025 – DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Paper:** [DeepSeek-AI et al., arXiv:2501.12948v1](https://arxiv.org/abs/2501.12948v1). This note covers the original January 2025 technical report.

## Summary

> DeepSeek-R1 separates two findings: verifiable rewards can substantially improve reasoning from a pretrained base without an initial supervised fine-tuning stage, and a usable assistant benefits from a broader pipeline that includes curated demonstrations and general preference training. Its small distilled models add a third lesson: successful reasoning traces can transfer more effectively than the particular small-model RL recipe tested here.

## Core Insights

### R1-Zero isolates a useful training intervention

R1-Zero starts from DeepSeek-V3-Base and applies [GRPO](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html). Several answers to the same question provide a group-relative reward baseline, avoiding a separately trained critic. The experiment removes the initial supervised fine-tuning stage; it still depends on a capable pretrained language model and multiple sampled completions per training question.

The rewards check final answers or executable tests and require a structured reasoning format. R1-Zero does not use a neural process or outcome reward model. This makes the feedback relatively inspectable where correctness is verifiable, while leaving the model freedom to discover intermediate strategies. The original report records longer outputs and self-checking examples during training. Those observations support behavioral change, but a selected “aha” example cannot establish which internal mechanism produced it.

![DeepSeek-R1-Zero AIME accuracy rises during training, with separate single-answer and consensus curves.](/assets/images/deepseek-r1-source-figure-2.png)
*Fig 1: The blue curve measures average single-answer accuracy; the red curve uses consensus over 16 samples. Their separation shows why a result needs its sampling budget attached. The dashed comparison lines also use different answer budgets. | source: [Paper, Figure 2](https://arxiv.org/html/2501.12948v1#S2.F2)*

[Open the source figure at full size](/assets/images/deepseek-r1-source-figure-2.png).

Reported AIME 2024 pass@1 rises from 15.6% to 71.0%. The report's separate final consensus result reaches 86.7% using 64 samples. That is additional inference computation, not another estimate of single-answer accuracy.

### The full R1 recipe solves problems that Zero leaves behind

R1-Zero can mix languages and produce difficult prose. Full R1 therefore uses four stages: supervised fine-tuning on thousands of cold-start reasoning examples; reasoning-focused RL; a new supervised stage using approximately 600,000 accepted reasoning traces plus 200,000 non-reasoning examples; and a final RL stage for broader assistant behavior. The intermediate 800,000-example dataset trains DeepSeek-V3-Base for two epochs.

Language-consistency rewards improve readability with a reported small reasoning tradeoff. The final general RL stage combines rule-based reasoning rewards with neural preference rewards for broader tasks. “No supervised stage” and “no neural reward model” describe R1-Zero, not the complete R1 assistant.

### Distillation transfers the trajectories, with teacher cost outside the comparison

The authors fine-tune Qwen and Llama models from 1.5B to 70B on the curated 800,000 examples without further RL. R1-Distill-Qwen-32B scores 72.6% on AIME 2024, compared with 47.0% for the tested direct-RL Qwen-32B baseline. This supports distillation as a strong route when a capable teacher exists. It does not establish a universal advantage under matched total compute: producing the teacher and its traces has a cost outside the small-model training comparison.

Final R1 reports 79.8% AIME pass@1 and 97.3% on MATH-500. Evaluation allows up to 32,768 generated tokens; pass@1 is estimated by averaging sampled answers, typically 4–64 per question, with temperature 0.6 and top-p 0.95. Averaging estimates single-answer reliability; selecting or voting among answers changes the decision procedure.

The limitations matter for use: function calling, structured output, and multi-turn interaction lag DeepSeek-V3 in the authors' assessment. Language mixing remains possible, few-shot demonstrations can hurt, and expensive software-engineering evaluations limit the scale of RL there. The demonstrated recipe is strongest where useful feedback can be obtained cheaply and repeatedly.

## High-Level Takeaways

- R1-Zero isolates RL without an initial supervised fine-tuning stage, not learning without pretraining.
- Full R1 combines supervised data, verifiable rewards, and broader preference training.
- Distillation is effective here, but its teacher and data-generation costs remain part of the system.
- Report answer budget, generation limit, and aggregation rule alongside reasoning scores.
