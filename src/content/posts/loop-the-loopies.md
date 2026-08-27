---
title: 'Loop the Loopies!'
date: '2026-07-17T09:00:00.000Z'
section: paper-shorts
postSlug: loop-the-loopies
legacyPath: /paper shorts/2026/07/17/loop-the-loopies.html
tags:
  - Language Models
  - Mixture of Experts
  - Scaling
field: 'Language Models'
summary: '2026 – Loop the Loopies!'
---

## 2026 – Loop the Loopies!

**arXiv:** [2607.16051](https://arxiv.org/abs/2607.16051)

## Summary

> Loopie makes recurrent depth compete with ordinary parameter scaling by changing both the loop schedule and the hardware budget. Each Transformer block runs twice before the next stored block, which lowers stored activation depth enough to double the per-device microbatch; the saved wall-clock time is reinvested in width and depth. Loopie-20B-A2B overtakes a reproduced Qwen3-like 30B-A3B baseline after about 600 billion pretraining tokens under matched optimizer-step time. This is a wall-clock match, not a FLOP match: Loopie's leading-order block-work proxy is 1.424× higher.

## Core Insights

![Loopie layer-loop recurrence compared with whole-model recurrence](/assets/images/loopie-layer-loop-vs-model-loop.png)
*Fig 1: Layer-loop applies each Attention/MoE block repeatedly before advancing through the stack; model-loop traverses the whole stack and then starts again. The local schedule improves parameter reuse and avoids a cyclic pipeline path. | source: [Loopie paper](https://arxiv.org/abs/2607.16051)*

![Figure 5 from Loop the Loopies!](/assets/images/loop-the-loopies-source-figure-5.webp)
*Fig 2: Layer-loop ablation for Loopie-6B-A0.6B. We report the average score across eight downstream benchmarks for Loopie-6B-A0.6B and Loopie-6B-A0.6B-Ablation. | source: [Loop the Loopies!](https://arxiv.org/abs/2607.16051)*

![Figure 7 from Loop the Loopies!](/assets/images/loop-the-loopies-source-figure-7.webp)
*Fig 3: Composition of the Stage-2 high-quality annealing data pool: multiple data sources make up the 1.26T-token annealing recipe. | source: [Loop the Loopies!](https://arxiv.org/abs/2607.16051)*


### Recurrence is scheduled within each stored layer

For three stored layers and two recurrent steps, model-loop executes

$$
L_1 \rightarrow L_2 \rightarrow L_3 \rightarrow L_1 \rightarrow L_2 \rightarrow L_3,
$$

whereas Loopie's layer-loop executes

$$
L_1 \rightarrow L_1 \rightarrow L_2 \rightarrow L_2 \rightarrow L_3 \rightarrow L_3.
$$

Both schedules reuse parameters across effective depth, but layer-loop keeps repeated applications adjacent. That shortens the reuse distance for weights and gradients, keeps the repetitions inside one pipeline stage, and shares a block across neighboring effective depths rather than positions separated by a full model traversal. In a 6B-A0.6B experiment, layer-loop initially trails model-loop but passes it after roughly 1.2 trillion tokens.

### The Loopie Recipe matches realized training time

The large comparison starts from a Qwen3-like 30B-A3B MoE with 48 stored layers. The recurrent seed halves stored depth to 24 and applies every layer twice. Under the paper's checkpointing scheme, dominant activation memory scales with stored depth rather than executed depth, so this seed retains 48 block applications while cutting the activation-memory proxy in half.

Loopie then uses the memory headroom to double the per-device microbatch and halve gradient-accumulation steps, keeping tokens per optimizer update fixed. The authors sweep aligned widths and depths and select 27 stored layers, width 2,304, and two loops because that configuration matches the baseline's measured end-to-end optimizer-step time in Megatron-LM. The resulting model has 20B total and 2B active parameters.

| Comparison axis | Qwen3-like baseline | Loopie-20B-A2B |
| --- | ---: | ---: |
| Stored layers | 48 | 27 |
| Recurrent steps | 1 | 2 |
| Hidden width | 2,048 | 2,304 |
| Relative block-work proxy | 1.000× | 1.424× |
| Best reported throughput | 189.65 TFLOPS/s | 261.53 TFLOPS/s |
| Per-device microbatch | 1 | 2 |

This operational match is the paper's most important qualification. Hardware allocation, sequence length, tokens per step, updates, data, optimizer, and checkpointing are held fixed; theoretical FLOPs are not. The larger microbatch turns more nominal work into the same step time on the tested systems. A different accelerator, parallelism plan, or kernel stack can move that boundary.

The main 800-billion-token run crosses the vanilla baseline near 600 billion tokens. Four smaller matched-wall-time pairs, spanning 0.15B to 1B baseline parameter scales, also favor Loopie by 0.6 to 2.2 average benchmark points. The authors choose two loops because the marginal advantage over adding stored layers falls as the loop count rises. The sweep does not establish that two loops are universally optimal.

### Post-training is a second, separate contribution

After roughly 3.5 trillion pretraining tokens, the paper applies two trillion tokens of supervised pretraining (SPT). SPT masks prompt and context tokens as conventional supervised fine-tuning does, but uses pretraining-scale batches and sequences: at least 1,024 examples per global batch, 128K context, and about 128 million token positions per update. Reasoning and general benchmarks rise through ten data epochs, but the paper says a comprehensive SPT ablation was not computationally feasible.

Math and code reinforcement learning then uses GSPO with DAPO-style asymmetric clipping and dynamic prompt filtering. The final model is strong relative to several similarly active MoE models, yet those external comparisons mix architecture, data volume, and post-training. They cannot isolate layer-loop as the cause of the final reasoning scores.

## High-Level Takeaways

- Loopie informs whether a training system should spend its budget on more stored parameters or on repeated computation with better memory locality. The answer depends on realized optimizer-step time, not parameter count or theoretical FLOPs alone.
- The controlled result supports two-step layer-loop recurrence plus a hardware-aware width-depth recipe. It does not support naive looping at fixed stored size, and its 1.424× nominal-work caveat should travel with every “compute-matched” claim.
- At ten times the scale, pipeline scheduling, expert communication, checkpointing semantics, and microbatch efficiency are likely to decide whether the recipe still pays. Inference memory and latency were not systematically studied.
- A decisive replication should match energy, accelerator-hours, training tokens, data, and inference cost across recurrent and non-recurrent MoE models on more than one hardware stack. Reject the scaling claim if the advantage disappears under energy or inference-budget matching, or if a tuned vanilla model recovers the same throughput.
