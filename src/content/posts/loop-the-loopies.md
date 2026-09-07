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





### Recurrence is scheduled within each stored layer

For three stored layers and two recurrent steps, model-loop executes

$$
L_1 \rightarrow L_2 \rightarrow L_3 \rightarrow L_1 \rightarrow L_2 \rightarrow L_3,
$$

whereas Loopie's layer-loop executes

$$
L_1 \rightarrow L_1 \rightarrow L_2 \rightarrow L_2 \rightarrow L_3 \rightarrow L_3.
$$

![Source comparison of layer-loop and whole-model recurrence schedules](/assets/images/loopie-layer-loop-vs-model-loop.png)
*Fig 1: Layer-loop finishes repeated applications within one stored block before moving onward. Whole-model recurrence returns to earlier blocks after traversing the stack, changing where repeated computation sits in the execution schedule. | source: [Loopie paper](https://arxiv.org/abs/2607.16051)*

Read the diagram as an execution order, not as extra independent weights. Reusing a block twice means applying the same transformation to two different intermediate states. It creates additional computation while tying the parameters across those effective depths.

Both schedules reuse parameters across effective depth, but layer-loop keeps repeated applications adjacent. That shortens the reuse distance for weights and gradients, keeps the repetitions inside one pipeline stage, and shares a block across neighboring effective depths rather than positions separated by a full model traversal. In a 6B-A0.6B experiment, layer-loop initially trails model-loop but passes it after roughly 1.2 trillion tokens.

### The Loopie Recipe matches realized training time

The large comparison starts from a Qwen3-like 30B-A3B MoE with 48 stored layers. The recurrent seed halves stored depth to 24 and applies every layer twice. Under the paper's checkpointing scheme, dominant activation memory scales with stored depth rather than executed depth, so this seed retains 48 block applications while cutting the activation-memory proxy in half.

The checkpointing boundary is the crucial implementation detail: all recurrent applications of one stored layer sit inside the same checkpointed unit. Intermediate work is recomputed during backpropagation, rather than retaining an independent layer-boundary activation for every loop. The memory claim would not automatically hold in an implementation that stores each recurrent step separately.

Loopie then uses the memory headroom to double the per-device microbatch and halve gradient-accumulation steps, keeping tokens per optimizer update fixed. The authors sweep aligned widths and depths and select 27 stored layers, width 2,304, and two loops because that configuration matches the baseline's measured end-to-end optimizer-step time in Megatron-LM. The resulting model has 20B total and 2B active parameters.

| Comparison axis | Qwen3-like baseline | Loopie-20B-A2B |
| --- | ---: | ---: |
| Stored layers | 48 | 27 |
| Recurrent steps | 1 | 2 |
| Hidden width | 2,048 | 2,304 |
| Relative block-work proxy | 1.000× | 1.424× |
| Best reported throughput | 189.65 TFLOPS/s | 261.53 TFLOPS/s |
| Per-device microbatch | 1 | 2 |

The selected model's activation proxy at the *reference* microbatch is $2304\times27/(2048\times48)\approx0.633$. That number cannot by itself prove a doubled microbatch fits: parameters, optimizer states, temporary buffers, and communication workspaces still contribute to peak memory. Candidate feasibility is measured on the real system.

This operational match is the paper's most important qualification. Hardware allocation, sequence length, tokens per step, updates, data, optimizer, and checkpointing are held fixed; theoretical FLOPs are not. The larger microbatch turns more nominal work into the same step time on the tested systems. A different accelerator, parallelism plan, or kernel stack can move that boundary.

The main 800-billion-token run crosses the vanilla baseline near 600 billion tokens. Four smaller matched-wall-time pairs, spanning 0.15B to 1B baseline parameter scales, also favor Loopie by 0.6 to 2.2 average benchmark points. The authors choose two loops because the marginal advantage over adding stored layers falls as the loop count rises. The sweep does not establish that two loops are universally optimal.

### Read token efficiency separately from hardware throughput

Figure 5 compares the reported layer-loop variant with an ablation that the authors describe as retaining the backbone, data, token budget, and overall looped computation budget while removing the layer-loop pattern. The separation of the curves supports a contribution from the schedule beyond simply adding nominal computation.

![Loopie source Figure 5 comparing downstream score against pretraining tokens for the recurrence ablation](/assets/images/loop-the-loopies-source-figure-5.webp)
*Fig 2: The horizontal axis counts training tokens. The upper curve reaches comparable benchmark scores earlier, which is a sample-efficiency comparison rather than a direct measurement of inference or training throughput. | source: [Loopie, Figure 5](https://arxiv.org/abs/2607.16051)*

The plot's “2.14× speedup” annotation marks a horizontal separation at a chosen score; it should not be read as a universal wall-clock multiplier. There is also a source labeling inconsistency: the legend says 5B while the caption and surrounding discussion describe 6B-A0.6B. The qualitative comparison is visible, but that inconsistency limits precise attribution of the plotted configuration.

The loop-count sweep has a different caveat. Section 2.8 says the small two-times-stored-layer baseline consumes substantially more training compute than the two-loop model. Its curve therefore does not establish that adding independent layers dominates two loops at equal compute. It supports the authors' practical choice within the reported sweep, with the matching limitation attached.

### Post-training is a second, separate contribution

After roughly 3.5 trillion pretraining tokens, the paper applies two trillion tokens of supervised pretraining (SPT). SPT masks prompt and context tokens as conventional supervised fine-tuning does, but uses pretraining-scale batches and sequences: at least 1,024 examples per global batch, 128K context, and about 128 million token positions per update. Reasoning and general benchmarks rise through ten data epochs, but the paper says a comprehensive SPT ablation was not computationally feasible.

Math and code reinforcement learning then uses GSPO with DAPO-style asymmetric clipping and dynamic prompt filtering. The final model is strong relative to several similarly active MoE models, yet those external comparisons mix architecture, data volume, and post-training. They cannot isolate layer-loop as the cause of the final reasoning scores.

## High-Level Takeaways

- Layer-loop ties weights across adjacent effective depths, changing computation order without creating an independent block for each application.
- The activation saving depends on grouping recurrent applications inside a checkpointed unit.
- The main comparison matches measured optimizer-step time while Loopie performs about 1.424 times the nominal block work.
- Token-efficiency curves, loop-count controls, and hardware timing answer different questions and have different qualifications.
- Final post-training scores mix architecture with additional data and optimization; inference-time efficiency remains insufficiently studied.
