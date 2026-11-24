---
title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity'
date: '2021-01-11T09:00:00.000Z'
section: paper-shorts
postSlug: switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity
legacyPath: /paper shorts/2021/01/11/switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity.html
tags:
  - Mixture of Experts
  - Sparsity
  - Training Systems
field: "Language Models"
summary: "2021 – Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
---

## 2021 – Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**Paper:** [Fedus, Zoph and Shazeer, arXiv:2101.03961v3](https://arxiv.org/abs/2101.03961v3) · **Journal:** [JMLR, 2022](https://www.jmlr.org/papers/v23/21-0998.html)

## Summary

> Switch replaces selected Transformer feed-forward blocks with banks of experts. The router picks one expert per token, so the model can add weights without running every added block on every token. The paper shows a large pretraining advantage on the authors' TPU setup. Expert traffic, uneven utilization and downstream adaptation still determine whether that extra capacity is worth the systems cost.

## Core Insights

### Route one token to one expert

In a normal T5 feed-forward layer, every token uses the same weights. A Switch layer contains multiple feed-forward networks. For each token, the router computes a softmax distribution across the experts and selects the expert with the highest probability. The output of that expert is multiplied by its probability under the full distribution. Routing can differ by token even within a single sequence; self-attention remains a separate operation in the standard design. Using one expert rather than two reduces per-token work and traffic.

![A Switch encoder block routes two tokens to different feed-forward experts and scales the returned outputs by their gates.](/assets/images/switch-transformers-source-figure-2.svg)
*Fig 1: Tokens from the same layer input can use different feed-forward weights. Each selected output is scaled by its router gate. | source: [Fedus et al., Figure 2](https://arxiv.org/html/2101.03961v3#S2.F2)*

[Open the figure at full size](/assets/images/switch-transformers-source-figure-2.svg) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The device still needs to know how much space to reserve per expert. The paper takes the number of tokens per batch, divides by the number of experts and multiplies by a capacity factor. If a token's expert is full, the residual path carries its representation forward without expert computation. More headroom reduces those bypasses but can waste computation and memory. An auxiliary loss penalizes the dot product of expert assignment fractions and average router probabilities. The hard counts do not supply gradients; the probability term gives the optimizer a way to discourage concentrated routing. To improve stability, the authors also cast local router computation to float32 without sending expert traffic at that higher precision.

### Measure time to a target, then check transfer

On 32 TPUv3 cores, the 64-expert Switch-Base reaches a matched C4 pretraining perplexity at approximately one-seventh the wall time of T5-Base (Figure 5). The result uses the paper's span-corruption objective, hardware and implementation. It is not a sevenfold increase in tokens per second or a sevenfold inference speedup. In Table 9, Switch-XXL and T5-XXL are both reported at 6.3T FLOPs per sequence. The 1.57T-parameter Switch-C uses narrower per-token computation and is reported at 890B FLOPs per sequence. Its fourfold time-to-perplexity gain against T5-XXL involves a different model design as well as sparse routing.

Switch-XXL's downstream results are mixed. In §5.6, it reaches 34.4 exact match on closed-book Natural Questions versus 32.8 for T5-XXL without salient span masking, but scores 87.5 on SuperGLUE versus 89.3 for T5. This is preliminary evidence from a partially pretrained checkpoint with joint fine-tuning, not a matched comparison at identical token counts. Upstream loss does not tell us which downstream tasks benefit.

Expert sharding makes room for more weights, but dispatch and combine send activations between devices. Top-1 routing does not remove that traffic or let one device hold all of a trillion-parameter model's weights. I would compare time to a fixed downstream target alongside resident memory, routing overflow and latency at the batch sizes that matter.

## High-Level Takeaways

- Top-1 routing adds model capacity without activating all expert weights per token.
- Capacity limits and load balancing trade overflow against idle expert slots.
- The headline speed gains measure time to a C4 pretraining target on a specified TPU setup.
- Better pretraining perplexity does not guarantee better downstream reasoning or faster serving.
