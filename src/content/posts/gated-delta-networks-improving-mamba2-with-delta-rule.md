---
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
date: '2024-12-09T00:00:00.000Z'
section: paper-shorts
postSlug: gated-delta-networks-improving-mamba2-with-delta-rule
legacyPath: /paper shorts/2024/12/09/gated-delta-networks-improving-mamba2-with-delta-rule.html
tags:
  - Attention
  - Sequence Modeling
field: Language Models
summary: "2024 \u2013 Gated Delta Networks: Improving Mamba2 with Delta Rule"
---

## 2024 – Gated Delta Networks: Improving Mamba2 with Delta Rule

**Paper:** [Yang, Kautz, and Hatamizadeh, arXiv:2412.06464v3](https://arxiv.org/abs/2412.06464v3) · **Code:** [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

## Summary

> Gated DeltaNet combines broad memory decay with a targeted correction to the value associated with the current key. In matched 1.3B-parameter, 100B-token experiments, this improves several language-modeling and retrieval results over Mamba2 and DeltaNet. The remaining retrieval gap matters: the pure recurrent model averages 30.6 on the paper's short-context recall suite versus 37.0 for its Transformer comparator; the H2 hybrid reaches 40.1.

## Core Insights

The model stores history in a fixed-size matrix $S$, rather than a separate cached key/value pair for every token. Ordinary additive writes can superimpose conflicting associations. DeltaNet corrects the association addressed by the current key; Mamba2-style decay can clear old information broadly. These operations solve different memory-management problems.

The gated delta update combines them:

$$
S_t=\alpha_t S_{t-1}+\beta_t\left(v_t-\alpha_t S_{t-1}k_t\right)k_t^\top.
$$

Here $\alpha_t$ controls decay and $\beta_t$ controls the correction. After decay, the state predicts a value for key $k_t$; the residual between that prediction and $v_t$ determines the write. This targeted update can reduce interference, but it does not make the finite state an unlimited store of independent facts.

The source figure shows how the recurrent operation enters a complete language model and two hybrids.

![Source Figure 1 shows Gated DeltaNet's block and H1/H2 hybrid stacks.](/assets/images/attention-hybrid-gated-deltanet.png)
*Fig 1: H1 interleaves Gated DeltaNet with sliding-window attention; H2 also includes Mamba2. The block keeps separate decay and write controls, with normalized query/key paths and an output gate. | source: [Gated Delta Networks, Figure 1](https://arxiv.org/html/2412.06464v3#S3.F1)*

[Open the source figure at full size](/assets/images/attention-hybrid-gated-deltanet.png)

Unlike [GQA](/paper%20shorts/2023/05/22/gqa-training-generalized-multi-query-transformer-models.html), which reduces the number of cached heads while retaining token positions, the recurrent branch must decide what survives in shared state. The attention layers give the hybrids another retrieval path within their retained windows.

### Better forgetting does not win every retrieval test

The main comparison trains all models on the same 100B-token FineWeb-Edu subset with a Llama2 tokenizer, 4K training sequences, and AdamW. Hybrids use a 2K attention window. Their reported zero-shot commonsense averages are 55.32 for Gated DeltaNet, 54.89 for Mamba2, and 52.14 for DeltaNet. These are small pretrained models, not instruction-tuned assistants.

The needle tests expose the trade-off more clearly. With a repeated synthetic distractor and an 8K sequence, ungated DeltaNet retains the passkey at 98.8% while Gated DeltaNet reaches 91.8%. With natural-text distractors, decay helps manage competing information. On the 4K numerical-needle task, Gated DeltaNet reaches 92.2% against DeltaNet's 18.6%, but it falls to 29.6% at 8K. Forgetting helps select useful state; it can also erase a fact that should have survived.

The algorithm extends DeltaNet's chunkwise parallel formulation so training uses matrix multiplications instead of a serial token loop. H100 measurements show throughput close to DeltaNet and slightly below Mamba2. Optimized short-window attention can make hybrids faster than the pure recurrent model, so a simpler asymptotic description does not determine measured throughput.

Appendix ablations at 400M parameters and 15B tokens show that short convolutions, output gating, and L2 query/key normalization also matter. The gain cannot be attributed to inserting a decay scalar into an otherwise arbitrary block. My deployment comparison would hold data, state size, retained attention window, and kernels fixed, then measure retrieval failures as well as throughput.

## High-Level Takeaways

- Decay and delta correction provide complementary control over a finite associative memory.
- Natural distractors and repeated synthetic distractors reward different retention behavior.
- Hybrid attention closes a retrieval gap that the recurrent update alone does not remove.
- The efficient chunkwise algorithm and surrounding block design are part of the method's practical result.
