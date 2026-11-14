---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
date: '2021-04-20T00:00:00.000Z'
section: paper-shorts
postSlug: roformer-enhanced-transformer-with-rotary-position-embedding
legacyPath: /paper shorts/2021/04/20/roformer-enhanced-transformer-with-rotary-position-embedding.html
tags:
  - Attention
  - Position Encoding
field: "Language Models"
summary: "2021 \u2013 RoFormer: Enhanced Transformer with Rotary Position Embedding"
---

## 2021 – RoFormer: Enhanced Transformer with Rotary Position Embedding

**Paper:** [Su et al., arXiv:2104.09864v5](https://arxiv.org/abs/2104.09864v5) · **Code:** [RoFormer](https://github.com/ZhuiyiTechnology/roformer)

## Summary

> Rotary position embedding makes an attention score depend on relative displacement by rotating queries and keys using their absolute positions. The construction preserves vector norms and avoids learning a separate embedding for every position. RoFormer's translation and long-document experiments support its usefulness, but mixed GLUE results and length-specific training prevent a claim of universal improvement or unrestricted context extrapolation.

## Core Insights

[The original Transformer](/paper%20shorts/2017/06/01/attention-is-all-you-need.html) adds position information to token representations. RoPE instead acts on the projected query and key. Split each vector into pairs of channels; rotate each pair by a position-dependent angle, using different frequencies across pairs.

The source figure follows that operation from a two-channel pair to the complete vector.

![Source Figure 1 shows position-dependent rotations applied to pairs of query and key channels.](/assets/images/roformer-source-figure-1.png)
*Fig 1: Each channel pair uses a fixed frequency and a rotation angle proportional to token position. Applying the corresponding rotations to both queries and keys makes their dot product depend on relative displacement. | source: [RoFormer, Figure 1](https://arxiv.org/html/2104.09864v5#S3.F1)*

[Open the figure at full size](/assets/images/roformer-source-figure-1.png)

Let $q$ and $k$ be projected content vectors, and $R_m$ the block-diagonal rotation at position $m$. The important identity is

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k.
$$

Absolute positions construct the rotations; their difference remains in the interaction. Moving both tokens by the same offset leaves this positional relation unchanged. The operation does not supply metric coordinates or object identity: a sequence index still has the meaning assigned by the input interface.

The paper also adapts this construction to linear attention. It rotates feature-mapped queries and keys in the numerator while leaving the denominator unrotated to avoid problematic normalization. This is a specific algebraic modification, not permission to insert rotations anywhere in a recurrent implementation.

### The empirical boundary is narrower than the algebra

On WMT14 English–German, matched Transformer-base training yields 27.5 BLEU with RoPE versus 27.3 for the baseline. English pretraining uses BookCorpus and Wikipedia, 100,000 steps, batch size 64, and sequence length 512. The paper reports faster masked-language-model convergence, but downstream GLUE results improve on only three of six evaluated datasets.

The long-document experiment uses a Chinese model pretrained on approximately 34 GB with staged sequence lengths, including 1,536-token stages. On CAIL2019-SCM case matching, RoFormer scores 68.29% test accuracy at a 512 cutoff and 69.79% at 1,024. The longer-input result combines the positional mechanism with access to more document content and length-aware pretraining; it does not isolate zero-shot extrapolation beyond training length.

RoPE's distance analysis concerns the behavior of its frequency construction and a bound on interactions. It should not be read as a guarantee that every learned attention score decreases monotonically with distance. My practical check would separate three claims: the rotation identity, accuracy at trained lengths, and retrieval at longer unseen lengths. Only the first follows directly from the construction.

## High-Level Takeaways

- RoPE encodes position through rotations of queries and keys, not an added coordinate vector.
- Relative displacement appears in the dot product because the rotations compose by subtraction.
- Translation and long-text gains coexist with regressions on several GLUE tasks; the evidence is task-dependent.
- A representable position is not automatically a position at which the trained model retrieves reliably.
