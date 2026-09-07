---
title: 'Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding'
date: '2026-08-04T11:07:19.000Z'
section: paper-shorts
postSlug: hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding
legacyPath: /paper shorts/2026/08/04/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding.html
tags: [Other]
field: 'Vision-Language Models'
summary: '2026 – Hi-Token makes bounding-box coordinates coarse-to-fine sequences rather than unrelated location symbols'
---

## 2026 – Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding

**arXiv:** [2608.03471](https://arxiv.org/abs/2608.03471)

## Summary

> Hi-Token changes the coordinate language of a generative VLM while leaving its backbone intact. A normalized coordinate is quantized to 1,000 bins and emitted as axis-specific hundreds, tens, and ones tokens, so each box becomes a 12-token coarse-to-fine sequence drawn from 60 coordinate types. Under a matched Qwen2.5-VL-3B recipe on 80K RefCOCO examples, Hi-Token SFT raises RefCOCO P@0.95 from 23.0 for flat coordinate tokens to 31.7. Hi-GAR then adds geometry-aware GRPO rewards and lifts the same model to 33.4 P@0.95 and 84.3 mIoU. The gains support a representation change plus a refinement stage; they do not isolate hierarchy, axis separation, and vocabulary size from one another.

## Core Insights

Generative grounding has a peculiar failure mode: the model can identify the right object and still emit a box that is slightly too wide, too narrow, or shifted onto a neighbor. A flat vocabulary treats `<323>` and `<324>` as unrelated classes, even though their numerical meanings are adjacent. A shared vocabulary also leaves the difference between an $x$ coordinate and a $y$ coordinate implicit. Hi-Token makes those relationships visible to the autoregressive model without adding a specialist detection head.

![Hi-Token representation and Hi-GAR reward](/assets/images/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding-source-figure-1.webp)
*Source Figure 1. The left side decomposes each box into axis-specific hundreds, tens, and ones tokens; the right side combines format validity, IoU, tiered coordinate checks, and strict-IoU bonuses under a validity gate during GRPO. [Hi-Token](https://arxiv.org/abs/2608.03471)*

For a coordinate $v\in[0,1]$, the paper first maps it to $I_v=\lfloor v(1000-1)\rfloor$, then emits three tokens. A box therefore uses 12 tokens: three each for $x_{min}$, $y_{min}$, $x_{max}$, and $y_{max}$. The hundreds token anchors a coarse location, the tens token adjusts the interval, and the ones token aligns the boundary. This is only a local numerical bias—the representation is not globally topology-preserving—but it gives the model a better set of reusable pieces. On the 80K training split, flat coordinates use 1,000 token types and four tokens per box; Hi-Token uses 60 types and 12 tokens. The resulting mean raw supervision density is 50 times higher, and the average number of distinct training examples per type is 15,426 versus 318.

The controlled comparison is the important part. With the same Qwen2.5-VL-3B backbone, 80K paired examples, optimizer, schedule, decoding, and evaluator, flat SFT gives 58.3 mIoU, 64.1 P@0.5, and 23.0 P@0.95 on RefCOCO. Hi-Token SFT gives 72.4, 79.0, and 31.7. A specially tuned flat baseline reaches 68.5/74.0/26.3, narrowing the gap but changing the recipe; it is a stress test rather than a matched causal estimate. The result is therefore evidence for the structured output under the fixed training recipe, with a fair warning that better flat optimization accounts for part of the difference.

Hi-GAR addresses a different mismatch. SFT rewards the likelihood of individual tokens, while IoU cares about the shape of the complete box. The reward combines format validity, continuous IoU, coordinate checks at 5%, 1%, and 0.3% tolerances, and bonuses at IoU 0.5, 0.9, and 0.95. Its gate activates coordinate-level rewards only when predicted IoU exceeds 0.01; otherwise a lucky corner match should not overpower the fact that the box is effectively invalid.

The ablation separates refinement from representation. Starting from Hi-Token SFT, IoU-only GRPO reaches 76.1 mIoU, 89.0 P@0.5, and 30.3 P@0.95. Full Hi-GAR without the gate reaches 80.4/91.9/31.3; with the gate it reaches 84.3/93.1/33.4. The reward's largest effect is on coarse and medium overlap: in the RefCOCO IoU distribution, predictions below 0.5 fall from 20.5% for Hi-Token SFT to 7.0% for Hi-R1, while the share at IoU ≥0.95 rises from 31.6% to 33.7%. Hi-GAR repairs bad boxes more decisively than it creates ultra-tight ones.

![Hi-Token boundary and scale diagnostics](/assets/images/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding-source-figure-4.webp)
*Source Figure 4. Boundary groups, object scale, and coordinate perturbation expose where the representation is fragile: near-hundreds transitions lose 2.4 P@0.95 points relative to the interior group, and one-bin or three-bin coordinate errors reduce IoU more for small objects than large ones. [Hi-Token](https://arxiv.org/abs/2608.03471)*

The diagnostics keep the claim honest. On a five-split scale aggregate, small objects reach 1.30 P@0.95 versus 17.5 for medium and 39.0 for large objects. Hi-GAR raises small-object P@0.95 from 1.07 to 1.30, a real but modest repair. Coordinate-boundary effects also remain: near-hundreds examples have 30.8 P@0.95 versus 33.2 for interior examples. The representation helps, but its fixed 1,000-bin grid still turns a one-bin error into a large relative displacement for a tiny object.

## High-Level Takeaways

- Hi-Token makes numerical proximity and axis role reusable in the output vocabulary: 60 coordinate types receive far denser supervision than 1,000 flat location types.
- The matched SFT result is 31.7 versus 23.0 P@0.95 on RefCOCO; the extra-tuned flat baseline at 26.3 is the paper's important control, not a footnote.
- Hi-GAR is a training-only repair. The validity gate cuts low-overlap predictions, moving the <0.5 IoU share from 20.5% to 7.0%, while strict P@0.95 moves only from 31.7 to 33.4.
- Small objects and digit boundaries remain failure surfaces: the five-split small-object P@0.95 is 1.30, and near-hundreds coordinates trail interior ones by 2.4 points.
- The causal story is coupled. A clean follow-up would vary digit hierarchy, axis vocabularies, vocabulary size, output length, and compute separately, then report calibration and latency alongside IoU.
