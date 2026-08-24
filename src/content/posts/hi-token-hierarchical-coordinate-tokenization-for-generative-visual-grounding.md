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

> Hi-Token changes the output representation, not the VLM backbone: each normalized box coordinate becomes axis-specific hundreds, tens, and ones tokens. The representation gives autoregressive grounding a coarse-to-fine order and much denser supervision per coordinate type; a training-only GRPO reward then improves low-overlap predictions. The reported strict-localization gains are substantial, but the paper does not isolate hierarchy, axis-specific vocabularies, and vocabulary size from one another.

## Core Insights

### A box is twelve structured tokens

Flat coordinate vocabularies make nearby positions such as 323 and 324 unrelated symbols, often sharing one vocabulary across horizontal and vertical axes. Hi-Token instead emits three digit tokens for each coordinate and separates the $x$ and $y$ vocabularies. A box therefore uses 12 tokens drawn from 60 coordinate types. This increases output length, but it exposes place value and axis role while reusing each token far more often.

![Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding source figure: Overview of Hi-Token and the geometry-aware post-training framework.](/assets/images/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding-paper-figure.webp)
*Overview of Hi-Token and the geometry-aware post-training framework. source: [Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding](https://arxiv.org/abs/2608.03471)*

![Figure 4 from Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding](/assets/images/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding-source-figure-4.webp)
*Figure 4 Hi-Token remains stable at tens transitions and retains useful small-object localization at moderate thresholds, while hundreds transitions and deterministic perturbations reveal localized geometric sensitivities. Panel (a) reports RefCOCO P@0.95 changes relative to the interior group (33.2) for two disjoint boundary groups. Panel (b) gives a value-labeled matrix of mIoU and representative threshold accuracies through P@0. source: [Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding](https://arxiv.org/abs/2608.03471)*

![Figure 1 from Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding](/assets/images/hi-token-hierarchical-coordinate-tokenization-for-generative-visual-grounding-source-figure-1.webp)
*Figure 1 Overview of Hi-Token and the geometry-aware post-training framework. Left: Hi-Token represents each bounding box as a 12-token sequence by decomposing each coordinate into axis-specific hundreds, tens, and ones tokens. Right: The model is optimized with GRPO using the Hi-GAR reward. The reward combines format validity, IoU feedback, tiered coordinate verification, and strict-IoU bonuses: . The validity gate applies coordinate-level rewards only when the predicted box has sufficient overlap with the target. source: [Hi-Token: Hierarchical Coordinate Tokenization for Generative Visual Grounding](https://arxiv.org/abs/2608.03471)*


Under a matched Qwen2.5-VL-3B setup with 80,000 RefCOCO training examples, Hi-Token SFT raises RefCOCO P@0.95 from 23.0 for flat-token SFT to 31.7. The paper reports a 50-fold mean increase in raw supervision density per coordinate type. A separately tuned flat baseline reaches 26.3, so tuning narrows the gap but does not close it in the reported comparison.

### Hi-GAR targets broad geometric failures

Hi-GAR combines box IoU, coordinate accuracy at multiple tolerances, threshold bonuses, and a validity gate that turns off coordinate-level rewards for nearly non-overlapping boxes. Relative to Hi-Token SFT, full Hi-GAR raises RefCOCO mIoU from 72.4 to 84.3 and P@0.5 from 79.0 to 93.1, while P@0.95 moves from 31.7 to 33.4. Its main role is therefore reducing poor-overlap outputs, not replacing the representation as the source of strict localization.

| Setting on RefCOCO | mIoU | P@0.5 | P@0.95 |
| --- | ---: | ---: | ---: |
| Flat SFT, matched recipe | 58.3 | 64.1 | 23.0 |
| Hi-Token SFT | 72.4 | 79.0 | 31.7 |
| Full Hi-GAR with gate | 84.3 | 93.1 | 33.4 |

The final Hi-R1 model also reports strong results across RefCOCO, RefCOCO+, and RefCOCOg. Cross-family leaderboard comparisons are contextual rather than controlled, because the systems do not share data or training regimes.

## High-Level Takeaways

- Coordinate tokenization is an architectural decision for generative grounding: it determines whether numerical proximity and axis semantics must be rediscovered from data.
- The matched SFT comparison supports the tokenization change, while the reward ablation supports Hi-GAR as a low-IoU repair. Neither establishes which part of the representation—digit hierarchy, axis separation, or smaller vocabulary—causes the gain.
- The fixed 1,000-bin space leaves a visible small-object boundary: ultra-strict localization is highly scale-sensitive, and the paper reports a P@0.95 of 1.30 for its small-object aggregate.
- A decisive follow-up would vary hierarchy, axis vocabularies, vocabulary size, output length, and compute independently, then measure calibration and latency as well as IoU.
- Generative grounding improves when the output language carries geometry instead of encoding every location as an unrelated word.
