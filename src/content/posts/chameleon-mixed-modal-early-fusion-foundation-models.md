---
title: 'Chameleon: Mixed-Modal Early-Fusion Foundation Models'
date: '2024-05-16T00:00:00.000Z'
section: paper-shorts
postSlug: chameleon-mixed-modal-early-fusion-foundation-models
legacyPath: /paper shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html
tags: [Multimodal AI]
field: 'Omni-Model Architectures'
summary: "2024 – Chameleon: Mixed-Modal Early-Fusion Foundation Models"
---

## 2024 – Chameleon: Mixed-Modal Early-Fusion Foundation Models

**arXiv:** [2405.09818](https://arxiv.org/abs/2405.09818)  
**Conference:** Technical report

## Summary

> Chameleon asks whether a transformer can treat an image-text document as one language-like sequence from the beginning. The paper's answer is promising: a single early-fusion model reports competitive text, captioning, VQA, and mixed-modal generation results. The price is paid in visual token count, tokenizer fidelity, and unusually delicate shared-modality training.

## Core Insights

Chameleon's central decision is to make images discrete before the language model sees them. Its image tokenizer maps a 512 × 512 image to 1,024 tokens from an 8,192-entry codebook, while the text vocabulary also contains those image-codebook entries. One autoregressive transformer can then place an image before a question, between paragraphs, or after an answer without changing its sequence-processing machinery.

![Chameleon architecture: text and images represented as interleaved discrete tokens for one transformer](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-paper-figure.png)
*Fig 1: Chameleon represents text, images, and code as discrete tokens and feeds the interleaved sequence to one transformer trained end to end. | source: [Chameleon, Figure 1](https://arxiv.org/abs/2405.09818)*

The interface is simple, but its cost is visible in the sequence length. A single 512 × 512 image already consumes 1,024 autoregressive positions. The paper also reports that its tokenizer reconstructs images containing substantial text poorly, which limits OCR-heavy use. Early fusion therefore buys arbitrary image-text ordering by asking the codec to preserve both semantic evidence and appearance under a fixed token budget.

![Chameleon task categories and example prompts used in the mixed-modal human evaluation](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-source-figure-19.webp)
*Fig 2: The human-evaluation prompt set is unevenly distributed: brainstorming is 18.6%, explanation 14.4%, how-to 12.5%, advice 10.2%, identification 9.3%, and reasoning 2.1%, with the remaining categories making up the rest. | source: [Chameleon, Figure 8](https://arxiv.org/abs/2405.09818)*

That composition matters when reading the aggregate human result. The authors' Figure 8 includes everyday mixed-modal prompts, but OCR is explicitly absent from the evaluation set. Chameleon performs well in brainstorming, comparison, and hypothetical prompts, while identification and reasoning are weaker. A capability headline drawn from this set mostly measures interactive generation and visual prompting; it is not a broad test of text-in-image understanding.

![Chameleon inter-annotator agreement counts for the absolute evaluation](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-source-figure-23.webp)
*Fig 3: Three annotators judge each absolute-evaluation question. Unanimous agreement dominates objective properties such as objectionable content, while task fulfillment and relevance show more close two-of-three disagreements; “no agreement” remains a small category. | source: [Chameleon, Figure 10](https://arxiv.org/abs/2405.09818)*

The agreement figure supplies the boundary around the human evaluation. Every question is judged by three annotators, and the final answer uses majority vote. In the reported 52B comparison, Chameleon fully fulfills 55.2% of tasks, versus 37.6% for Gemini+ and 44.7% for GPT-4V+. Crowdworkers prefer RL-CAI-style behavior at similar helpfulness/harmfulness tradeoffs, but the result is still a judgment protocol over a category mix, not a universal visual-grounding score.

The data and stability recipes are equally structural. The first training stage contains 2.9T text-only tokens, 1.4B text-image pairs producing about 1.5T text-image tokens, and 400B interleaved image-text tokens. The second stage downweights the first mixture and adds higher-quality and instruction data. Above 8B parameters and roughly 1T tokens, late divergence appeared when modalities with different entropy competed through shared softmax layers. QK normalization, norm reordering, dropout choices, and z-loss form the stabilization recipe; uncontrolled growth in the final-layer output norm predicted later loss divergence. The paper shows how to make early fusion work at scale, not that it is intrinsically stable.

| Decision | Evidence in Chameleon | Cost or boundary |
| --- | --- | --- |
| Representation | Discrete image and text tokens in one stream | Image fidelity and OCR depend on the visual tokenizer. |
| Sharing | One transformer processes arbitrary image-text orderings | Different modality entropies can interfere during training. |
| Objective | Next-token prediction for all modalities | Long visual sequences consume context and decoding steps. |

## High-Level Takeaways

- Chameleon is a clean baseline for early fusion: one autoregressive sequence can contain text, images, and code without a separate image-generation route.
- The 1,024-token image representation is flexible, but the tokenizer is the hard information boundary; the paper itself identifies text-heavy images as a weakness.
- The human evaluation is broad but uneven, with no OCR category and a three-annotator majority-vote protocol; its 55.2% task-fulfillment result should be read in that context.
- The stability result is operational: monitor output norms and reproduce the QK-norm/z-loss recipe before scaling a mixed-modality run.
- The core trade is clear in deployment terms: arbitrary mixed documents versus visual-token context, codec quality, and shared-modality interference.
