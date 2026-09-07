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

### Method and reported result

Chameleon trains token-based, early-fusion models that can understand and generate images and text in arbitrary order within one sequence. Images become discrete tokens, text remains a BPE sequence, and a single autoregressive transformer is trained on text-only, paired, and interleaved documents. The paper reports competitive text performance, strong image captioning and VQA results, and non-trivial image and mixed-modal generation from the same model family.

## Summary

> Chameleon asks a clean architectural question: can a transformer treat an image-text document as one language-like sequence from the beginning? The answer is promising, but the price is paid in visual token count, tokenizer fidelity, and unusually delicate shared-modality training.

## Core Insights

![Chameleon architecture: text and images represented as interleaved discrete tokens for one transformer](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-paper-figure.png)
*Fig 1: Chameleon represents text, images, and code as discrete tokens and feeds the interleaved sequence to one transformer trained end to end. | source: [Chameleon, Figure 1](https://arxiv.org/abs/2405.09818)*

The unification happens at the token interface. Chameleon's image tokenizer maps a 512 × 512 image to 1,024 discrete tokens from an 8,192-entry codebook, while the text vocabulary also contains those image-codebook entries. That is a wonderfully simple interface: the model can place an image before a question, between paragraphs, or after an answer without changing the transformer. It also makes the bottleneck visible. A 512 × 512 image already consumes 1,024 autoregressive positions, and the paper reports that the tokenizer reconstructs images with substantial text poorly. Early fusion therefore buys ordering flexibility by spending context and by asking one codebook to preserve both semantic content and appearance.

![Chameleon evaluation task categories and example prompts](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-source-figure-19.webp)
*Fig 2: The paper's mixed-modal evaluation spans prompt types such as advice, explanation, comparison, identification, and reasoning, with examples that interleave text and images. | source: [Chameleon, Figure 19](https://arxiv.org/abs/2405.09818)*

![Chameleon inter-annotator agreement counts for the absolute evaluation](/assets/images/chameleon-mixed-modal-early-fusion-foundation-models-source-figure-23.webp)
*Fig 3: Inter-annotator agreement counts for questions in the paper's absolute evaluation, showing how much of the human judgment signal is shared across raters. | source: [Chameleon, Figure 23](https://arxiv.org/abs/2405.09818)*

The data recipe is as important as the diagram. The first training stage includes 2.9 trillion text-only tokens, 1.4 billion text-image pairs producing about 1.5 trillion text-image tokens, and 400 billion tokens of interleaved image-text data. The second stage lowers the weight of the first-stage mixture and adds higher-quality and instruction data. This matters because “one sequence” only becomes useful when the training distribution contains the orderings and compositions the model will see at inference.

The paper also shows why a shared transformer is not automatically stable. Above 8B parameters and roughly 1T tokens, late divergence appeared when modalities with different entropy competed through shared softmax layers. QK normalization, norm reordering, dropout choices, and z-loss regularization form a stability recipe; uncontrolled growth in the final-layer output norm was a strong warning signal for future loss divergence. This is a useful operational lesson, not evidence that early fusion is intrinsically stable. The modality mixture, normalization, optimizer, and model scale are coupled in that experiment.

| Decision | Evidence in Chameleon | Cost or boundary |
| --- | --- | --- |
| Representation | Discrete image and text tokens in one stream | Image fidelity and OCR depend on the visual tokenizer. |
| Sharing | One transformer processes arbitrary image-text orderings | Different modality entropies can interfere during training. |
| Objective | Next-token prediction for all modalities | Long visual sequences consume context and decoding steps. |

## High-Level Takeaways

- Chameleon is the clean baseline for early fusion: one autoregressive sequence can contain text, images, and code without a separate image-generation route.
- The decisive engineering variable is visual bandwidth. The 1,024-token representation is flexible, but image detail and OCR quality are constrained by the tokenizer before the language model sees the input.
- The stability figures turn “multimodal interference” into a measurable training concern: monitor output norms and test the normalization recipe before scaling the mixture.
- Its benchmark breadth shows that the interface can support many tasks; it does not establish that discrete early fusion is compute-optimal against continuous visual encoders or diffusion decoders.
- A fair follow-up would match data, parameters, and FLOPs across early fusion, a decoupled visual route, and a continuous-image hybrid, then measure both mixed-document quality and long-context cost.
