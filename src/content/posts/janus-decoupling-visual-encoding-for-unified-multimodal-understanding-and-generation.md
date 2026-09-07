---
title: 'Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation'
date: '2024-10-17T00:00:00.000Z'
section: paper-shorts
postSlug: janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation
legacyPath: /paper shorts/2024/10/17/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation.html
tags: [Multimodal AI]
field: 'Omni-Model Architectures'
summary: "2024 – Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation"
---

## 2024 – Janus

**arXiv:** [2410.13848](https://arxiv.org/abs/2410.13848)  
**Conference:** Technical report

### Method and reported result

Janus keeps one autoregressive transformer but uses separate visual encoders for understanding and image generation. The understanding route uses a SigLIP encoder and adaptor for semantic features; the generation route uses a VQ tokenizer and adaptor for discrete image tokens. The paper reports strong multimodal-understanding results for its 1.3B model and competitive text-to-image generation, arguing that the visual interface—not the shared transformer—is where the main conflict lies.

## Summary

> Janus separates the things that need to be invariant from the things that need to be reconstructed. Understanding wants a compact semantic description; generation wants a fine-grained image code. Sharing the transformer preserves cross-modal reasoning while allowing each visual route to keep the information its task actually needs.

## Core Insights

![Janus architecture with separate visual understanding and generation encoders feeding one autoregressive transformer](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-paper-figure.png)
*Fig 1: Janus decouples visual encoding for understanding and generation while routing both representations through one unified autoregressive transformer. | source: [Janus, Figure 2](https://arxiv.org/abs/2410.13848)*

A single visual representation has an awkward job. For question answering, the encoder should discard nuisance detail and expose high-level semantics; for image generation, it must preserve local appearance well enough to reconstruct pixels or image codes. Janus makes that tension explicit. SigLIP features are flattened and mapped into the language-model space for understanding, while a VQ tokenizer turns target images into codebook IDs for generation. The language model can therefore share sequence reasoning without forcing one encoder to optimize two incompatible information bottlenecks.

![Janus multimodal understanding and visual-generation benchmark results](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-source-figure-1.webp)
*Fig 2: Janus's reported understanding and visual-generation results are compared with similarly sized multimodal and image-generation systems. | source: [Janus, Figure 1](https://arxiv.org/abs/2410.13848)*

The training schedule is part of the design, not a footnote. Stage I freezes the encoders and language model while learning the two adaptors and image head. Stage II unfreezes the language model for unified pretraining on pure text, multimodal understanding, and visual-generation data. Stage III performs mixed supervised fine-tuning while keeping the generation encoder fixed. This staged path first teaches interfaces to speak the language model's embedding space, then lets the shared reasoning core adapt, and finally aligns behavior. It also means that “decoupling” is tested together with a particular optimization curriculum.

![Janus three-stage training procedure and module update schedule](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-source-figure-3.webp)
*Fig 3: The training diagram marks which adaptors, heads, encoders, and the language model are updated or frozen across the three stages. | source: [Janus, Figure 3](https://arxiv.org/abs/2410.13848)*

The radar chart is useful as a capability map rather than a single score. Janus's 1.3B system expands across POPE, VQA, GQA, MMMU, MMBench, MM-Vet, SEED-Bench, and GenEval while using the same transformer for both task families. The result supports the practical compromise, but it does not isolate the cause: Janus also has two encoders, adaptors, a generation head, staged data, and a particular base language model. A capacity- and compute-matched comparison against a stronger shared tokenizer is still needed.

| Design question | Janus's answer | Cost or boundary |
| --- | --- | --- |
| What is shared? | The autoregressive language-model core | Shared layers can still experience task interference. |
| What is specialized? | Semantic understanding and generative visual encoders | More modules and separate token spaces must be maintained. |
| What is optimized? | Text loss for understanding and image-token loss for generation | The curriculum and data ratios affect attribution. |

## High-Level Takeaways

- Janus is a useful middle point between a fully shared visual representation and two entirely separate models.
- Its deepest idea is information allocation: semantic understanding and pixel reconstruction should not be forced through the same visual code.
- The three-stage schedule explains how the routes become usable, but it also makes the final gains a property of architecture plus curriculum.
- A decisive follow-up would match total parameters, data, and training FLOPs against one stronger dual-purpose encoder and against separate transformers.
- Share the expensive reasoning core when transfer is valuable; specialize the visual interfaces when their information requirements diverge.
