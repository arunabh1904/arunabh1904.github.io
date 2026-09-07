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

## Summary

> Janus tests a practical design hypothesis: in a unified multimodal model, visual understanding and image generation may need different information bottlenecks even when their reasoning core can be shared. It uses a SigLIP route for understanding, a VQ-token route for generation, and one autoregressive transformer. Its 1.3B model reports 69.4 on MMBench, 63.7 on SEED-Bench, 87.0 on POPE, 61% on GenEval, and 8.53 FID on COCO-30K.

## Core Insights

### Specialized visual routes share one transformer

Janus does not claim that every understanding system must use invariant semantic features or that every generator must preserve the same level of detail. Its narrower argument is architectural: in this setup, using one visual encoder for both jobs creates a tradeoff worth removing. SigLIP features are flattened and mapped into the language-model space for understanding, while a VQ tokenizer turns target images into codebook IDs for generation. Both routes then enter one shared autoregressive transformer.

![Janus architecture with separate visual understanding and generation encoders feeding one autoregressive transformer](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-paper-figure.png)
*Fig 1: Janus decouples visual encoding for understanding and generation while routing both representations through one unified autoregressive transformer. | source: [Janus, Figure 2](https://arxiv.org/abs/2410.13848)*

The radar chart is useful because it separates the two sides of the claim. In the paper's reported 1.3B result, Janus reaches 69.4 MMBench, 63.7 SEED-Bench, and 87.0 POPE, exceeding the listed LLaVA-v1.5 and Qwen-VL-Chat 7B comparisons on those metrics. On generation, the same model reaches 61% GenEval accuracy and 8.53 FID on COCO-30K, compared with 53% GenEval for unified Show-o and 55% for SDXL. The comparison is encouraging, but the model has two encoders, adaptors, a generation head, and a staged curriculum; the numbers do not isolate which addition produced the gain.

![Janus multimodal understanding and visual-generation benchmark results](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-source-figure-1.webp)
*Fig 2: Janus's reported understanding and visual-generation results are compared with similarly sized multimodal and image-generation systems. | source: [Janus, Figure 1](https://arxiv.org/abs/2410.13848)*

The schedule is part of the method. Stage I freezes the visual encoders and language model while learning the understanding adaptor, generation adaptor, and image head. Stage II unfreezes the language model for unified pretraining on pure text, multimodal understanding, and visual-generation data. Stage III performs mixed supervised fine-tuning while keeping the generation encoder fixed. First the routes learn to speak the language model's embedding space; then the shared reasoning core adapts; finally the model is aligned for dialogue and generation.

![Janus three-stage training procedure and module update schedule](/assets/images/janus-decoupling-visual-encoding-for-unified-multimodal-understanding-and-generation-source-figure-3.webp)
*Fig 3: The training diagram marks which adaptors, heads, encoders, and the language model are updated or frozen across the three stages. | source: [Janus, Figure 3](https://arxiv.org/abs/2410.13848)*

This curriculum makes the interface hypothesis testable, but it also limits attribution. The base is DeepSeek-LLM 1.3B with a 4,096-token context; the understanding encoder is SigLIP-Large-Patch16-384, and the generation encoder uses a 16,384-entry codebook with 16× downsampling. A capacity-matched comparison against one stronger dual-purpose encoder would need to hold the parameter count, visual-token budget, data, and training steps fixed.

| Design question | Janus's answer | Cost or boundary |
| --- | --- | --- |
| What is shared? | The autoregressive language-model core | Shared layers can still experience task interference. |
| What is specialized? | Semantic understanding and generative visual encoders | More modules and token spaces must be maintained. |
| What is optimized? | Text loss for understanding and image-token loss for generation | The curriculum and data ratios affect attribution. |

## High-Level Takeaways

- Janus is a middle point between a fully shared visual representation and two entirely separate models.
- Its evidence supports a design hypothesis about this model family; it does not prove that semantic and generative encoders are universally incompatible.
- The 1.3B results are unusually strong against larger baselines, but the extra encoders, adaptors, head, and training stages belong in the comparison.
- The three-stage diagram explains how the routes become usable and why the final behavior cannot be attributed to decoupling alone.
- The practical choice is whether the transfer gained from a shared reasoning core outweighs the maintenance and inference cost of specialized visual interfaces.
