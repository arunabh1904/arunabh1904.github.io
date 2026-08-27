---
title: Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)
date: '2023-10-05T00:00:00.000Z'
section: paper-shorts
postSlug: improved-baselines-with-visual-instruction-tuning-llava-1-5
legacyPath: /paper shorts/2023/10/05/improved-baselines-with-visual-instruction-tuning-llava-1-5.html
tags:
  - Multimodal AI
field: 'Vision-Language Models'
summary: '2023 – Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)'
---

## 2023 – Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)

**Paper:** [arXiv 2310.03744](https://arxiv.org/abs/2310.03744)

**Conference:** [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html)

**Code:** [LLaVA](https://github.com/haotian-liu/LLaVA)

## Summary

> LLaVA-1.5 strengthens the original [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) recipe with a two-layer MLP projector, a 336-pixel CLIP encoder, academic VQA data, and explicit response-format prompts. The paper reports state-of-the-art results across 11 benchmarks using 1.2 million public training examples and about one day on eight A100 GPUs. Its stepwise ablation matters more than the final score: replacing the linear projector with an MLP improves GQA from 46.8 to 47.3, MME from 1323.8 to 1355.2, and MM-Vet from 26.3 to 27.8.

## Core Insights

The original LLaVA projects every selected CLIP patch feature into the language model with one linear layer. LLaVA-1.5 replaces that layer with a two-layer MLP. The connector still preserves the patch-token sequence, but it can now learn a nonlinear mapping between the vision encoder and Vicuna's embedding space.

![LLaVA-1.5 benchmark comparison, training sample counts, and MLP projector architecture](/assets/images/improved-baselines-visual-instruction-tuning-paper-figure-1.png)
*Fig 1: Combines the final benchmark comparison with the architecture and training-sample counts. The connector is a small part of the system, while the complete recipe also changes data, prompting, visual resolution, and language-model scale. | source: [Improved Baselines with Visual Instruction Tuning](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf)*

![Figure 3 from Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](/assets/images/improved-baselines-with-visual-instruction-tuning-llava-1-5-source-figure-3.webp)
*Fig 2: Ablation on LLM choices. Data points represent the relative performance of the best performing variant for each dataset. | source: [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744)*

![Figure 2 from Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](/assets/images/improved-baselines-with-visual-instruction-tuning-llava-1-5-source-figure-2.webp)
*Fig 3: LLaVA-1.5-HD. Scaling LLaVA-1.5 to higher resolutions by splitting the image into grids and encoding them independently. | source: [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744)*


The paper builds the recipe one change at a time. Adding a response-format prompt improves MME from 1197.0 to 1323.8 while slightly reducing GQA and MM-Vet. Replacing the linear connector with the MLP then improves all three reported metrics. Later rows add OCR and open-knowledge VQA data, region-level examples, higher resolution, GQA, ShareGPT, and a 13B language model. The final checkpoint is therefore evidence for the whole recipe, while the MLP row isolates a smaller connector gain.

This comparison also clarifies the difference from [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html). BLIP-2 uses a 188-million-parameter Q-Former to compress image features into 32 learned query outputs before a frozen language model. LLaVA-1.5 sends projected patch features directly into a language model that is updated during instruction tuning. The connector can remain small because the language model participates in the alignment.

[MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) later compared connector families while controlling more of the visual interface. Its experiments found that visual-token count and image resolution mattered more than the connector type. Taken together, the papers support a practical rule: begin with a direct MLP projector when the token budget is acceptable, and add learned compression only when reducing or selecting visual tokens solves a measured constraint.

## High-Level Takeaways

- A two-layer MLP improves the original linear projector within the LLaVA recipe without introducing a separate query transformer.
- The MLP maps features but does not reduce their number. Visual compression remains a separate architectural decision.
- LLaVA-1.5's final result combines connector, data, prompting, resolution, and model-scale changes, so the full gain cannot be assigned to the projector.
- The reported ablation favors the MLP over the linear layer for this model. A matched study such as MM1 is still needed before treating that result as a universal connector ranking.
