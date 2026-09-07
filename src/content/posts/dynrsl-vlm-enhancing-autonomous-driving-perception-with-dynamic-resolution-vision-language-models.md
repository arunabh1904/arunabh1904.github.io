---
title: 'DynRsl-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models'
date: '2025-03-14T10:19:24.000Z'
section: paper-shorts
postSlug: dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models
legacyPath: /paper shorts/2025/03/14/dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2025 – DynRsl-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models"
---
## 2025 – DynRsl-VLM

**arXiv:** [2503.11265](https://arxiv.org/abs/2503.11265)

## Summary

> DynRsl-VLM changes the visual interface of a driving VLM. Instead of accepting a heavily downsampled image, it uses dynamic-resolution processing to retain entity detail while keeping the Vision Transformer input tractable. A custom image-text alignment module replaces a Q-Former for the resulting variable-resolution features. On NuInstruct, the full model reaches 35.5 on planning with reasoning and the ablations show accuracy drops when either added module is removed; the study evaluates VQA-style driving understanding rather than a control policy.

## Core Insights

Resolution is a deployment decision, not merely a vision-backbone setting. Fixed downsampling spends roughly the same visual budget everywhere and can discard the objects that matter most in driving. DynRsl-VLM instead keeps a flexible number of image features, then aligns them to text with an interface designed for that representation. The intended gain is perceptual coverage without an unbounded token cost.

The input construction is more specific than simply zooming detected boxes. YOLOv8 proposes vehicle and pedestrian regions in the high-resolution image; DynRsl-VLM keeps each ROI, also forms combined regions whose boxes contain pairs or groups of entities, and includes a low-resolution full image for global context. A frozen ViT processes the resulting views, and two projection heads align multiple resolution-specific image features to one text feature with symmetric InfoNCE. The three figures therefore trace one idea at three levels: alignment losses, the full model path, and the ROI-plus-global image set that preserves both object detail and relations.

![DynRsl-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models source figure: Architecture of the alignment module and the losses employed during model training.](/assets/images/dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models-paper-figure.webp)
*Fig 1: Architecture of the alignment module and the losses employed during model training. | source: [DynRsl-VLM, Figure 4](https://arxiv.org/abs/2503.11265)*

![Figure 1 from DynRsl-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models](/assets/images/dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models-source-figure-1.webp)
*Fig 2: The architecture of our model that acquires multi-resolution images, performs visual-text alignment, and conducts efficient computations. | source: [DynRsl-VLM, Figure 1](https://arxiv.org/abs/2503.11265)*

![Figure 2 from DynRsl-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models](/assets/images/dynrsl-vlm-enhancing-autonomous-driving-perception-with-dynamic-resolution-vision-language-models-source-figure-2.webp)
*Fig 3: Method for obtaining Region Images. This diagram illustrates the approach for acquiring Region Images, which include both individual entity regions and combined regions. | source: [DynRsl-VLM, Figure 2](https://arxiv.org/abs/2503.11265)*


The paper does report a task-level check. On the NuInstruct test set, the full model reaches 35.5 on planning with reasoning, compared with 35.2 for BEV-InMLLM and 31.4 for MV-MLLM. In the module ablation, the reported aggregated accuracy is 34.8 for the full model, 33.2 without dynamic-resolution extraction, 34.1 without dynamic image-text alignment, and 31.1 without both. The training protocol uses three uniformly sampled frames per video, 224 × 224 crops, a frozen base MLLM with only the added components optimized, AdamW at 1e−4, cosine annealing, and 20 epochs. This connects the figures to a measurable claim: the ROI path and alignment path each retain signal that disappears when both are removed.

The alignment figure should also be read precisely. The pretraining objective combines image-guided text generation (ITG), symmetric image-text contrastive learning (ITC/InfoNCE), and image-text matching (ITM); it is not an InfoNCE-only interface. The language decoder then consumes the aligned multi-resolution features. YOLOv8 proposes vehicle and pedestrian boxes, combined boxes preserve relations, and the downsampled full image supplies global context. The system therefore trades a fixed resize for detector-dependent cropping and extra views. The evaluation is still VQA-style NuInstruct rather than closed-loop driving, and it does not report a matched wall-clock or token-budget comparison against adaptive cropping or fixed high-resolution baselines.

## High-Level Takeaways

- DynRsl-VLM makes preservation of small and distant driving evidence the primary representation decision, then adapts the language-alignment interface to that variable input.
- The abstract establishes a perceptual motivation but does not yet establish that dynamic resolution improves action quality or safety at a fixed latency.
- A matched token- and wall-clock-budget study should compare dynamic resolution with fixed-resolution and adaptive-cropping baselines; the claim weakens if any of them recover the same small-object evidence.
