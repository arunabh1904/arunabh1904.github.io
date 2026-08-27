---
title: 'PaLI: A Jointly-Scaled Multilingual Language-Image Model'
date: '2022-09-14T00:00:00.000Z'
section: paper-shorts
postSlug: pali-jointly-scaled-multilingual-language-image-model
legacyPath: /paper shorts/2022/09/14/pali-jointly-scaled-multilingual-language-image-model.html
tags: [Vision-Language Models, Multimodal Scaling]
field: 'Multimodal Scaling & Data Mixtures'
summary: '2022 – PaLI: A Jointly-Scaled Multilingual Language-Image Model'
---

## 2022 – PaLI: A Jointly-Scaled Multilingual Language-Image Model

**arXiv:** [2209.06794](https://arxiv.org/abs/2209.06794)

## Summary

> PaLI scales the vision encoder and encoder-decoder language model together, then trains them on a multilingual task mixture. Its WebLI dataset contains 10 billion image-text examples in more than 100 languages. The paper reports state-of-the-art results across captioning, visual question answering, and scene-text understanding while keeping one text-generation interface.

## Core Insights


![Figure 1 from PaLI: A Jointly-Scaled Multilingual Language-Image Model](/assets/images/pali-jointly-scaled-multilingual-language-image-model-source-figure-1.webp)
*Fig 1: The PaLI main architecture is simple and scalable. It uses an encoder-decoder Transformer model, with a large-capacity ViT component for image processing. | source: [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794)*

![Figure 2 from PaLI: A Jointly-Scaled Multilingual Language-Image Model](/assets/images/pali-jointly-scaled-multilingual-language-image-model-source-figure-2.webp)
*Fig 2: PaLI scaling for a number of tasks. We report CIDEr scores for captioning tasks, and accuracy scores for VQA tasks. | source: [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794)*


PaLI asks whether vision capacity must grow with language capacity. The authors train a 4-billion-parameter ViT because contemporary language backbones were much larger than their visual encoders. The reported gains support joint scaling within PaLI's data and compute regime.

The design is modular, but its evidence does not isolate architecture from data scale. A 10-billion-example multilingual mixture changes both coverage and optimization. Reproducing the result therefore requires more than matching parameter count.

## High-Level Takeaways

- PaLI treats visual capacity as a first-class scaling variable.
- One text-generation interface absorbs many visual and multilingual tasks.
- The scale of WebLI makes data composition inseparable from the reported architecture gains.
