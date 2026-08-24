---
title: 'InternVL3: Native Multimodal Pretraining'
date: '2025-04-14T00:00:00.000Z'
section: paper-shorts
postSlug: internvl3-native-multimodal-pretraining
legacyPath: /paper shorts/2025/04/14/internvl3-native-multimodal-pretraining.html
tags: [Vision-Language Models, Multimodal Pretraining]
field: 'Vision-Language Models'
summary: '2025 – InternVL3: Native Multimodal Pretraining'
---

## 2025 – InternVL3: Native Multimodal Pretraining

**arXiv:** [2504.10479](https://arxiv.org/abs/2504.10479)

## Summary

> InternVL3 develops language and multimodal capability in one pretraining stage instead of attaching vision only after text pretraining. It adds variable visual position encoding for long multimodal contexts, mixed preference optimization after supervised tuning, and test-time scaling. InternVL3-78B reports 72.2 on MMMU.

## Core Insights

![InternVL3 benchmark comparison with open and closed multimodal models](/assets/images/internvl3-paper-figure-1.svg)
*The reported comparison places InternVL3 across a broad multimodal benchmark suite. source: [InternVL3](https://arxiv.org/abs/2504.10479)*

![Figure 1 from InternVL3: Native Multimodal Pretraining](/assets/images/internvl3-native-multimodal-pretraining-source-figure-1.webp)
*Figure 1 Multimodal performance of the InternVL series and other advanced MLLMs. The InternVL series has consistently exhibited progressive enhancements in multimodal capabilities. The newly released InternVL3 significantly outperforms existing open-source MLLMs. Moreover, even in comparison with state-of-the-art closed-source commercial models, InternVL3 continues to demonstrate highly competitive performance. source: [InternVL3: Native Multimodal Pretraining](https://arxiv.org/abs/2504.10479)*

![Figure 2 from InternVL3: Native Multimodal Pretraining](/assets/images/internvl3-native-multimodal-pretraining-source-figure-2.webp)
*Figure 2 Performance of various MLLMs on the OpenCompass multimodal academic leaderboard. The enhanced InternVL series—InternVL3—demonstrates outstanding multimodal capabilities, significantly outperforming both the Qwen2.5-VL series and closed-source models such as Step-1o, GLM-4v-Plus, and GPT-4o. Remarkably, InternVL3-78B also remains highly competitive with the state-of-the-art Gemini-2.5-Pro. source: [InternVL3: Native Multimodal Pretraining](https://arxiv.org/abs/2504.10479)*


Here, native multimodal pretraining describes the training schedule. Multimodal and text data appear together during the main pretraining stage, though the system still initializes strong pretrained components. Vision is no longer confined to a small adapter trained after the language model is complete.

This enables deeper adaptation but weakens attribution. Data mixture, positional encoding, post-training, test-time compute, and model scale all move together. The reported endpoint is strong; a matched ablation is needed to isolate which stage produced each gain.

## High-Level Takeaways

- InternVL3 moves multimodal learning into the main pretraining stage.
- Variable visual positions support longer interleaved contexts.
- End-to-end capability improves, while the coupled recipe makes causal attribution harder.
