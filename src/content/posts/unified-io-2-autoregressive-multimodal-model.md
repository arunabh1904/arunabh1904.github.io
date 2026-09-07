---
title: 'Unified-IO 2: Scaling Autoregressive Multimodal Models'
date: '2023-12-28T00:00:00.000Z'
section: paper-shorts
postSlug: unified-io-2-autoregressive-multimodal-model
legacyPath: /paper shorts/2023/12/28/unified-io-2-autoregressive-multimodal-model.html
tags: [Omni-Model Architectures, Multimodal Generation]
field: 'Omni-Model Architectures'
summary: '2023 – Unified-IO 2: Scaling Autoregressive Multimodal Models'
---

## 2023 – Unified-IO 2: Scaling Autoregressive Multimodal Models

**arXiv:** [2312.17172](https://arxiv.org/abs/2312.17172)

## Summary

> Unified-IO 2 makes 220 tasks over 120 datasets share a typed token interface. Text, images, audio, video, coordinates, dense maps, and robot actions still keep their own codecs, but one encoder-decoder transformer learns to route among them. The breadth is real: Unified-IO 2 improves over Unified-IO by 2.7 points on GRIT and reaches 84.6 versus 81.2 on a same-source VQA comparison. The cost is equally concrete: adding video destabilizes the shared mixture late in training, and every codec adds its own information bottleneck.

## Core Insights

### Modalities meet in a shared autoregressive sequence

Unified-IO 2's unification happens at serialization time. Text becomes BPE tokens and points and boxes become coordinate tokens. Image inputs use continuous ViT patch embeddings, while generated images and dense outputs use discrete visual codes; audio similarly has separate input and output representations. Dynamic packing puts those sequences into one encoder context; dynamic unpacking sends decoder outputs to the appropriate detokenizer. The system is broad because the transformer sees a common contract, not because pixels, waveforms, and actions have been forced into one raw representation.

![Unified-IO 2 architecture for packing text, image, audio, history, and structured outputs into one encoder-decoder transformer](/assets/images/unified-io-2-autoregressive-multimodal-model-source-figure-2.webp)
*Fig 1: Unified-IO 2 encodes text, images, audio, and histories into packed embeddings, then decodes text, actions, structured vision targets, images, and audio through modality-specific heads. | source: [Unified-IO 2, Figure 2](https://arxiv.org/abs/2312.17172)*

This typed interface changes what a task head means. A prompt can ask for a caption, segmentation mask, depth map, image, audio clip, or action sequence through the same instruction format. The model therefore supports composition—“look at this image, answer a question, and emit a mask”—without a new end-to-end network for every benchmark. The boundary is the codec: if the visual code loses a small symbol or the coordinate vocabulary is too coarse, the shared transformer cannot recover that evidence.

The paper's breadth also has a measurable engineering footprint. The training mixture spans more than 120 datasets and 220 tasks, and several outputs are long discrete sequences. Audio is limited to roughly 4.08 seconds in the reported setup, while image and video token rates make context length and decoding cost input-dependent. Unified-IO 2 is most useful when a product benefits from composable outputs; a narrow high-resolution workload may get better throughput from specialists.

![Unified-IO 2 loss and gradient-norm curves as modality mixtures are added](/assets/images/unified-io-2-autoregressive-multimodal-model-source-figure-3.webp)
*Fig 2: The reported training curves compare image-only, image-text, and image-text-video mixtures, including loss, gradient norm, and next-token accuracy across streams. | source: [Unified-IO 2, Figure 3](https://arxiv.org/abs/2312.17172)*

The training curves are the paper's sharpest warning. The left panels compare short runs: image generation alone and image-text training remain stable, while adding video makes gradient norms grow. The right panels show a separate XXL run over all modalities whose loss explodes after 350k steps and whose token accuracy falls around 400k. These are failures before the proposed architectural improvements. The repair combines 2D rotary embeddings, query/key LayerNorm, scaled cosine attention in the resampler, float32 attention logits, and delayed unfreezing of the pretrained image and audio encoders. The mechanism is intuitive: a shared next-token objective does not average away different token entropies, sequence lengths, and gradient scales. A universal interface still requires mixture-aware optimization.

The reported task results show why the authors accept that complexity. Unified-IO 2 gains 2.7 points over Unified-IO on GRIT and reaches 84.6 versus 81.2 on the paper's same-source VQA comparison. For image generation, its TIFA score is close to minDALL-E and roughly ten points ahead of the cited CoDi and Emu comparisons. These are breadth results across different protocols, not evidence that the model is the best specialist for each modality.

| Design choice | What it buys | What to monitor |
| --- | --- | --- |
| Shared encoder-decoder | One instruction interface across many tasks | Modality interference and long contexts |
| Discrete codecs | Autoregressive outputs for images, audio, and dense maps | Reconstruction fidelity and token rate |
| Dynamic packing | Multi-modal examples in one sequence | Padding, batching, and latency tails |
| QK and resampler stabilization | More reliable mixed training | Added complexity and tuning surface |

## High-Level Takeaways

- Unified-IO 2 is a typed serialization system around a shared transformer; its modality-specific codecs remain part of the model's behavior.
- Its strongest product case is task composition across many outputs, where routing among specialists would be brittle or expensive to maintain.
- Video destabilizes the short mixture comparison, while the separate all-modality XXL run fails after 350k steps; both motivate testing stability before scaling the shared model.
- The 2.7-point GRIT gain and 84.6 versus 81.2 VQA result support breadth, while codec limits and input-dependent sequence cost define the deployment boundary.
- A fair specialist comparison must match task coverage, data, total activated compute, and output quality rather than compare one headline score.
