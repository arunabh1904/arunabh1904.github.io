---
title: 'Flamingo: A Visual Language Model for Few-Shot Learning'
date: '2022-04-29T00:00:00.000Z'
section: paper-shorts
postSlug: flamingo-visual-language-model-for-few-shot-learning
legacyPath: /paper shorts/2022/04/29/flamingo-visual-language-model-for-few-shot-learning.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2022 – Flamingo: A Visual Language Model for Few-Shot Learning'
---

## 2022 – Flamingo: A Visual Language Model for Few-Shot Learning

**arXiv:** [2204.14198](https://arxiv.org/abs/2204.14198)

## Summary

> Flamingo teaches a frozen language model to use interleaved images, video, and text through newly trained visual resampling and gated cross-attention layers. Once trained, it can switch tasks by reading a few image-answer examples in its prompt. The central design problem is how to add visual evidence without disrupting the language model, while allowing a prompt to contain more images than were seen together during training.

## Core Insights

### Compress each visual input before asking the language model to read it

Flamingo uses a frozen NFNet-F6 image encoder and a frozen Chinchilla language model. Between them, a trainable Perceiver Resampler turns a variable-length visual feature sequence into 64 output tokens per image or video. Learned latent queries repeatedly attend to the visual features to collect the information that will be useful for text prediction. For video, frames are sampled at one frame per second, encoded separately, and given temporal embeddings before resampling.

The fixed output count controls the cost of the later text-to-vision attention. A longer video can create more encoder features, but each downstream cross-attention block still reads 64 resampled tokens for that video. The encoder and resampler do not become free, and those 64 tokens must represent a larger amount of evidence as the input grows. This creates a practical distinction between accepting a longer visual input and retaining every small object or brief event in it.

![Flamingo's frozen encoders, trainable resamplers, and interleaved visual-text sequence](/assets/images/flamingo-source-figure-3-resampler.png)
*Fig 1: Each visual input is encoded and resampled before entering newly trained blocks among the frozen language-model layers. The text stream retains the positions of the image markers. | source: [Paper, Figure 3](https://arxiv.org/abs/2204.14198)*

In Figure 1, follow the dog and cat through the separate visual paths, then look at the text sequence on the right. The first image and its description provide an example of the desired behavior; the second image is the query. At inference, changing that demonstration changes the task context without updating weights. The resampler does not itself perform few-shot learning: that behavior depends on the language model using the interleaved sequence and the training mixture having taught it how such sequences work.

### Start the added visual pathway at zero contribution

Flamingo inserts cross-attention and feed-forward blocks among the pretrained language layers. In a cross-attention block, the current language state supplies queries and the resampled image features supply keys and values. Its output is multiplied by a learned scalar passed through tanh before being added to the residual stream. That scalar starts at zero. The new feed-forward contribution has its own gate.

![Flamingo's gated cross-attention and feed-forward additions to a frozen language block](/assets/images/flamingo-source-figure-4-gated-attention.png)
*Fig 2: Purple modules add trainable visual conditioning; blue modules retain the pretrained language computation. Zero-initialized tanh gates initially suppress the added residual contributions. | source: [Paper, Figure 4](https://arxiv.org/abs/2204.14198)*

Read Figure 2 from the bottom up. The visual branch can compute an attention result, but at initialization its gate contributes zero to the language state. The original pretrained computation is therefore preserved before the new pathway has learned anything useful. As training opens the gates, visual information can influence the frozen language layers. Freezing weights does not freeze their activations: the same language parameters process increasingly visual-conditioned inputs.

The ablation supports the gate's role under this training setup. In the shorter Flamingo-3B experiment, removing tanh gating lowers the normalized overall score from 70.7 to 66.5 and produces training instability. The score averages performance relative to prior best results across five development benchmarks; it is not an accuracy percentage. Likewise, training the pretrained language model instead of freezing it yields 62.7 in that sweep. These results favor preserving the pretrained model during this multimodal training regime, without establishing that language-model fine-tuning is always harmful.

### Directly read the latest image, remember earlier ones through language

At each text token, Flamingo's cross-attention mask exposes the visual tokens of the immediately preceding image or video. It does not directly expose every earlier image to that text token. Earlier visual information can still influence the answer through causal self-attention over language states that have already incorporated it.

This makes the repeated unit manageable: read one visual input, incorporate it into the language stream, then continue. Training on M3W uses at most five images per sequence, yet evaluation benefits from up to 32 image/video demonstrations. The architecture supports that longer interleaving without expanding the set of images visible to an individual cross-attention operation. It is not a guarantee of unlimited context or perfect comparison across distant images; the language context and the representations carried through it still matter.

### Interleaved training data teaches the structure of the eventual prompt

The training mixture combines approximately 43 million M3W webpages, 1.8 billion ALIGN image-text pairs, 312 million LTIP image-text pairs, and 27 million VTP video-text pairs. M3W preserves the placement of images among text, using document structure to construct the sequence. Paired caption datasets provide broad visual-language coverage, while interleaved webpages supply examples of repeatedly moving between visual and textual context.

Removing M3W drops the short-run ablation score from 70.7 to 53.4, compared with 60.9 after removing paired image-text data. Both sources matter. The comparison is not a pure experiment on interleaving with data volume and content held fixed: removing a dataset also changes the training distribution and work performed per update. Still, the large loss from removing M3W supports the paper's central connection between training-sequence structure and multimodal prompting.

### More demonstrations help, but the comparison budget needs care

The largest Flamingo combines a 70B language model with added components for roughly 80B total parameters. Its Table 1 reports the following results without task-specific weight updates:

| Task and metric | Four demonstrations | Thirty-two demonstrations |
| --- | --- | --- |
| OK-VQA accuracy | 57.4 | 57.8 |
| VQAv2 accuracy | 63.1 | 67.6 |
| COCO captioning CIDEr | 103.2 | 113.8 |

The gains vary substantially by task. More examples improve COCO captioning and VQAv2 much more than OK-VQA here; providing additional demonstrations does not automatically supply missing world knowledge. Across the full table, the 32-shot model exceeds the listed task-fine-tuned best result on seven tasks. The paper's separate fine-tuning experiments update model weights and should not be folded into that claim.

Even the “zero-shot” rows need their appendix definition. For open-ended tasks, the prompt contains two task examples with their images or videos removed, providing text-only examples of the expected output. Closed-ended answer scoring does not require those examples. This is zero visual demonstrations, rather than a completely example-free instruction prompt.

The contribution is rapid task adaptation after substantial multimodal pretraining. It reduces the need for a new weight-training run for each downstream task, while retaining the cost of a large pretrained system, visual processing, and longer prompts. Its architecture makes those costs easier to separate and study.

## High-Level Takeaways

- Resampling fixes each image or video's downstream visual-token budget at 64, while leaving encoder cost and information compression as separate concerns.
- Zero-initialized gates introduce visual conditioning gradually into frozen language layers; their ablation supports this stabilization strategy.
- Per-image cross-attention and language self-attention play different roles: direct access is local to the latest visual input, while earlier context persists through language states.
- Interleaved web training is central to the few-shot result, and the strongest comparisons use a large pretrained model with explicitly counted demonstrations.
