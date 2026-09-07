---
title: 'ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations'
date: '2019-08-06T00:00:00.000Z'
section: paper-shorts
postSlug: vilbert-pretraining-task-agnostic-visiolinguistic-representations
legacyPath: /paper shorts/2019/08/06/vilbert-pretraining-task-agnostic-visiolinguistic-representations.html
tags:
  - Vision-Language Models
  - Multimodal Pretraining
field: 'Vision-Language Models'
summary: '2019 – ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations'
---

## 2019 – ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks

**arXiv:** [1908.02265](https://arxiv.org/abs/1908.02265)

## Summary

> ViLBERT makes visual grounding a pretraining problem. It keeps detector regions and words in separate Transformer streams, then lets them exchange information through co-attention layers. Two proxy tasks—masked multimodal prediction and image-text alignment—transfer the resulting representation to VQA, visual commonsense reasoning, referring expressions, and image retrieval. The transfer is broad, but its visual vocabulary is still fixed by the upstream region detector.

## Core Insights

### Separate streams let each modality spend depth differently

![ViLBERT architecture with separate visual and language streams connected by co-attention](/assets/images/vilbert-paper-figure-1.png)
*Fig 1: ViLBERT processes region features and word tokens in parallel, inserting sparse co-attention blocks where one stream can read the other. The dashed repeat block makes the different visual and linguistic depths explicit. | source: [ViLBERT, Figure 1](https://arxiv.org/abs/1908.02265)*

ViLBERT starts with a set of image regions rather than pixels. A Faster R-CNN with a ResNet-101 backbone, pretrained on Visual Genome, supplies 10–36 high-scoring regions per image. Each region is a mean-pooled visual feature plus a projected five-dimensional location vector containing normalized top-left and bottom-right coordinates and the fraction of image area it covers. A special `IMG` region is the mean-pooled whole-image feature. The language side starts from BERT-base embeddings and processes WordPiece tokens.

The two streams do not share every layer. A co-attentional Transformer layer keeps each modality’s queries but swaps keys and values across the streams. The visual update is therefore conditioned on language, and the language update is conditioned on regions; residual connections and feed-forward layers then finish each Transformer block. The paper gives the text stream more standalone processing because its words need sentence context, while detector regions already arrive as high-level visual summaries. That is the useful architectural bet: interaction is placed where it can align the modalities without forcing a region feature through the same depth as a word sequence.

The model uses a 1024-dimensional visual stream with eight attention heads and initializes the linguistic stream from BERT-base. The resulting joint representation is not a single universal output head. Downstream tasks attach small classifiers or matching layers and fine-tune the whole model, so the pretraining interface stays shared while the decision rule changes.

### The pretraining signal asks for semantics and compatibility

![ViLBERT pretraining tasks for masked multimodal learning and image-text alignment](/assets/images/vilbert-pretraining-task-agnostic-visiolinguistic-representations-source-figure-3.png)
*Fig 2: ViLBERT’s two proxy tasks either reconstruct masked words or region categories from the remaining pair, or classify whether the complete image-caption pair belongs together. | source: [ViLBERT, Figure 3](https://arxiv.org/abs/1908.02265)*

The masked multimodal task hides about 15% of words and regions. Text masking follows BERT. For a masked region, the feature is zeroed 90% of the time and left unchanged 10% of the time; the target is a distribution over detector semantic classes, trained with KL divergence rather than direct feature regression. The choice is deliberate: a caption can identify “dog” without reconstructing the exact region vector. In the alignment task, the model uses the `IMG` and `CLS` outputs, combines them elementwise, and predicts whether the pair matches. Since Conceptual Captions supplies aligned pairs only, negatives come from replacing either the image or caption with one from another example.

The data are broad but weak. The authors train on about 3.1 million usable pairs from the 3.3 million-image Conceptual Captions release, whose alt text can be editorial, vague, or only loosely visual. Pretraining runs for ten epochs with batch size 512 on eight TitanX GPUs, using Adam with an initial learning rate of $10^{-4}$ and equal task-loss weights. The model is then transferred to four tasks, with a fifth zero-shot retrieval diagnostic that uses the pretrained alignment score without task-specific fine-tuning.

### Transfer is broad, while depth and pretraining both matter

| Transfer task | ViLBERT result | Protocol boundary |
| --- | ---: | --- |
| VQA 2.0 | 70.55 test-dev (70.92 test-standard) | 3,129-answer soft-target classification |
| VCR | 72.42 Q→A / 74.47 QA→R / 54.04 Q→AR | four-way multiple choice on 290K questions |
| RefCOCO+ | 72.34 val / 78.52 testA / 62.61 testB | reranks Mask R-CNN proposals; IoU threshold 0.5 |
| Flickr30K image retrieval | 58.20 / 84.90 / 91.52 R@1/5/10 | fine-tuned pair scoring |
| Flickr30K zero-shot retrieval | 31.86 / 61.12 / 72.80 R@1/5/10 | pretrained alignment score only |

The full model beats the paper’s task-specific comparisons across the four transfer tasks, but the ablations make the attribution more useful. On VQA, the pretrained two-stream model reaches 70.55 versus 68.93 for an unpretrained ViLBERT-shaped model and 65.90 for the single-stream unpretrained baseline. On fine-tuned Flickr30K retrieval, the corresponding two-stream score is 58.20 R@1 versus 45.50 for the single-stream unpretrained model. Zero-shot retrieval reaches 31.86 R@1, showing that the Conceptual Captions objective learned a meaningful alignment before seeing Flickr30K, while also showing how much task-specific supervision still contributes.

The depth study does not support one magic number. Six repeated co-attention blocks give the best VQA result among the tested depths (70.55), while retrieval continues to rise from 55.68 R@1 at two blocks to 58.78 at eight. VCR and RefCOCO+ are slightly better with shallower variants. Data scale is similarly regular: zero-shot retrieval R@1 rises from 0 with no Conceptual Captions pretraining to 20.40, 26.76, and 31.86 when using 25%, 50%, and 100% of the corpus. The model benefits from more alignment data, but the task-dependent depth trend says that “more fusion” is not itself the claim.

### The detector still decides what the model can see

ViLBERT makes visual-language interaction transferable while retaining a fixed visual front end. A caption can help identify a proposed region, but co-attention cannot retrieve visual evidence absent from the detector outputs. Its bidirectional BERT-style core also serves discriminative heads rather than directly generating open-ended answers. The depth ablation and detector dependence together locate the contribution: learning where to exchange semantic information, given an existing region vocabulary.

## High-Level Takeaways

- ViLBERT pretrains grounding with separate region and language streams joined by sparse co-attention.
- Masked region semantics and image-text matching provide complementary local and global supervision.
- Transfer improves across VQA, VCR, referring expressions, and retrieval, while the best fusion depth depends on the task.
- Detector proposals and noisy web captions remain the main limits; the method cannot recover visual evidence the detector never emits.
