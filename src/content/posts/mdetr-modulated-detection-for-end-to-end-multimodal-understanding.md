---
title: 'MDETR: Modulated Detection for End-to-End Multimodal Understanding'
date: '2021-04-26T00:00:00.000Z'
section: paper-shorts
postSlug: mdetr-modulated-detection-for-end-to-end-multimodal-understanding
legacyPath: /paper shorts/2021/04/26/mdetr-modulated-detection-for-end-to-end-multimodal-understanding.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2021 – MDETR: Modulated Detection for End-to-End Multimodal Understanding'
---

## 2021 – MDETR: Modulated Detection for End-to-End Multimodal Understanding

**arXiv:** [2104.12763](https://arxiv.org/abs/2104.12763)

**Code:** [ashkamath/mdetr](https://github.com/ashkamath/mdetr)

## Summary

> MDETR makes detection conditional on a free-form text query. It early-fuses image features and RoBERTa tokens, then uses DETR object queries to predict boxes aligned to text spans rather than fixed category IDs. Pretraining on 1.3M aligned image-text pairs transfers to phrase grounding, referring expressions, segmentation, visual question answering, and few-shot detection. The method removes the frozen detector bottleneck, but pays for dense phrase-box supervision and a fused image-text pass for every query.

## Core Insights

### The query defines what counts as an object

![MDETR output for the text query “A pink elephant”](/assets/images/mdetr-modulated-detection-for-end-to-end-multimodal-understanding-source-figure-1.webp)
*Fig 1: Given “A pink elephant,” MDETR selects the pink elephant and labels its box through the query, even though the training data contain neither pink nor blue elephants as fixed detector classes. The colors shown are the pixels, not segmentation masks. | source: [MDETR, Figure 1](https://arxiv.org/abs/2104.12763)*

Earlier vision-language systems usually begin with a detector trained on a fixed object vocabulary. MDETR makes the detector itself text-conditioned. A convolutional backbone produces spatial image features with 2D positional embeddings. RoBERTa produces text features. Modality-specific linear layers project both sequences into one space, concatenate them, and feed them to a Transformer encoder. A DETR-style decoder then lets learned object queries cross-attend to the fused image-text sequence and predict boxes.

The output is not a category label such as “dog” or “chair.” For each matched query, MDETR predicts a distribution over the token positions that refer to the box. Unmatched queries predict the `∅` no-object class. Hungarian matching pairs predicted queries with ground-truth boxes, while L1 and generalized-IoU losses supervise the geometry. This changes the output contract: the same model can answer “the person in the blue shirt,” “objects to the left of the table,” or a new attribute combination without adding a classifier head.

### Alignment needs both token spans and representation similarity

The first alignment loss is soft token prediction. For a matched box, the target is a uniform distribution over all text tokens that refer to that object; multiple boxes may share a phrase, and one object can be associated with several words. The second is a symmetric contrastive loss between decoder object embeddings and the corresponding cross-encoder token embeddings, with temperature $0.07$. Soft spans tell a query where its language evidence occurs; contrastive alignment makes the visual and textual representations close. The ablations show why both are needed: on CLEVR-Medium, removing contrastive alignment drops class-agnostic detection AP from 99.0 to 83.2, while removing soft-token classification drops it to 87.7.

![MDETR’s text-aligned boxes supporting a visual question answer](/assets/images/mdetr-modulated-detection-for-end-to-end-multimodal-understanding-source-figure-5-white.png)
*Fig 2: On the question “What is on the table?”, MDETR assigns boxes to relevant question words and predicts “laptop,” making the visual evidence for the answer inspectable. | source: [MDETR, Figure 5](https://arxiv.org/abs/2104.12763)*

The figure is useful because it shows the shared interface doing two jobs. The boxes are conditioned on the question, and the answer head can use those query representations rather than a detector’s unrelated object inventory. The paper’s GQA extension adds question-type-specific queries and heads for relation, object, global, category, and attribute questions. It does not simply append a classifier to a frozen detector.

### Dense aligned text makes the loss non-trivial

MDETR combines annotations from Flickr30K Entities, COCO referring expressions, Visual Genome regions, and GQA. A graph-coloring procedure joins sentences that refer to the same image only when the boxes attached to their phrases have $\mathrm{GIoU}\le 0.5$; it also caps each combined sentence at 250 characters and removes downstream validation and test images from pretraining. The resulting 1.3M image-text pairs cover about 200K images and contain multiple phrase-box relationships per example. Dense captions matter: if a sentence contains one object only, a model can often predict the sentence root without looking carefully at the image. Multiple people and objects force it to disambiguate which span belongs to which box.

The natural-image experiments use a ResNet-101 or EfficientNet-B3/B5 backbone, a frozen-batchnorm visual path, and a pretrained RoBERTa-base text encoder. The main pretraining run lasts 40 epochs on 32 V100 GPUs with effective batch size 64 and takes about a week. Downstream fine-tuning uses the same text-conditioned detection interface. Referring expressions rank boxes by $1-P(\emptyset)$; PhraseCut adds a segmentation head after box training; GQA adds question-specific queries; phrase grounding ranks up to 100 predicted boxes by their soft token alignment. For the LVIS extension, the model queries all 1.2k category names separately and merges the boxes, which costs about 10 seconds per image on a GPU.

### The gains appear across grounding and reasoning protocols

| Evaluation | MDETR result | Protocol boundary |
| --- | ---: | --- |
| RefCOCO | 87.51/90.40/82.67 accuracy | ENB3; val/testA/testB |
| RefCOCO+ | 81.13/85.52/72.96 accuracy | ENB3; val/testA/testB |
| RefCOCOg | 83.35/83.31 accuracy | ENB3; val/test |
| Flickr30K phrase grounding | 83.6/93.4/95.1 R@1/5/10 on val | ENB5 with pretraining, ANY-BOX protocol |
| PhraseCut segmentation | 53.1 mean IoU; 56.1 Pr@0.5 | ResNet-101 backbone |
| GQA | 62.95 test-dev / 62.45 test-standard | ENB5 visual backbone |

The referring-expression numbers exceed the proposal-reranking baselines in the table. Direct box prediction also removes the limit imposed by a fixed proposal set, although the comparison does not isolate that architectural choice from training differences. The paper flags a leakage issue for several prior systems: their Bottom-Up-and-Top-Down detector saw some RefCOCO validation and test images during detector training. MDETR’s pretraining excludes those images, so the comparison is not only about architecture.

The CLEVR results show another boundary. MDETR reaches 99.7% on the main task and 81.7% on CLEVR-Humans after fine-tuning, but CoGenT test-B accuracy falls to 76.7 from 99.8 on test-A. The model therefore handles text-conditioned boxes and synthetic reasoning well while retaining compositional biases. A curriculum that first trains modulated detection reaches 99.7% QA accuracy; removing that curriculum falls to 68.2. Replacing separate QA heads with one shared head falls to 90.1, so the GQA/CLEVR gains depend on the task-specific decoder design as well as the pretrained detector.

### Free-form queries still need aligned supervision

MDETR moves the visual boundary from a fixed detector vocabulary to the phrases supplied at inference. Its boxes and QA outputs expose a useful grounding interface, but learning that interface takes phrase-box annotations, not just arbitrary web captions. The LVIS timing makes the other cost concrete: querying many category names repeats multimodal processing. The model expands what can be requested while leaving annotation cost and query volume as practical limits.

## High-Level Takeaways

- MDETR replaces fixed detector classes with text-conditioned boxes and token-span alignment.
- Removing either soft token prediction or contrastive alignment sharply lowers detection AP in the CLEVR ablation.
- Dense phrase-box data transfer to referring expressions, phrase grounding, segmentation, and VQA.
- The method trades detector vocabulary limits for aligned annotation cost and per-query multimodal compute.
