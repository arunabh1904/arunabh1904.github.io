---
title: 'UNITER: Universal Image-Text Representation Learning'
date: '2019-09-25T00:00:00.000Z'
section: paper-shorts
postSlug: uniter-universal-image-text-representation-learning
legacyPath: /paper shorts/2019/09/25/uniter-universal-image-text-representation-learning.html
tags:
  - Vision-Language Models
  - Multimodal Pretraining
field: 'Vision-Language Models'
summary: '2019 – UNITER: Universal Image-Text Representation Learning'
---

## 2019 – UNITER: Universal Image-Text Representation Learning

**arXiv:** [1909.11740](https://arxiv.org/abs/1909.11740)

**Code:** [ChenRocks/UNITER](https://github.com/ChenRocks/UNITER)

## Summary

> UNITER puts detector regions and wordpieces into one Transformer and makes both global pair matching and local word-region alignment training targets. Its four-task recipe combines masked language modeling, three masked-region objectives, image-text matching, and optimal-transport alignment. The resulting base and large models transfer across VQA, VCR, NLVR2, visual entailment, retrieval, and referring expressions. The stronger fusion comes with a clear price: every image-text pair must be jointly processed, and the visual stream still begins with detector proposals.

## Core Insights

### One stream makes every region-word pair available

![UNITER model and its pretraining objectives](/assets/images/uniter-universal-image-text-representation-learning-source-figure-1.webp)
*Fig 1: UNITER projects detector regions and WordPiece tokens into one sequence, then applies a shared Transformer to support masked language, masked regions, matching, and word-region alignment. | source: [UNITER, Figure 1](https://arxiv.org/abs/1909.11740)*

The image embedder starts with Faster R-CNN features from a Visual Genome detector. For each region, UNITER projects both the pooled visual feature and a seven-dimensional location vector—normalized coordinates, width, height, and area—then sums and layer-normalizes them. The text embedder does the same for WordPiece and position embeddings. A modality embedding tells the shared Transformer which tokens came from which side. Once concatenated, all regions and words can attend to one another at every layer.

That early fusion is a deliberate contrast with the two-stream models in this lineage. It makes fine-grained interaction cheap inside a single pair: a phrase can update its representation using the region it describes, and a region can use the surrounding words. It also makes retrieval expensive at scale, because an image and a caption cannot be encoded independently before matching. UNITER’s base model has 12 layers, 768 hidden dimensions, and 86M parameters; the large model doubles the depth to 24 layers, uses 1,024 hidden dimensions, and has 303M parameters.

### Conditional masking and transport supervise different alignments

UNITER masks one modality at a time. Masked language modeling replaces 15% of the words and predicts them from the remaining words and all regions. Masked region modeling zeros 15% of the region features and predicts them from the text and visible regions in three variants: feature regression (MRFR), hard detector-label classification (MRC), and a soft KL-divergence version (MRC-kl). The conditional mask matters: masking a word and the only region that names it at the same time can turn a cross-modal task into an impossible missing-missing pair.

Image-text matching adds an instance-level test. A `[CLS]` representation is classified as a match or mismatch after replacing an image or sentence with one from another sample. Word-region alignment (WRA) adds the local test that matching alone cannot provide. It treats contextualized words and regions as two discrete distributions and finds a transport plan with cosine distance as its cost. The IPOT approximation is efficient enough for training; the plan is self-normalized and sparse, so it supplies an interpretable soft correspondence rather than only one global score.

The training schedule samples one objective per mini-batch, so the same encoder alternates between reconstructing missing content, judging pair compatibility, and aligning local tokens. This makes the ablation readable: the paper can remove WRA or conditional masking without changing the encoder or data pipeline.

### The data split is part of the transfer claim

![UNITER downstream data splits and pretraining overlap controls](/assets/images/uniter-universal-image-text-representation-learning-source-figure-4-white.png)
*Fig 2: The source split diagram shows which image-text examples belong to training, validation, and test for the downstream datasets, with COCO evaluation images removed from pretraining. | source: [UNITER, Figure 4](https://arxiv.org/abs/1909.11740)*

UNITER pretrains on image-sentence pairs from COCO Captions, Visual Genome Dense Captions, Conceptual Captions, and SBU Captions. To make the in-domain split fair, the authors merge the raw COCO training and validation data, remove downstream evaluation images, and use URL matching to remove overlapping Flickr30K images; 222 images are eliminated through that split construction. They apply the same URL matching to the out-of-domain Conceptual Captions and SBU mixture, excluding 109 images from training. The out-of-domain mixture tests whether data that look less like the target images can substitute for in-domain pairs.

The downstream suite contains VQA, VCR, NLVR2, SNLI-VE visual entailment, Flickr30K and COCO image-text retrieval, and RefCOCO/RefCOCO+/RefCOCOg referring-expression comprehension. UNITER-base and large use the same pretraining objectives, then add task heads and fine-tune end to end. VCR receives a second pretraining stage on the VCR data; NLVR2 receives a small pair-level adaptation because the task has two images while the pretraining input has one.

### The objective mixture transfers, with measurable caveats

| Evaluation | UNITER result | Protocol boundary |
| --- | ---: | --- |
| VQA v2.0 | 72.91 base / 74.02 large test-standard | answer classification after pair fusion |
| VCR | 58.20 base / 62.80 large Q→AR | two-stage pretraining includes VCR for the reported model |
| NLVR2 | 77.85 base / 79.98 large test-P | pair-biattention adaptation for two images |
| SNLI-VE | 78.28 base / 79.38 large test | three-way entailment classification |
| Flickr30K image retrieval | 72.52/92.36/96.08 base R@1/5/10 | jointly scored image-text pairs |
| RefCOCO+ | 83.66/86.19/78.89 base val/testA/testB; detected 75.31/81.30/65.58 | ground-truth objects; `d` marks detected-proposal evaluation |

The paper reports UNITER-large as state of the art across its six task families, but the objective ablation explains where the gain comes from. On the in-domain split, the full MLM+ITM+MRC-kl+MRFR+WRA recipe reaches a Meta-Sum of 400.93, versus 399.97 without WRA. Removing conditional masking lowers the corresponding score to 396.51. MRC-kl beats hard-label MRC in the combined objective, consistent with detector labels being uncertain rather than ground truth.

Data choice matters as much as the loss. The same full objective on out-of-domain SBU+Conceptual Captions reaches 396.91, below the 400.93 in-domain result, while combining both mixtures reaches 405.24. This is evidence for both domain similarity and scale. The two-image NLVR2 adaptation is also informative: a pair model reaches 75.85 development accuracy and adding bidirectional attention between the image representations reaches 77.18, beating a triplet input that asks the single-image pretrained encoder to absorb a new interaction pattern.

### Decision test and boundary

UNITER is the right baseline when local word-region grounding matters more than independent embedding speed. Compare it with two-stream and dual-encoder models using the same detector, image-overlap controls, and pair-scoring budget; report WRA and conditional-masking ablations separately. The detector still fixes the available visual vocabulary, and VCR’s second-stage pretraining means its best score is not a pure transfer result. UNITER shows that a single fused sequence plus explicit transport can improve alignment; it does not make open-world detection or large-scale retrieval cheap.

## High-Level Takeaways

- UNITER fuses detector regions and words in one Transformer so local alignment is available at every layer.
- Masked content, global matching, and optimal-transport alignment target different failure modes.
- WRA, conditional masking, in-domain data, and the two-image NLVR2 adaptation each have measurable effects.
- Joint fusion improves reasoning and grounding at the cost of detector dependence and pairwise retrieval compute.
