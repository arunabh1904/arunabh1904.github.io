---
title: 'LocCa: Visual Pretraining with Location-aware Captioners'
date: '2024-03-28T00:00:00.000Z'
section: paper-shorts
postSlug: locca-visual-pretraining-with-location-aware-captioners
legacyPath: /paper shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2024 – LocCa: Visual Pretraining with Location-aware Captioners"
---

## 2024 – LocCa: Visual Pretraining with Location-aware Captioners

**arXiv:** [2403.19596](https://arxiv.org/abs/2403.19596)

## Summary

> LocCa adds “where” to visual captioning pretraining. One Vision Transformer and one autoregressive decoder learn ordinary captions alongside automatic referring expressions and grounded captions, with boxes represented as text. The same encoder therefore has to preserve scene semantics and object locations before downstream transfer. On clean RefCOCO splits, the frozen LocCa encoder is far ahead of captioning and contrastive baselines while its holistic classification, captioning, OCR, and VQA results remain competitive. The gain comes with a precise limit: proxy boxes are not pixel supervision, and the standard RefCOCO splits are heavily contaminated by image overlap.

## Core Insights

### Location enters through the same language interface

![Figure 1 from LocCa showing captioning, automatic referring expression, and grounded captioning pretraining tasks](/assets/images/locca-visual-pretraining-with-location-aware-captioners-paper-figure.png)
*Fig 1: Overview of LocCa. LocCa consists of a standard vision transformer and a transformer decoder. The vision transformer takes image pixel as input, produces visual tokens as cross attention input to the transformer decoder. The transformer decoder is trained to read out rich information from the visual tokens. We adopt the following three task for pretraining: Cap, AREF and GCAP. | source: [LocCa, Figure 1](https://arxiv.org/abs/2403.19596)*

Figure 1 is a single encoder with three output contracts. Cap maps an image to a caption. AREF (automatic referring expression) maps a generated region description to its box. GCAP (grounded captioning) maps a box to its regional caption. During pretraining, the model does not receive the box as a privileged input for one task and the caption as a privileged input for the other. It predicts both sides sequentially from the image, using task prefixes to indicate the sequence:

- “Cap:” followed by the image caption;
- “ARef: {caption} : {box}”;
- “GCap: {box} : {caption}”.

The losses cover the whole sequence after the prefix, so AREF teaches the decoder to identify a region and then regress its coordinates, while GCAP teaches it to identify a region and describe what is inside. The two tasks share the same visual tokens and decoder weights. At inference, the same interface can be used with only “ARef:” to propose and describe a region, or with “ARef: a black and white cat :” to condition on text and request only a location.

The model is deliberately ordinary: a ViT-L/14 image encoder, a 12-block Transformer-L decoder, and roughly 600M parameters. Half of the ordinary captioning examples use parallel prediction, where the decoder cannot rely on preceding caption tokens and must extract more of the caption from visual features. This prevents the language model from doing all of the work while keeping the computation of location-aware tasks close to an ordinary captioner.

### Pseudo boxes are enough to teach object sensitivity

LocCa uses one billion English WebLI image/alt-text pairs after text filtering and de-duplicates the training images against every evaluation set. An OWL-ViT CLIP L/14 detector supplies pseudo boxes from alt-text n-grams and PaLI object categories. Boxes below confidence 0.3 are discarded, then one box-caption or box-category pair is sampled for AREF and one for GCAP. The pretraining run sees about nine billion image/alt-text examples—roughly nine passes over the tailored subset—with batch size 8,192, 224 × 224 inputs, AdaFactor, a learning rate of $10^{-3}$, and 10,000 warmup steps.

This supervision is weak in two ways. The locations come from a detector, not human boxes, and each example samples only one region for each location-aware task. Yet the extra sequences force the shared visual representation to answer questions that a global caption loss can ignore: which object was named, where it sits, and which words belong to it. The model is not learning a separate region proposal network or a contrastive region-text matrix. It is learning to make the visual encoder useful to a decoder whose next token may be a coordinate or a word.

The clean RefCOCO evaluation isolates that transfer. RefCOCO, RefCOCO+, and RefCOCOg training images overlap heavily with one another: 61.2% of RefCOCO validation images, 60.5% of testA, and 65.1% of testB also appear in the combined training pool, with the same ratios for RefCOCO+ and 48.8%/48.3% for RefCOCOg val-u/test-u. LocCa removes all validation and test images from the combined training sets and also removes COCO images and near-duplicates from pretraining. This makes the “clean” numbers lower than contaminated comparisons, but much easier to interpret.

![Figure 2 from LocCa showing COCO detection before and after reward tuning](/assets/images/locca-visual-pretraining-with-location-aware-captioners-source-figure-2.png)
*Fig 2: Result on COCO detection with a limit of 25 output boxes. For reward tuned models we show both the results before (dark blue and orange) and after (light blue and orange) reinforce tuning. | source: [LocCa, Figure 2](https://arxiv.org/abs/2403.19596)*

Figure 2 shows a second transfer interface. For COCO detection, LocCa uses an Objects365-pretrained decoder that can emit up to 25 box sequences, first trains by likelihood and then applies reinforcement tuning against an mAP-related reward. The pretrained visual encoder is what changes the starting point: location-aware LocCa is already more object-sensitive before reward tuning, and the gap remains after it. The result is not zero-shot detection from the three pretraining tasks; it is evidence that a location-aware visual encoder makes a downstream autoregressive detector easier to train.

### The localization gain does not erase global features

| Frozen encoder transfer | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCOg test-u |
| --- | ---: | ---: | ---: | ---: | ---: |
| CapPa | 64.17 | 69.90 | 58.25 | 56.14 | 59.91 |
| LocCa | 88.34 | 91.20 | 85.10 | 79.39 | 82.64 |
| LocCa, clean combined train | 89.70 | 92.75 | 85.30 | 83.85 | 85.86 |

These referring-expression comprehension scores use a randomly initialized decoder with the vision encoder frozen. The same pattern appears for referring-expression segmentation: LocCa reaches 64.98/65.39/64.09 mIoU on RefCOCO val/testA/testB, 57.85/60.92/52.72 on RefCOCO+, and 55.84/56.95 on RefCOCOg val-u/test-u. The clean protocol matters because methods using COCO-pretrained detector components may have seen the nominal test images even when their RefCOCO training split was disjoint.

On holistic transfer, LocCa reaches 84.5 ImageNet-1k, 96.0 Resisc-45, 127.1 COCO CIDEr, 90.7 Flickr30K CIDEr, 64.5 OCR-VQA, 72.8 VQAv2, and 61.8 GQA in the LiT-Decoder setup. It is close to CapPa on classification and better on captioning, OCR, and VQA, especially the object-centric GQA task. The comparison is not a claim that every location task improves every global metric: LocCa slightly trails CapPa on Oxford-Pet in the reported table, and LocCaG is a larger ViT-G/14 variant with separate numbers.

The encoder also transfers into PaLI-3. With a 224-pixel LocCa-L encoder, the PaLI-3 transfer reaches 138.9 COCO CIDEr, 77.6 VQAv2, 58.4 OKVQA, 49.2 TextVQA, 50.9 ST-VQA, and 79.3/64.1 on simple/complex TallyQA. The corresponding SigLIP-L row is 135.8, 75.6, 57.5, 41.1, 46.2, and 74.9/61.4. The improvement is largest on visually situated text and counting-style tasks, which is exactly where a global caption objective has fewer reasons to preserve precise object identity and position.

![Figure 3 from LocCa showing resolution and coordinate-token ablations](/assets/images/locca-visual-pretraining-with-location-aware-captioners-source-figure-3.png)
*Fig 3: Ablation studies on (a) impact of different pretrained image resolutions on string token; and (b) string vs special token of box coordinates with pretrained res 224. The results are the average Acc@0.5 of the val&test splits on RefCOCO/+. | source: [LocCa, Figure 3](https://arxiv.org/abs/2403.19596)*

Figure 3 separates two implementation questions from the pretraining idea. Transferring a 224-pixel encoder at 384 or 640 pixels improves RefCOCO, and pretraining at the larger resolution improves it again; the model benefits from both more image detail and the encoder’s native positional geometry. String-token coordinates perform about as well as special coordinate tokens, so LocCa’s simple decision to tokenize integer coordinates with the same SentencePiece vocabulary is not the source of the localization gain.

### Location-aware pretraining has a real ceiling

LocCa’s zero-shot Visual Genome predictions show the ceiling clearly. The decoder can identify foreground regions, but without non-maximum suppression many boxes overlap because pretraining samples one object at a time. Increasing sampling noise produces more varied boxes but degrades caption quality. The model has learned an object-sensitive representation, not a complete set-level detector.

The same distinction appears in segmentation. Referring-expression segmentation adds a “Mask:” suffix with 16 VQ-VAE tokens representing a 64 × 64 mask inside the predicted box. A full LocCa-L fine-tune reaches 76.98 RefCOCO val, 78.25 testA, 72.90 testB, 71.25/76.52/63.67 on RefCOCO+, and 69.51/70.44 on RefCOCOg. It is competitive with much larger systems, but the pretraining itself has no pixel-level labels. The paper explicitly leaves zero-shot segmentation for future work.

### Location supervision is the transfer advantage

Use LocCa when a caption-pretrained encoder must later support grounding, detection, OCR, or object-sensitive VQA while retaining one generative interface. Report the clean RefCOCO protocol, separate proxy-box pretraining from downstream boxes, and distinguish frozen-encoder transfer from full-model fine-tuning. The key comparison is ordinary captioning at the same image-text compute with and without AREF/GCAP; the paper’s ablations show that either location-aware task helps, GCAP alone gives a large gain, and their combination is complementary. The durable claim is specific: adding coordinate-bearing language during visual pretraining changes what the encoder preserves. It does not remove the need for task-specific set prediction, pixel labels, or leakage-resistant evaluation.

## High-Level Takeaways

- LocCa adds AREF and GCAP to captioning so one generative decoder learns descriptions and coordinates from shared visual tokens.
- OWL-ViT pseudo boxes at 0.3 confidence are enough to produce large clean RefCOCO transfer gains without a region-specific architecture.
- Holistic classification, captioning, OCR, VQA, and PaLI-3 transfer remain competitive, with the largest gains on object- and text-sensitive tasks.
- The method learns object sensitivity rather than complete set or pixel prediction; overlapping boxes, coordinate transfer, and RefCOCO leakage remain evaluation boundaries.
