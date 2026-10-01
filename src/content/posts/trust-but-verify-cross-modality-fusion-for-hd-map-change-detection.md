---
title: "Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection"
date: '2022-12-14T00:00:00.000Z'
section: paper-shorts
postSlug: trust-but-verify-cross-modality-fusion-for-hd-map-change-detection
legacyPath: /paper shorts/2022/12/14/trust-but-verify-cross-modality-fusion-for-hd-map-change-detection.html
tags: [Autonomous Driving, Mapping]
topics: [autonomy]
field: Mapping
summary: "2021 \u2013 Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection"
---

# 2021 – Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection

**Paper:** [2212.07312](https://arxiv.org/abs/2212.07312)

**Project and data:** [TbV](https://tbv-dataset.github.io/) · **Code:** [official implementation](https://github.com/johnwlambert/tbv)

This is the NeurIPS 2021 dataset paper, posted to arXiv in December 2022. The site date follows that arXiv posting; the research year follows the venue.

## Summary

> Trust, but Verify constructs a benchmark for deciding whether sensor observations still agree with an onboard HD map. Synthetic changes supply training negatives, while real fleet-mined changes supply evaluation. Early fusion outperforms the tested late-fusion alternatives, but the reported TbV-Beta BEV model reaches only 67.28% mean class accuracy on proximity-based real test examples. The useful contribution is a visibility-aware, real-change evaluation contract, not an automatic map-repair system.

## Core Insights

### Rare real changes are reserved for evaluation

TbV-1.0 contains 1,043 logs, 7.8 million images, and approximately 559,400 LiDAR sweeps from six North American cities. The split has 799 training, 111 validation, and 133 test logs. Training logs have accurate maps; the validation and test collections contain real lane-geometry and crossing changes mined from months of fleet operations. Three independent human-review stages identify, confirm, and characterize changes with spatial annotations.

The released maps describe what was onboard when each log was captured, with local semantic geometry around the trajectory. Change polygons and polylines support match/mismatch labels based on proximity. They do not constitute a complete replacement HD map suitable for scoring every repaired vector. This is why [MapEX](/paper%20shorts/2023/11/17/mapex-mind-the-map.html) treats TbV as a change-detection benchmark rather than a direct map-reconstruction test.

The figure shows actual evaluation examples. Sensor imagery and the onboard map disagree about markings or a crossing even though most of the scene remains unchanged.

![TbV source Figure 2 illustrates crossing removal and changed lane markings](/assets/images/tbv-source-figure.png)
*Fig 1: Each example compares observed appearance with mapped semantics. The target is a localized disagreement, which can occupy only a small part of an otherwise correct road map. | source: [Trust, but Verify, Figure 2; figure crop, CC BY-NC-SA 4.0](https://arxiv.org/abs/2212.07312)*

### Render map and sensor evidence into comparable views

Training examples pair sensor data, a map, and a binary agreement label. Synthetic negatives alter map vectors before rendering: crossing insertion/deletion, lane-marking changes, and related edits obey road-layout constraints so the model cannot solve the problem only by spotting implausible graphics. Evaluation uses real alterations rather than the same corruption generator.

The paper compares early channel concatenation with late Siamese fusion, using ImageNet-initialized ResNet-18 or ResNet-50 backbones. Ego-view models project maps into the front camera and use interpolated LiDAR depth to suppress occluded elements. BEV models ray-cast camera pixels onto a pre-generated ground surface and aggregate 70 images across seven cameras and ten timesteps. This is rendered orthoimagery, not a learned BEVFormer feature field, and accurate pose is assumed.

Some variants add semantic segmentation masks for road and marking categories. Binary classification learns whether the pair agrees; gradients can help localize suspicious regions. Neither a classification score nor a gradient visualization produces a verified replacement lane graph.

### Visibility and class balance change the meaning of accuracy

Experiments use TbV-Beta, an earlier release than the public TbV-1.0. The paper reports mean per-class accuracy to avoid letting the frequent unchanged class dominate. It also separates proximity evaluation, where a change lies within 20 meters, from visibility evaluation, where the ego camera can see it.

| TbV-Beta configuration | Test mean class accuracy | Evaluation |
| --- | ---: | --- |
| Ego-view early RGB/semantics/map fusion | 67.24% | Proximity |
| Ego-view early RGB/semantics/map fusion | 72.34% | Visibility |
| Ego-view late RGB/map fusion | 49.30% | Proximity |
| BEV early RGB/semantics/map fusion | 67.28% | Proximity |

The ego early-fusion model's proximity changed-class accuracy is 57%, versus 77% for unchanged examples. BEV and ego rows also use different backbones, and the late-fusion rows lack semantic inputs, so these are tested system configurations rather than one isolated fusion-operator comparison. The large validation-to-test gap reinforces the synthetic-to-real difficulty.

The taxonomy emphasizes permanent lane geometry and crossings. Temporary cones and blockades are largely assigned to object perception, and accurate localization remains an assumption. My adoption test would preserve the real-change holdout, vary pose error and occlusion, and measure false alerts and delay per changed element. This benchmark makes those questions concrete without claiming to solve all of them.

## High-Level Takeaways

- Synthetic training and real evaluation should remain distinct when changes are rare.
- Observability matters: nearby but unseen geometry is a different test from visible disagreement.
- Keep TbV-Beta results separate from the released TbV-1.0 dataset statistics.
- Detecting a stale map does not supply a corrected map or justify a permanent edit by itself.
