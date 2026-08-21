---
title: 'DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather'
date: '2026-04-09T00:00:00.000Z'
section: paper-shorts
postSlug: dinorade-full-spectral-radar-camera-fusion
legacyPath: /paper shorts/2026/04/09/dinorade-full-spectral-radar-camera-fusion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – DinoRADE: dense radar-camera fusion with DINOv3 features'
---
## 2026 – DinoRADE

**arXiv:** [2604.08074](https://arxiv.org/abs/2604.08074)

**Paper:** [CVPR 2026 Workshop](https://openaccess.thecvf.com/content/CVPR2026W/DriveX/html/Leitgeb_DinoRADE_Full_Spectral_Radar-Camera_Fusion_with_Vision_Foundation_Model_Features_CVPRW_2026_paper.html)

**Code:** [chr-is-tof/RADE-Net](https://github.com/chr-is-tof/RADE-Net)

### Method and reported result

DinoRADE starts from dense range-azimuth-Doppler radar tensors, not a sparse point export. It projects radar reference points into the image, gathers nearby DINOv3 features with deformable cross-attention, and uses a learned weighted lift to return the fused evidence to radar BEV space.

## Summary

> The model uses radar to define metric support and a vision foundation model to supply semantic detail. Cross-attention is the alignment mechanism; the image is not flattened into a globally fused auxiliary channel.

## Core Insights

On K-RADAR v2.1 across five classes and all weather conditions, the paper reports 36.99 3D mAP and 39.61 BEV mAP. Against RADE-Net on the same table, the absolute gain is 16.06 3D mAP. For the sedan-only K-RADAR v1.1 comparison, the paper reports a 12.1-point 3D AP gain over recent radar-camera methods. Replacing DINOv3 with a fine-tuned ResNet-50 reduces both 3D and BEV AP by about three points.

![DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather source figure: Overview of the DinoRADE architecture.](/assets/images/dinorade-full-spectral-radar-camera-fusion-paper-figure.webp)
_Overview of the DinoRADE architecture. Source: [DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather](https://arxiv.org/abs/2604.08074), Figure 1, via arXiv HTML._


| Configuration | 3D AP | BEV AP |
| --- | ---: | ---: |
| Radar only | 61.65 | 66.68 |
| Radar + camera | 69.61 | 74.96 |
| Radar + camera + weighted lift | 71.38 | 75.32 |
| ResNet-50 substitution | 68.43 | 72.42 |

Class and weather imbalance complicate the aggregate. Some vulnerable-road-user cells contain fewer than 2% of the training objects, and the paper notes missing annotations that can penalize correct detections.

## High-Level Takeaways

- Dense radar preserves spectral structure that point-cloud preprocessing can discard.
- Foundation-model image features help most when fusion queries a geometrically local region rather than assuming pixel-perfect projection.
- Multi-class adverse-weather reporting is valuable, but sparse class counts make several per-weather comparisons unstable.
- The result supports radar-centered semantic refinement; it does not establish that DINOv3 is the best accuracy-latency choice for an embedded vehicle stack.
