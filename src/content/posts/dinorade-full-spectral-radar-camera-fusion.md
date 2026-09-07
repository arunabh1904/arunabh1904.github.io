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

**arXiv:** [2604.08074](https://arxiv.org/abs/2604.08074)

**Paper:** [CVPR 2026 Workshop](https://openaccess.thecvf.com/content/CVPR2026W/DriveX/html/Leitgeb_DinoRADE_Full_Spectral_Radar-Camera_Fusion_with_Vision_Foundation_Model_Features_CVPRW_2026_paper.html)

**Code:** [chr-is-tof/RADE-Net](https://github.com/chr-is-tof/RADE-Net)

## Summary

> DinoRADE lets radar define the metric support and asks a frozen DINOv3 image encoder for semantic detail only where the two sensors are geometrically related. Dense range–azimuth–Doppler radar is lifted into BEV with learned spectral weights; radar queries sample four nearby image features through deformable cross-attention; an adaptive gate then decides how much camera evidence to retain. The reported gains are large, but they span two K-RADAR versions and a severe long-tail class/weather regime.

## Core Insights

### Radar preserves the coordinate system

The starting point is a dense 4D radar tensor, not a sparse point export. RAD and RAE projections retain range, azimuth, elevation, and Doppler structure; a modified RADE-Net processes the spectral representation and produces a 256×112 BEV feature map. Ten learned elevation segments are weighted from the RAE distribution before being lifted into the detection space. That choice keeps radar’s metric support in charge of where an object can be, instead of asking image features to invent depth first.

DINOv3 ViT-S/16 is frozen and its multi-level features are upsampled through an FPN. For each radar reference point, calibration maps the query into the camera plane. Deformable cross-attention samples four offsets around that reference rather than pooling the whole image. The camera branch therefore contributes local semantic evidence while radar determines the query’s 3D neighborhood.

![DinoRADE radar-camera architecture with spectral radar processing and local cross-attention](/assets/images/dinorade-full-spectral-radar-camera-fusion-paper-figure.webp)
*Fig 1: DinoRADE preserves the radar BEV stream, samples DINOv3 features around projected 3D queries, and adaptively fuses the two modalities before a CenterPoint head; this is the paper’s Figure 1. | source: [DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather, Figure 1](https://arxiv.org/abs/2604.08074)*

### Local alignment is a better use of a vision foundation model

The cross-attention offsets are learned around a calibrated reference, which gives the model room for imperfect projection, object extent, and annotation noise without discarding geometry. A gated adaptive fusion then lets the radar feature dominate when the image is unreliable. The ablation makes this ordering visible on K-RADAR v2.1: radar-only scores 61.65 3D AP / 66.68 BEV AP; adding camera features reaches 69.61 / 74.96; adding the learned weighted lift reaches 71.38 / 75.32. Replacing DINOv3 with a fine-tuned ResNet-50 gives 68.43 / 72.42, so the comparison supports the feature choice under this training recipe, not a universal foundation-model advantage.

### Read the headline through the dataset split

The sedan-only K-RADAR v1.1 comparison reports 70.8 total 3D AP for DinoRADE, 12.1 points above the strongest listed radar-camera result. The broader v2.1 table reports 36.99 3D mAP and 39.61 BEV mAP across Sedan, Bus/Truck, Pedestrian, Bicycle, and Motorcycle. Motorcycle reaches only 3.77 AP and Bicycle 21.89, while the authors note that some vulnerable-road-user cells contain fewer than 2% of training objects and that annotations can be missing.

![DinoRADE occlusion examples](/assets/images/dinorade-full-spectral-radar-camera-fusion-source-figure-4.webp)
*Fig 2: The source’s Figure 4 shows partially, heavily, and fully occluded examples, the cases where radar’s metric support is most valuable. | source: [DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather, Figure 4](https://arxiv.org/abs/2604.08074)*

The headline result should therefore be read as two related tests: a controlled sedan comparison on v1.1 and a five-class, weather-diverse evaluation on v2.1. They do not share exactly the same label space or difficulty. Weather and class slices are essential before deciding whether the system improves a deployed stack, especially when the camera gate may help one regime and hide another.

## High-Level Takeaways

- DinoRADE’s central decision is to keep radar as the geometric query space and use camera features as local semantic evidence.
- Dense spectral radar and the weighted elevation lift are part of the reported gain; a point-cloud-only reproduction would test a different representation.
- The weighted-lift ablation and ResNet substitution are informative matched comparisons, while the v1.1 sedan result and v2.1 five-class result should not be merged into one score.
- Occlusion and adverse weather are the right stress cases for this design, but long-tail classes and missing annotations make aggregate mAP fragile.
- The next deployment check is an accuracy–latency study with camera dropout, calibration noise, and per-class weather slices against radar-only and simpler local-fusion baselines.
