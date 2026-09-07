---
title: 'DeepInteraction: 3D Object Detection via Modality Interaction'
date: '2022-08-23T00:00:00.000Z'
section: paper-shorts
postSlug: deepinteraction-3d-object-detection-via-modality-interaction
legacyPath: /paper shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – DeepInteraction: 3D Object Detection via Modality Interaction'
---
## 2022 – DeepInteraction

**arXiv:** [2208.11112](https://arxiv.org/abs/2208.11112)

**Code:** [fudan-zvg/DeepInteraction](https://github.com/fudan-zvg/DeepInteraction)

## Summary

> DeepInteraction treats modality fusion as an exchange protocol rather than a one-time concatenation. An image stream and a LiDAR BEV stream remain distinct through the encoder; bidirectional correspondence and attention update both, and a predictive decoder alternates between them around object queries. On nuScenes, the base R50 model reaches 70.8 mAP/73.4 NDS on test at 4.9 FPS on an A100, while the ablations show that both representational and predictive interaction contribute. The cost is explicit 2D–3D calibration and a larger, less efficiency-focused model.

## Core Insights

### Keep two representations alive

Most multimodal detectors fuse image and LiDAR features into one hybrid tensor before decoding. DeepInteraction keeps an image-perspective representation and a LiDAR-BEV representation through the whole pipeline. Separate backbones first build the two streams. Each encoder layer then contains three parts: multi-modal representational interaction (MMRI), intra-modal learning (IML), and representational integration. The output is still two modality-specific tensors, but each has received information from the other.

![DeepInteraction source Figure 1: one fused stream versus two interacting streams](/assets/images/deepinteraction-paper-figure-1.png)
*Fig 1: The paper contrasts one-shot feature fusion with DeepInteraction's two live representations, which exchange information in the encoder and again during predictive decoding. | source: [DeepInteraction, Figure 1](https://arxiv.org/abs/2208.11112)*

The reason to preserve the split is physical. Image features have broad semantic coverage but ambiguous depth; LiDAR features have metric geometry but missing and irregular returns. A single fused tensor may perform well while hiding which sensor supplied a cue and while erasing the neighborhood structure native to either view. DeepInteraction makes the interaction rule shared, not the representation itself.

### Cross-modal sampling is geometry-aware and directional

MMRI builds dense mappings between image coordinates and BEV coordinates. For image-to-LiDAR interaction, the model projects LiDAR points into the cameras, completes the resulting sparse depth map, back-projects image pixels into 3D, and maps their ground-plane coordinates to BEV neighbors. A LiDAR BEV feature can then query image features that correspond to its location and receive visual context that fills sparse geometry. For LiDAR-to-image interaction, the model projects the LiDAR points in each BEV pillar into the cameras, so an image feature queries the BEV features that actually generated its geometric evidence. The cross-attention is applied over these mapped neighbors rather than over every image and BEV position.

![DeepInteraction source Figure 2: image-to-LiDAR and LiDAR-to-image interaction](/assets/images/deepinteraction-3d-object-detection-via-modality-interaction-source-figure-2.webp)
*Fig 2: The two panels show the directional exchange: image features are sampled into LiDAR BEV in one direction, and LiDAR context is sampled back into image features in the other. | source: [DeepInteraction, Figure 2](https://arxiv.org/abs/2208.11112)*

This is not symmetric averaging. The query remains in its native stream, and the other modality supplies keys and values. In practice, that lets the image stream sharpen the distinction between a tiny or occluded object and background while the LiDAR stream supplies metric context that a perspective feature cannot infer reliably. Repeating the operation across two representational interaction layers is an explicit choice to let the streams correct one another progressively.

### Prediction alternates between the streams

The decoder receives object queries and the boxes predicted by the previous layer. Its multi-modal predictive interaction (MMPI) layer crops a region of one modality around each current box, performs self-attention and cross-attention, and decodes the updated query. The layers alternate between image and LiDAR representations. For the LiDAR RoI, the projected box is enlarged twofold because driving objects occupy very few cells in BEV. This is object-conditioned interaction: the query chooses which local evidence matters after it has a provisional box.

![DeepInteraction source Figure 3: predictive interaction decoder and MMPI layer](/assets/images/deepinteraction-3d-object-detection-via-modality-interaction-source-figure-3.webp)
*Fig 3: The decoder generates predictions while successive MMPI layers interact with image and LiDAR RoIs; the model does not decode from one collapsed feature map. | source: [DeepInteraction, Figure 3](https://arxiv.org/abs/2208.11112)*

The decoder ablation separates the two ideas. A LiDAR-only first layer reaches 65.1 mAP/70.1 NDS on nuScenes validation. Replacing it with the full alternating stack reaches 69.9/72.6; using five layers is the best point in the reported sweep, while six slightly reduces NDS. The gains are not simply “more decoder layers”: a DETR-style decoder on both streams reaches 68.6/71.6, one MMPI direction reaches 69.3/72.1, and MMPI on both streams reaches 69.9/72.6.

### The results isolate interaction from backbone choice

The base model uses nuScenes's six 1600×900 cameras and 32-beam LiDAR; it rescales the images to half resolution before the image branch. Its ResNet-50 image backbone is initialized from a Cascade Mask R-CNN model, and it uses two representational interaction layers, five predictive layers, and 200 queries. On nuScenes validation it reaches 69.9 mAP/72.6 NDS at 4.9 FPS on an A100, 3.1 FPS on an A6000, and 2.6 FPS on a V100. On the test split, the same base setting reaches 70.8/73.4. The larger Swin-Tiny model with test-time augmentation reaches 74.1/75.5, and the ensemble reaches 75.6/76.3; those rows include extra capacity or evaluation machinery.

The encoder ablation provides the sharper mechanism test. IML alone reaches 68.1/71.9, MMRI alone 69.5/72.5, and both together 69.9/72.6. Replacing representational interaction with conventional representational fusion drops validation performance from 68.7/71.9 to 67.5/71.3. The same pattern survives a LiDAR backbone change: the displayed table rises from TransFusion-L's 54.5 to 60.0 mAP with PointPillars (+5.5), and from 65.1 to 69.9 with VoxelNet (+4.8). The surrounding paper text summarizes the latter comparison as +5.5 for voxel and +4.4 for pillars, so the table's row values are the safer boundary for the claim.

The category breakdown explains where the retained streams matter most. Relative to the LiDAR-only TransFusion-L baseline, DeepInteraction gains 11.8 mAP on bicycles, 6.9 on motorcycles, and 5.9 on traffic cones. These are small, rare, or weakly observed objects for which image semantics and LiDAR geometry have unusually complementary failure modes.

## High-Level Takeaways

- DeepInteraction keeps image and LiDAR representations separate while sharing bidirectional interaction operators in the encoder and decoder.
- MMRI alone reaches 69.5/72.5 mAP/NDS on validation, MMPI alone 69.3/72.1, and their combination 69.9/72.6, showing that representation and prediction interaction are complementary.
- The base R50 model reaches 70.8/73.4 on nuScenes test; larger and ensemble variants reach higher scores with different evaluation costs.
- The strongest category gains are on bicycles (+11.8 mAP), motorcycles (+6.9), and traffic cones (+5.9), while calibration quality and interaction cost remain explicit limits.
