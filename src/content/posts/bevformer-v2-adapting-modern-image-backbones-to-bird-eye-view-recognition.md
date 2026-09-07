---
title: 'BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision'
date: '2022-11-18T00:00:00.000Z'
section: paper-shorts
postSlug: bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition
legacyPath: /paper shorts/2022/11/18/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2022 – BEVFormer v2: strengthening BEV learning with perspective supervision'
---
## 2022 – BEVFormer v2

**arXiv:** [2211.10439](https://arxiv.org/abs/2211.10439)

**Paper:** [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.html)

## Summary

> BEVFormer v2 addresses an optimization mismatch in camera-BEV models: a BEV loss reaches the image backbone only after projection, attention, and sparse object decoding. A DD3D-style perspective head supplies dense image-aligned 3D supervision, then its filtered proposals become scene-conditioned reference points for a second-stage BEV decoder. The matched ResNet-101 ablation improves NDS from 42.6 to 45.1 and mAP from 35.5 to 37.4; the 63.4 NDS / 55.6 mAP InternImage-XL result is a larger, separately configured test point. Perspective supervision improves backbone adaptation, but it does not remove projection, calibration, occlusion, or online-temporal limits.

## Core Insights

### BEV supervision reaches the backbone too indirectly

A BEV detector first converts multi-view image features into grid-shaped BEV features and then asks a small set of object queries to predict 3D boxes. The gradient from a box loss therefore travels through view transformation, spatial cross-attention, BEV encoding, and a sparse decoder before it teaches the image backbone about depth or orientation. BEVFormer v2 calls this supervision implicit and sparse with respect to image features: only image locations sampled by attended BEV references contribute directly to the final error.

The paper adds a perspective 3D detection head directly on the image features. Its FCOS3D-like head predicts 2D location, 3D center depth, projected-center offset, size, and orientation, along with confidence. The total objective is $\mathcal{L}_{total}=\lambda_{bev}\mathcal{L}_{bev}+\lambda_{pers}\mathcal{L}_{pers}$, with both weights set to one in the reported experiments. The perspective loss gives the backbone dense, direct feedback about the properties the BEV stage needs, while the BEV head and its original representation remain in place.

![Figure 2 from BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-source-figure-2.webp)
*Fig 1: Perspective supervision compares image features directly with per-pixel 3D predictions, whereas BEV supervision reaches them through projected BEV features and a sparse set-prediction head. | source: [BEVFormer v2, Figure 2](https://arxiv.org/abs/2211.10439)*

This framing explains why the paper focuses on backbone adaptation rather than only on a new BEV operator. Modern 2D backbones pretrained on COCO have strong visual features but no reason to encode driving-specific depth and orientation. The auxiliary head turns those missing properties into an explicit training signal without requiring extra LiDAR or depth-estimation pretraining.

### The auxiliary head becomes a proposal generator

Joint supervision would already help optimization, but BEVFormer v2 also uses the perspective head’s predictions in a two-stage detector. It filters perspective boxes with per-view 2D NMS at IoU 0.75, keeps the top $k_1=100$ proposals per camera, projects them into BEV, applies BEV NMS at IoU 0.3 to remove duplicate views, and keeps the top $k_2=100$ proposals. Their projected box centers become image-conditioned reference points for the BEV decoder.

Those references are combined with learned content queries and learned positional embeddings. The per-image proposals tell deformable attention where an object is likely to be in this scene; the learned queries preserve a dataset-level spatial prior and can recover objects missed by the perspective head. This is why the method keeps both query types rather than replacing the learned bank entirely.

![BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision source figure: Overall architecture of BEVFormer v2.](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-paper-figure.webp)
*Fig 2: The perspective head and BEV head share the image backbone; perspective predictions provide the auxiliary loss and scene-conditioned proposals that join learned queries in the temporal BEV decoder. | source: [BEVFormer v2, Figure 1](https://arxiv.org/abs/2211.10439)*

![Figure 3 from BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-source-figure-3.webp)
*Fig 3: Projected centers from filtered perspective proposals become per-image reference points, while learned content and positional embeddings preserve a stable query prior for the second-stage BEV decoder. | source: [BEVFormer v2, Figure 3](https://arxiv.org/abs/2211.10439)*

The proposal path has a clear failure mode. An occluded object or one near the boundary of adjacent camera views may be missed by the perspective head, so the learned references are necessary as a fallback. Conversely, a false perspective proposal is given a privileged place to be sampled by the second stage. The query design moves scene association earlier in the decoder; it does not make the proposal set ground truth.

### Longer temporal spacing changes the online contract

BEVFormer v2 also replaces BEVFormer’s recurrent temporal self-attention with a warp-and-concatenate encoder. A historical BEV feature is bilinearly warped into the current frame with the relative $SE(3)$ transform, concatenated with the current BEV feature, and reduced with residual blocks. The model keeps the same number of historical features as the original design but samples them at a longer interval: 2 seconds rather than 0.5 seconds. That provides larger object displacement and more diverse ego positions without linearly increasing the number of stored features.

The paper also allows future BEV features in its offline 3D detection setting. That is useful for leaderboard accuracy but changes the deployment contract: a real-time system cannot use frames that have not arrived. The temporal encoder’s architecture and the perspective-supervision ablations should therefore be read separately. Table 2 and Table 3 are single-frame comparisons without temporal information; the main test configuration uses the temporal encoder and a different training/evaluation setup.

### The matched ablations support an optimization explanation

On the nuScenes validation split with a ResNet-101 backbone and no temporal information, the BEV-only detector reaches 42.6 NDS and 35.5 mAP. Perspective-only reaches 41.2/32.3, while Perspective & BEV reaches 45.1/37.4. Replacing the perspective head with a second BEV head gives 42.8/35.0. The two-stage structure by itself is therefore not enough; the useful ingredient is the perspective view’s direct and dense supervision plus its image-conditioned proposals.

The effect repeats across backbones initialized from COCO. For ResNet-50, NDS rises from 40.0 to 42.8 and mAP from 32.7 to 34.9; for DLA-34, 40.3 to 43.5 and 33.8 to 35.8; for VoVNet-99, 44.1 to 46.7 and 36.7 to 39.6; for InternImage-B, 45.5 to 48.5 and 39.8 to 41.7. Training also converges faster: after 24, 48, and 72 epochs, the ResNet-50 BEV-only model reaches 37.9/40.0/41.0 NDS, while Perspective & BEV reaches 41.4/42.8/42.8. More BEV-only epochs narrow the gap only slightly.

The full test table uses stronger and differently pretrained components. BEVFormer v2 with InternImage-B reaches 62.0 NDS and 54.0 mAP; InternImage-XL reaches 63.4 and 55.6, surpassing the cited BEVStereo entry by 2.4 NDS and 3.1 mAP. The source contrasts these COCO-initialized backbones with V2-99 backbones pretrained on depth estimation and then fine-tuned with DD3D. The scale result is compelling, but it is not the same controlled comparison as the ResNet-101 ablation.

## High-Level Takeaways

- BEVFormer v2 adds a dense perspective 3D loss because a sparse BEV decoder gives the image backbone an indirect geometric signal.
- The perspective head also supplies filtered, per-image proposal centers; learned queries remain as a fallback for missed or occluded objects.
- The matched ResNet-101 comparison supports the mechanism: Perspective & BEV reaches 45.1 NDS / 37.4 mAP, while a two-BEV-head control stays at 42.8 / 35.0.
- Gains repeat across COCO-initialized backbones and shorten convergence, while the InternImage-XL 63.4 NDS / 55.6 mAP point uses a separate scale and temporal contract.
- The warp-and-concatenate temporal encoder can use future frames offline, and the perspective route still inherits calibration, occlusion, and projection limits.
