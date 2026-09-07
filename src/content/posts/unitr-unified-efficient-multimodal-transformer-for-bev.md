---
title: "UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation"
date: '2023-08-15T00:00:00.000Z'
section: paper-shorts
postSlug: unitr-unified-efficient-multimodal-transformer-for-bev
legacyPath: /paper shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2023 – UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation"
---

## 2023 – UniTR

**ArXiv:** [2308.07732](https://arxiv.org/abs/2308.07732)

**Code:** [Haiyang-W/UniTR](https://github.com/Haiyang-W/UniTR)

## Summary

> UniTR shares the expensive transformer backbone across camera patches and LiDAR voxels while keeping their tokenizers and neighborhoods modality-specific. An intra-modal block processes the two streams in parallel; alternating 2D perspective and 3D geometric partitions then let them exchange information before LiDAR tokens are pooled to BEV. On nuScenes validation, the camera-LiDAR model reaches 73.1 NDS and 70.0 mAP at a reported 88.7 ms, versus 71.4 NDS, 68.5 mAP, and 130.5 ms for the reproduced BEVFusion comparison. The result depends on explicit coordinate structure: parameter sharing works because the model does not pretend that an image patch and a voxel have the same neighborhood.

## Core Insights

### Share the transformer, preserve the sensor physics

What does “unified” mean when the inputs are not even laid out in the same space? UniTR begins with two tokenizers: an 8×8 image-patch tokenizer for six camera views and a dynamic voxel feature encoder for LiDAR. The tokens share a feature width, but they keep their sensor identity and coordinates. A modality-agnostic DSVT-style block can therefore process image and LiDAR local sets in parallel without forcing either sensor through the other's representation.

![UniTR source Figure 3: modality-specific tokenizers and shared intra- and inter-modal transformer blocks](/assets/images/unitr-paper-figure-3.png)
*Fig 1: UniTR uses different tokenizers, then shares transformer computation for intra-modal learning and cross-modal interaction before pooling enriched LiDAR tokens to BEV. | source: [UniTR, Figure 3](https://arxiv.org/abs/2308.07732)*

The intra-modal block partitions each modality in its native space. Image tokens use 2D windows within each camera; LiDAR tokens use sparse 3D windows. Both are passed through the same attention weights in parallel. The design reduces duplicated encoder work, but the important constraint is that the partition function remains modality-aware. Sharing weights without sharing neighborhoods is what makes the representation reusable.

### Fuse in the two spaces that each sensor understands

A single projection loses something. The camera plane preserves dense semantic neighborhoods; LiDAR space preserves metric proximity and height. UniTR alternates two inter-modal blocks that use both. For 2D interaction, LiDAR tokens are projected into the first camera view they hit, and mixed local sets exchange information in perspective space. For 3D interaction, image tokens are assigned approximate depth from precomputed virtual grid points and unprojected into the LiDAR coordinate system. The model alternates the resulting partitions rather than appending a separate late fusion module.

The standard configuration is one intra-modal block followed by inter2D, inter2D, and inter3D blocks. Enriched LiDAR tokens are then pooled to BEV for detection or map segmentation. This order matters: the paper reports that putting 3D fusion before 2D or fusing before intra-modal representation learning is slightly worse, which supports a progression from native features to semantic alignment and finally geometric consolidation.

The matched ablation makes the role of each space visible:

| Backbone variant | NDS | mAP | What it adds |
| --- | ---: | ---: | --- |
| LiDAR only | 70.5 | 65.9 | Geometric baseline. |
| Add 2D perspective interaction | 72.5 | 69.0 | Dense camera semantics. |
| Add 3D geometric interaction | 72.0 | 68.5 | Cross-modal metric neighborhoods. |
| Use both 2D and 3D interaction | 73.1 | 70.0 | Complementary view and geometry. |
| Add LSS BEV fusion after UniTR | 73.3 | 70.5 | A small task-specific extension. |

### The speed gain is a scheduling decision

UniTR trains camera and LiDAR together in one end-to-end stage instead of separately pretraining each modality and adding a fusion stage. The experiments use nuScenes, with 40,157 annotated samples, six cameras, and a 32-beam LiDAR. The detection setup uses 256×704 images, LiDAR voxel size 0.3×0.3×8 m, ImageNet and nuImage initialization, AdamW on eight A100 GPUs, batch size 24, and ten epochs. The segmentation setup trains for twenty epochs.

In the isolated parallel-backbone ablation, a serial camera-LiDAR encoder takes 51 ms and reaches 72.2 NDS/68.5 mAP; the parallel shared encoder takes 33 ms and reaches 72.4/69.0. The paper notes that this table measures the transformer backbone on the same A100 workstation, while the 88.7 ms comparison includes the reported full model latency. TensorRT lowers the latter to 50.2 ms at the same validation accuracy. The efficiency claim is therefore about a concrete deployment path, not only a smaller FLOP count.

Map segmentation tests whether the representation carries semantic structure beyond boxes. UniTR reaches 73.2 mIoU and its LSS-enhanced variant 74.7 mIoU on nuScenes validation, compared with 62.7 for the cited BEVFusion result. Low-beam experiments also show that the shared backbone remains useful as geometry becomes sparse: at 1-beam LiDAR, parallel processing reaches 59.5 NDS versus 57.6 for the serial variant; at 32 beams, parallel and serial processing are nearly tied at 73.3 and 73.2, both above the 70.5 LiDAR-only row.

### The boundary is modality flexibility, not arbitrary sensor switching

UniTR handles camera and LiDAR together, and the paper evaluates camera or LiDAR malfunctions using the BEVFusion protocol. It does not provide a mixture-of-experts switch that turns the same backbone into a camera-only, LiDAR-only, or camera-LiDAR encoder at inference. The authors identify this as an open challenge. Radar and other token types are suggested extensions, but the reported experiments do not establish their normalization or neighborhood rules.

## High-Level Takeaways

- UniTR's unit of sharing is the transformer block, while tokenizers, coordinate partitions, and task heads retain the structure each sensor needs.
- The decisive fusion ablation is 2D plus 3D interaction: 70.5 NDS for LiDAR only becomes 73.1 when both perspective and geometric neighborhoods are used.
- Parallel scheduling lowers the isolated backbone latency from 51 to 33 ms and slightly improves the matched result. The comparison does not include every tokenizer and partitioning cost in that table.
- The model is a strong camera-LiDAR backbone for fixed sensor families. A deployment that must change modalities at runtime still needs explicit routing, normalization, and failure training.
