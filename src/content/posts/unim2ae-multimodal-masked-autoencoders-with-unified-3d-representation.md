---
title: 'UniM²AE: Multi-Modal Masked Autoencoders with Unified 3D Representation for Autonomous Driving'
date: '2023-08-21T00:00:00.000Z'
section: paper-shorts
postSlug: unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation
legacyPath: /paper shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – UniM²AE: Multi-Modal Masked Autoencoders with Unified 3D Representation for Autonomous Driving'
---
## 2023 – UniM²AE

**arXiv:** [2308.10421](https://arxiv.org/abs/2308.10421)

**Code:** [hollow-503/UniM2AE](https://github.com/hollow-503/UniM2AE)

### Method and reported result

UniM²AE pretrains camera and LiDAR encoders by masking both inputs, projecting visible features into a shared 3D volume, exchanging information there, and reconstructing each modality through its own decoder. Extending BEV along height preserves more 3D structure than a flat grid while giving both sensors a common pretraining workspace.

## Summary

> The paper addresses a scale problem: labeled 3D boxes and maps are expensive, but synchronized images and point clouds are abundant on instrumented fleets. A masked reconstruction objective can consume those pairs before task-specific fine-tuning.

## Core Insights

The camera branch divides images into patches; the LiDAR branch voxelizes the point cloud. Both mask a large fraction of tokens before their encoders, which reduces pretraining cost. Token-to-volume projection aligns visible camera and LiDAR features in the ego frame. The Multi-modal 3D Interaction Module exchanges semantic and geometric evidence, then volume-to-token projection returns features to modality-specific decoders for reconstructing the masked image and point-cloud inputs.

On nuScenes, the paper reports improvements of 1.2 NDS for 3D object detection and 6.5 mIoU for BEV map segmentation after pretraining. The result supports cross-modal masked pretraining, but the objective still depends on synchronized camera-LiDAR data and reconstruction fidelity is only a proxy for downstream perception.

![Figure 1 from UniM²AE: Multi-Modal Masked Autoencoders with Unified 3D Representation for Autonomous Driving](/assets/images/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation-source-figure-1.webp)
*Fig 1: Early input alignment fuses modalities before their encoders, whereas UniM²AE first extracts modality-specific features and then lets them interact in a shared 3D volume. | source: [UniM²AE: Multi-Modal Masked Autoencoders with Unified 3D Representation for Autonomous Driving](https://arxiv.org/abs/2308.10421)*

![Masked camera and LiDAR tokens interacting in a shared 3D volume before modality-specific reconstruction](/assets/images/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation-source-figure-2.webp)
*Fig 2: Camera patches and LiDAR voxels are masked separately, projected into a shared 3D volume for cross-modal completion, and decoded back into their original modalities during pretraining. | source: [UniM²AE: Multi-Modal Masked Autoencoders with Unified 3D Representation for Autonomous Driving](https://arxiv.org/abs/2308.10421)*


| Pretraining choice | Purpose | Cost or uncertainty |
| --- | --- | --- |
| Independent input masking | Reduces encoder tokens and creates missing evidence | Mask ratio can make one modality dominate reconstruction. |
| Shared 3D volume | Aligns camera semantics with LiDAR geometry | Adds calibrated projection and height-axis memory. |
| Cross-modal interaction | Lets one sensor help reconstruct the other | May learn shortcuts from synchronized pairs. |
| Modality-specific decoders | Preserve different reconstruction targets | Decoder quality does not guarantee downstream transfer. |

## High-Level Takeaways

- UniM²AE informs whether synchronized unlabeled sensor logs should pretrain separate encoders independently or through a shared geometric bottleneck. Its atomic units are image patches and LiDAR voxels; encoders and decoders are modality-specific, while the interaction volume is shared.
- The missing matched controls equalize visible tokens and compute across camera-only MAE, LiDAR-only MAE, late feature alignment, and shared-volume pretraining. At 10× fleet data, synchronization, calibration drift, data diversity, and storage dominate. The shared-volume objective would fail if future prediction or task-aware distillation transfers better under the same data and compute, or if the deployed camera-only encoder depends too heavily on LiDAR co-occurrence during pretraining.
- UniM²AE represents the masked-reconstruction branch of driving pretraining; UniWorld, ViDAR, and DriveWorld instead ask the representation to predict occupancy or future geometry and dynamics.
- Pretraining at scale needs a target that forces semantics and geometry to meet before expensive task labels are introduced.
