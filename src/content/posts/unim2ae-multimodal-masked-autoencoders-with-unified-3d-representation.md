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

**ArXiv:** [2308.10421](https://arxiv.org/abs/2308.10421)

**Code:** [UniM2AE](https://github.com/hollow-503/UniM2AE)

## Summary

> UniM²AE pretrains camera and LiDAR encoders together without flattening their geometry into the image plane or a flat BEV. It masks image patches and LiDAR voxels independently, projects visible features into a shared 3D volume that preserves height, lets them interact through deformable 3D attention, and reconstructs each modality through its own decoder. With the paper's pre-trained MMIM variant, BEVFusion-SST gains 1.2 NDS and 1.5 mAP on nuScenes validation, while BEV map segmentation gains 6.5 mIoU. The objective is useful when synchronized unlabeled camera-LiDAR logs are available; its own limitations are random masking, redundant adjacent frames, and the gap between reconstruction and a deployed sensor set.

## Core Insights

### Preserve height before asking modalities to agree

A camera patch can contain several objects at different depths, while a LiDAR voxel already carries metric position. Projecting both into the image plane merges geometry; collapsing them into a flat BEV removes height. UniM²AE uses a 3D volume as the meeting space instead. LiDAR tokens are placed directly by ego-frame coordinates. For image features, 3D volume queries are projected into each camera view, and deformable attention samples the corresponding 2D feature maps. The volume keeps x, y, and z long enough for a traffic sign above the road and a car on the road to remain different objects.

![UniM²AE source Figure 1: image-plane alignment versus unified 3D interaction](/assets/images/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation-source-figure-1.webp)
*Fig 1: The source comparison shows why UniM²AE aligns features in a 3D volume rather than forcing masked camera and LiDAR inputs to share an image-plane layout. | source: [UniM²AE, Figure 1](https://arxiv.org/abs/2308.10421)*

The shared volume can then be mapped back to each modality. After interaction, the fused volume is sampled at masked LiDAR voxel positions and projected to the corresponding camera coordinates. That gives each modality a decoder target in its native space while allowing the encoder to borrow information from the other sensor. The shared object is therefore geometric enough for correspondence and detailed enough for reconstruction.

### Mask each stream, then exchange evidence in the volume

The camera branch splits six views into patches; the LiDAR branch voxelizes the point cloud. The default masking ratios are 75% for camera tokens and 70% for LiDAR tokens. Separate Swin-T and SST encoders process the visible tokens. Token-to-volume projection creates two feature volumes, one from each modality. The Multi-modal 3D Interaction Module concatenates them along channels, applies three stacked deformable self-attention blocks over the volume, then splits the result back into modality-specific features.

![UniM²AE source Figure 2: masked token branches, 3D interaction, and native reconstruction decoders](/assets/images/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation-source-figure-2.webp)
*Fig 2: Masked camera patches and LiDAR voxels meet in the height-preserving volume, exchange information through MMIM, and return to their original modalities for reconstruction. | source: [UniM²AE, Figure 2](https://arxiv.org/abs/2308.10421)*

The pretraining setup covers a perception-sized region of 100×100 m in x and y and -5 to 3 m in z, with 0.5×0.5×4 m voxels. The models train for 200 epochs on eight GPUs with a base learning rate of 2.5e-5. The LiDAR decoder reconstructs point content with Chamfer distance and predicts empty voxels with binary cross-entropy; the camera decoder uses pixel MSE. The reconstruction heads disappear during downstream fine-tuning.

### The strongest evidence is data efficiency plus the interaction ablation

UniM²AE is evaluated on nuScenes for 3D detection and BEV map segmentation. For data efficiency, the authors fine-tune with 20%, 40%, 60%, 80%, or 100% of the labeled data. At 20% labeled camera-LiDAR data, random initialization reaches 51.5 mAP/50.9 NDS, while UniM²AE reaches 55.9/52.8. The comparison is not only a full-data leaderboard: it tests whether the pretraining signal helps when labels are scarce.

The downstream full-data table uses SST-based variants and reports a protocol detail that changes the interpretation. BEVFusion-SST reaches 68.2 mAP/71.5 NDS; the plain UniM²AE row reaches 68.4/71.9, while the daggered row fine-tuned with the pre-trained MMIM reaches 69.7/72.7. The latter is the source of the paper's 1.5 mAP and 1.2 NDS improvement. For BEV map segmentation, UniM²AE improves BEVFusion-SST by 6.5 mIoU and beats X-Align by 2.1 in the reported setup.

Table 4 isolates where the gain comes from. Training from scratch reaches 61.8 NDS; adding modality-specific pretraining, multimodal initialization, and interaction progressively improves it, with the complete 3D-volume/MMIM setting reaching 65.2. Replacing the volume with BEV lowers the result, supporting the height-preservation explanation. The result is a causal clue about the representation, although the comparisons also inherit the different initialization paths.

The masking ablation finds a broad optimum around the default ratios: lower masking weakens the reconstruction challenge, while excessively high masking also hurts. Increasing the z-axis layers beyond two raises FLOPs and GPU memory without a consistent accuracy gain. Random masking does not coordinate complementary missing evidence across sensors, and the paper does not model temporal redundancy between adjacent driving frames.

## High-Level Takeaways

- UniM²AE's shared object is a calibrated 3D volume, while camera and LiDAR tokenizers, encoders, and decoders remain modality-specific.
- The useful downstream signal appears under label scarcity: at 20% labels, the multimodal model reaches 55.9 mAP/52.8 NDS versus 51.5/50.9 from scratch.
- The reported headline gain depends on the daggered MMIM fine-tuning row; the plain BEVFusion-SST comparison is smaller. Keep those initialization paths separate.
- The next test is whether volume interaction still helps with unsynchronized sensors, camera-only deployment, or temporal prediction targets that are less directly tied to reconstruction.
