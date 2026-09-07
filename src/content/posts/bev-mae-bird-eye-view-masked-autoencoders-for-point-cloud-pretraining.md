---
title: "BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining"
date: '2022-12-12T05:00:00.000Z'
section: paper-shorts
postSlug: bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining
legacyPath: /paper shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2022 – BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining"
---
## 2022 – BEV-MAE

**arXiv:** [2212.05758](https://arxiv.org/abs/2212.05758)

**Code:** [VDIGPKU/BEV-MAE](https://github.com/VDIGPKU/BEV-MAE)

## Summary

> BEV-MAE makes the masked unit match the spatial unit consumed by an outdoor LiDAR detector: a non-empty ground-plane cell. During pretraining it replaces the cell’s points with a shared token, asks a light decoder to reconstruct local point geometry and range-dependent density, and then discards that decoder for detection. On Waymo, the method improves the 20%-pretraining row by 1.10 L2 mAP / 1.04 L2 APH and the 100%-pretraining row by 1.42 / 1.34, with every detector fine-tuned on 20% labels. Gains grow in the low-label regime, while the full-label and cross-sensor results expose the limits of treating scan density as a transferable location cue.

## Core Insights

### Mask the representation the detector will consume

Voxel masking is a natural analogue of masked image modeling, but outdoor LiDAR detectors do not ultimately reason over arbitrary voxel patches. They encode sparse points and project them into a BEV feature map. BEV-MAE first defines an $X\times Y$ ground-plane grid whose spacing matches the 3D encoder’s downsample ratio. Each point is assigned to a cell by its $(x,y)$ coordinates. The method randomly selects a large fraction of non-empty cells as masked cells and leaves the remaining cells visible.

The masked points are replaced with one shared learnable point token. The sparse convolution still receives the full point-cloud coordinates as its indexing structure, while the first-layer feature for masked points is the token and the later sparse-convolution layers are unchanged. This matters because removing masked points would also remove the occupied sparse-convolution sites and shrink the receptive field during pretraining. The token keeps the communication pattern similar to fine-tuning without revealing the hidden cells’ coordinates or content.

The BEV-aligned mask also makes the decoder target fixed in space. One BEV feature corresponds to one masked cell, so a one-layer $3\times3$ convolution can predict its reconstruction without sparse upsampling or a multi-scale decoder.

![BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining source figure: Overall pipeline of BEV-MAE.](/assets/images/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining-paper-figure.webp)
*Fig 1: Points are grouped into non-empty BEV cells, masked cells receive a shared point token, and a lightweight decoder predicts local geometry and density from the encoded BEV features. | source: [BEV-MAE, Figure 3](https://arxiv.org/abs/2212.05758)*

### Reconstruct shape and density, not just occupancy

For each masked cell, the decoder predicts a fixed set of $L=20$ points even though the original cell can contain a different number of returns. The target points are represented by offsets from the cell center and normalized by the BEV cell size. A Chamfer-distance loss compares the predicted set with the variable-cardinality target, so the decoder learns local surface structure without requiring an arbitrary point ordering. Coordinate normalization prevents cells at different absolute positions from producing unstable regression scales.

The second target is point density. For a masked cell, BEV-MAE counts its points and divides by the occupied 3D volume, then applies a Smooth-L1 loss to the predicted density. Outdoor LiDAR becomes sparser with distance from the sensor, so density carries a coarse location prior that point coordinates alone do not make explicit. The authors compare this with predicting the raw number of points, which varies from one to hundreds and is less stable.

This target is a useful inductive bias and a potential domain assumption at the same time. Density can tell the encoder that a feature is far away, but a new LiDAR’s beam pattern, returns, and accumulation strategy can change that relationship. The pretext task is therefore learning scan geometry as well as object shape.

### The clean Waymo gain depends on the pretraining fraction

Waymo Table 1 keeps fine-tuning fixed at 20% of the labeled training data. BEV-MAE with 20% of the Waymo data for pretraining takes 5 hours and reaches 66.70 L2 mAP / 64.25 L2 APH, which is +1.10 / +1.04 over the from-scratch baseline at 65.60 / 63.21. With 100% pretraining data and 30 epochs, it reaches 67.02 / 64.55, or +1.42 / +1.34. The two gains describe different data-availability points; the larger pair should not be attached to the 20%-pretraining row.

The efficiency comparison is part of the claim. The 20%-pretraining BEV-MAE run uses 5 hours, while its 100%-pretraining run is estimated at 38 hours; the paper reports 63% of GA-MAE’s pretraining cost for the matched comparison and a stronger result than ProposalContrast. The masked target and one-layer decoder reduce the cost enough that unlabeled-data pretraining is practical within the paper’s hardware setup.

![Figure 1 from BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining](/assets/images/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining-source-figure-1.webp)
*Fig 2: BEV-MAE’s Waymo accuracy–pretraining-time curve compares methods after CenterPoint fine-tuning on 20% labels; the points encode both representation quality and the cost of acquiring it. | source: [BEV-MAE, Figure 1](https://arxiv.org/abs/2212.05758)*

### Pretraining matters most when labels are scarce

The data-efficiency table makes the transfer pattern clearer than the headline row. With 5% labeled Waymo data, BEV-MAE raises L2 mAP from 44.41 to 51.63 and L2 APH from 40.34 to 47.77, gains of 7.22 and 7.43. At 10%, the gains are 3.85 and 4.29; at 20%, 2.72 and 3.19; at 50%, 0.73 and 0.71; and at 100%, only 0.85 and 0.92. Once the detector sees all labels, detector capacity becomes a stronger bottleneck than initialization.

The component ablation points to why the full recipe works. Coordinate reconstruction without normalization reaches 65.66/63.09, while normalized coordinates reach 66.20/63.71. Density alone reaches 65.80/63.27; replacing density with point-count prediction falls to 65.32/62.88. Combining normalized coordinates and density reaches 66.49/63.99, and adding the shared token reaches 66.70/64.25. The token’s 0.21 mAP / 0.26 APH gain is small but consistent with its intended role: preserving the encoder’s receptive-field pattern rather than adding semantic information.

The decoder study supports simplicity. A one-layer convolution reaches 66.70/64.25 at $1\times$ cost, a residual convolution block reaches 66.61/64.09 at $1.2\times$, and a transformer block reaches 65.80/63.26 at $1.4\times$. More decoder capacity does not improve the representation here, likely because the sparse-convolution encoder and transformer decoder have mismatched inductive biases.

### The mask and sensor domain are part of the result

Against random occupied-voxel masking, BEV-guided masking reaches 66.70 L2 mAP / 64.25 L2 APH versus 66.63 / 64.16, while reducing GPU memory from 12.6 GB to 4.1 GB and training cost from $1.4\times$ to $1\times$. The method works across a broad 50–80% mask range, with 70% best in the reported ablation. Its efficiency is therefore not only a consequence of masking fewer points; it comes from choosing cells that line up with the downstream BEV feature and decoder.

The nuScenes result is strong but has a separate model contract. Vanilla BEV-MAE with TransFusion-L reports 71.7 NDS / 67.0 mAP, while the modified-model variant reports 73.6 / 69.6. The paper’s transfer experiments use coordinates only so that pretrained representations can cross sensor types. Pretraining and fine-tuning on the same dataset work better, and the authors note that Waymo’s point density is about five times nuScenes’. A combined nuScenes+Waymo pretraining set improves both targets, but the cross-sensor gap means the density prior is not automatically universal.

## High-Level Takeaways

- BEV-MAE aligns the pretext task with the detector’s BEV representation: mask non-empty ground-plane cells, preserve sparse-convolution sites with a shared token, and decode each cell directly.
- Normalized local coordinates and density are complementary targets; raw point count and heavier decoders are weaker in the reported ablations.
- At 20% labeled fine-tuning, the 20%-pretraining row gains 1.10/1.04 and the 100%-pretraining row gains 1.42/1.34 L2 mAP/APH; the contexts are distinct.
- The largest practical benefit appears with 5% labels, where L2 mAP rises 7.22 points, while gains are marginal at full labels.
- The method’s strongest assumptions are outdoor scan geometry and density. The 73.6 NDS / 69.6 mAP nuScenes point uses a modified model, and transfer across new sensors remains a domain question.
