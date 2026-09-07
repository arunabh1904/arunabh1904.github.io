---
title: 'VoTr: Voxel Transformer for 3D Object Detection'
date: '2021-09-06T04:00:00.000Z'
section: paper-shorts
postSlug: votr-voxel-transformer-for-3d-object-detection
legacyPath: /paper shorts/2021/09/06/votr-voxel-transformer-for-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – VoTr: Voxel Transformer for 3D Object Detection'
---
## 2021 – VoTr

**arXiv:** [2109.02497](https://arxiv.org/abs/2109.02497)

## Summary

> VoTr asks whether a sparse voxel should receive context from a fixed geometric kernel or from a content-dependent list of occupied neighbors. Its local and dilated attention mechanisms enlarge the receptive field without making all roughly 90,000 Waymo voxels attend to one another. Fast Voxel Query is part of the design: the GPU hash table makes neighbor retrieval executable. The gains are strongest for sparse, distant objects, while irregular indexing lowers the paper's throughput relative to SECOND.

## Core Insights

### The query follows occupied geometry

VoTr replaces selected sparse-convolution blocks with voxel self-attention. A query voxel projects its feature to $Q$; each attending voxel contributes a key and value augmented with a relative position embedding derived from the difference between their 3D centers. Attention is computed only over an explicitly constructed neighbor set. This matters because a full transformer over the non-empty voxels would still be too large: the Waymo frame has nearly 90,000 occupied voxels, although they occupy less than 0.1% of the dense grid.

The paper keeps two kinds of voxel modules. A submanifold module updates features only at already occupied positions, preserving the measured scene structure. A sparse module can create features at empty positions, which is needed when downsampling moves voxel centers or when the active support should expand. At an empty query position there is no input feature from which to form $Q$, so VoTr approximates it by max-pooling the features of the attending voxels before applying attention.

![VoTr architecture with sparse and submanifold voxel modules](/assets/images/votr-voxel-transformer-for-3d-object-detection-paper-figure.webp)
*Fig 1: VoTr replaces the 3D convolutional backbone with sparse and submanifold voxel modules, then sends the resulting features to the usual BEV backbone and detection head. | source: [VoTr, Figure 2](https://arxiv.org/abs/2109.02497)*

This distinction is the structural intuition behind the architecture. The sparse module changes the support when the backbone needs it; the submanifold module lets attention spread information without filling every empty cell. The rest of the detector remains familiar: voxel features become a BEV map, proposals are generated, and the two-stage VoTr-TSD variant can reuse the backbone for ROI refinement.

### Local and dilated attention spend a bounded budget

Local Attention uses a three-dimensional radius of $(1,1,1)$ and searches every occupied voxel in that neighborhood. Dilated Attention divides a larger range into progressively wider shells and increases the search stride with distance. Near neighbors are sampled densely, while far neighbors are sampled sparsely. With the paper's configuration, the range can exceed 15 m while each query still attends to fewer than 50 voxels.

![Local and dilated attention in VoTr](/assets/images/votr-voxel-transformer-for-3d-object-detection-source-figure-3.webp)
*Fig 2: Local attention keeps all occupied neighbors in a nearby 3D range; dilated attention reaches farther by increasing the spacing between candidate locations. | source: [VoTr, Figure 3](https://arxiv.org/abs/2109.02497)*

The comparison with convolution is about coverage, not a claim that every distant voxel is useful. A fixed 3D kernel repeatedly samples the same geometric offsets and can miss the disconnected returns of an incomplete object. VoTr's query first enumerates candidate coordinates, keeps the occupied ones, and then lets feature similarity weight them. The source illustration is drawn in two dimensions, but the search rule is three-dimensional.

![VoTr source Figure 1(a): fixed convolution neighborhood](/assets/images/votr-voxel-transformer-for-3d-object-detection-source-figure-1.webp)
*Fig 1(a): A fixed 3D convolution samples a local geometric neighborhood around a query; VoTr replaces that fixed active-neighbor pattern with local or dilated occupied-voxel retrieval. | source: [VoTr, Figure 1(a)](https://arxiv.org/abs/2109.02497)*

### Fast Voxel Query turns retrieval into parallel lookup

The sparse coordinate list is not ordered so that a neighbor's integer coordinate can be found directly. Scanning all $N_{\mathrm{sparse}}$ coordinates for every query would cost $O(N_{\mathrm{sparse}})$ per lookup; storing a dense index over the entire scene would require more than $10^7$ cells. Fast Voxel Query builds a GPU hash table from integer voxel coordinates to rows in the sparse feature array. Each query generates its local or dilated candidate coordinates, hashes them in parallel, rejects empty entries, and gathers the surviving features.

That implementation is a useful boundary on the method. The attention matrix is small by construction, but the hash-table build, irregular reads, and candidate generation are still real work. The paper's claim is a better receptive-field/compute trade-off for occupied voxels, not free global context.

### Long-range context shows up in the far-range tables

On Waymo's 202 validation sequences, simply replacing the SECOND backbone with VoTr-SSD improves Level-1 vehicle mAP by 1.05 points and mAPH by 1.11 points. The range breakdown is more informative than the average: the mAP gains are 0.08 points at 0–30 m, 1.42 at 30–50 m, and 1.72 beyond 50 m. Replacing the PV-RCNN backbone with VoTr-TSD gives larger gains, including 3.37 and 4.83 points in the two farther ranges. Level 1 counts boxes with more than five LiDAR points; Level 2 includes boxes with at least one point.

On the KITTI validation comparison, VoTr-SSD reaches 78.27 moderate car AP at 14.65 Hz, compared with SECOND at 76.48 AP and 20.73 Hz. PV-RCNN reaches 83.69 AP at 9.25 Hz, while VoTr-TSD reaches 84.04 AP at 7.17 Hz. Increasing each query's attended-voxel budget from 24 to 48 adds 1.19 moderate AP, and adding dilated attention to local attention adds 2.79 points in the paper's ablation. These results support a specific intuition: the extra context helps when an object is represented by a few separated returns, but its indexing cost is visible in throughput.

## High-Level Takeaways

- VoTr uses separate sparse and submanifold modules so attention can expand support when needed while preserving occupied geometry elsewhere.
- Local attention keeps fine detail; dilated attention reaches beyond 15 m with fewer than 50 candidates per query.
- Fast Voxel Query is essential infrastructure, using a GPU hash table instead of a dense scene index or a full sparse-list scan.
- The largest reported gains are at 30 m and beyond, where incomplete objects need context beyond a fixed local kernel.
- The paper's KITTI trade-off—14.65 Hz for 78.27 AP versus SECOND's 20.73 Hz for 76.48—exposes the cost of content-dependent retrieval.
