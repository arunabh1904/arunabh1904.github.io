---
title: 'VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking'
date: '2023-03-20T04:00:00.000Z'
section: paper-shorts
postSlug: voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking
legacyPath: /paper shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking'
---
## 2023 – VoxelNeXt

**arXiv:** [2303.11301](https://arxiv.org/abs/2303.11301)

**Code:** [dvlab-research/VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)

## Summary

> VoxelNeXt asks whether sparse 3D features have to become dense at the moment they predict objects. Its answer is a voxel-to-object head: additional downsampling gives sparse features a larger receptive field, sparse height compression connects the 3D backbone to a 2D sparse head, and active voxels directly score and decode boxes. Sparse max pooling removes local duplicate queries without a dense heatmap or NMS. On nuScenes, the single-frame model reaches 64.5 mAP/70.0 NDS at 66 ms, but the paper also shows why FLOP savings do not translate into the same latency on every device.

## Core Insights

### Dense heatmaps spend work where there are no objects

CenterPoint turns sparse voxel features into a dense BEV map and predicts a heatmap at every cell. VoxelNeXt begins with a simple observation: on the nuScenes validation set, fewer than 1% of the car heatmap locations have a meaningful response. A dense head still computes over the other locations and then needs NMS to remove duplicate predictions. The paper therefore removes the center or anchor proxy and lets the active voxel set define the prediction domain.

![VoxelNeXt source Figure 1: sparse input and CenterPoint heatmaps](/assets/images/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking-source-figure-1.webp)
*Fig 1: Most CenterPoint car heatmap values are nearly zero even though the dense head evaluates the whole BEV grid. VoxelNeXt uses the non-empty voxel support as its starting set. | source: [VoxelNeXt, Figure 1](https://arxiv.org/abs/2303.11301)*

The change is more than replacing one convolution. A regular dense head gives every spatial cell the same computational status, while a sparse head can discard locations before box regression. That makes “where should the detector spend capacity?” an explicit part of the architecture.

### Receptive field grows through sparse downsampling

Direct prediction cannot rely on a dense head to broaden context. VoxelNeXt adapts a six-stage sparse CNN with strides $1,2,4,8,16,32$. The first four stages are the usual backbone; two additional downsampling stages enlarge the effective receptive field, and features from the last three stages are aligned and concatenated at the stride-8 resolution without another parameterized fusion layer. Sparse height compression then places active 3D voxels onto the ground plane and sums features at coincident $(x,y)$ positions. The result is a 2D sparse feature set, so the detector can keep a 3D backbone while making the prediction head cheaper.

![VoxelNeXt source Figure 4: sparse backbone, height compression, voxel selection, and box regression](/assets/images/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking-paper-figure.webp)
*Fig 2: The detailed source diagram shows the four decisions in sequence: extra downsampling, sparse height compression, active-voxel selection, and direct box regression. The retained asset is the paper's Figure 4. | source: [VoxelNeXt, Figure 4](https://arxiv.org/abs/2303.11301)*

The ablation makes the receptive-field tradeoff concrete. With only three downsampling stages, direct prediction falls to 46.7 mAP; adding one stage reaches 52.3, and the default five-stage variant reaches 56.5 mAP/64.5 NDS in the pilot study. A $5\times5\times5$ kernel also recovers some accuracy, but costs 225 ms versus 66 ms for the extra-downsampling design. Enlarging context by keeping more sparse locations would erase the efficiency gain, so the paper chooses depth and stride instead.

The backbone also prunes spatial support by feature magnitude. With a default pruning ratio of 0.5, only the top half of active voxels are dilated during downsampling. Ratios up to 0.5 barely change validation performance, while 0.7 and 0.9 cause much larger drops. This is a learned computational budget: the network preserves high-response locations while refusing to expand every background voxel into a larger kernel neighborhood.

### Query voxels predict boxes without being object centers

The sparse prediction head first scores each active voxel, performs a class-wise sparse max pool over local neighborhoods, and decodes boxes only at the surviving queries. A $3\times3$ submanifold sparse convolution gives the default head more local context than a fully connected $1\times1$ head; the latter is faster but loses some accuracy. Because local maxima are selected on the active set, this stage can replace dense heatmap peak extraction and NMS. In the validation ablation, sparse max pooling alone reaches 56.2 mAP, NMS alone reaches 56.0, and using both also reaches 56.2. The comparable numbers support removing NMS without claiming that the two operations are identical.

The query need not be near the geometric center. On high-quality predictions with IoU above 0.7, only 9.9% of queries are near the center, 72.8% are near the box boundary, and 17.3% are outside the box. The distribution changes by class: sparse pedestrians and some traffic cones are often predicted from outside voxels. This is an important intuition about voxel features. A surface return can encode an object's class and extent even when the object center has no return; forcing every prediction through a center proxy would throw away that evidence.

### Sparse queries also repair center-biased tracking

VoxelNeXt extends CenterPoint's velocity-based tracking by associating the query voxels that actually produced each box. It follows the voxel index back to an input voxel instead of using only a predicted center, then matches query positions between frames by L2 distance. Because those voxels lie on observed surfaces, their relative position to an object can be more stable than a biased center prediction. Voxel association adds 1.1 AMOTA on the nuScenes validation ablation. On the test split, the single-frame model reaches 69.5 AMOTA; with double-flip detection results it reaches 71.0 and ranks first among the reported LiDAR-only entries.

### The speed claim is hardware-shaped

On nuScenes test, VoxelNeXt reports 64.5 mAP and 70.0 NDS at 66 ms; the double-flip variant reaches 66.2 mAP and 71.4 NDS. The fully 2D sparse variant is faster at 61 ms but drops to 53.4 mAP/62.6 NDS in the pilot comparison. VoxelNeXt uses 38.7G FLOPs versus CenterPoint's 186.6G in the paper's accounting, yet the authors explicitly warn that sparse-kernel implementation and hardware determine realized latency. The Argoverse2 range plot makes the same point: CenterPoint's overall and head latency rises sharply from 50 to 200 m, while VoxelNeXt stays nearly flat.

![VoxelNeXt source Figure 3: latency across Argoverse2 perception ranges](/assets/images/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking-source-figure-3.webp)
*Fig 3: Across 50–200 m perception ranges, the source plot shows CenterPoint latency growing much faster than VoxelNeXt's overall and head latency. | source: [VoxelNeXt, Figure 3](https://arxiv.org/abs/2303.11301)*

## High-Level Takeaways

- VoxelNeXt keeps the active voxel set through height compression and prediction, removing sparse-to-dense conversion, center/anchor proxies, and the need for NMS.
- Two extra sparse downsampling stages recover receptive field for direct prediction; the pilot ablation rises from 46.7 to 56.5 mAP without using dense head computation.
- Only 9.9% of high-quality query voxels are near object centers, so boundary and even outside voxels carry useful box evidence.
- The nuScenes result is 64.5 mAP/70.0 NDS at 66 ms, while the paper's FLOP and range studies show that sparse speed remains implementation and hardware dependent.
