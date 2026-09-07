---
title: 'VoxelNet: End-to-End Point Cloud 3D Detection'
date: '2017-11-17T05:00:00.000Z'
section: paper-shorts
postSlug: voxelnet-end-to-end-point-cloud-3d-detection
legacyPath: /paper shorts/2017/11/17/voxelnet-end-to-end-point-cloud-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2017 – VoxelNet: learn point-cloud features inside metric voxels'
---
## 2017 – VoxelNet

**arXiv:** [1711.06396](https://arxiv.org/abs/1711.06396)

## Summary

> VoxelNet makes the first representation decision explicit: divide the metric scene into voxels, learn a descriptor for the points inside each occupied voxel, and only then hand a volumetric tensor to a detector. Its VFE block retains local shape that hand-crafted height and density channels discard. The KITTI results show the value of that choice most clearly in full 3D detection, while the 225 ms TitanX runtime shows why the dense 3D middle network quickly became the next bottleneck.

## Core Insights

### A voxel token carries both point identity and local shape

VoxelNet starts from a point $\mathbf{p}_i=[x_i,y_i,z_i,r_i]$ and augments it with the offset from the mean point in its voxel. The resulting seven-dimensional input is $[x_i,y_i,z_i,r_i,x_i-v_x,y_i-v_y,z_i-v_z]$. A shared linear layer, batch normalization, and ReLU map each point to a feature. Elementwise max pooling produces a voxel summary, and that summary is concatenated back to every point before another VFE layer. The final max pool turns the set into one voxel feature.

![Figure 3 from VoxelNet: End-to-End Point Cloud 3D Detection](/assets/images/voxelnet-end-to-end-point-cloud-3d-detection-source-figure-3.webp)
*Fig 1: The VFE layer applies the same pointwise transform to every point, pools a voxel-level feature, and returns that context to the point features before the next layer. | source: [VoxelNet, Figure 3](https://arxiv.org/abs/1711.06396)*

The max operation makes the final representation insensitive to the order in which points arrived, while the centroid offsets preserve local surface geometry. A flat road patch and a vertical object can occupy similar voxel volume but produce different patterns of relative offsets. This is the paper's core departure from a fixed occupancy, height, or density statistic: the local descriptor is learned jointly with the box objective. In the reported car model, the stack is VFE(7, 32) followed by VFE(32, 128); pedestrians and cyclists use a separate point cap because their smaller objects need more of the available returns.

### Voxelization makes a variable cloud tensor-shaped, then reintroduces density

A high-definition scan may contain roughly 100,000 points, but a voxel can contain anything from one point to many. VoxelNet randomly samples at most $T$ points in an overfull voxel and stores only a bounded number of non-empty voxels. For the KITTI car setting, the crop is $[-3,1]\times[-40,40]\times[0,70.4]$ m along $z,y,x$, the voxel size is $0.4\times0.2\times0.2$ m, and the grid is $10\times400\times352$. The car model uses $T=35$ points per voxel; the pedestrian and cyclist setting uses $T=45$ and a shorter range.

![VoxelNet architecture from raw point cloud to 3D detections](/assets/images/voxelnet-end-to-end-point-cloud-3d-detection-paper-figure.webp)
*Fig 2: The end-to-end pipeline first learns voxel features, then applies 3D convolutional middle layers and an RPN to predict 3D boxes. | source: [VoxelNet, Figure 2](https://arxiv.org/abs/1711.06396)*

Only non-empty voxels are encoded, so the intermediate representation is a sparse four-dimensional tensor. The car configuration's two VFE layers feed a $64\times2\times400\times352$ middle tensor, which is reshaped to $128\times400\times352$ after the vertical dimension is reduced. The RPN has three convolutional branches with different strides, upsamples them to a common resolution, concatenates them, and predicts classification and seven-parameter box residuals. This is a carefully chosen handoff: VFE handles unordered local sets, the middle network aggregates neighboring voxels, and the RPN receives a regular map. It is also where the cost returns. More than 90% of the possible voxels are empty, yet the middle layers operate after the sparse-to-dense transition.

### The strongest evidence is in full 3D detection

VoxelNet evaluates both bird's-eye-view and full 3D boxes on the KITTI validation split, then reports an official test submission. The commonly used split has 3,712 training frames and 3,769 validation frames. The controlled hand-crafted baseline uses the same broad detector design and replaces the learned VFE features with BEV-style statistics. VoxelNet improves that baseline especially for pedestrians and cyclists, where local 3D shape matters more than ground-plane localization.

For cars, the paper's Table 2 3D comparison is unusually revealing: LiDAR-only VoxelNet exceeds the cited MV method, which uses LiDAR and RGB, by 10.68, 2.78, and 6.29 percentage points on easy, moderate, and hard difficulty. Those differences belong to the **3D** table; they are not BEV gains. For pedestrian and cyclist detection, the paper reports roughly an 8% improvement over its hand-crafted comparison in BEV and roughly 12% in 3D. The widening gap is consistent with the representation hypothesis: height, surface orientation, and partial shape become part of the decision instead of being summarized before learning.

The qualitative source panel makes the task concrete. It shows one raw LiDAR scene with predicted 3D boxes colored by class for cars, pedestrians, and cyclists. The panel is a point-cloud visualization of the LiDAR detections; it does not show an RGB camera projection or imply that RGB was an input to the model.

![VoxelNet source Figure 1: LiDAR detections with class-colored boxes](/assets/images/voxelnet-end-to-end-point-cloud-3d-detection-source-figure-1.webp)
*Fig 1: A raw LiDAR scene with predicted 3D boxes for cars, pedestrians, and cyclists, colored by class. | source: [VoxelNet, Figure 1](https://arxiv.org/abs/1711.06396)*

### The representation has a measurable systems boundary

The reported inference time is 225 ms on a TitanX GPU with a 1.7 GHz CPU: 5 ms to compute voxel input features, 20 ms in the feature-learning network, 170 ms in the convolutional middle layers, and 30 ms in the RPN. The breakdown explains the lineage that follows. Learning local shape is not the dominant cost; dense 3D context aggregation is. Random point caps also make the representation depend on range and occupancy, and the KITTI crop is much smaller than a full surround-sensor scene. VoxelNet establishes the learned voxel interface, but its own timing makes the case for sparse middle layers and early height compression.

## High-Level Takeaways

- VFE replaces fixed voxel statistics with a permutation-invariant learned descriptor built from pointwise features and pooled local shape.
- The model is sparse during voxel encoding but pays for a dense 3D middle tensor after the VFE stage.
- Its most diagnostic gains are in KITTI full 3D detection, including a LiDAR-only car result that exceeds the cited LiDAR-plus-RGB MV baseline.
- The 225 ms breakdown places the next research problem in the convolutional middle layers, not in the local set encoder.
