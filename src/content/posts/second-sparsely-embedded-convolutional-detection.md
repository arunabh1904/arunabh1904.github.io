---
title: 'SECOND: Sparsely Embedded Convolutional Detection'
date: '2018-10-06T04:00:00.000Z'
section: paper-shorts
postSlug: second-sparsely-embedded-convolutional-detection
legacyPath: /paper shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2018 – SECOND: Sparsely Embedded Convolutional Detection'
---
## 2018 – SECOND

**Paper:** [Sensors 18(10), 3337](https://doi.org/10.3390/s18103337)

## Summary

> SECOND keeps the useful 3D part of VoxelNet while refusing to convolve through empty road volume. Sparse convolution and GPU rule generation preserve occupied coordinates until height has been reduced to a BEV map. A sine-based angle loss and a direction classifier repair a separate box ambiguity, while ground-truth database sampling changes the amount of foreground seen during training. The paper's 20–40 FPS result is the combination of these execution and supervision choices.

## Core Insights

### Sparsity is an execution rule with a deliberate boundary

SECOND has three blocks: a voxelwise feature extractor, a sparse convolutional middle layer, and an RPN. Its VFE stage follows VoxelNet, including two linear-batch-normalization-ReLU VFE layers and a final linear projection. The difference begins in the middle. A normal sparse convolution can create output coordinates adjacent to active inputs; a submanifold convolution restricts outputs to locations that were already active. The model uses both: sparse layers downsample the vertical axis and submanifold layers propagate features without allowing the active set to explode.

![SECOND detector from point cloud through voxel features and sparse convolution to three RPN heads](/assets/images/second-paper-figure.webp)
*Fig 1: SECOND converts points to voxel features and coordinates, processes them with VFE and sparse CNN layers, then predicts class, box, and direction outputs through an RPN. | source: [SECOND, Figure 1](https://doi.org/10.3390/s18103337)*

The two-phase middle extractor first keeps the 3D coordinate structure while learning along $z$. Each phase contains submanifold layers and one ordinary sparse convolution with stride two in height. Once the vertical dimension has been reduced to one or two cells, the sparse tensor is converted to a dense feature map and reshaped into a 2D BEV tensor. The RPN then follows an SSD-like pattern: three stages downsample, apply convolutions, upsample to a common resolution, concatenate, and emit class, regression, and direction maps. SECOND is therefore sparse where occupancy is expensive and dense where the detector benefits from regular 2D kernels.

The car crop is $[-3,1]\times[-40,40]\times[0,70.4]$ m along $z,y,x$, with voxel size $0.4\times0.2\times0.2$ m and at most 35 points per voxel. Pedestrian and cyclist detection uses a shorter crop and $T=45$ points per voxel because small objects need more returns. These limits determine both the active set and the final BEV resolution.

### GPU rule generation is part of the contribution

A sparse convolution does not multiply a dense kernel over every coordinate. It must build rules that map active input coordinates to valid output coordinates, gather the feature rows for each kernel offset, and scatter the results. Earlier implementations generated these rules on the CPU, adding transfer and synchronization overhead. SECOND's GPU rule-generation algorithm constructs the coordinate relations in parallel, removes duplicate output positions, and uses a lookup buffer to map the rules to gather/scatter indices.

On KITTI, the authors report a factor-of-four training speedup and a factor-of-three inference speedup over a dense convolution implementation. The small network runs in about 0.025 s on a GTX 1080 Ti, and the reported large and small models run at approximately 20 and 40 FPS. These are hardware and occupancy-dependent measurements: the coordinate bookkeeping, irregular memory access, and eventual sparse-to-dense conversion remain in the path. The speedup is a property of an active voxel pattern and a particular implementation, not a universal multiplier for every scene.

### Orientation needs a second target

A 3D box can have the same footprint when its yaw differs by $\pi$. Directly regressing the raw angle treats those equivalent boxes as far apart and creates a large error at the wraparound. SECOND replaces the direct angular residual with a sine-error objective, which makes the overlap-related loss continuous across that equivalence. But sine alone cannot tell which semantic direction the object faces. A small auxiliary direction classifier supplies that missing bit and is trained with a softmax loss.

This division of labor matters. The regression term can concentrate on the geometry that determines box overlap, while the direction head handles the front-versus-back convention. The paper notes that the direction loss is given a smaller weight so difficult examples do not make the detector sacrifice box localization merely to resolve heading.

### Database sampling changes the training distribution

The authors build a database of labeled objects and the LiDAR points inside their boxes. During training they sample complete objects, transform their points and boxes together, concatenate them into the current scene, and remove any sample that collides with an existing object. This increases the number of positive instances per frame and exposes the network to more layouts without inventing points with a separate renderer. Global scaling and rotation are also applied to the scene and boxes.

The convergence plot and ablation section attribute a large part of the improvement to this foreground resampling. It is a useful reminder that a sparse backbone can be limited by the number of objects it sees, not only by the way it stores empty voxels. The method still depends on the database's object distribution and collision rule; pasted returns can be statistically unusual outside the KITTI setting.

### The benchmark supports the combined design

On the official KITTI test set, the large SECOND model reports 0.05 s per frame and car 3D AP of 83.13, 73.66, and 66.20 for easy, moderate, and hard difficulty. The paper emphasizes that these are LiDAR-only results and that several comparison methods use images or different train/validation splits. Its BEV results are slightly behind the strongest multimodal systems while remaining ahead of LiDAR-only VoxelNet, which makes the execution trade-off visible rather than hiding it inside one headline score. The authors also report that pedestrian and cyclist results are competitive but may benefit from image information, motivating camera fusion as future work.

## High-Level Takeaways

- SECOND preserves 3D sparsity through height reduction and densifies only at the BEV handoff.
- GPU rule generation turns sparse convolution from a conceptual operation into a usable execution path.
- Sine angle regression optimizes box overlap while the direction classifier restores heading semantics.
- Database sampling addresses foreground scarcity and is a major part of the reported convergence and accuracy gains.
- Its speed and accuracy numbers are tied to KITTI occupancy, a fixed crop, and the cost of the final dense BEV map.
