---
title: 'CaDDN: Categorical Depth for Monocular 3D Detection'
date: '2021-03-01T05:00:00.000Z'
section: paper-shorts
postSlug: caddn-categorical-depth-for-monocular-3d-detection
legacyPath: /paper shorts/2021/03/01/caddn-categorical-depth-for-monocular-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2021 – CaDDN: supervise a depth distribution that lifts image features into 3D'
---

## 2021 – CaDDN

**arXiv:** [2103.01100](https://arxiv.org/abs/2103.01100)
**Code:** [TRAILab/CaDDN](https://github.com/TRAILab/CaDDN)

## Summary

> CaDDN turns a camera feature into a probability-weighted ray, then builds a metric voxel grid for 3D detection. Its contribution is not just predicting depth: it supervises a categorical depth distribution, preserves that distribution during lifting, and lets the detection loss refine it. On KITTI validation, depth supervision raises moderate-car 3D AP from 6.43 to 14.03 in the component ablation; in a separate experiment, retaining the full distribution improves over choosing its most likely bin by another 2.60 points. LiDAR-derived depth labels are a training dependency; inference uses one image.

## Core Insights

### A feature can occupy several depths without being copied equally everywhere

A pixel tells us which camera ray an object lies on, but not where it lies along that ray. Repeating the same feature at every depth creates a streak of plausible object locations. Choosing one estimated depth avoids the streak but commits to a potentially wrong range. CaDDN instead multiplies a semantic feature by the probability of each depth bin:

$$
G(u,v,d,c)=D(u,v,d)\,F(u,v,c).
$$

At one feature pixel, a length-$D$ probability vector and a length-$C$ feature vector form a $D\times C$ outer product. A confident depth estimate concentrates the feature near one bin; an ambiguous estimate spreads weaker evidence over several bins. The method preserves uncertainty about location without inventing a different semantic feature for every possible depth.

The architecture follows that distinction. ResNet-101 supplies quarter-resolution image features; one branch predicts depth probabilities through a DeepLabV3-style network, while another reduces the feature channels from 256 to 64 before the outer product. This reduction matters because every channel is replicated across the depth axis. The resulting frustum has image-plane coordinates and discrete depths, rather than equally sized cells in physical space.

![CaDDN source architecture showing depth probabilities weighting image features, frustum sampling into voxels, and a BEV detection head](/assets/images/caddn-source-figure-2-architecture.png)
*Fig 1: The depth branch controls where the image features enter the frustum volume. Camera calibration then connects that volume to metric voxels, and height-channel compression produces the BEV input to the detector. | source: [CaDDN, Figure 2](https://arxiv.org/abs/2103.01100)*

### Geometry changes the grid; supervision changes what survives the transformation

CaDDN fills a regular voxel grid by projecting each voxel center into the frustum and sampling its features with trilinear interpolation. This is a backward sampling operation: the metric grid asks which image location and depth bin can supply its feature. It is different from pushing every image feature forward and accumulating it wherever it lands.

Image-feature resolution limits the useful voxel resolution. If many voxels sample the same coarse feature neighborhood, a finer voxel grid merely repeats similar evidence. The feature-resolution ablation reflects this: quarter-resolution Block1 features reach 16.31 moderate-car AP, while the tested eighth-resolution features reach roughly 14.65–14.94. After sampling, CaDDN concatenates height slices into channels and learns a $1\times1$ projection, then applies a PointPillars-derived BEV backbone and detection head. Height is compressed through learned channel mixing rather than simply discarded.

The visual comparison below shows why explicit depth supervision matters. The unsupervised representation smears evidence along rays; the supervised representation localizes stronger responses near the visible vehicles. This figure is qualitative evidence for sharper geometry. The component ablation measures how much each training choice helps.

![Input image above unsupervised and depth-supervised CaDDN BEV features, showing smeared rays becoming localized object evidence](/assets/images/caddn-categorical-depth-for-monocular-3d-detection-source-figure-1.webp)
*Fig 2: The lower-left BEV map spreads similar features along viewing rays. The lower-right map, trained with depth-distribution supervision, concentrates responses near object locations. | source: [CaDDN, Figure 1](https://arxiv.org/abs/2103.01100)*

| KITTI validation component ablation | Moderate-car 3D AP, IoU 0.7 |
| --- | ---: |
| Repeat features along depth | 5.66 |
| Predict depth distributions without depth labels | 6.43 |
| Add explicit depth-distribution supervision | 14.03 |
| Weight foreground depth pixels more heavily | 15.10 |
| Use linearly increasing depth-bin widths | 16.31 |

These are cumulative steps from Table 3, not independent effect estimates. The largest step comes from supervising where the feature belongs. Foreground weighting then directs more of that learning toward detection: the focal-loss weight is 3.25 inside annotated 2D object boxes and 0.25 outside. Those boxes identify foreground for weighting; they are not pixel-accurate object masks.

The depth targets also deserve care. The paper projects LiDAR into the image, runs depth completion, downsamples to the feature resolution, and converts the resulting depths to one-hot bin labels. Thus the supervision is denser than the original projected returns but is not a dense direct range measurement. An additional out-of-range category participates in depth training and is removed before frustum construction. Linearly increasing bin widths allocate finer depth resolution nearby and coarser resolution farther away; the appendix compares this with uniform and logarithmically increasing spacing.

### Sharp training targets and uncertain predictions are compatible

One-hot labels encourage a sharp distribution at the correct depth, yet uncertain predictions remain useful. Table 4 separates that benefit from end-to-end training. With CaDDN depth estimates and hard bin selection, separate depth/detection training reaches 12.26 moderate-car AP; joint training reaches 13.71. Keeping the full distribution instead of its argmax reaches 16.31. Learning task-relevant depth and retaining ambiguity are two distinct improvements.

The entropy plot shows where ambiguity remains. Entropy is lowest around six meters and generally rises farther away. It also rises at very short distances, which the authors attribute to fewer training pixels in that range. Foreground estimates have slightly higher entropy than background estimates despite their larger loss weight. Uncertainty therefore follows both geometric difficulty and the training distribution; this plot does not establish calibrated probabilities for every depth or object.

![CaDDN depth-distribution entropy against ground-truth depth, with foreground and background curves and confidence bands](/assets/images/caddn-source-figure-6-depth-entropy.png)
*Fig 3: Mean entropy reaches a minimum near six meters and generally increases with distance. Foreground and background curves differ, and the shaded bands show 95% confidence intervals within ground-truth depth bins. | source: [CaDDN, Figure 6](https://arxiv.org/abs/2103.01100)*

The benchmark boundary is just as important as the mechanism. KITTI test moderate-car AP is 13.41 for 3D boxes and 18.91 for BEV boxes; these measure different overlap conditions and should not be mixed with the validation ablations. On the front-camera Waymo evaluation, vehicle 3D mAP at LEVEL_1 and IoU 0.7 is 5.03 overall, with 14.54 within 30 meters and 1.47 at 30–50 meters. The representation substantially improves the paper's monocular baseline, while absolute long-range detection remains difficult.

## High-Level Takeaways

- The depth distribution is a feature-placement rule: it controls how much semantic evidence is assigned to each possible location along a camera ray.
- Explicit depth supervision creates the largest gain in the component ablation, while preserving the full distribution adds a separate benefit over hard depth selection.
- Joint training and foreground weighting make depth useful for detection rather than optimizing all pixels equally. Completed LiDAR targets remain an imperfect supervision source.
- Voxel resolution cannot recover detail missing from the image features, and distributional lifting still carries a large intermediate-volume cost.
- Entropy reveals where the model remains uncertain, but the weak long-range Waymo result prevents treating that uncertainty representation as a solution to monocular ambiguity.
