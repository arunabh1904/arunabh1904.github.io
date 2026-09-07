---
title: "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
date: '2022-03-31T00:00:00.000Z'
section: paper-shorts
postSlug: bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers
legacyPath: /paper shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2022 – BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
---
## 2022 – BEVFormer

**arXiv:** [2203.17270](https://arxiv.org/abs/2203.17270)

**Code:** [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

## Summary

> BEVFormer turns six camera streams into a persistent metric workspace by making every BEV cell a learned query. Each query first reads ego-motion-aligned history, then samples a small set of image features at projected 3D heights. On the paper's nuScenes test setup it reaches 56.9 NDS and 0.378 m/s mean velocity error. The contribution is a reusable spatial state with a measured accuracy/compute trade-off; the paper still identifies camera depth and dense BEV cost as limitations.

## Core Insights

### A BEV cell is a geometric address

The model starts with a learnable query grid $Q\in\mathbb{R}^{H\times W\times C}$. A query at grid position $(x,y)$ is mapped to a ground-plane location in the ego frame. The paper samples several predefined heights above that location, projects those 3D reference points through each camera's calibration matrix, and uses deformable attention to read only the corresponding regions of the image feature maps. This is the important distinction from a flat image token: the query already asks for evidence at a metric place, while the camera feature only answers what is visible there.

The architecture figure is easiest to read in three passes. The left side is the six-view image pyramid; the middle is the learned BEV grid; and the two arrows into each encoder layer are different evidence paths. Temporal self-attention first reads the prior BEV, then spatial cross-attention samples the current cameras, followed by a feed-forward update. The output remains a $H\times W$ feature field, so a detector can decode boxes and velocity while a map head decodes drivable area and lane classes from the same scene state.

![Figure 2 from BEVFormer showing the BEV encoder with spatial cross-attention and temporal self-attention](/assets/images/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers-paper-figure.png)
*Fig 1: The encoder first reads the previous BEV at each query, then samples projected image features around several 3D heights before producing the next BEV feature field. | source: [BEVFormer, Figure 2](https://arxiv.org/abs/2203.17270)*

### Memory helps because it is aligned before it is attended

A previous BEV cannot simply be concatenated with the current grid: the ego vehicle moved, so the same cell index no longer denotes the same world location. BEVFormer warps the previous feature map according to ego-motion and lets each current query attend to both its present query and nearby locations in the aligned history. The first frame falls back to ordinary self-attention because no prior BEV exists. This turns the recurrent state into a spatial memory rather than a raw frame cache.

The visibility experiment makes that mechanism concrete. The source figure divides nuScenes validation objects into 0–40%, 40–60%, 60–80%, and 80–100% visible subsets. Compare the vertical gaps between the temporal and single-frame curves: the largest recall improvement is in the most occluded group. Translation, orientation, and velocity errors also improve. Scale error changes little between the single-frame and temporal models; attribute error improves for partially visible objects but is worse in the most-visible group. History is supplying missing observations and motion cues; it is not making the camera image intrinsically better at estimating object size.

![Figure 3 from BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](/assets/images/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers-source-figure-3.webp)
*Fig 2: Temporal BEV features raise recall most in the 0–40% visibility group and reduce translation, orientation, and velocity errors. Scale changes little; the attribute gain is not uniform across visibility groups. | source: [BEVFormer, Figure 3](https://arxiv.org/abs/2203.17270)*

### The ablation separates receptive field, memory, and latency

The spatial cross-attention comparison is more informative than the headline score. Replacing deformable attention with global attention consumes too much GPU memory; restricting every query to its single geometrically projected reference point loses context around the object. The paper's learned offsets sample a local region around several projected heights, which is an interpretation of how the model can cover object extent and residual feature misalignment—not a claim that the calibrated reference point itself is inaccurate. The method's sparse sampling is therefore a bounded approximation to image-wide attention, not a free removal of geometric uncertainty.

Temporal memory explains the jump from the single-frame BEVFormer-S model to the full system. On the test set, BEVFormer reaches 0.378 m/s mAVE, a large improvement over earlier camera-only systems that were nearly unable to estimate velocity. The same temporal state also raises recall for occluded objects. But the model still has a dense feature map at every timestamp, and it is the camera backbone rather than only the BEV encoder that becomes the main efficiency bottleneck.

The scale table gives usable operating points. With one encoder layer and the full 200×200 BEV, configuration C reaches 50.1 NDS at 25 ms versus the default 51.7 NDS at 130 ms for the BEV encoder. Configuration D combines one layer, single-scale features, and a 100×100 BEV; its BEV encoder is 7 ms and reaches 47.8 NDS. The corresponding component rows are 391/130/19 ms for the default and 387/7/18 ms for D (backbone/BEV encoder/head). The paper reports 1.7 FPS and 2.3 FPS for those configurations; the component sums are useful for locating the bottleneck but should not be treated as measured end-to-end latency because runtime overhead is reported separately. These measurements use a V100 with a 900×1600 input and R101-DCN backbone, so they are a design curve for this setup rather than a portable product latency claim.

### What the qualitative result does and does not show

The final visualization pairs 3D boxes in the camera views with their BEV projections. Read the same object across views before looking at the top-down panel: the shared BEV lets detections remain consistent when a vehicle is near a camera boundary. The paper also calls out remaining mistakes on small and remote objects. That failure is consistent with the method's geometry: projected reference points can land near a weak or absent image feature, and a dense grid does not create pixels that the cameras never observed.

![Figure 4 from BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](/assets/images/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers-source-figure-4.webp)
*Fig 3: The same predicted 3D boxes are shown in multiple camera views and in the BEV workspace; the examples make cross-view consistency visible while retaining small and distant-object mistakes. | source: [BEVFormer, Figure 4](https://arxiv.org/abs/2203.17270)*


_BEV queries, spatial cross-attention, temporal self-attention, and detection/segmentation heads. source: [BEVFormer paper](https://arxiv.org/abs/2203.17270)


**What to look at:**
- BEV queries define the dense bird's-eye grid.
- Spatial cross-attention connects each BEV cell to camera evidence.
- Temporal self-attention carries history without recomputing a long video window.

## High-Level Takeaways

- A calibrated BEV query gives detection, velocity, and map heads one persistent spatial state; temporal self-attention reads aligned history before deformable image sampling.
- The occlusion gain comes from memory, while the spatial cross-attention gain comes from bounded multi-height sampling; neither removes camera depth ambiguity.
- The latency table separates the 130 ms versus 7 ms BEV encoder from the 391/19 ms and 387/18 ms backbone/head components; the reported throughput is 1.7 versus 2.3 FPS, with overhead outside those sums.
- The source's strongest qualitative and visibility evidence is improved cross-view consistency and recall for 0–40% visible objects, while small and remote objects remain failure cases.
