---
title: 'MGMap: Mask-Guided Learning for Online Vectorized HD Map Construction'
date: '2024-04-01T00:00:00.000Z'
section: paper-shorts
postSlug: mgmap-mask-guided-learning-for-online-vectorized-hd-map-construction
legacyPath: /paper shorts/2024/04/01/mgmap-mask-guided-learning-for-online-vectorized-hd-map-construction.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2024 – MGMap: Mask-Guided Learning for Online Vectorized HD Map Construction"
---

**arXiv:** [2404.00876](https://arxiv.org/abs/2404.00876) · **Code:** [xiaolul2/MGMap](https://github.com/xiaolul2/MGMap)

## Summary

> MGMap gives sparse vector-map queries a denser view of the geometry they are trying to recover. Learned instance masks initialize queries with a whole map element's shape, while local patches of a binary mask help refine individual points. The strongest evidence is the controlled progression from a MapTR baseline through enhanced BEV features, auxiliary segmentation, and actual mask-guided decoding. Camera-only nuScenes mAP rises from 51.1 to 61.4 under the matched 30-epoch setting, at a measurable runtime cost.

## Core Insights

### A query needs to find the bend before it can represent it

A road boundary is thin relative to the surrounding image and BEV grid. Sparse deformable-attention samples can miss the few cells that distinguish a straight segment from a sharp bend. Regressing the right number of points cannot recover that detail if every point query reads mostly background.

MGMap first builds enhanced multi-scale BEV features with a pyramid of residual blocks and channel/spatial attention. It then learns an instance mask for each candidate map element. Multiplying each mask by the BEV features aggregates the selected region into an instance query. Shared learned point embeddings are added to this query, giving every point both an instance identity and its own position in the predicted sequence.

![MGMap's BEV feature pyramid, mask-activated queries, and local patch refinement](/assets/images/mgmap-mask-guided-learning-for-online-vectorized-hd-map-construction-paper-figure.png)
*Fig 1: The middle panel uses a whole-instance mask to initialize shape-aware queries. The right panel returns to neighborhoods around predicted points, allowing local geometry to correct a broadly plausible shape. | source: [MGMap, Figure 2](https://arxiv.org/abs/2404.00876)*

The instance masks receive bipartite-matched raster supervision during training. A deformable transformer still performs the main coordinate decoding; masks do not replace the vector output with a segmentation map. They change the evidence carried into that decoder.

### Local mask patches provide a second chance to locate a boundary

The refinement branch predicts a two-channel binary mask, expands it into features, and combines it with BEV features and normalized coordinate channels. Around each decoded point, ROIAlign extracts a 5×5 patch representation. The point query attends to this local evidence and predicts a coordinate correction. The selected patch size is 0.1 in normalized coordinates, with two refinement stages.

This creates a useful hierarchy of questions. The instance mask asks which pixels belong to the same map element. The point patch asks where the boundary actually passes near the current estimate. A patch that is too small can miss a displaced boundary; a large one mixes competing structures. The ablation reflects that trade: patch sizes 0.08, 0.10, and 0.12 with two stages give 60.8, 61.4, and 60.1 mAP. A third refinement stage also falls to 60.6.

![A mask preserves a local road-boundary bend that a sparse decoder smooths away](/assets/images/mgmap-mask-guided-learning-for-online-vectorized-hd-map-construction-source-figure-1.webp)
*Fig 2: Compare the highlighted green boundary with ground truth: the mask-guided version preserves the local change in direction. The example connects a missed image detail to a specific error in the vector map. | source: [MGMap, Figure 1](https://arxiv.org/abs/2404.00876)*

The standard output uses at most 50 instances with 20 points per instance on a 200×100 BEV grid. The paper evaluates dividers, road boundaries, and pedestrian crossings. It does not predict directed lane connectivity merely by producing these vectors.

### The ablation separates better features from better use of masks

At 30 epochs with a ResNet-50 camera backbone, the main comparison improves Chamfer-based mAP from MapTR's 51.1 to 61.4. Raster-based mAP also rises from 25.6 to 37.2, so the gain survives a metric sensitive to the shape of the entire rasterized element. Chamfer AP averages 0.5, 1.0, and 1.5 m matching thresholds; it should not be read as centimeter-level map accuracy.

| Controlled variant | nuScenes camera mAP |
| --- | ---: |
| MapTR baseline | 51.1 |
| Enhanced multi-level BEV features | 57.6 |
| BEV features plus parallel segmentation | 59.3 |
| Instance and point mask guidance | 61.4 |

The feature neck supplies much of the total gain. Auxiliary segmentation explains another part, while actively using the masks adds 2.1 points beyond that supervision-only control. Within the mask ablation, instance guidance alone gives 59.5 and point guidance alone 60.2; their combination reaches 61.4. The paper therefore tests the distinction that matters: learning a mask and using its features are different interventions.

The method also transfers to a stronger MapTRv2-based implementation, 61.5→64.8 mAP at 24 epochs, and an Argoverse2 challenge subset, 57.4→62.8. Separate LiDAR and camera–LiDAR configurations reach 67.9 and 71.7 mAP. Those are different sensor budgets, not additional camera-only gains.

### Extra localization work has a cost and an evidence limit

The matched runtime breakdown grows from 64.7 ms for MapTR to 84.3 ms: the neck adds 6.9 ms and refinement 12.7 ms. The main table reports 15.7 versus 11.6 FPS on a V100. Mask guidance improves geometry while spending more time on it.

Natural-condition subsets also show why aggregate robustness claims need care. In the separate 24-epoch camera experiment, sunny mAP improves 53.5→65.2, but nighttime improves only 35.7→37.6. Dense guidance cannot manufacture useful visual evidence under severe darkness or occlusion. These within-dataset slices support qualified robustness, while temporal integration, new geographies, and downstream navigation remain separate tests.

## High-Level Takeaways

- Whole-instance masks and point-centered patches solve different localization problems: identifying a shape and recovering its local detail.
- The segmentation-only control matters. The final gain combines stronger BEV features, dense supervision, and active mask-guided decoding.
- Report shape metrics and latency together; improved vector accuracy requires additional feature extraction and refinement.
- Masks can direct attention toward evidence, but cannot guarantee a correct map when that evidence is absent.
