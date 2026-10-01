---
title: "MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping"
date: '2024-03-23T00:00:00.000Z'
section: paper-shorts
postSlug: maptracker-tracking-with-strided-memory-fusion-for-consistent-vector-hd-mapping
legacyPath: /paper shorts/2024/03/23/maptracker-tracking-with-strided-memory-fusion-for-consistent-vector-hd-mapping.html
tags: [Autonomous Driving, Mapping]
topics: [autonomy]
field: Mapping
summary: "2024 \u2013 MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping"
---

# 2024 – MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping

**Paper:** [2403.15951](https://arxiv.org/abs/2403.15951)

**Project:** [MapTracker](https://map-tracker.github.io/) · **Code:** [official implementation](https://github.com/woodfrog/maptracker)

## Summary

> MapTracker turns online vector mapping into a tracking problem with separate memories for BEV features and individual road elements. Distance-strided retrieval selects useful historical viewpoints rather than only the latest frames. On its temporally corrected nuScenes labels, the full model reaches 76.1 mAP and 69.1 consistency-aware mAP, compared with 70.4 and 56.4 for reproduced StreamMapNet. The gain costs runtime, and the method does not yet handle map-element splits and merges correctly.

## Core Insights

### A stable map needs stable element identity

[StreamMapNet](/paper%20shorts/2023/08/24/streammapnet-streaming-mapping-network-for-vectorized-online-hd-map-construction.html) conditions detection on history, but independently reassigned detections can still change identity or geometry across frames. MapTracker propagates each tracked element's query and supervises it against the same ground-truth track. Hungarian matching handles newly appearing elements. This changes the training target from a sequence of independent sets to corresponding road elements over time.

The figure shows two complementary memories. BEV memory stores a spatial field around the vehicle; vector memory stores the latent representation of each crossing, divider, or boundary. Geometry is decoded from those vector latents.

![MapTracker source Figure 2 shows raster and vector memory buffers with distance-strided fusion](/assets/images/maptracker-source-figure.png)
*Fig 1: BEV memory preserves spatial evidence while vector memory preserves element identity. Selected historical entries are motion-aligned before they refine the current field and its tracked instances. | source: [MapTracker, Figure 2; figure crop](https://arxiv.org/abs/2403.15951)*

Each buffer holds the last 20 frames. The default retrieval chooses up to four distinct states whose vehicle positions are nearest to 1, 5, 10, and 15 meters from the current position. Dense features are geometrically warped and convolutionally fused; vector latents use a pose-conditioned MLP and per-instance cross-attention. Recent history stabilizes the current estimate, while separated viewpoints reduce redundancy.

Training uses five-frame clips sampled from the current frame and four of the previous ten. It first pretrains image/BEV features with segmentation, warms the vector decoder, and then trains jointly. Focal and Dice raster losses, track-aware classification and curve regression, and transformation supervision support the two memories. Pose perturbation during training exposes the vector branch to alignment error.

### Consistency must exist in the labels before it can be scored

The benchmark contribution repairs inconsistent crossing and divider preprocessing, then establishes ground-truth tracks by motion alignment, raster overlap, adjacent-frame matching, and chaining. These are algorithmically constructed correspondences over map annotations, not independently hand-labeled identities. Better labels improve competing methods too, so scores must identify which ground-truth processing they use.

Consistency-aware mAP first matches geometry per frame, then removes matches inconsistent with earlier ancestors of the same track. Baselines without tracking receive tracks from a matching procedure. This tests temporal correspondence as well as geometry, but the resulting score also depends on that track construction.

| nuScenes, corrected labels | mAP | C-mAP |
| --- | ---: | ---: |
| Reproduced StreamMapNet | 70.4 | 56.4 |
| Tracking, no history-buffer fusion | 70.8 | 62.4 |
| Tracking plus recent-frame fusion | 74.9 | 68.1 |
| Tracking plus distance-strided fusion | 76.1 | 69.1 |

The tracking formulation moves consistency more than ordinary AP; buffered fusion improves both. On the reduced-overlap nuScenes split at 60 × 30 m, MapTracker reaches 40.3/32.5 mAP/C-mAP versus 33.5/22.2 for StreamMapNet. The absolute drop relative to familiar geography remains substantial.

### Memory spacing has a failure boundary

Changing strides to 5, 10, 15, and 20 meters lowers C-mAP to 51.5 from 69.1. More distant views lose the recent anchor and differ from the training setup. On one RTX 6000 at batch size one, MapTracker runs at 11.5 FPS versus StreamMapNet's 14.2; per-instance loops in memory fusion contribute implementation overhead.

A U-shaped boundary can split into two visible fragments as the crop moves. The paper explicitly says its tracking and labels do not handle such splits and merges properly. Stable IDs are therefore useful but insufficient for persistent map maintenance. My next test would evaluate identity transitions, pose errors, and actual changes separately before adding more memory.

## High-Level Takeaways

- Tracking supervision preserves correspondence that history-conditioned detection alone does not guarantee.
- Spatial and instance memories justify separate representations and separate audits.
- Corrected annotations and consistency-aware metrics are part of the reported result, not neutral preprocessing.
- Distance-strided history needs a recent anchor; larger gaps can substantially damage consistency.
- Split/merge identity and permanent change remain outside the demonstrated maintenance capability.
