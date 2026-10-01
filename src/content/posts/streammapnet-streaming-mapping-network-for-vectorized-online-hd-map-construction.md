---
title: "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
date: '2023-08-24T00:00:00.000Z'
section: paper-shorts
postSlug: streammapnet-streaming-mapping-network-for-vectorized-online-hd-map-construction
legacyPath: /paper shorts/2023/08/24/streammapnet-streaming-mapping-network-for-vectorized-online-hd-map-construction.html
tags: [Autonomous Driving, Mapping]
topics: [autonomy]
field: Mapping
summary: "2023 \u2013 StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
---

# 2023 – StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction

**Paper:** [2308.12570](https://arxiv.org/abs/2308.12570)

**Code:** [official implementation](https://github.com/yuantianyuan01/StreamMapNet)

## Summary

> StreamMapNet carries dense BEV features and sparse map queries across frames, while multi-point attention gathers evidence along each elongated map element. On the paper's new nuScenes split at 60 × 30 m, it reaches 33.9 mAP versus 20.9 for MapTR. Its equally consequential contribution is the geographic evaluation audit: the same method reaches 62.9 on the original split. Memory improves observation reuse, but the split determines whether the model is also being rewarded for familiar geography.

## Core Insights

### A long boundary needs more than one attention center

A single reference point works naturally for a compact object. A road boundary may curve across much of the local map. StreamMapNet assigns one query to an entire instance, predicts its sampled points, and uses those points as distributed references for deformable cross-attention. Each decoder layer retrieves evidence along the current shape and predicts absolute coordinates. A control with conventional center-based attention fails to converge in the reported long-range ablation.

The architecture figure separates spatial retrieval from temporal reuse. Follow the memory buffer into both the BEV field and the decoder: the former preserves regional evidence, while the latter retains candidate elements.

![StreamMapNet source Figure 2 shows image features, BEV, decoder queries, and a recurrent memory buffer](/assets/images/streammapnet-source-figure.png)
*Fig 1: The memory buffer supplies both BEV features and refined map queries. Their separate paths retain spatial context and candidate geometry while the current images provide new observations. | source: [StreamMapNet, Figure 2; figure crop](https://arxiv.org/abs/2308.12570)*

The previous BEV is warped using ego motion and fused through a GRU followed by layer normalization. High-confidence queries are propagated through a pose-conditioned residual MLP; their reference polylines are transformed geometrically. Fresh queries remain available for newly visible elements. An auxiliary transformation loss asks the propagated latent to decode the correctly transformed geometry rather than merely preserve a useful class embedding.

Compared with [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html), the change is therefore not just a larger input window. The model processes one current frame while a recurrent state summarizes history. Training stops gradients through the previous frame and begins with four single-frame epochs before streaming. The main schedules use 24 nuScenes epochs and 30 Argoverse 2 epochs, with eight RTX 3090 GPUs and ordinary classification, curve, and transformation losses.

### Memory and image resolution contribute separate gains

The cumulative ablation uses the new Argoverse 2 split and a 100 × 50 m region. At 384 × 384 image resolution, direct coordinate prediction reaches 41.7 mAP. Query propagation raises it to 42.8; transformation supervision to 43.7; BEV fusion to 46.1. Raising image resolution to 608 × 608 reaches 51.2. The full improvement cannot be assigned to temporal memory alone.

The new nuScenes split yields 33.9 mAP at 60 × 30 m and 23.0 at 100 × 50 m. Corresponding MapTR values are 20.9 and 14.8. The paper reports 13.2 FPS for these nuScenes configurations; its 14.2-FPS headline belongs to a different setting. Range, resolution, and dataset must remain attached to the result.

### Geographic separation is part of the benchmark

The appendix estimates that over 85% of original nuScenes validation locations overlap training coverage, falling to roughly 11% under the adopted replacement split. Argoverse 2's original overlap is approximately 54%, and its new split removes overlap. Calling both new splits perfectly geographically disjoint would overstate the nuScenes construction.

The original nuScenes result of 62.9 mAP and the new-split result of 33.9 reveal how strongly map prediction depends on geography. They are models trained under their respective splits, not a single checkpoint evaluated twice. The method predicts dividers, boundaries, and crossings; it does not establish lane topology or permanent change detection. [MapTracker](/paper%20shorts/2024/03/23/maptracker-tracking-with-strided-memory-fusion-for-consistent-vector-hd-mapping.html) subsequently makes persistent element identity and consistency-aware evaluation explicit.

## High-Level Takeaways

- Distributed attention matches a long map element better than one object-centered reference under the reported ablation.
- BEV recurrence and query propagation preserve different information; transformation supervision makes the latter geometrically meaningful.
- Keep image resolution, range, and geographic split fixed before attributing a gain to memory.
- A useful follow-up would test pose corruption and real structural changes, where preserving history can preserve an error.
