---
title: 'SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos'
date: '2023-08-18T04:00:00.000Z'
section: paper-shorts
postSlug: sparsebev-high-performance-sparse-3d-object-detection
legacyPath: /paper shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos'
---
## 2023 – SparseBEV

**arXiv:** [2308.09244](https://arxiv.org/abs/2308.09244)

**Code:** [MCG-NJU/SparseBEV](https://github.com/MCG-NJU/SparseBEV)

**Summary:** SparseBEV keeps the BEV prior without building a dense BEV map. Pillar-shaped queries interact through scale-adaptive self-attention, predict several spatiotemporal sample points, retrieve multiscale camera features at those points, and decode them with query-conditioned channel and point mixing.

The paper argues that sparse camera detectors lagged dense ones because their retrieval and mixing were too rigid, not because sparsity itself was the wrong representation.

## Paper Insights

Scale-adaptive attention gives each query head a learned BEV receptive field. Large classes learn broader neighborhoods. Adaptive sampling predicts support around the query, projects it across eight timestamps and visible cameras, and weights feature scales. Dynamic mixing lets the query decide how sampled points and channels should interact.

| Component | mAP | NDS | Contribution |
| --- | ---: | ---: | --- |
| Baseline without mixing | 38.6 | 49.1 | Sparse sampled features with fixed aggregation. |
| Static channel→point mixing | 41.9 | 51.8 | Better decoding without query-conditioned weights. |
| Adaptive channel→point mixing | 45.4 | 55.6 | Query-specific decoding of support and semantics. |

Pillar queries add 1.4 mAP over 3D points. Scale-adaptive attention adds 4.0 mAP over vanilla query self-attention. Aligning ego motion raises NDS by about ten points; adding a constant-velocity object correction raises it further to 55.6. The ResNet-50 configuration reports 44.8 mAP and 55.8 NDS at 23.5 FPS. The paper's 67.5 NDS test result uses future frames and should not be compared with causal online systems.

## Decision Lens

SparseBEV informs whether a camera detector can retain metric BEV structure as a query prior instead of a dense field. Its atomic unit is a BEV pillar query with adaptive spatial extent. Capacity scales with queries, sample points, feature levels, and timestamps rather than the full metric area.

The rejection test compares SparseBEV with a dense BEV model under causal frames, the same backbone, and corrupted ego pose. SparseBEV loses if pose noise destroys temporal support, if query initialization misses weak actors, or if its nominally sparse gathers compile poorly. The paper also notes that latency still grows linearly with frames because sampled features are stacked across time.

**Context:** DETR3D samples one projected query point; Sparse4D samples structured 3D keypoints; SparseBEV adds adaptive BEV receptive fields and query-conditioned mixing. Each expands sparse support without materializing the full grid.

**Takeaway:** Sparse camera detection becomes competitive when the query can adapt where it looks, how much context it uses, and how it mixes the evidence.
