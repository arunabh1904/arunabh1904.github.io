---
title: 'DSVT: Dynamic Sparse Voxel Transformer'
date: '2023-01-15T05:00:00.000Z'
section: paper-shorts
postSlug: dsvt-dynamic-sparse-voxel-transformer
legacyPath: /paper shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – DSVT: bounded local attention over variable-density sparse voxels'
---
## 2023 – DSVT

**arXiv:** [2301.06051](https://arxiv.org/abs/2301.06051)

**Code:** [Haiyang-W/DSVT](https://github.com/Haiyang-W/DSVT)

### Method and reported result

DSVT partitions the variable number of occupied voxels in each window into fixed-size local sets, applies attention within those sets, and changes the partition orientation between blocks. It also introduces an attention-style 3D pooling stage. The result is a sparse transformer that can be deployed through standard tensor operations rather than custom sparse-convolution kernels.

## Summary

> Dynamic Sparse Window Attention addresses a practical mismatch: fixed geometric windows contain wildly different token counts in near and far regions, while accelerators prefer bounded regular workloads.

## Core Insights

Rotated set partitioning lets tokens exchange information beyond one grouping without paying global attention cost. The paper applies the same backbone to voxel and pillar forms and reports a TensorRT implementation at 27 Hz, making deployment path part of the contribution.

![DSVT: Dynamic Sparse Voxel Transformer source figure: Top : An illustration of the Dynamic Sparse Voxel Transformer block, including one X-Axis DSVT Layer and one Y-Axis DSVT Layer with different set…](/assets/images/dsvt-dynamic-sparse-voxel-transformer-paper-figure.webp)
*Fig 1: Top: An illustration of the Dynamic Sparse Voxel Transformer block, including one X-Axis DSVT Layer and one Y-Axis DSVT Layer with different set…. | source: [DSVT: Dynamic Sparse Voxel Transformer](https://arxiv.org/abs/2301.06051)*

![Figure 1 from DSVT: Dynamic Sparse Voxel Transformer](/assets/images/dsvt-dynamic-sparse-voxel-transformer-source-figure-1.webp)
*Fig 2: Detection performance (mAPH/L2) vs speed (Hz) of different methods on Waymo PeiSun2020ScalabilityIP validation set. All the speeds are evaluated on an NVIDIA A100 GPU with AMD EPYC 7513 CPU. | source: [DSVT: Dynamic Sparse Voxel Transformer](https://arxiv.org/abs/2301.06051)*

![Figure 2 from DSVT: Dynamic Sparse Voxel Transformer](/assets/images/dsvt-dynamic-sparse-voxel-transformer-source-figure-2.webp)
*Fig 3: A demonstration of dynamic sparse window attention in our DSVT block. In the X-Axis DSVT layer, the sparse voxels will be split into a series of window-bounded and size-equivalent subsets in X-Axis main order, and self-attention is computed within each set. | source: [DSVT: Dynamic Sparse Voxel Transformer](https://arxiv.org/abs/2301.06051)*


| Design | Purpose | Limit |
| --- | --- | --- |
| Fixed-size sparse sets | Bound attention work | Requires sorting and padding logic. |
| Rotated partitions | Cross-set exchange | Context still grows layer by layer. |
| 3D pooling | Learn height compression | Pooling remains irreversible. |
| Reported pillar result | 71.14 mAP / 68.59 mAPH | Benchmark-specific, not a universal speed claim. |

## High-Level Takeaways

- DSVT is a strong candidate when custom sparse operators complicate export or when active-voxel density makes submanifold convolutions context-limited. The decisive profiling unit is active tokens per set across real scenes, including worst-case crowding, not average FLOPs.
- For unified sensing, DSVT provides a reusable interaction primitive only after each modality has produced meaningful tokens. It does not imply that camera patches and LiDAR voxels should share their tokenizer.
- SST establishes single-stride sparse attention; DSVT regularizes the workload and later informs UniTR's shared multimodal transformer blocks.
- Efficient sparsity is an execution contract as well as a mathematical one: bound the active set and use operators the deployment stack can actually accelerate.
