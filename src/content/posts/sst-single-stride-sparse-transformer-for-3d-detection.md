---
title: 'SST: Single-Stride Sparse Transformer for 3D Detection'
date: '2021-12-13T05:00:00.000Z'
section: paper-shorts
postSlug: sst-single-stride-sparse-transformer-for-3d-detection
legacyPath: /paper shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2021 – SST: preserve high-resolution sparse LiDAR features without a downsampling pyramid'
---
## 2021 – SST

**arXiv:** [2112.06375](https://arxiv.org/abs/2112.06375)

## Summary

> SST treats network stride as a representation choice rather than inheriting the image-detector pyramid. Waymo objects are tiny relative to the 150 m scene, so the model keeps one high-resolution sparse voxel grid and uses local attention to recover the receptive field that single-stride convolutions lose. Shifted regional grouping crosses window boundaries, while two dense convolutions repair empty object centers before the detection head. The resulting gain is clearest for pedestrians and strict localization thresholds.

## Core Insights

### The pilot study links stride to the object scale

Waymo frames cover roughly $150\text{ m}\times150\text{ m}$, while a vehicle is about 4 m long and a pedestrian can be about 1 m. The paper quantifies the mismatch with $S_{\mathrm{rel}}=\sqrt{A_o/A_s}$, where $A_o$ is the object area and $A_s$ is the scene area. Only 0.54% of Waymo objects exceed the 0.04 threshold, compared with 73.03% of COCO objects. A 2D-style downsampling pyramid therefore removes a larger fraction of the evidence for the objects that matter in driving.

![Relative object size in COCO and Waymo](/assets/images/sst-single-stride-sparse-transformer-for-3d-detection-source-figure-2.webp)
*Fig 1: The relative-size distribution is shifted sharply toward small objects in Waymo: 0.54% exceed $S_{\mathrm{rel}}=0.04$, versus 73.03% in COCO. | source: [SST, Figure 2](https://arxiv.org/abs/2112.06375)*

The authors first vary stride in a PointPillars pilot trained on 20% of Waymo. The four stage strides are D3: $\{1,2,4,8\}$, D2: $\{1,2,4,4\}$, D1: $\{1,2,2,2\}$, and D0: $\{1,1,1,1\}$. Performance improves from D3 to D1 for all classes. D0 helps cyclists and only slightly changes pedestrians, but vehicle performance falls because a fully single-stride convolution has too little receptive field for larger objects. D0 with dilation 2 in its last stages recovers vehicle performance while harming small-object detail; D0 with $5\times5$ kernels improves all classes but has the highest latency. The pilot establishes both parts of the design: preserving resolution helps, but attention must supply context.

### Sparse Regional Attention restores context without filling the grid

SST voxelizes the point cloud and treats each non-empty voxel as a token. Regional Grouping partitions the 3D space into fixed, non-overlapping regions; Sparse Regional Attention (SRA) attends only to tokens in the same region. To batch regions with different token counts, the method groups them by powers of two and pads each group to the next power, masking the added tokens. This uses standard tensor operations while keeping empty road cells out of the attention input.

A second SRA block shifts every region by half its size. A target near a region boundary therefore meets a different set of neighbors in the next block, much like shifted windows in 2D vision. The backbone preserves each token's spatial location instead of downsampling and later interpolating it. The receptive field grows through alternating local sets, while the cost scales with occupied tokens and the padded set capacity.

SST uses six SRA blocks with two attention modules per block, eight heads, 128 input channels, and 256 hidden channels. Its default region is $3.84\text{ m}\times3.84\text{ m}\times6\text{ m}$ and its BEV pillar size is $0.32\text{ m}\times0.32\text{ m}\times6\text{ m}$. The detection head is inherited from PointPillars, including smooth L1 localization, focal classification, and direction losses.

![SST architecture with regional and shifted regional attention](/assets/images/sst-single-stride-sparse-transformer-for-3d-detection-paper-figure.webp)
*Fig 2: SST keeps sparse tokens at one stride, alternates regional and shifted-regional SRA blocks, then recovers a dense feature map for the existing detection head. | source: [SST, Figure 4](https://arxiv.org/abs/2112.06375)*

The dense handoff has a subtle problem. LiDAR observes object surfaces, so the voxel at an object's center is often empty. A conventional head can therefore receive a zero feature exactly where it wants to place a box. SST adds two $3\times3$ convolutions before the head to fill most of these holes. This makes the final dense operation a learned recovery layer rather than evidence that the backbone itself has become dense.

### Pedestrian results expose the resolution benefit

On Waymo validation, the one-frame SST model reports 78.71 Level-1 AP and 70.02 Level-2 AP for pedestrians; the three-frame version reaches 82.42 and 75.14. The two-stage SST_TS_3f model reaches 83.81 Level-1 AP and 75.94 Level-2 AP. With the stricter pedestrian IoU threshold of 0.6, the best model reports 65.06 AP. These values are tied to the paper's official validation protocol and should not be mixed with the 20% data ablations.

The distance breakdown clarifies where the architecture helps. SST_1f gains 12.8 pedestrian AP over the PointPillars counterpart at 0–30 m, while its advantage at 50 m and beyond is small. Aligning and concatenating three frames raises the long-range SST result by 10.4 AP over SST_1f, because multiple sweeps provide the sparse tokens that attention can connect. For vehicles, recall also rises over PointPillars across length groups: 41.31/80.85/13.41 versus 40.60/73.11/10.59 for objects shorter than 4 m, 4–8 m, and longer than 8 m. The single-stride design does not only help small pedestrians; its context mechanism keeps large objects viable.

The alternative comparison uses 20% Waymo data and a 2080 Ti. PointPillars-SS reports 64.01 vehicle AP and 60.85 pedestrian AP at 60 ms; SparsePillars-SS falls to 51.57 and 61.55 at 67 ms because submanifold sparse convolution cannot propagate through empty gaps; SST reaches 67.86 and 70.94 at 97 ms with 1.6M parameters and 6.8 GB reported memory. The comparison isolates why attention is needed in the single-stride setting: keeping tokens high-resolution without a mechanism for crossing empty space is not enough.

### The cost is memory and unresolved sparse resampling

SST's authors report slightly higher memory than baseline models and leave memory-efficient variants for future work. They also identify sparse downsampling and upsampling as open problems: merging a variable number of tokens into a smaller set and recovering their locations later both require choices about what information to preserve. SST avoids those choices by keeping one stride, which makes the model's resolution advantage clear but also limits the range of architectures it can represent.

## High-Level Takeaways

- Waymo's object-size distribution gives a concrete reason to question image-style downsampling in LiDAR detectors.
- Single stride improves resolution only when sparse attention supplies a larger receptive field; D0 convolutions alone lose vehicle context.
- Regional and shifted SRA preserve occupied-token resolution and cross window boundaries without global attention.
- The strongest reported result is 83.81 Level-1 pedestrian AP for SST_TS_3f, with gains concentrated at short range and strengthened by aligned multi-frame input.
- The design pays in memory and avoids, rather than solves, the general problem of sparse downsampling and upsampling.
