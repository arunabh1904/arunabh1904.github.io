---
title: 'Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving'
date: '2023-04-27T00:00:00.000Z'
section: paper-shorts
postSlug: occ3d-large-scale-3d-occupancy-prediction-benchmark
legacyPath: /paper shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – Occ3D: visibility-aware dense 3D occupancy benchmarks'
---

**arXiv:** [2304.14365](https://arxiv.org/abs/2304.14365)

**Project and data:** [Occ3D](https://tsinghua-mars-lab.github.io/Occ3D/)

**Code:** [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

## Summary

> Occ3D makes occupancy prediction a measurable camera task by releasing dense, visibility-aware voxel labels. Its most important design choice is epistemic: an unobserved voxel is kept as unobserved instead of being silently counted as free or as a hallucinated completion.
>
> The benchmark is also a reconstruction pipeline. Multi-frame LiDAR, object motion, ray visibility, mesh filling, and camera semantics all shape the target before a model sees it. That makes Occ3D useful for evaluating models while keeping the provenance of each voxel in view.

## Core Insights

### Occupancy expands the question beyond boxes

A 3D detector answers where a known object is with a compact box. Occ3D asks for the state of every voxel: occupied with a semantic class, free, or unobserved. The representation can retain an irregular construction vehicle arm and can place out-of-vocabulary objects into a General Object (GO) class. It is richer than a box ontology, but it is not the same as semantic scene completion: the authors explicitly do not require a model to infer invisible regions.

The two releases make the scale concrete. The first row is the native Occ3D-nuScenes label grid; the second is the much finer Occ3D-Waymo label grid published by the benchmark:

| Benchmark | Split and data | Classes | Voxel volume and size |
| --- | --- | ---: | --- |
| Occ3D-nuScenes | 600 train / 150 val / 150 test scenes; 40,000 frames | 16 + GO | 200 x 200 x 16; 0.4 m |
| Occ3D-Waymo | 798 train / 202 val sequences; 200,000 frames | 14 + GO | 3200 x 3200 x 128; 0.05 m |

![Occ3D's semi-automatic label generation pipeline](/assets/images/occ3d-large-scale-3d-occupancy-prediction-benchmark-source-figure-2.webp)
*Fig 1: Occ3D densifies LiDAR, assigns labels, reconstructs surfaces, reasons about occlusion, and applies image-guided refinement before producing occupancy labels. | source: [Occ3D, Figure 2](https://arxiv.org/abs/2304.14365)*

### The label is a chain of geometric decisions

The pipeline first separates dynamic objects from static scene points. It aggregates dynamic points in object coordinates, aggregates static points in global coordinates, uses KNN to label otherwise unannotated frames, and fills remaining surface holes with mesh reconstruction. Naively accumulating all points would smear a moving car across time; separating the coordinate systems is what keeps density from becoming motion blur.

Visibility reasoning then casts rays from LiDAR origins. A voxel that reflects a return is occupied; a voxel traversed by a ray is free; a voxel reached by neither is unobserved. Camera visibility is a second mask: evaluation is performed only where both the LiDAR and camera views observe the voxel. This is a useful contract for comparing camera models because it avoids rewarding guesses in regions the input could not see.

![Occ3D's visibility masks and image-guided refinement](/assets/images/occ3d-large-scale-3d-occupancy-prediction-benchmark-source-figure-3.webp)
*Fig 2: LiDAR and camera rays distinguish occupied, free, and unobserved voxels; image-guided refinement clears voxels before the first depth with a matching pixel label. | source: [Occ3D, Figure 3](https://arxiv.org/abs/2304.14365)*

The last refinement step addresses a subtle failure: pose drift and LiDAR noise can make an object's reconstructed surface too thick. Along a ray from a camera pixel, the first voxel with the same semantic label is retained and previously traversed occupied voxels are marked free. The 3D-2D consistency check in the paper shows why this is more than cosmetic; multi-frame aggregation improves recall but can lower precision, while voxel size, mesh reconstruction, and refinement trade those errors in different directions.

### CTF-Occ spends computation where geometry is uncertain

Occ3D also introduces CTF-Occ. Its pyramid voxel encoder predicts whether a voxel is empty, selects foreground or uncertain tokens, refines only the top-k candidates with spatial cross-attention, and upsamples between levels. An implicit MLP decoder can query a semantic label at an arbitrary coordinate. In the matched Occ3D-nuScenes table, CTF-Occ reaches 28.53 mIoU versus BEVFormer's 26.88, a 1.65 point gain. On Occ3D-Waymo it reaches 18.73 versus 16.76, a 1.97 point gain; the paper's corresponding vehicle comparison is 28.09 IoU versus TPVFormer's 17.86, a 10.23 point gain.

The label grid and the model evaluation grid should not be conflated. Occ3D-Waymo publishes 5 cm labels, but the paper's CTF-Occ experiments use a 0.4 m voxel size on both datasets. The 18.73 mIoU result therefore evaluates the benchmark at that coarser model grid; it is not a 5 cm-resolution camera prediction score.

The ablation explains the source of that gain: on Waymo, the combination of OHEM and top-k token selection reaches 18.43 mIoU, whereas removing both targeted selection and hard-example weighting gives 14.06. The model is not simply “using a finer grid”; it is reallocating cross-attention toward occupied and ambiguous regions in a space dominated by empty voxels.

## High-Level Takeaways

- Occ3D's central contribution is a visibility-aware evaluation contract, not only a larger label file.
- Separating moving objects before temporal aggregation prevents density from turning into geometric smear.
- General Objects and irregular geometry expose failures that fixed box categories hide, while unobserved labels prevent unsupported completion from looking correct.
- CTF-Occ's coarse-to-fine token selection is a useful compute pattern for sparse 3D space; its gains should still be read alongside the benchmark's calibration, pose, deformable-object, and auto-labeling limits.
