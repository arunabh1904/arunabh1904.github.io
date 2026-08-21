---
title: 'OccAny: Generalized Unconstrained Urban 3D Occupancy'
date: '2026-03-24T00:00:00.000Z'
section: paper-shorts
postSlug: occany-generalized-unconstrained-urban-3d-occupancy
legacyPath: /paper shorts/2026/03/24/occany-generalized-unconstrained-urban-3d-occupancy.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – OccAny: metric occupancy from out-of-domain, uncalibrated urban images'
---
## 2026 – OccAny

**arXiv:** [2603.23502](https://arxiv.org/abs/2603.23502)

**Project:** [OccAny](https://valeoai.github.io/OccAny/)

**Code:** [valeoai/OccAny](https://github.com/valeoai/OccAny)

## Summary

> OccAny trades target-rig specialization for geometric generalization. One model accepts monocular, sequential, or surround-view images without target camera calibration and predicts metric occupancy in unseen datasets, but its semantic mIoU remains low and its strongest completion gain depends on extra rendered views at inference.

The model is not supervision-free. It learns metric pointmaps from projected LiDAR across five source datasets—Waymo, DDAD, PandaSet, Virtual KITTI 2, and ONCE—then evaluates zero-shot on SemanticKITTI and Occ3D-nuScenes. The contribution is that the target dataset supplies neither training samples nor known intrinsics and extrinsics, not that metric geometry appears without metric source supervision.

## Core Insights

OccAny extends a multi-view geometry model with a scene memory and two occupancy-specific constraints. **Segmentation Forcing** distills SAM2-like features alongside pointmaps so object and surface boundaries regularize sparse geometric supervision. **Novel-View Rendering** freezes the reconstruction network, projects its geometry into sampled camera poses, and trains a second encoder-decoder to complete missing pointmaps. At inference, test-time view augmentation renders viewpoints along the predicted trajectory, aggregates reconstructed and rendered pointmaps, and voxelizes them into a dense metric grid.

![OccAny: Generalized Unconstrained Urban 3D Occupancy source figure: OccAny is a generalized 3D occupancy model that is trained once and can operate on out-of-domain sequential, monocular, or surround-view urban images.](/assets/images/occany-generalized-unconstrained-urban-3d-occupancy-paper-figure.webp)
_OccAny is a generalized 3D occupancy model that is trained once and can operate on out-of-domain sequential, monocular, or surround-view urban images. Source: [OccAny: Generalized Unconstrained Urban 3D Occupancy](https://arxiv.org/abs/2603.23502), Figure 1, via arXiv HTML._


Across out-of-domain sequence inputs, OccAny reports 25.91 IoU on SemanticKITTI and 23.55 on Occ3D-nuScenes. The strongest compared zero-shot baselines reach 15.93 and 19.30 respectively. With a single SemanticKITTI image, OccAny reports 24.03 IoU, compared with 13.03 for the strongest zero-shot baseline in the table. With one surround-view timestep on Occ3D-nuScenes, it reaches 34.15 IoU—well above the 20.78 zero-shot baseline, but below in-domain SelfOcc at 45.01 and GaussTR at 45.19.

| Setting | Best compared zero-shot baseline | OccAny | Strong in-domain result in the table |
| --- | ---: | ---: | ---: |
| SemanticKITTI, five-frame sequence | 15.93 IoU | 25.91 IoU | Not reported in this table |
| SemanticKITTI, one image | 13.03 IoU | 24.03 IoU | 22.81 IoU |
| Occ3D-nuScenes, surround view | 20.78 IoU | 34.15 IoU | 45.19 IoU |

The ablation identifies where the gain comes from. Removing test-time view augmentation drops sequence IoU by 6.27 points and monocular IoU by 12.48 points. Removing Segmentation Forcing costs 1.68 and 2.30 points. The method therefore generalizes partly by moving computation into inference-time geometry completion, not only by learning a stronger feed-forward reconstruction model. In the reported six-input, six-render-view surround setting, reconstruction takes 93.8 ms and rendering 123.2 ms; the forward path contains about 651 million parameters.

Geometry and semantics also separate sharply. OccAny reports 25.91 geometric IoU but only 7.28 semantic mIoU on SemanticKITTI sequences, and 34.15 IoU but 6.66 mIoU on Occ3D-nuScenes surround views. The [Occ3D benchmark](/paper%20shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html) defines a calibrated, visibility-aware target; OccAny shows how far metric geometry can travel beyond that contract, not that it preserves the same class quality or operational guarantees.

## High-Level Takeaways

- OccAny changes the deployment contract from a known camera rig to images plus a learned metric reconstruction prior; calibration is predicted rather than supplied.
- Its largest measured gain comes from novel-view completion at test time, which improves occluded geometry while adding latency, memory, and dependence on predicted poses.
- Metric occupancy generalizes much better than fine-grained semantic occupancy in the reported experiments, so the output is more credible as a geometric scaffold than as a complete planning-ready scene state.
- A result that closes the semantic gap to in-domain methods under matched compute—and remains stable under moving objects, poor pose recovery, and adverse weather—would be needed to justify replacing a calibrated production occupancy stack.
