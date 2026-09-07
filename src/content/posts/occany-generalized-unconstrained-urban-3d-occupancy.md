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

**arXiv:** [2603.23502](https://arxiv.org/abs/2603.23502)

**Project:** [OccAny](https://valeoai.github.io/OccAny/)

**Code:** [valeoai/OccAny](https://github.com/valeoai/OccAny)

## Summary

> OccAny changes the deployment contract for 3D occupancy: a single model accepts sequential, monocular, or surround-view urban images without a target camera rig and predicts metric occupancy in unseen datasets. It learns pointmaps and SAM2-like features from five source datasets, then completes missing geometry with novel-view rendering at inference. Zero-shot geometric IoU reaches 25.91 on SemanticKITTI sequences, 24.03 from one image, and 34.15 on Occ3D-nuScenes surround views, while semantic mIoU remains much lower and in-domain methods still lead.

## Core Insights

### Generalization comes from a metric scene memory

OccAny is not supervision-free. It trains on projected LiDAR pointmaps from Waymo, DDAD, PandaSet, Virtual KITTI 2, and ONCE. What is unconstrained is the target rig: evaluation uses out-of-domain SemanticKITTI and Occ3D-nuScenes without target calibration, training samples, or a fixed camera layout.

The reconstruction stage starts from MUSt3R, freezes its encoder, and trains a decoder to produce metric pointmaps, confidence, RGB, and segmentation-like features. These outputs are stored in a scene memory in a common reference frame. Segmentation Forcing distills SAM2-like features alongside geometry, so boundaries and semantic regions regularize a pointmap objective that would otherwise care mostly about 3D coordinates.

![OccAny accepts sequential, monocular, and surround-view images and produces occupancy and promptable features](/assets/images/occany-generalized-unconstrained-urban-3d-occupancy-paper-figure.webp)
*Fig 1: The source’s Figure 1 shows the generalized input modes and the metric occupancy and segmentation outputs of OccAny. | source: [OccAny: Generalized Unconstrained Urban 3D Occupancy, Figure 1](https://arxiv.org/abs/2603.23502)*

### Novel views turn occlusion into an inference-time resource

The rendering stage receives sampled camera poses along the predicted trajectory and reconstructs pointmaps and segmentation features for those novel views. A smaller rendering encoder is initialized from the reconstruction stage and distilled from its larger 24-block encoder. At inference, Test-Time View Augmentation samples forward and lateral shifts, aggregates original and rendered pointmaps, and voxelizes them with trilinear interpolation. It is a completion operation: a view that was never captured can expose an object surface hidden from the input cameras.

![OccAny two-stage reconstruction and novel-view-rendering training pipeline](/assets/images/occany-generalized-unconstrained-urban-3d-occupancy-source-figure-2.webp)
*Fig 2: The source’s Figure 2 shows reconstruction, scene-memory formation, Segmentation Forcing, and the novel-view rendering stage trained from sampled poses. | source: [OccAny: Generalized Unconstrained Urban 3D Occupancy, Figure 2](https://arxiv.org/abs/2603.23502)*

The result is a deliberate accuracy–compute trade. In the reported six-input, six-render-view surround configuration, reconstruction takes 93.8 ms and rendering 123.2 ms, with about 651M parameters in the forward path. The strongest ablation changes happen when test-time view augmentation is removed, which means the headline generalization is partly an inference-time computation result rather than a purely feed-forward representation result.

### Geometry travels farther than semantics

On SemanticKITTI sequence input, OccAny reaches 25.91 IoU; the strongest compared zero-shot baseline is 15.93. On Occ3D-nuScenes sequence input it reaches 23.55 versus 19.30 for the strongest zero-shot entry. For one SemanticKITTI image it reaches 24.03 IoU versus 13.03, and for a surround-view Occ3D timestep it reaches 34.15 versus 20.78 among zero-shot methods. In-domain methods remain higher in the same table: SelfOcc reaches 45.01 and GaussTR 45.19 on the surround benchmark.

The semantic readout shows a different boundary. OccAny reports 7.28 mIoU / 13.53 super-class mIoU on SemanticKITTI sequences and 6.66 / 10.32 on Occ3D-nuScenes surround views. Segmentation Forcing is still useful—the no-forcing variant is lower—but the model’s strongest claim is metric geometry under changing camera contracts, not planning-ready class labels.

This distinction matters for deployment. The target benchmark’s calibrated, visibility-aware occupancy labels encode a sensor and annotation contract that OccAny intentionally does not assume. The method can provide a geometric scaffold across rigs, but moving from that scaffold to reliable semantic occupancy still requires better class supervision, promptable feature calibration, and dynamic-scene tests.

## High-Level Takeaways

- OccAny learns a reusable metric scene memory and predicts target camera geometry instead of requiring a known target rig.
- Segmentation Forcing adds semantic boundaries to geometric training, while Novel-View Rendering completes the surfaces that the input views cannot see.
- The largest reported generalization gains depend on test-time view augmentation, so latency and pose-error sensitivity are part of the method’s identity.
- Geometric IoU transfers much better than fine-grained semantic mIoU, and in-domain occupancy specialists still lead the comparison.
- A convincing next step is a dynamic, uncalibrated evaluation with moving objects, pose drift, adverse weather, and matched inference budgets against calibrated occupancy models.
