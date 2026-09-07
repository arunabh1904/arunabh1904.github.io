---
title: 'RPGFusion: 4D Radar Prior-Guided Multi-Modal Fusion'
date: '2026-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: rpgfusion-4d-radar-prior-guided-fusion
legacyPath: /paper shorts/2026/06/01/rpgfusion-4d-radar-prior-guided-fusion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – RPGFusion: use 4D-radar priors to localize and densify camera evidence'
---

**Paper:** [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_RPGFusion_4D_Radar_Prior-Guided_Multi-Modal_Fusion_for_3D_Detection_CVPR_2026_paper.html)

## Summary

> RPGFusion makes 4D radar a geometric prior for camera-to-BEV sampling. Radar returns are converted into confidence and depth maps, those maps initialize and guide image BEV queries, and a robust sparse-to-dense radar encoder supplies a second BEV stream. Spatial alignment and semantic fusion then reconcile the two modalities. The method reports 69.31 mAP over the entire View-of-Delft annotated area and 86.20 mAP in its driving corridor, but the result depends on radar priors, calibration, and region-specific IoU protocols.

## Core Insights

### Radar should guide where the camera looks

A conventional camera-to-BEV transform must infer depth along each viewing ray. RPGFusion uses 4D radar to narrow that ambiguity before image features are sampled. Each radar point contributes through a Gaussian kernel to a BEV confidence map and a depth map. The confidence map describes where radar evidence is spatially reliable; the depth map supplies an explicit range cue. A small prior encoder combines them with position and base query embeddings to initialize image BEV queries.

The image sampler then projects each BEV cell into the camera using the radar-provided height and calibration, adds learned local offsets, and gathers nearby image features. A three-layer iterative update uses radar confidence to emphasize reliable regions and depth to keep the samples near the correct spatial support. This is the useful intuition: radar does not merely become another feature after view transformation; it changes the geometric search that creates the image BEV in the first place.

![RPGFusion radar-prior-guided image sampling and unified fusion](/assets/images/rpgfusion-paper-figure.webp)
*Fig 1: Radar confidence and depth priors initialize image BEV queries and guide sampling before radar and camera features undergo spatial alignment and semantic fusion; the displayed framework is the paper’s Figure 2. | source: [RPGFusion: 4D Radar Prior-Guided Multi-Modal Fusion for 3D Detection, Figure 2](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_RPGFusion_4D_Radar_Prior-Guided_Multi-Modal_Fusion_for_3D_Detection_CVPR_2026_paper.html)*

### Densification repairs the radar branch before fusion

The radar point cloud is both sparse and noisy. RPGFusion first aggregates returns into pillars with range, height, Doppler, reflectivity, and point-count statistics. An RCS-based neighborhood confidence downweights isolated or inconsistent echoes. A sparse-to-dense module propagates features to empty BEV cells using spatial proximity and feature similarity, producing a radar BEV with smoother structural support without discarding the physical confidence cue.

After image and radar BEV features are built, spatial alignment predicts local offsets and uses deformable attention so each modality can correct small calibration or sampling errors. Semantic fusion is bidirectional: camera features can update radar features with semantic cues, and radar features can update camera features with geometric evidence. A learned gate then mixes the aligned streams rather than treating them as equally reliable everywhere.

### Ablations identify the physical priors

On View-of-Delft validation, RPGFusion reaches 69.31 EAA-mAP over the entire annotated area and 86.20 DCA-mAP in the driving corridor. The table uses IoU 0.5 for cars and 0.25 for pedestrians and cyclists. On TJ4DRadSet test, it reports 43.05 3D mAP and 46.86 BEV mAP, improving over CVFusion by 3.05 and 2.79 points.

The prior-map ablation is unusually diagnostic. On VoD, using both confidence and depth priors gives 69.31 EAA / 86.20 DCA; removing confidence gives 64.72 / 81.36, removing depth gives 63.46 / 80.77, and removing both gives 58.29 / 75.44. The same pattern appears on TJ4DRadSet. For robust radar encoding and densification, removing both gives 54.10 EAA / 75.42 DCA on VoD and 33.58 / 36.26 on TJ4DRadSet, while the full branch reaches 69.31 / 86.20 and 43.05 / 46.86.

The fusion ablation also separates modality presence from fusion quality. On VoD, camera-only is 56.09 EAA / 73.92 DCA, radar-only 45.71 / 65.84, simple concatenation 63.25 / 79.66, and the unified fusion 69.31 / 86.20. These comparisons support the role of radar-guided geometry, but they do not establish robustness to a corrupted prior: a false radar echo can steer the camera sampler before semantic fusion has a chance to correct it.

## High-Level Takeaways

- RPGFusion’s main decision is to use 4D radar before camera lifting, where confidence and depth can reduce image-to-BEV ambiguity.
- Sparse-to-dense radar encoding and prior-guided sampling solve different problems: one repairs radar support, the other changes where image evidence is queried.
- Both confidence and depth maps matter, and their removal causes larger drops than a generic modality concatenation baseline.
- View-of-Delft’s entire-area and driving-corridor scores use different spatial coverage, so they should be reported separately from TJ4DRadSet’s 3D and BEV metrics.
- The next stress test should inject multipath ghosts, calibration drift, camera or radar dropout, and tangentially moving actors to measure whether a wrong prior can erase useful camera evidence.
