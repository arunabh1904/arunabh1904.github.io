---
title: 'CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer'
date: '2022-09-14T04:00:00.000Z'
section: paper-shorts
postSlug: craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer
legacyPath: /paper shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer'
---
## 2022 – CRAFT

**arXiv:** [2209.06535](https://arxiv.org/abs/2209.06535)

## Summary

> CRAFT treats radar as a noisy geometric correction to a camera proposal. A camera detector proposes a 3D box, Soft Polar Association selects radar returns in the coordinate system where radar is most precise, and two cross-attention encoders exchange image semantics and radar range/velocity evidence. A fusion head can refine the proposal when the radar support is credible or fall back to the camera prediction when it is not. On nuScenes test, CRAFT reports 41.1 mAP and 52.3 NDS, 8.7 and 10.8 points above the paper's camera-only baseline, at 4.1 FPS.

## Core Insights

### The proposal is the unit of fusion

CRAFT begins with separate modality-specific features. A DLA-34 image backbone and a lightweight camera 3D detector predict a set of image proposals. Each proposal contains a projected center, depth and its variance, dimensions, yaw, and velocity; camera intrinsics and extrinsics convert it into the vehicle frame. A PointNet++ radar branch embeds each radar point together with its position, radar cross section, and ego-motion-compensated radial velocity. The authors accumulate six radar sweeps and cap the input at 2,048 points, so the fusion module receives a small, explicitly object-conditioned set rather than a dense radar image.

![CRAFT source Figure 2: proposal-level camera-radar fusion architecture](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-paper-figure.webp)
*Fig 1: The source pipeline first predicts camera proposals, associates radar points around each proposal in polar coordinates, exchanges image and radar features with consecutive cross-attention layers, and then predicts a fusion score and box offsets. | source: [CRAFT, Figure 2](https://arxiv.org/abs/2209.06535)*

The proposal boundary makes the design's responsibility clear. The camera supplies object coverage and semantic context; radar is asked mainly to correct location and velocity. A proposal with no trustworthy radar support is still allowed to pass through as a camera prediction. This fallback is part of the model rather than a post-hoc rule for handling missing sensors.

### Soft Polar Association matches the sensor error

Radar is not a low-resolution LiDAR. The paper describes roughly 0.4 m radial resolution and about 4.5 degrees of azimuth resolution; at 50 m, that angular uncertainty spans roughly 4 m. A Cartesian neighborhood treats those directions as equally reliable. CRAFT instead transforms the eight proposal corners and radar points from Cartesian to polar coordinates. It keeps points between the proposal's left and right azimuth boundaries and within an adaptive radial interval. The radial interval uses the front and back proposal corners, the predicted depth variance, a minimum range parameter $\gamma=5$, and a scale parameter $\delta=10$.

This association intentionally admits some clutter. Radar can miss a low-reflectivity object or return a multipath point behind it, so making the interval too tight would discard the evidence that fusion needs. The transformer must learn which of the selected points are useful. The association ablation shows why the coordinate choice matters: at the strict 0.5 m car matching threshold, RoI pooling reaches 29.8 AP with 50.7% association recall, a Cartesian ball query reaches 39.5 AP with 78.2% recall, and Soft Polar Association reaches 41.3 AP with 77.1% recall. The polar rule is slightly less permissive than the ball query but reduces radial error from 1.73 to 1.26 m and azimuth error from 0.52 to 0.25 in the coordinate ablation.

### Cross-attention separates where from what

Soft association only defines a candidate set. Spatio-Contextual Fusion Transformer (SCFT) then decides how to use it in two directions. In the Image-to-Radar encoder, each radar point is projected into the image and receives an image patch around that location. The patch grows with distance, is resized, and is read with deformable multi-head cross-attention. A radar query can therefore attend to a small neighborhood when projection is imperfect, rather than trusting a single pixel.

The Radar-to-Image encoder gives each image proposal access to its associated radar features. It uses the proposal's polar coordinates, omits the unreliable radar height coordinate, and adds a zero key/value so the attention can fall back when the radar set is empty. The fused representation predicts a class score, location offsets, center-ness, and velocity. The fusion score decides whether the refined proposal should replace the camera box. This division of labor explains why the model can use radar's range and Doppler cues without pretending that radar gives reliable object shape or orientation.

### The gains are localization gains

On the nuScenes validation split, CRAFT-I, the camera-only version with the same DLA-34 setup, reports 33.2 mAP. CRAFT reaches 41.1, a 7.9-point increase; the largest class gains are on car (+17.2), bus (+17.3), and truck (+11.9), while traffic-cone AP slightly decreases by 0.4. On the test table, CRAFT reports 41.1 mAP, 52.3 NDS, 0.467 mATE, 0.268 mASE, 0.456 mAOE, 0.519 mAVE, and 0.114 mAAE at 4.1 FPS. The camera-only CRAFT-I validation model runs at 4.7 FPS, so the fusion machinery has a measurable speed cost.

The strict-distance table makes the role of radar easier to see than the aggregate mAP. For cars at a 0.5 m center-distance threshold, CRAFT reaches 50.4 AP on test versus 22.3 for CenterFusion; at 1 m it reaches 66.9 versus 45.7. On validation, CRAFT improves CRAFT-I from 19.6 to 51.9 AP at 0.5 m and from 45.4 to 68.8 at 1 m. The method is correcting metric localization, not merely adding a class score to a camera detection.

![CRAFT source Figure 4: performance by object distance and radar support](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-source-figure-4.webp)
*Fig 2: The source analysis separates performance by object distance and by the number of associated radar points. The large improvements at longer range and with more returns show when radar has enough support to correct camera depth. | source: [CRAFT, Figure 4](https://arxiv.org/abs/2209.06535)*

The paper reports a 32.2% relative improvement for objects beyond 35 m in the distance analysis. The point-count curve gives the complementary boundary: with few or no valid radar points, the gain is much smaller; with more points, the fusion has enough range evidence to move an imprecise camera proposal. This is why the architecture keeps a camera fallback instead of forcing every proposal through a radar update.

### Radar sparsity remains a class-dependent ceiling

The qualitative panel marks radar-refined predictions in blue and camera-only predictions in red when no valid associated return exists. A blue box should be read as evidence that the proposal had usable radar support, not as proof that radar improved every object. The per-class analysis reports that only 53.4% of traffic-cone instances contain a radar point, and although 70.0% of traffic-cone image proposals have an association, only 40.2% have a valid association. Non-metallic objects, sidewalk clutter, and multipath returns therefore limit the correction mechanism.

![CRAFT source Figure 5: camera-only and radar-refined qualitative detections](/assets/images/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer-source-figure-5.webp)
*Fig 3: Blue circles identify proposals refined with radar and red circles identify proposals that remain camera-only because no valid radar return was found. The panel makes the model's fallback visible in complex, rainy, and night scenes. | source: [CRAFT, Figure 5](https://arxiv.org/abs/2209.06535)*

The detection-head ablation supports the same interpretation. In the car-only nuScenes validation ablation, removing the location-offset head drops car AP from 69.6 to 53.1, and removing velocity drops it to 65.4 while worsening mAVE from 0.31 to 1.21. Adding dimension or rotation regression also lowers car AP to 65.1 or 65.7. The irregular, ambiguous radar points help correct position and motion more reliably than they help estimate object size or heading. CRAFT's practical boundary is therefore the quality of the camera proposal, the calibration and timestamps used to associate sensors, and whether the scene supplies a valid radar return.

## High-Level Takeaways

- CRAFT fuses around a camera 3D proposal: radar supplies a candidate set and the transformer decides whether its range and velocity evidence is trustworthy.
- Soft Polar Association reflects radar's strong radial resolution and weak azimuth resolution; its ablation reaches 41.3 car AP at a 0.5 m threshold with 77.1% association recall.
- The reported gain is primarily localization: CRAFT-I to CRAFT rises from 19.6 to 51.9 car AP on validation at a 0.5 m threshold, while the full test result is 41.1 mAP/52.3 NDS at 4.1 FPS.
- Missing or ambiguous radar returns are a real ceiling: traffic cones have 70.0% associated proposals but only 40.2% valid associations, so the camera fallback remains essential.
