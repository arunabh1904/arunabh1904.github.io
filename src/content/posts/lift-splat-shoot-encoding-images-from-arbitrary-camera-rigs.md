---
title: 'Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D'
date: '2020-08-13T00:00:00.000Z'
section: paper-shorts
postSlug: lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs
legacyPath: /paper shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2020 – Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D'
---
## 2020 – Lift, Splat, Shoot

**arXiv:** [2008.05711](https://arxiv.org/abs/2008.05711)

**Project and code:** [nv-tlabs.github.io/lift-splat-shoot](https://nv-tlabs.github.io/lift-splat-shoot/)

## Summary

> Lift-Splat-Shoot (LSS) makes the vehicle-centered BEV grid the common address for evidence from an arbitrary number of calibrated cameras. Each pixel keeps a distribution over possible depths instead of committing to one guessed 3D point; the distributions are splatted into pillars and processed by a BEV network. The paper shows strong camera-only segmentation and a useful planning interface, while its oracle-depth comparison still exposes the cost of far-range and nighttime ambiguity. “Shoot” ranks fixed trajectory templates in the learned cost map, so the result is an interpretable perception-to-planning interface rather than a learned closed-loop controller.

## Core Insights

### Put every camera on one spatial address

The problem LSS isolates is a coordinate mismatch. A camera feature lives at an image pixel, but a planner needs a map in the ego vehicle’s frame. Given image $X_k$, calibration $(E_k, I_k)$, and a discrete depth set $D$, LSS creates candidate points $(h,w,d)$ along every pixel ray. The geometry is fixed by calibration; the learned part decides how much feature evidence to assign to each candidate.

For a pixel, a small network predicts a context vector $c\in\mathbb{R}^C$ and a categorical depth distribution $\alpha$. The feature at depth $d$ is $c_d=\alpha_d c$. A one-hot $\alpha$ behaves like pseudo-LiDAR, placing the context at one range. A uniform $\alpha$ behaves like an orthographic feature transform, spreading it along the ray. The learned distribution can therefore express uncertainty instead of turning an ambiguous image cue into a single, overconfident 3D location.

The calibrated candidates from each camera are transformed into the ego frame and assigned to ground-plane pillars. Sum pooling produces one $C	imes X	imes Y$ tensor, independent of how many cameras supplied the candidates. A standard BEV CNN can then fuse overlapping views and support semantic segmentation or a cost-map head. This is the durable interface: the downstream model queries a metric map, while the view transformer absorbs the camera-specific coordinate systems.

![Figure 1 from Lift, Splat, Shoot, showing surround-camera images mapped into a vehicle-centered BEV prediction](/assets/images/lift-splat-shoot-source-figure-1-overview.png)
*Fig 1: Surround-camera images are converted into vehicle, drivable-area, and lane semantics in one BEV frame; the colored BEV predictions are projected back onto the input views for visualization. | source: [Lift, Splat, Shoot, Figure 1](https://arxiv.org/abs/2008.05711)*

### The lift keeps alternatives; the splat pays for them

The outer product creates a frustum feature for every pixel and every depth bin. That is expressive, but it can be large before pooling. LSS follows a PointPillars-style representation in which points with the same ground-plane bin are summed. Instead of padding every pillar to a common number of points, the implementation sorts features by bin ID, takes a cumulative sum, and subtracts values at bin boundaries. The paper derives an analytic gradient for the whole pooling layer and reports a 2× training speedup over backpropagating through the individual operations.

Read the lifting diagram as a factorization: the context vector supplies feature channels, and the depth distribution supplies a weight for each column of the frustum. The network does not predict an unrelated context vector at every depth. It reuses one pixel’s visual evidence at several possible ranges, scaling each copy before calibration places it in the shared map.

![LSS depth distribution and context vector forming a lifted frustum feature](/assets/images/lift-splat-shoot-source-figure-3-lift.png)
*Fig 2: The outer product of a pixel’s context vector and depth distribution creates a channel-by-depth feature, retaining alternative ranges before BEV pooling. | source: [Lift, Splat, Shoot, Figure 3](https://arxiv.org/abs/2008.05711)*

The concrete model uses EfficientNet-B0 image encoders and a ResNet-style BEV network. Input images are resized and cropped to $128	imes352$; the BEV grid spans $[-50,50]$ metres in both axes at $0.5$-metre cells, and depth candidates cover 4–45 metres in 1-metre steps. The final network has 14.3M trainable parameters and runs at 35 Hz on a Titan V in the paper’s setup. Those numbers describe a compact research configuration, while the frustum’s depth resolution and camera count remain the main memory levers.

### Calibration is part of the learned representation

LSS does not require a fixed camera index to mean a fixed spatial location. It receives each camera’s intrinsics and extrinsics, so the same lifting rule can map different rigs into the same ego grid. The experiments make this more than an architectural claim. A model trained with four of nuScenes’ six cameras improves car-segmentation IoU from 26.53 with only the training cameras to 27.94 when both unseen cameras are added at evaluation. Training on nuScenes and evaluating on the entirely different Lyft rig gives 21.35 car IoU and 22.59 vehicle IoU, compared with 7.00 and 8.06 for the plain CNN baseline.

The robustness experiment also separates redundancy from coverage. Dropping cameras during training improves performance when cameras are missing at test time, and the best full-rig model is trained with one random camera removed from each sample. Dropping the rear camera hurts most because its wider field of view leaves a larger uncovered region. The BEV network can inpaint some missing evidence, but the full-rig score remains an upper bound wherever no camera sees the scene.

![Figure 7 from Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](/assets/images/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs-source-figure-7.webp)
*Fig 3: Car-segmentation IoU after removing each camera shows the largest loss for the wide-field rear camera; sensor placement determines the missing spatial evidence, not simply the number of remaining views. | source: [Lift, Splat, Shoot, Figure 7](https://arxiv.org/abs/2008.05711)*

### The benchmark shows an interface, not solved depth

On nuScenes, LSS reaches 32.06 car IoU and 32.07 vehicle IoU; on Lyft it reaches 43.09 and 44.64. The map benchmark in Table 2 reports 72.94 drivable-area IoU and 19.96 lane-boundary IoU on nuScenes. The later oracle-depth comparison lists a different LSS row, 70.81/19.58, so its paired comparison is kept separate below. The model beats the paper’s CNN, frozen-encoder, and OFT baselines, showing that the architecture learns both useful context and a useful implicit depth distribution.

The oracle-depth comparison puts the remaining ambiguity in view. With one LiDAR scan, the PointPillars reference reaches 74.91 drivable-area IoU, 25.12 lane-boundary IoU, 40.26 car IoU, and 44.48 vehicle IoU, above LSS’s 70.81, 19.58, 32.06, and 32.07. The gap grows at night, and both camera and LiDAR models degrade roughly linearly with distance. LSS gives the planner a spatial map, but the camera does not acquire LiDAR’s direct range measurement.

The planning experiment is deliberately narrower than its title suggests. The model sums the learned spatial cost along each of 1,000 K-means trajectory templates, each five seconds long at 0.25-second intervals. Negated trajectory costs become softmax logits. During training, the expert trajectory’s nearest template under L2 distance supplies the classification target, so the planning loss can train the camera representation end to end. Its top-5, top-10, and top-20 accuracies are 15.52, 19.94, and 27.99, below the one-scan LiDAR reference at 19.27, 28.88, and 41.93. The demonstration establishes that BEV semantics can be queried by a planner; it does not train a policy to discover arbitrary controls or evaluate closed-loop interventions.

## High-Level Takeaways

- LSS’s key abstraction is a calibrated pixel-by-depth feature: a probability distribution keeps multiple 3D hypotheses alive until all views share the ego-frame BEV grid.
- The cumsum pooling layer makes that frustum practical, but depth bins, BEV extent, and camera count still determine the representation’s memory budget.
- Camera calibration is an input to the spatial mapping. Unseen-camera and cross-rig results support that design, while dropout tests show that coverage gaps cannot be fully recovered by the BEV CNN.
- The oracle-depth, night, and range comparisons preserve a real limitation: camera-only lifting supplies a useful map without making metric depth observable everywhere.
- “Shoot” ranks a fixed template set in the cost map. It validates an interface to planning, not closed-loop autonomy.
