---
title: 'Perception for Autonomous Driving'
date: '2026-07-31T18:00:00.000Z'
section: blog
postSlug: how-unified-sensor-models-are-built-for-autonomous-driving
legacyPath: /blog/2026/07/31/how-unified-sensor-models-are-built-for-autonomous-driving.html
tags:
  - Research
  - Autonomous Driving
  - Sensor Fusion
topics:
  - autonomy
  - multimodal
summary: How camera, LiDAR, and radar encoders preserve different measurements, align them in metric space, fuse them, and carry scene state through time.
---

# Perception for Autonomous Driving in 2026

An autonomous vehicle receives several partial descriptions of the same scene. Cameras provide dense appearance and semantic detail, but depth has to be inferred and image quality falls under glare, darkness, rain, or fog. LiDAR measures range directly and resolves 3D shape well, but the point cloud becomes sparse with distance and can be degraded by weather, reflectivity, occlusion, and motion during a scan. Radar measures range and radial velocity at long distance and is comparatively tolerant of poor visibility, but its angle estimates are coarse and multipath produces ambiguous returns.

The central architectural question is therefore not simply how to combine sensors. It is where to preserve their differences, where to establish a shared geometric representation, and where to spend the information that the model can no longer recover. The progression in this article follows that dependency:

<aside class="revision-insight" aria-label="The autonomous-driving perception pipeline">
  <header class="revision-insight__header">
    <div class="revision-insight__badge" aria-hidden="true">→</div>
    <h3 class="revision-insight__title">The progression to keep in mind</h3>
  </header>
  <div class="revision-insight__content">
    <p><strong>sensor-specific encoders → metric 3D representation → temporal state → task heads</strong></p>
  </div>
</aside>

Fusion sits between the second and third steps. The model should preserve modality-specific evidence first, align it in the vehicle frame, and only then decide whether the shared state should be a dense scene field, a sparse set of object queries, or a learned mixture of both.

## From measurements to a driving scene

The diagram below shows the runtime path. Each sensor first passes through an encoder designed for its sampling pattern. Intrinsics describe how a camera maps 3D rays to pixels; extrinsics describe where every sensor sits relative to the vehicle; timestamps and ego poses place measurements at a common time. These transformations let the model ask whether an image edge, a LiDAR return, and a radar detection refer to the same location. If calibration is wrong, fusion creates structured errors rather than random noise: a vehicle can acquire a displaced image feature, a lane boundary can shift across the BEV grid, and displacement from an angular error grows with range.

After alignment, the model stores the scene as a dense bird's-eye-view (BEV) field, a sparse set of 3D queries, or a mixture of both. BEV gives every ground-plane location a persistent cell, which suits occupancy, free space, lanes, and maps. Queries allocate state to selected actors or map elements, which suits detection, tracking, and motion prediction. A temporal module then aligns previous state to the current ego frame. This step reduces frame-to-frame flicker, exposes velocity through displacement, and preserves evidence through a short occlusion; it can also propagate stale detections or pose error if state is not aged and refreshed carefully.

[![Autonomous-driving perception pipeline from sensor-specific encoders through calibrated metric representations, dense or sparse temporal state, and task heads, with learned latent tokens and a separate training-only graph](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_The runtime path moves from sensor-specific features to a calibrated BEV field, query set, or learned latent state, then carries that state into task heads. The lower path shows supervision that may exist during training without becoming a deployed sensor input._

## Sensor encoders preserve different evidence

The first step is not fusion. It is choosing an encoder that does not erase the measurement that makes a modality useful.

### Cameras: dense semantics at several scales

Camera encoding must recognize visual patterns without discarding the spatial detail needed for 3D placement. Convolutional backbones build local features with translation-equivariant kernels and map well to optimized inference libraries; production-oriented systems such as [NVAutoNet](https://openaccess.thecvf.com/content/WACV2024/html/Pham_NVAutoNet_Fast_and_Accurate_360deg_3D_Visual_Perception_for_Self_WACV_2024_paper.html) use efficient CNNs for both image and BEV processing. Vision transformers use content-dependent attention to connect distant image regions, which can help when recognition depends on wider context, but global attention at full surround-camera resolution is expensive. Both families still produce dense per-view feature maps, and neither removes the need for calibrated 3D reasoning downstream.

Feature pyramids are especially useful in driving because apparent object size changes sharply with range. A high-resolution level preserves the few pixels that represent a distant pedestrian or traffic light. Lower-resolution levels combine evidence over a larger receptive field and are cheaper to use for large vehicles, road layout, and scene context. [EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) is a clear 2D example of learned multiscale fusion. [BEVFormer v2](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.html) shows a separate issue in 3D: modern image backbones benefit from perspective-view supervision because a loss applied only after BEV conversion gives the image features a weak and indirect training signal.

For a 3D point $X$, camera $i$ uses its intrinsics $K_i$ and extrinsics $(R_i,t_i)$ to compute

$$
u_i=\pi(K_i,R_i,t_i,X).
$$

The same point appears at a different pixel in each camera. If pyramid level $l$ has stride $s_l$, the feature is sampled at $u_i/s_l$. This projection is a correspondence rule, not a source of visual information: it can retrieve a fine boundary only if the backbone retained that boundary, and it can retrieve broad context only if the receptive field encoded it.

[![Animation showing one metric 3D point projected to different pixels in two calibrated cameras and sampled at stride-adjusted feature-pyramid coordinates](/assets/images/autonomous-perception-vision-encoder.gif)](/assets/images/autonomous-perception-vision-encoder.gif)
_A single 3D reference point lands at different pixels in different cameras and at different coordinates on each pyramid level. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) uses this operation for object queries; [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) uses it for BEV queries._

Camera encoders are strongest where appearance matters: class identity, lane paint, signs, lights, and object boundaries. Their main 3D weakness is not a lack of semantics but depth ambiguity. Increasing input resolution or backbone capacity can improve recognition without fixing that ambiguity. The next decision is therefore where to place image evidence in metric space.

### LiDAR: sparse geometry before dense BEV

A LiDAR sweep is already metric, but it is irregular, sparse, and strongly range-dependent. Point encoders preserve individual returns but make neighborhood construction expensive. Pillar encoders group points in vertical columns and collapse height early. Voxel encoders retain a 3D grid for longer, which preserves overpasses, trucks, poles, and other height-dependent structure. The main encoder decision is where to trade 3D detail for the speed and regularity of a 2D BEV backbone.

[PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) makes that trade early: it pools each occupied pillar, scatters the result into a dense pseudo-image, and performs nearly all later computation with 2D convolutions. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) keeps sparse 3D voxels through a middle encoder and densifies only after height has been compressed. Sparse convolution is effective because it shares local kernels over occupied cells, but coordinate-map construction and irregular memory access remain real system costs.

Sparse attention changes how occupied cells communicate. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) partitions variable-density voxels into bounded local sets, applies attention inside each set, and rotates the partition between layers so information crosses set boundaries. Attention can connect non-adjacent occupied cells more directly than a small convolutional kernel, while bounded sets keep the workload deployable. It does not make the representation free: sorting, padding, token density, and the eventual height compression still determine latency.

[![Animation comparing early densification in PointPillars, late BEV densification in SECOND and DSVT, and box prediction from active voxels in VoxelNeXt](/assets/images/autonomous-perception-lidar-encoder.gif)](/assets/images/autonomous-perception-lidar-encoder.gif)
_PointPillars compresses height before its dense 2D backbone. SECOND and DSVT retain sparse 3D cells for longer, then construct BEV. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) keeps the active-voxel representation through the prediction head._

The right encoder depends on the operating envelope. Pillars are a strong latency baseline on mostly planar roads. Sparse voxels are preferable when vertical separation, long range, or dense 3D structure matters. Fully sparse heads save BEV work when occupied cells remain rare, but the wall-clock gain can be much smaller than the FLOP reduction on hardware with weak sparse-kernel support. In every case, intensity, timestamp, and point age should remain attached to the geometry: LiDAR range is direct, but a rotating scan is not an instantaneous snapshot.

### Radar: range and motion under uncertainty

Radar should be modeled around the measurements that distinguish it from LiDAR. A return may contain range, azimuth, elevation, radial velocity, Radar Cross Section (RCS), timestamp, and a sensor-specific confidence estimate. Doppler can reveal a moving actor before image-only temporal inference has enough baseline, and millimeter-wave sensing is less sensitive than cameras to darkness and often more tolerant than cameras or LiDAR in rain and fog. The cost is weak angular localization, sparse semantic evidence, multipath, and ghost returns. Radar supplies useful constraints, not clean 3D boxes.

Three encoder families correspond to three fusion strategies. A point encoder keeps individual returns and is useful when association happens around an object proposal. A polar or range-azimuth tensor preserves the sensor's native sampling pattern and can exploit local radar structure before Cartesian resampling. A radar-BEV encoder creates an independent metric feature map for later alignment with camera or LiDAR BEV. Rasterizing too early into binary occupancy removes Doppler, RCS, confidence, and return-level ambiguity—the exact information needed to reject clutter.

[![Animation comparing camera-radar interaction at the proposal, depth, and BEV stages in CRAFT, CRN, and RCBEVDet](/assets/images/autonomous-perception-radar-encoder.gif)](/assets/images/autonomous-perception-radar-encoder.gif)
_[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates a soft set of radar returns with each camera proposal. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to refine camera depth before lifting and aligns both BEV maps with deformable attention. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) combines point and transformer radar paths before BEV fusion._

Recent radar-camera systems improve less by treating radar as an extra image channel than by deciding where its range and Doppler should change the computation. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) found that retaining radar metadata, disabling an aggressive outlier filter, and accumulating aligned sweeps all affected performance. [Doppler-Aware LiDAR–Radar Fusion](https://openaccess.thecvf.com/content/ICCV2025/html/Chae_Doppler-Aware_LiDAR-RADAR_Fusion_for_Weather-Robust_3D_Detection_ICCV_2025_paper.html) processes radar power and Doppler as distinct signals during multimodal interaction, while [DinoRADE](https://openaccess.thecvf.com/content/CVPR2026W/DriveX/html/Leitgeb_DinoRADE_Full_Spectral_Radar-Camera_Fusion_with_Vision_Foundation_Model_Features_CVPRW_2026_paper.html) combines dense radar tensors with DINOv3 image features in adverse-weather detection. A useful comparison should report adverse-weather and long-range accuracy together with false associations and velocity error; higher average detection is not enough if ghost returns attach to the wrong actor.

The encoders now preserve the right evidence for each modality. The next question is how a camera feature becomes a metric 3D feature before it meets LiDAR or radar.

## Camera to metric 3D: LSS and BEVDepth

A pixel fixes a ray through the camera center, not a distance along that ray. Camera-based 3D perception must decide where along the ray to place its feature. The first important family makes that uncertainty explicit.

[Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a categorical depth distribution for each image location, copies the image feature along the corresponding depth bins, and pools those lifted features into BEV:

image pixel → ray → depth distribution → 3D feature cloud → BEV grid

The depth distribution in the original LSS is a latent variable. It is not directly supervised; downstream BEV task losses push the distribution toward geometry that helps detection and other outputs. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) adds projected LiDAR depth supervision, so the camera branch receives both a direct geometric signal and the downstream task signal. LiDAR can therefore be present during training and disappear at inference. That is learning with privileged information; it becomes teacher-student or cross-modal distillation only when a separate teacher explicitly transfers knowledge to a student.

These forward-projection methods preserve dense image evidence, which is useful for occupancy and maps, but a depth error writes the feature into the wrong metric cell.

![Figure 1 from Lift, Splat, Shoot, showing multiview evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_Lift, Splat, Shoot predicts depth along every camera ray and pools the lifted features into a vehicle-centered BEV grid. Source: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

Pull-based methods start in metric space. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) projects each 3D voxel into the cameras and bilinearly samples the visible image features. Its controlled experiments found that input resolution and effective batch size changed vehicle-segmentation IoU more than the lifting operator in that setup. The result is a useful warning against reading every benchmark gain as evidence for a better view transformer; training recipe, image resolution, and sensor inputs must be matched first.

The metric representation can also be queried sparsely. That creates a second branch of the progression.

## Sparse object-centric 3D: DETR3D, PETR, and StreamPETR

[DETR](https://arxiv.org/abs/2005.12872) introduced the basic object-query idea for 2D detection: a bounded set of learned query vectors asks whether there is an object to represent, and a transformer decoder turns those queries into a set of class and box predictions. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) attaches each query to a 3D reference point $(x,y,z)$, projects that point into every camera, and samples image evidence around the projections.

The difference from dense lifting is where the model spends its geometric budget. LSS represents every image location and every candidate depth bin. DETR3D asks only about a bounded set of candidate objects. It can therefore be efficient for detection, but it also inherits a recall risk: if no query acquires support for an actor, later refinement cannot recover evidence that was never represented.

[PETR](/paper%20shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html) moves more of the geometry into the image features themselves. Multi-view features otherwise know their appearance and camera index, but not directly which region of physical 3D space they describe. PETR injects 3D positional information into those features, so subsequent attention can reason over image evidence with explicit geometric context. A useful shorthand is that DETR3D puts much of the geometry into the query and reference-point mechanism, while PETR makes the features more explicitly geometry-aware.

[StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) extends the object-query representation through time. Instead of rediscovering every actor from each frame, it retains a bounded queue of foreground queries, conditions them on ego pose, elapsed time, and velocity, and introduces fresh queries for newly visible actors. The temporal state is sparse and object-centric: it improves stability and occlusion handling without storing a full scene grid, but query birth, duplicate removal, aging, and stale false positives become part of the model.

The sparse branch gives us an efficient answer for selected actors. Driving perception also needs a representation for everything between those actors: lanes, free space, occupancy, and map structure.

## Dense scene-centric 3D: BEVFormer

BEVFormer changes the unit of representation from an object query to a BEV query. Imagine a learned token for every bird's-eye-view location. Each token asks the image features what appears to exist at that physical location, using calibration-aware attention to sample several heights along the corresponding vertical pillar.

This builds a dense spatial world representation using attention rather than LSS-style depth lifting. Temporal attention can update the current BEV representation from previous BEV representations, so the model carries scene context rather than only selected actors. The price is that the number of BEV cells, cameras, heights, and attention samples can dominate compute.

Dense BEV is useful beyond bounding boxes because the same spatial field can support detection, occupancy, map segmentation, lanes, free space, motion, and planning. The representation choice is therefore a task decision:

[![Animation comparing depth along an image ray in LSS, a 3D object-query reference point in DETR3D, and vertical reference points from a BEV cell in BEVFormer](/assets/images/autonomous-perception-camera-lifting.gif)](/assets/images/autonomous-perception-camera-lifting.gif)
_The highlighted variable is the source of metric support: depth along an image ray in LSS, a 3D object reference point in DETR3D, or several heights above a fixed BEV cell in BEVFormer._

| 3D construction | Where the metric hypothesis starts | Best fit | Main cost | Typical failure |
| --- | --- | --- | --- | --- |
| Depth lift and splat | Every image location predicts depth | Dense occupancy, maps, and detection | Pixels × depth bins, followed by BEV processing | Wrong depth moves evidence to the wrong BEV cell |
| Voxel-to-image sampling | Every metric voxel projects into each camera | Simple dense BEV baselines | Voxels × visible cameras | A sampled pixel may contain mixed depths or occlusion |
| Object queries | A bounded set of 3D actor hypotheses | Detection, tracking, and motion | Queries × views × samples | An object is missed when no query acquires support |
| BEV queries | Every ground-plane cell samples several heights | Dense scene state with selective image retrieval | BEV cells × heights × views | Weak or incorrect projected support creates false or empty cells |

No row dominates every task. Dense BEV says “represent the world”; object queries say “represent the candidate actors.” The important question is what evidence each representation has thrown away, and whether a later module can recover it.

## Fusion: make the sensors meet in metric space

Once camera, LiDAR, and radar have metric support, the model must decide where they interact. Early fusion combines near-raw inputs, but pixels, points, and Doppler returns do not naturally share a representation. Late fusion combines independent object predictions, which is modular but loses feature-level complementarity. Intermediate fusion lets each sensor use its own encoder first, then aligns the resulting features and combines them. For heterogeneous driving sensors, this is the most useful conceptual baseline.

[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) attaches camera class scores to LiDAR points before voxelization. This is cheap and geometrically direct, but image evidence survives only where LiDAR produced a return. [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) samples camera, LiDAR, and radar features around the same 3D object reference point. It matches an object-centric output but does not retain a complete background field.

[BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) gives each modality an appropriate encoder, transforms the resulting features into a shared BEV grid, and fuses there:

The path is camera images → camera encoder → camera BEV; LiDAR point cloud → point or voxel encoder → LiDAR BEV; then camera BEV + LiDAR BEV → fusion → task heads.

BEV becomes a natural meeting room because a camera feature at $(x,y)$ and a LiDAR feature at $(x,y)$ refer to approximately the same physical location. The model does not make the camera behave like LiDAR or LiDAR behave like a camera; it preserves their inductive biases until physical alignment makes interaction meaningful.

[![Animation showing the same actor and lane evidence entering point, object-query, and dense-BEV fusion](/assets/images/autonomous-perception-fusion-granularity.gif)](/assets/images/autonomous-perception-fusion-granularity.gif)
_Point fusion retains camera features only at measured points. Query fusion gathers evidence around selected actors. Dense BEV fusion keeps actor evidence and the surrounding lane or occupancy field._

## Proposal-conditioned fusion and deeper interaction

The shared BEV path is not the only way to fuse. Proposal-based fusion spends compute selectively: [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) uses LiDAR proposals to retrieve image evidence around candidate objects instead of trusting one calibrated pixel. This can be efficient and robust to small correspondence errors, but it exposes a structural failure mode. If LiDAR misses an object that the camera sees, a LiDAR-controlled proposal stage may never give the camera evidence a chance to recover it.

[![Animation comparing LiDAR-controlled proposals, merged proposals from multiple modalities, and shared-BEV fusion before detection](/assets/images/autonomous-perception-proposal-recall.gif)](/assets/images/autonomous-perception-proposal-recall.gif)
_Proposal-conditioned fusion is selective, but proposal recall can become a ceiling: a missed LiDAR proposal may prevent camera evidence from entering the refinement path. Multi-proposal fusion restores recall at the cost of matching and deduplication; shared BEV fusion preserves both modality fields before detection._

One response is to let each modality generate proposals and merge them before refinement. That creates a different set of problems—duplicate proposals, conflicting confidence, inconsistent localization, and cross-modal matching—but it removes the single-modality recall ceiling. A dense shared representation makes the opposite choice: fuse world evidence first and generate detections afterward. It spends more compute, but it preserves a path for one modality to recover what another missed.

Fusion can also happen repeatedly rather than in one terminal block. A simple architecture runs a camera encoder and a LiDAR encoder to completion and applies one fusion block. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) allows the streams to update one another across representation stages; [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) applies shared transformer blocks after modality-specific tokenization. The important distinction is not merely that cross-attention exists. It is that modality interaction is allowed before feature extraction is completely finished, so one stream can refine what the other treats as relevant.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion keeps camera and LiDAR encoding separate until both modalities occupy the same BEV grid, then shares that grid across detection and map heads. Source: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

The right fusion design therefore depends on the output and the failure being optimized. Point fusion is tied to measured support, proposal fusion is tied to query recall, and dense BEV fusion is tied to grid cost. The next problem is that even a good fused representation describes only the current sensor window.

### Missing, degraded, and misaligned sensors

A fusion network trained only with all sensors often becomes dependent on the strongest stream. Zeroing LiDAR at inference does not turn such a network into a competent camera-only model. [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) demonstrates this directly: its model trained only in fused mode reaches 3.0 camera-only mAP in the reported ablation, while modality dropout and fusion normalized over the streams that remain produce a usable camera-only path. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) similarly trains full, camera-only, and LiDAR-only modes and lets BEV queries attend to whichever encoders are available. Its reported missing-LiDAR result improves 35.5 NDS over a vanilla BEVFusion comparison, showing that graceful fallback is primarily a training-distribution and routing problem.

[![Animation contrasting modality availability in UniBEV and MetaBEV with reliability gating in Grace-BEV](/assets/images/autonomous-perception-modality-dropout.gif)](/assets/images/autonomous-perception-modality-dropout.gif)
_Modality dropout covers discrete cases in which a stream is absent. A reliability gate is needed for the harder case shown in the final phase: the camera still arrives, but its evidence is degraded and should receive less weight._

Sensor absence is only one failure mode. Blur, saturation, fog, reduced LiDAR beams, blocked fields of view, packet delay, and calibration drift leave a tensor present but unreliable. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) adds reliability-aware gating, while [MetaBEV](https://openaccess.thecvf.com/content/ICCV2023/html/Ge_MetaBEV_Solving_Sensor_Failures_for_3D_Detection_and_Map_Segmentation_ICCV_2023_paper.html) evaluates several corruptions in addition to missing streams. A production model needs both mechanisms: modality dropout for supported sensor configurations, and corruption training plus an observable health signal for partial degradation. Calibration noise should be included as a separate test because every projected association can be confidently wrong in the same direction.

## Time: sparse objects or a dense scene?

Camera features and a single LiDAR sweep do not directly provide a complete object velocity, preserve evidence through an occlusion, or stabilize a noisy depth estimate. Radar supplies radial velocity, but not the full motion state of every actor. Temporal modeling fills the remaining gap by comparing evidence across frames. Old features must first be expressed in the current coordinate frame: ego-motion compensation aligns roads and static structures, while independently moving actors require velocity or learned motion updates. Timestamp error, pose error, and rolling-shutter effects appear as residual motion after alignment.

Dense temporal memory stores scene-level BEV features. [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) warps the previous BEV feature into the current ego frame, concatenates it with the current feature, and lets a BEV encoder learn displacement cues. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) uses temporal attention to update a recurrent BEV field. [SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) keeps a short high-resolution history for fine stereo correspondence and a longer low-resolution BEV history for depth and velocity. Dense state retains roads, free space, and weak background evidence that may later support a new detection, but its memory and warp cost scale with BEV area and history.

Sparse temporal memory stores selected objects or queries. The StreamPETR queue introduced above is the clearest example: it keeps a bounded set of foreground representations and introduces fresh queries for actors not already in memory. [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) transforms prior instances into the current frame and combines them with fresh anchors. Sparse memory is lighter and naturally object-centric, but it can discard free space, road structure, undetected objects, and context that has not yet become a confident query.

[![Animation comparing a warped dense BEV field, transformed recurrent instances with fresh anchors, and a bounded foreground-query queue](/assets/images/autonomous-perception-temporal-memory.gif)](/assets/images/autonomous-perception-temporal-memory.gif)
_Dense recurrence carries every BEV cell. Sparse4D v2 transforms recurrent object instances and adds fresh anchors. StreamPETR carries a bounded queue of foreground queries and introduces new queries for actors that were not already in memory._

[![Figure 3 from StreamPETR, showing object queries propagated through a temporal memory queue](/assets/images/streampetr-paper-figure-3.png)](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR transforms selected object queries into the current frame, updates them from current images, and keeps the strongest foreground queries for the next step. Query selection saves memory but can discard weak evidence before an actor is confidently detected. Source: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

[SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) keeps sparse object support but retrieves from several stored frames, so its cost still grows with history length. “Sparse temporal” can therefore mean compressing history into recurrent state or retaining history and reading only selected locations; the two designs have different memory, latency, and error-accumulation behavior. StreamPETR remembers objects; SOLOFusion remembers the scene.

Sparsity should be reported per component. A sparse LiDAR backbone avoids empty voxels; a sparse camera decoder avoids a full BEV field; a sparse temporal model avoids replaying every frame. None of them removes the dense surround-camera backbone. Sparse operators also pay for indexing, sorting, padding, gathers, and irregular memory access. FLOPs are therefore insufficient: the relevant deployment measurements are component latency, peak memory, P95/P99 end-to-end latency, active-token count in crowded scenes, and recall when the query budget saturates.

## Training contracts shape the deployed model

The runtime sensor graph is not always the training sensor graph. In [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), projected LiDAR supervises camera depth and disappears at inference. In [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), sparse depth remains a deployed input. In [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html), a camera-LiDAR teacher transfers features, relations, and outputs to a camera-radar student. These systems should not all be described as “using LiDAR”; they have different runtime contracts.

[![Animation separating LiDAR depth labels, runtime sparse-depth input, and a LiDAR-camera teacher](/assets/images/autonomous-perception-lidar-training-contracts.gif)](/assets/images/autonomous-perception-lidar-training-contracts.gif)
_LiDAR supplies labels to BEVDepth, remains a runtime input for Sparse-to-Dense, and belongs to the training-only teacher in CRKD. The deployed sensor contract differs in every column._

The same distinction applies to pretraining. Image classification teaches appearance but not calibration, metric depth, ego motion, or persistence. [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) reconstructs masked camera and LiDAR inputs through a shared 3D volume; [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) reconstructs masked LiDAR columns. [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html), [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html), and [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) add future occupancy, point-cloud prediction, or dynamic state. These targets are useful when they improve transfer across tasks and label budgets; a longer forecast is not automatically better because distant futures are increasingly ambiguous.

The same BEV feature may feed detection, occupancy, lanes, velocity, tracking, and planning heads. Sharing saves repeated sensor encoding and lets tasks exchange scene context, but their losses have different units, label densities, and learning speeds. Normalize each loss by a meaningful count before adjusting task weights. If one task still dominates shared layers, [uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) or [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) can change gradient magnitude; if task gradients point in opposing directions, [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html), adapters, or an earlier architectural split address a different problem. Multi-task learning is useful here, but it is an optimization constraint on the shared representation rather than a separate perception architecture.

[![Animation comparing loss-scale weighting, GradNorm's training-rate targets, and PCGrad's projection of conflicting gradients](/assets/images/autonomous-perception-multitask-gradients.gif)](/assets/images/autonomous-perception-multitask-gradients.gif)
_Loss weighting changes gradient magnitude; GradNorm adjusts weights using relative training rates; PCGrad changes direction when task gradients conflict. These methods solve different problems._

## Beyond BEV and queries: a compressed world state

Dense BEV memory preserves a great deal of spatial structure, while sparse queries preserve selected actors. A natural next question is whether the world can be compressed into a smaller learned state without collapsing it into one undifferentiated vector.

[![Animation comparing dense BEV memory, sparse object queries, learned latent tokens, and a single pooled embedding](/assets/images/autonomous-perception-latent-memory.gif)](/assets/images/autonomous-perception-latent-memory.gif)
_The compression ladder keeps the scene fixed while changing what persists: every BEV cell, selected actors, a small learned token set, or one pooled vector. The further right the representation moves, the more the model must learn which spatial evidence is worth retaining._

Let $X_t$ contain the current sensor features and let $Z_{t-1}$ be a compact set of learned latent tokens. A recurrent world state could be updated as

$$
Z_t = \operatorname{CrossAttention}(Z_{t-1}, X_t).
$$

The tokens are a bottleneck: the model must learn which parts of the observation are worth carrying forward for perception, forecasting, or planning. This is related to Perceiver-style bottlenecks, token compression, memory tokens, and latent world models. The benefit is compute that scales with the number of latents rather than every BEV cell or every stored object query.

Mean-pooling the entire map into one embedding is usually too aggressive for driving. One vector would need to preserve a pedestrian on the left, a stop sign 30 meters ahead, the curvature of the lane, a cyclist approaching from behind, an occluded vehicle, and the free-space topology. Mean pooling removes explicit spatial separation and is better suited to “what kind of scene is this?” than “where exactly is the pedestrian relative to the ego vehicle?”

A more realistic compression strategy is a small set of latent tokens whose specializations emerge through learning. Some may preserve nearby dynamic actors, road topology, far-field hazards, or route context, but those roles do not need to be hard-coded. The design space is a compression ladder:

dense BEV memory → sparse object memory → learned latent memory → single pooled embedding

Moving right saves memory and compute, but also increases the burden on the learning objective to decide what information matters. The central architectural question returns here in its most compressed form: what is discarded, at what stage, and can a downstream task recover it?

## Driving foundation models

The research direction is moving from separate perception heads toward models that share sensor representations across perception, prediction, simulation, and planning. Masked multimodal pretraining, future occupancy prediction, driving VLMs, and world models are separate steps toward that architecture. They differ in their training targets, but all try to reuse more of the geometric and temporal state learned from driving data.

Waymo has described one version of this system-level design. In [Waymo Co-CEO Dmitri Dolgov's talk, “The Demo Is Only 1% Of The Work”](https://www.youtube.com/watch?v=Gp4zrV3-6N8), the Waymo Foundation Model contains a Sensor Fusion Encoder, a Driving VLM, and a Generative World Decoder. The diagram routes multimodal sensor evidence into a shared world representation and then into actions and predictions. It should be read as an architecture overview; it does not establish that every displayed block runs unchanged in the deployed online controller.

![The Waymo Foundation Model diagram with sensor fusion, a driving VLM, and a generative world decoder](/assets/images/waymo-foundation-model-architecture.png)
_Waymo's diagram separates a fast sensor-fusion path from a slower language-conditioned path, then joins both inside a generative world decoder. Source: [Dmitri Dolgov's Waymo talk](https://www.youtube.com/watch?v=Gp4zrV3-6N8); see Waymo's public [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) and earlier [foundation-model overview](https://waymo.com/blog/2024/10/ai-and-ml-at-waymo/)._

The three blocks have different roles:

| Path | Input → intermediate representation | Output role |
| --- | --- | --- |
| Sensor Fusion Encoder | Camera, LiDAR, radar → objects and sensor embeddings | Fast, metric evidence and reactions |
| Driving VLM | Text prompts, sensor data, and autonomous-driving history → VLM embeddings, semantics, rationales, and text tokens | Slower semantic reasoning for rare or complex situations |
| Generative World Decoder | Both representations → a shared world model | Driving actions, agent predictions, and other predictions |

The fast path still performs the work developed throughout this article: preserve each measurement, align it in metric space, and update scene state at driving frequency. The VLM adds language-conditioned semantics and broader learned context on a slower path. The world decoder then combines those representations before producing actions or predictions. This separation matters because semantic reasoning can be useful without placing a large language model directly in the latency-critical sensor loop.

A generative world decoder also changes the learning problem. Instead of predicting only what occupies the scene now, it can model how agents and the environment may evolve, generate candidate trajectories, and produce signals used to evaluate them. Waymo's public description says its World Decoder predicts road-user behavior, generates maps and vehicle trajectories, and provides trajectory-validation signals. It also describes adapting large teacher models to the Driver, Simulator, and Critic before distilling smaller students. Those statements explain the slide's multi-stage training, but they do not reveal the exact online graph or demonstrate closed-loop safety by themselves.

The foundation-model label does not remove the engineering boundaries established earlier. Sensor encoders still have to preserve modality-specific evidence. Calibration and synchronization still determine whether features refer to the same place and time. Temporal state still needs freshness, birth, and reset rules. [Vision-Language Models: A Reading Guide](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) covers the adjacent progression from image-text alignment to grounding, video, and action; the driving problem adds metric geometry, real-time recurrence, and closed-loop validation.

The hard question is whether a shared decoder can use slower semantic reasoning without losing contact with calibrated sensor evidence, represent several plausible futures without collapsing them into one forecast, and still meet the latency and validation requirements of closed-loop driving. Progress toward a driving foundation model will depend on the same concrete decisions as the perception stack beneath it: what each encoder preserves, where geometry is established, which evidence is allowed to fuse, and what state is trusted at the next frame.
