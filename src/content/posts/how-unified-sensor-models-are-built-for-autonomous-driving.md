---
title: 'Autonomous-Vehicle Perception, circa 2026'
date: '2026-07-31T18:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: how-unified-sensor-models-are-built-for-autonomous-driving
legacyPath: /blog/2026/07/31/how-unified-sensor-models-are-built-for-autonomous-driving.html
tags:
  - Research
  - Autonomous Driving
  - Sensor Fusion
topics:
  - autonomy
  - multimodal
summary: How modern autonomous-driving systems preserve sensor-specific evidence, establish metric geometry, fuse modalities, carry world state through time, and connect perception to planning, simulation, and validation.
---
# Autonomous-Vehicle Perception, circa 2026

The perception task on an autonomous vehicle turns raw camera, LiDAR, radar, and other sensor measurements into a world state that the rest of autonomy can use. Calibration fixes the geometric relationship among those sensors. Fusion then has to combine their partial evidence into one coherent representation of the road, actors, free space, and motion. And it all has to run in real time. It is an incredibly hard task.

That clean description breaks as soon as evidence conflicts, an actor is occluded, timestamps drift, or one sensor degrades. A distant cyclist at dusk may appear as a few image pixels, two or three LiDAR returns, and a noisy radar detection with radial velocity. None of those measurements is the cyclist. Each is partial evidence with a different sampling pattern, uncertainty, and failure mode.

What still makes this problem interesting to me is that more unification can make the model worse. The tempting definition of a unified sensor model is one that converts every sensor into one tensor as early as possible. That is usually the wrong objective.

Early unification can erase the very signal that made a sensor useful: image texture, LiDAR height structure, radar Doppler, measurement age, or sensor-specific confidence. Once that information has been averaged away, a larger downstream model cannot reconstruct it.

The real architectural problem is deciding where each measurement becomes geometry, where different modalities are allowed to interact, what state survives through time, and which parts of that state are made explicit for prediction, planning, simulation, and validation. The path I keep coming back to is to preserve native evidence, establish metric support, let modalities interact, carry the resulting world state through time, and expose both structured and latent outputs before prediction and planning act.

<div class="compact-flow-diagram"><a href="/assets/images/perception-evidence-to-planning.svg"><img src="/assets/images/perception-evidence-to-planning.svg" alt="Compact six-stage perception path from sensor-native evidence to calibrated metric support, cross-modal interaction, temporal state, structured and latent outputs, and finally prediction and planning"></a></div>
_The stages name information obligations, not mandatory modules. A model can merge implementations, but it still has to preserve, align, interact, persist, expose, and act under a deadline._

This gives a stricter meaning to *unified*. A system can be unified along several independent axes:

| Axis of unification | What becomes shared | What can remain specialized |
| --- | --- | --- |
| Sensors | A vehicle-centered geometric support and downstream state | Native encoders, uncertainty models, sampling patterns, and health signals |
| Time | A recurrent world state | Update, aging, birth, deletion, and reset rules for different state elements |
| Tasks | Sensor computation and scene context | Task heads, losses, label spaces, and validation contracts |
| System | A common model family or world-state vocabulary across the Driver, Simulator, and Critic | Model size, execution frequency, supervision, and deployment constraints |

These axes should not be collapsed into one claim. Sharing a BEV backbone across detection and mapping does not imply that camera and radar should share an encoder. Backpropagating through perception and planning does not imply that every learned intermediate should remain opaque at runtime. Using one foundation-model family across driving and simulation does not imply that the same graph runs onboard and offline.

The central thesis of this article is simple: **preserve sensor-native evidence until geometry makes interaction meaningful, then preserve enough structured and latent state to make the next decision both capable and testable.**

This is a design map rather than an exhaustive paper catalog. It focuses on onboard camera, LiDAR, and radar, plus the learned world state that connects perception to prediction and planning. Localization, control, V2X, and sensor hardware are outside the main scope except where they constrain that state. Sources were checked through August 2026.

## The system contract: preserve, align, interact, persist, materialize
The runtime path begins with encoders matched to each measurement process. Intrinsics describe how a camera maps rays to pixels. Extrinsics describe where each sensor sits relative to the vehicle. Timestamps and ego poses place measurements at a common time. These transformations let the model ask whether an image edge, a LiDAR return, and a radar detection could refer to the same physical support.

Calibration does not add information. It creates a correspondence rule. If that rule is wrong, fusion produces structured errors rather than independent noise. A vehicle can inherit image evidence from an adjacent lane, a lane boundary can shift across the BEV grid, and a small angular error can become a large lateral displacement at long range. Timestamp error has a similar effect for moving actors: two correct measurements can disagree because they describe different moments.

After alignment, the model needs a scene state. A dense BEV field gives each ground-plane location a persistent cell and naturally supports occupancy, free space, lanes, and maps. Sparse queries allocate state to selected actors or map elements and naturally support detection, tracking, motion prediction, and vectorized road structure. Learned latent tokens compress the scene more aggressively and let the model decide what information is worth preserving. Most practical systems will mix these forms rather than choose one globally.

[![Autonomous-driving perception pipeline from sensor-specific encoders through calibrated metric representations, dense or sparse temporal state, and task heads, with learned latent tokens and a separate training-only graph](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_The runtime path moves from sensor-specific features to calibrated metric support, then into a dense field, sparse query set, or learned latent state. The lower path shows supervision and teacher models that can exist during training without becoming deployed sensor inputs._

“Fusion” is often used for three different operations:

1. **Alignment** establishes which measurements could refer to the same place and time.
2. **Interaction** determines how evidence from one modality changes features from another.
3. **Materialization** determines what shared state downstream tasks are allowed to consume.

Separating these operations clarifies many architecture comparisons. Two models may use the same BEV coordinates but perform very different interaction. A model may fuse features deeply while still materializing separate object, occupancy, and roadgraph outputs. Conversely, a model may concatenate aligned tensors once and call the result unified, even though one stream dominates and the others contribute little.

Shared coordinates also do not imply shared certainty. A fused feature should retain, explicitly or implicitly, which sensors support it, how old that support is, and whether the relevant sensor is degraded. A high-confidence prediction resting on one stale or corrupted stream is not equivalent to the same prediction supported by current camera, LiDAR, and radar evidence.

## Sensor encoders preserve different evidence
The first step is not fusion. It is choosing an encoder that does not erase the measurement that makes a modality useful.

### Cameras: dense semantics at several scales
Cameras provide the richest appearance signal in the stack. They distinguish lane paint from a crack in the road, read lights and signs, recognize unusual objects, and preserve boundaries that may occupy only a handful of pixels. Their weakness is metric ambiguity: a pixel identifies a ray through the camera center, not a distance along that ray.

Camera encoding must therefore retain both semantics and spatial detail. Convolutional backbones build local features with translation-equivariant kernels and map well to optimized inference libraries. Production-oriented systems such as [NVAutoNet](/paper%20shorts/2023/03/23/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving.html) use efficient CNNs for image and BEV processing. Vision transformers use content-dependent attention to connect distant regions, which can help when recognition depends on broader context, but global attention at full surround-camera resolution is expensive. Large pretrained image models improve appearance features; they do not remove the need for calibrated 3D reasoning.

Feature pyramids remain important because apparent object size changes sharply with range. High-resolution levels preserve a distant pedestrian, traffic light, or narrow lane marking. Lower-resolution levels aggregate larger context and are cheaper for vehicles, road layout, and scene semantics. [EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) is a clear 2D example of learned multiscale fusion.

[BEVFormer v2](/paper%20shorts/2022/11/18/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition.html) exposes a separate optimization issue. A loss applied only after BEV conversion gives the image backbone a weak and indirect training signal, so perspective-view supervision can materially improve the features before they enter 3D.

For a 3D point $X$, camera $i$ uses intrinsics $K_i$ and extrinsics $(R_i, t_i)$ to compute

$$
u_i = \pi(K_i, R_i, t_i, X).
$$

The same point appears at a different pixel in each camera. If pyramid level $l$ has stride $s_l$, its feature is sampled at $u_i/s_l$. Projection can retrieve a fine boundary only if the backbone retained that boundary, and it can retrieve broad context only if the receptive field encoded it.

[![Animation showing one metric 3D point projected to different pixels in two calibrated cameras and sampled at stride-adjusted feature-pyramid coordinates](/assets/images/autonomous-perception-vision-encoder.gif)](/assets/images/autonomous-perception-vision-encoder.gif)
_A single 3D reference point lands at different pixels in different cameras and at different coordinates on each pyramid level. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) applies this operation to object queries; [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) applies it to BEV queries._

The camera encoder is strongest where appearance matters. Its main 3D weakness is not missing semantics but uncertain depth. More image resolution or a larger backbone can improve recognition without resolving where along a ray the evidence belongs.

### LiDAR: sparse geometry before dense BEV
A LiDAR sweep is already metric, but it is irregular, sparse, and strongly range-dependent. Point encoders preserve individual returns but make neighborhood construction expensive. Pillar encoders group points into vertical columns and collapse height early. Voxel encoders retain a 3D grid for longer, preserving overpasses, trucks, poles, and other height-dependent structure. The main encoder decision is where to trade 3D detail for the speed and regularity of a 2D BEV backbone.

[PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) makes that trade early: it pools each occupied pillar, scatters the result into a dense pseudo-image, and performs nearly all later computation with 2D convolutions. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) keeps sparse 3D voxels through a middle encoder and densifies only after height has been compressed.

Sparse convolution is effective because it shares local kernels over occupied cells, but coordinate-map construction and irregular memory access are real system costs. Sparse attention changes how occupied cells communicate. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) partitions variable-density voxels into bounded local sets, applies attention inside each set, and rotates the partition between layers so information crosses set boundaries. Attention can connect non-adjacent occupied cells more directly than a small convolutional kernel, while bounded sets keep the workload controlled. Sorting, padding, token density, and the eventual height compression still determine latency.

[![Animation comparing early densification in PointPillars, late BEV densification in SECOND and DSVT, and box prediction from active voxels in VoxelNeXt](/assets/images/autonomous-perception-lidar-encoder.gif)](/assets/images/autonomous-perception-lidar-encoder.gif)
_PointPillars compresses height before its dense 2D backbone. SECOND and DSVT retain sparse 3D cells for longer, then construct BEV. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) keeps the active-voxel representation through the prediction head._

The right encoder depends on the operating envelope. Pillars are a strong latency baseline on mostly planar roads. Sparse voxels are preferable when vertical separation, long range, or dense 3D structure matters. Fully sparse heads save BEV work when occupied cells remain rare, but a large FLOP reduction may yield a modest wall-clock gain on hardware with weak sparse-kernel support.

Intensity, return type, timestamp, and point age should remain attached to the geometry. LiDAR range is direct, but a rotating scan is not an instantaneous snapshot. A point measured near the beginning of a sweep may already be stale by the time the full scan is processed.

### Radar: range and motion under uncertainty
Radar should be modeled around the measurements that distinguish it from LiDAR. A return may contain range, azimuth, elevation, radial velocity, Radar Cross Section (RCS), timestamp, and a sensor-specific confidence estimate. Doppler can reveal a moving actor before image-only temporal inference has enough baseline. Millimeter-wave sensing is also insensitive to darkness and is often more tolerant than cameras or LiDAR under rain and fog.

The cost is weak angular localization, sparse semantic evidence, multipath, and ghost returns. Radar supplies useful constraints, not clean 3D boxes.

Three encoder families correspond to three fusion strategies. A point encoder keeps individual returns and is useful when association happens around an object proposal. A polar or range-azimuth tensor preserves the native sampling pattern and can exploit local radar structure before Cartesian resampling. A radar-BEV encoder creates an independent metric feature map for later alignment with camera or LiDAR BEV.

Rasterizing too early into binary occupancy removes Doppler, RCS, confidence, and return-level ambiguity. Those are precisely the signals needed to distinguish motion from clutter.

[![Animation comparing camera-radar interaction at the proposal, depth, and BEV stages in CRAFT, CRN, and RCBEVDet](/assets/images/autonomous-perception-radar-encoder.gif)](/assets/images/autonomous-perception-radar-encoder.gif)
_[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates a soft set of radar returns with each camera proposal. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to refine camera depth before lifting and aligns both BEV maps with deformable attention. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) combines point and transformer radar paths before BEV fusion._

Recent camera-radar systems improve less by treating radar as an extra image channel than by deciding where range and Doppler should change the computation. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) found that retaining radar metadata, disabling an aggressive outlier filter, and accumulating aligned sweeps all affected performance. [Doppler-Aware LiDAR–Radar Fusion](/paper%20shorts/2025/10/23/doppler-aware-lidar-radar-fusion-for-weather-robust-3d-detection.html) processes radar power and Doppler as distinct signals during multimodal interaction. [DinoRADE](/paper%20shorts/2026/04/09/dinorade-full-spectral-radar-camera-fusion.html) combines dense radar tensors with DINOv3 image features for adverse-weather detection.

A useful radar comparison should report adverse-weather and long-range accuracy together with false associations and velocity error. Higher average detection is not enough if a ghost return is attached to the wrong actor.

The encoders now preserve the right evidence. The next question is where that evidence acquires metric support.

## Where metric geometry enters
There is no geometry-free sensor fusion. A model can hide geometry inside attention, depth logits, positional encodings, or learned correspondences, but camera evidence must eventually be assigned to physical support before it can interact coherently with LiDAR, radar, maps, or temporal state.

The main camera-to-3D families differ in where the metric hypothesis begins.

### Push image evidence into 3D: LSS and BEVDepth
A pixel fixes a ray through the camera center, not a distance along that ray. [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a categorical depth distribution for each image location, copies the image feature along the candidate depth bins, and pools those lifted features into BEV.

<div class="compact-flow-diagram"><a href="/assets/images/camera-lift-to-bev.svg"><img src="/assets/images/camera-lift-to-bev.svg" alt="Compact diagram of an image pixel becoming a camera ray, depth distribution, 3D feature cloud, and BEV grid"></a></div>
_LSS pushes image evidence into metric space. The depth distribution decides where along the ray that evidence can land; pooling then writes it into the vehicle-centered BEV grid._

The depth distribution in the original LSS is latent. It is not directly supervised; downstream BEV losses push it toward geometry that helps detection and other tasks. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) adds projected LiDAR depth supervision, so the camera branch receives both a direct geometric signal and a downstream task signal.

LiDAR can therefore be present during training and disappear at inference. That is learning with privileged information. It becomes teacher-student or cross-modal distillation only when a separate teacher transfers knowledge to a student.

Forward lifting preserves dense image evidence and is a natural fit for occupancy and maps. Its failure is equally direct: a depth error writes the feature into the wrong metric cell.

![Figure 1 from Lift, Splat, Shoot, showing multiview evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_Lift, Splat, Shoot predicts depth along every camera ray and pools the lifted features into a vehicle-centered BEV grid. Source: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

### Pull image evidence from metric space
Pull-based methods begin with a hypothesis in physical space and ask the images for supporting evidence. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) projects each 3D voxel into the cameras and bilinearly samples visible image features. Its controlled experiments found that input resolution and effective batch size changed vehicle-segmentation IoU more than the lifting operator in that setup. This is a useful warning: before attributing a benchmark gain to a new view transformer, match the backbone, image resolution, training schedule, batch size, and sensor inputs.

The metric hypothesis can also be sparse.

[DETR](/paper%20shorts/2020/05/26/end-to-end-object-detection-with-transformers.html) introduced object queries for 2D detection: a bounded set of learned vectors asks whether an object should be represented, and a transformer decoder turns those queries into a set of predictions. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) attaches each query to a 3D reference point, projects that point into every camera, and samples image evidence around the projections.

The difference from dense lifting is where the model spends its geometric budget. LSS represents every image location across candidate depth bins. DETR3D asks only about a bounded set of candidate objects. This can be efficient for detection, but it creates a recall dependency: if no query acquires support for an actor, later refinement cannot recover evidence that was never represented.

[PETR](/paper%20shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html) moves more geometry into the image features. Multi-view features otherwise know appearance and camera identity, but not directly which region of physical 3D space they describe. PETR injects 3D positional information into those features, allowing attention to reason over image evidence with explicit geometric context. A useful shorthand is that DETR3D places much of the geometry in the query and reference-point mechanism, while PETR makes the features themselves more geometry-aware.

[BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) changes the unit from an object query to a BEV query. A learned token for each bird's-eye-view location samples image evidence at several heights along the corresponding vertical pillar. This creates a dense spatial field without explicitly predicting a depth distribution for every pixel. The price is that BEV cells, cameras, heights, and attention samples can dominate computation.

[![Animation comparing depth along an image ray in LSS, a 3D object-query reference point in DETR3D, and vertical reference points from a BEV cell in BEVFormer](/assets/images/autonomous-perception-camera-lifting.gif)](/assets/images/autonomous-perception-camera-lifting.gif)
_The highlighted variable is the source of metric support: depth along an image ray in LSS, a 3D object reference point in DETR3D, or several heights above a fixed BEV cell in BEVFormer._

| 3D construction | Where the metric hypothesis begins | Best fit | Main cost | Characteristic failure |
| --- | --- | --- | --- | --- |
| Depth lift and splat | Every image location predicts depth | Dense occupancy, maps, and detection | Pixels × depth bins, followed by BEV processing | Wrong depth moves evidence to the wrong metric cell |
| Voxel-to-image sampling | Every metric voxel projects into visible cameras | Simple dense BEV baselines | Voxels × cameras | A sampled pixel may mix depths or lie behind an occluder |
| Object queries | A bounded set of 3D actor hypotheses | Detection, tracking, and motion | Queries × views × samples | An actor is missed when no query acquires support |
| BEV queries | Every ground-plane cell samples several heights | Dense scene state with selective image retrieval | BEV cells × heights × views | Weak projected support creates false or empty cells |

No row dominates every task. Dense methods say “represent the world.” Object queries say “represent candidate actors.” The useful comparison is not which abstraction is newer, but what evidence each one discards and whether a later module has any path to recover it.

## Fusion is alignment, interaction, and routing
Once camera, LiDAR, and radar features have metric support, the model must decide how they interact. Early fusion combines near-raw inputs, but pixels, points, and Doppler returns do not naturally share a sampling pattern. Late fusion combines independent predictions, which is modular and easy to validate but gives up feature-level complementarity. Intermediate fusion lets each modality preserve its own evidence first, then interact after geometric alignment.

For heterogeneous driving sensors, intermediate fusion is the most useful default. It is not one architecture. The decisive choice is the granularity at which sensors meet.

### Point, query, and dense-field fusion
[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) attaches camera class scores to LiDAR points before voxelization. This is cheap and geometrically direct, but image evidence survives only where LiDAR produced a return. Empty space and camera-only observations disappear from the shared representation.

[FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) samples camera, LiDAR, and radar features around the same 3D object reference point. This matches an object-centric output and spends computation selectively, but it does not retain a complete background field.

[BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) gives each modality an appropriate encoder, transforms the resulting features into a shared BEV grid, and fuses there. Camera images become camera BEV through an image encoder; LiDAR points become LiDAR BEV through a point or voxel encoder. Only then do the aligned representations meet before the task heads.

<div class="compact-flow-diagram"><a href="/assets/images/bevfusion-data-paths.svg"><img src="/assets/images/bevfusion-data-paths.svg" alt="Compact BEVFusion diagram with separate camera and LiDAR encoder paths meeting only after both streams reach aligned BEV coordinates"></a></div>
_The useful unification happens at the metric interface. Each encoder keeps its sensor's native evidence until camera and LiDAR features refer to comparable physical support._

BEV is a natural meeting room because a camera feature at $(x,y)$ and a LiDAR feature at $(x,y)$ refer to approximately the same physical support. The model does not need to make the camera behave like LiDAR or LiDAR behave like a camera. It preserves their inductive biases until physical alignment makes interaction meaningful.

[![Animation showing the same actor and lane evidence entering point, object-query, and dense-BEV fusion](/assets/images/autonomous-perception-fusion-granularity.gif)](/assets/images/autonomous-perception-fusion-granularity.gif)
_Point fusion retains camera features only at measured points. Query fusion gathers evidence around selected actors. Dense BEV fusion keeps actor evidence together with the surrounding lane, occupancy, and free-space field._

### Proposal recall is an architectural ceiling
Proposal-conditioned fusion spends compute selectively. [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) uses LiDAR proposals to retrieve image evidence around candidate objects instead of trusting one calibrated pixel. This can tolerate small correspondence errors, but it exposes a structural failure mode: if LiDAR misses an actor that the camera sees, a LiDAR-controlled proposal stage may never give the camera evidence a chance to recover it.

[![Animation comparing LiDAR-controlled proposals, merged proposals from multiple modalities, and shared-BEV fusion before detection](/assets/images/autonomous-perception-proposal-recall.gif)](/assets/images/autonomous-perception-proposal-recall.gif)
_Proposal-conditioned fusion is selective, but proposal recall can become a ceiling. Multi-proposal fusion restores a recovery path at the cost of matching and deduplication; shared-BEV fusion preserves both modality fields before detection._

One response is to let each modality generate proposals and merge them before refinement. That removes the single-modality recall ceiling, but introduces duplicates, conflicting confidence, inconsistent localization, and cross-modal matching. Dense shared fusion makes the opposite choice: preserve both modality fields before generating detections. It spends more computation but keeps a recovery path when one sensor misses an actor.

This creates a general diagnostic question for any fusion architecture: **which modality controls admission into the shared representation?** If the answer is one sensor's proposals, points, or confidence threshold, then that sensor's recall becomes a ceiling unless another path explicitly bypasses it.

In multimodal fusion, the hidden bottleneck is often admission rather than attention. The sensor that creates the proposals, points, or thresholds decides which evidence becomes eligible for downstream computation; without a bypass, its recall becomes the system's ceiling.

### Interaction can occur before either stream is finished
A simple architecture runs a camera encoder and a LiDAR encoder to completion, then applies one fusion block. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) lets the streams update one another across representation stages. [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) applies shared transformer blocks after modality-specific tokenization.

The important distinction is not merely that cross-attention exists. It is whether one modality can change what another stream chooses to preserve before feature extraction is complete. Repeated interaction can improve complementarity, but it also makes failure attribution harder. A corrupted stream may contaminate several layers instead of one terminal block, so deeper fusion increases the need for health-aware routing and modality-specific diagnostics.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion keeps camera and LiDAR encoding separate until both modalities occupy the same BEV grid, then shares that grid across detection and map heads. Source: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

### Missing, degraded, and misaligned sensors
A fusion network trained only with all sensors often becomes dependent on its strongest stream. Zeroing LiDAR at inference does not turn such a network into a competent camera-only model.

[UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) demonstrates the failure directly. In its reported ablation, a model trained only in fused mode collapses under camera-only inference, while modality dropout and normalization over the streams that remain produce a usable fallback path. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) trains full, camera-only, and LiDAR-only modes and lets BEV queries attend to whichever encoders are available.

[![Animation contrasting modality availability in UniBEV and MetaBEV with reliability gating in Grace-BEV](/assets/images/autonomous-perception-modality-dropout.gif)](/assets/images/autonomous-perception-modality-dropout.gif)
_Modality dropout covers discrete cases in which a stream is absent. Reliability gating covers the harder case in which a tensor is present but its evidence should receive less weight._

Sensor absence is only one failure mode. Blur, saturation, fog, reduced LiDAR beams, blocked fields of view, packet delay, interference, and calibration drift all leave a tensor present but unreliable. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) adds reliability-aware gating, while [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) evaluates several corruptions in addition to missing streams.

A production model needs at least three mechanisms:

- modality dropout for supported sensor configurations,
- corruption training and observable health signals for partial degradation,
- calibration and timing perturbations as separate tests.

The third category matters because miscalibration can make every projected association confidently wrong in the same direction. It is not equivalent to random feature noise.

A tensor being available is not the same as its evidence being trustworthy. Reliability should therefore survive fusion. At minimum, downstream state should preserve modality support, timestamp or age, and a health estimate. Otherwise, the network may emit a confident fused prediction without exposing that it rests entirely on one degraded stream.

## Time: what should survive the next frame?
A single camera frame or LiDAR sweep does not directly provide a complete motion state, preserve evidence through an occlusion, or stabilize an uncertain depth estimate. Radar supplies radial velocity, but not the full velocity of every actor. Temporal modeling fills the gap by comparing evidence across time.

Before old state can be reused, it must be expressed in the current coordinate frame. Ego-motion compensation aligns roads and static structures. Independently moving actors require velocity, motion hypotheses, or learned updates. Timestamp error, pose error, rolling shutter, and scan motion appear as residual displacement after alignment.

A useful abstraction is

$$
S_t = f\left(\operatorname{Align}(S_{t-1}, \Delta T_t), X_t, \Delta t, H_t\right),
$$

where $S_{t-1}$ is prior scene state, $X_t$ is current sensor evidence, $\Delta T_t$ is the ego-frame transformation, $\Delta t$ is elapsed time, and $H_t$ contains sensor-health information. The equation is simple; the hard choice is what $S_t$ contains.

### Dense scene memory
Dense temporal memory stores a scene-level BEV field. [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) warps the previous BEV feature into the current ego frame, concatenates it with the current feature, and lets a BEV encoder learn displacement cues. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) uses temporal attention to update a recurrent BEV representation.

[SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) separates temporal resolution: a short high-resolution history supports fine stereo correspondence, while a longer low-resolution BEV history supports depth and velocity. Dense state retains roads, free space, and weak background evidence that may later support a new detection. Its memory and warp cost scale with BEV area and history, and stale background evidence can persist if updates are too conservative.

### Sparse entity memory
Sparse temporal memory stores selected actors or queries. [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) retains a bounded queue of foreground queries, conditions them on ego pose, elapsed time, and velocity, and introduces fresh queries for newly visible actors. [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) transforms prior instances into the current frame and combines them with fresh anchors.

Sparse memory is lighter and naturally object-centric, but query birth, duplicate removal, aging, and deletion become part of the learned system. It can discard free space, road structure, undetected actors, and context that has not yet become a confident query.

[![Animation comparing a warped dense BEV field, transformed recurrent instances with fresh anchors, and a bounded foreground-query queue](/assets/images/autonomous-perception-temporal-memory.gif)](/assets/images/autonomous-perception-temporal-memory.gif)
_Dense recurrence carries every BEV cell. Sparse4D v2 transforms recurrent object instances and adds fresh anchors. StreamPETR carries a bounded foreground-query queue and introduces new queries for actors not already in memory._

[![Figure 3 from StreamPETR, showing object queries propagated through a temporal memory queue](/assets/images/streampetr-paper-figure-3.png)](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR transforms selected object queries into the current frame, updates them from current images, and keeps the strongest foreground queries for the next step. Query selection saves memory but can discard weak evidence before an actor is confidently detected. Source: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

[SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) keeps sparse object support but retrieves from several stored frames, so its cost still grows with history. “Sparse temporal” can mean either compressing history into recurrent state or retaining history and reading selected locations. Those designs have different memory, latency, and error-accumulation behavior. StreamPETR remembers selected objects; SOLOFusion remembers a scene field.

Dense and sparse memory fail in opposite directions. Dense state preserves weak evidence but can carry clutter and stale features. Sparse state limits cost but may delete evidence before it becomes important. A practical hybrid can keep a lower-resolution scene field for occupancy and topology, high-resolution queries for actors and map elements, and explicit age or confidence for both.

Sparsity should be reported per component. A sparse LiDAR backbone avoids empty voxels. A sparse camera decoder avoids a full BEV field. A sparse temporal model avoids replaying every location in every frame. None removes the dense surround-camera backbone, and sparse operators still pay for indexing, sorting, padding, gathers, and irregular memory access.

FLOPs are therefore insufficient. Deployment evaluation should include component latency, peak memory, P95 and P99 end-to-end latency, active-token count in crowded scenes, recall when the query budget saturates, and error after long occlusions or ego-pose drift.

## The world state is not one tensor
Dense BEV fields and sparse object queries are often presented as competing philosophies. They are better understood as different points on a compression ladder. The state can preserve every spatial cell, selected entities, a small set of learned latent tokens, or one pooled scene vector.

[![Animation comparing dense BEV memory, sparse object queries, learned latent tokens, and a single pooled embedding](/assets/images/autonomous-perception-latent-memory.gif)](/assets/images/autonomous-perception-latent-memory.gif)
_The scene remains fixed while the representation changes: every BEV cell, selected actors, a compact learned token set, or one pooled vector. Moving right saves computation but asks the learning objective to decide which spatial evidence can be discarded._

Let $X_t$ contain current sensor features and let $Z_{t-1}$ be a compact set of learned latent tokens. A recurrent latent state could be updated as

$$
Z_t = \operatorname{CrossAttention}(Z_{t-1}, X_t).
$$

The latent tokens form an information bottleneck. Compute scales with the number of retained tokens rather than every BEV cell or every stored observation. This is related to Perceiver-style bottlenecks, memory tokens, token compression, and latent world models.

Mean-pooling the entire map into one embedding is usually too aggressive for driving. One vector would need to preserve a pedestrian on the left, a stop sign ahead, lane curvature, a cyclist approaching from behind, an occluded vehicle, and free-space topology. Mean pooling is better suited to “what kind of scene is this?” than “where is the pedestrian relative to the ego vehicle?”

A small token set is more plausible because different tokens can specialize through learning. [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) explores this direction for end-to-end driving: camera-aware register tokens compress multi-camera features into a compact scene representation, then lightweight decoders generate and score candidate trajectories. The result is evidence that targeted token compression can support planning without carrying every camera token downstream. It is not evidence that all metric structure should disappear.

A 2025 preprint, [UniLION](/paper%20shorts/2025/11/03/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns.html), pushes unification into the backbone itself. Its linear group RNN supports LiDAR-only, temporal LiDAR, multimodal, and multimodal-temporal variants across perception, prediction, and planning tasks. The interesting claim is not that one operator has solved the stack, but that sensor, time, and task unification can be treated as one sequence-modeling problem. Whether one architecture remains optimal across hardware, calibration, and failure constraints is still empirical.

The compression ladder runs from a dense scene field, through sparse structured entities and learned latent registers, to one pooled embedding.

<div class="compact-flow-diagram"><a href="/assets/images/world-state-compression-ladder.svg"><img src="/assets/images/world-state-compression-ladder.svg" alt="Compact compression ladder from dense scene fields to sparse entities, learned latent registers, and a single pooled embedding"></a></div>
_Each move to the right saves memory and compute while asking the learning objective to preserve more information in fewer carriers._

Moving right saves memory and compute. It also increases the burden on the objective to preserve the right information. The design question is still the same: what is discarded, when is it discarded, and can any downstream component recover it?

### Materialized structure and latent state should coexist
Compression alone does not decide what the rest of the system can inspect. A driving model can expose several complementary views of the same learned world state:

| Representation | What it preserves well | What it tends to miss | Natural consumers |
| --- | --- | --- | --- |
| Objects and tracks | Dynamic agents, identity, kinematics, compact interaction state | Unmodeled geometry, unusual objects, amorphous hazards | Prediction, interaction modeling, behavior planning, validation |
| Vector roadgraph and map elements | Lane boundaries, connectivity, stop lines, route topology | Non-map obstacles and uncertain geometry | Routing, rule reasoning, planning, simulation |
| Occupancy and free space | Detailed geometry, unknown obstacles, traversability | Stable instance identity and long-range semantics | Collision checking, mapping, simulation, planning |
| Dense BEV features | Spatially organized residual evidence | Expensive to store and difficult to validate directly | Shared perception heads and local planning features |
| Latent tokens | High-bandwidth information under a fixed compute budget | Explicit geometry and independent interpretability | World decoders, policies, generative models |

Bounding boxes are efficient, but they assume that the relevant world can be divided into known instances. Occupancy asks a more basic question: which parts of 3D space are occupied, free, or unobserved? [Occ3D](/paper%20shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html) formalized dense, visibility-aware semantic occupancy benchmarks; [PanoOcc](/paper%20shorts/2023/06/16/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation.html) treats occupancy as a unified representation for camera-based 3D panoptic understanding. [OccAny](/paper%20shorts/2026/03/24/occany-generalized-unconstrained-urban-3d-occupancy.html) pushes the research frontier toward metric occupancy in out-of-domain and even uncalibrated urban scenes. These results do not remove calibration requirements from a production stack, but they show why occupancy is becoming a general scene interface rather than one auxiliary head.

Road structure has a different natural form. [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) models online vectorized map elements as structured point sets rather than raster pixels. A vector roadgraph preserves connectivity and shape in a form that planning and simulation can consume directly.

The strongest system contract is therefore not “structured outputs or learned embeddings.” It is structured outputs **and** learned embeddings. Materialized objects, roadgraph elements, occupancy, semantics, uncertainty, and timestamps provide compact interfaces for validation and simulation. Latent features preserve residual information that those schemas do not capture.

This distinction also resolves a common false choice. End-to-end learning does not require an unstructured runtime. A model can backpropagate through perception, prediction, and planning while still materializing selected state for independent checks. Conversely, a stack can have named modules yet remain difficult to validate if the interfaces carry poorly calibrated learned features.

## Training contracts shape the deployed model
The graph that learns can be richer than the graph that runs. Confusing the two makes architecture descriptions imprecise.

In [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), projected LiDAR supervises camera depth and disappears at inference. In [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), sparse depth remains a deployed input. In [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html), a camera-LiDAR teacher transfers features, relations, and outputs to a camera-radar student. These systems should not all be described as “using LiDAR.” Their runtime sensor contracts are different.

[![Animation separating LiDAR depth labels, runtime sparse-depth input, and a LiDAR-camera teacher](/assets/images/autonomous-perception-lidar-training-contracts.gif)](/assets/images/autonomous-perception-lidar-training-contracts.gif)
_LiDAR supplies labels to BEVDepth, remains a runtime input for Sparse-to-Dense, and belongs to the training-only teacher in CRKD. The deployed sensor contract differs in every column._

The same distinction applies to pretraining. Image classification teaches appearance, but not calibration, metric depth, ego motion, or temporal persistence. [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) reconstructs masked camera and LiDAR inputs through a shared 3D volume. [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) reconstructs masked LiDAR columns.

[UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html), [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html), and [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) add future occupancy, point-cloud prediction, or dynamic state. Their shared idea is that a useful scene representation should explain not only the current observation but also how the world evolves. A longer forecast is not automatically better, because distant futures are increasingly multimodal and supervision can reward averaging.

The same BEV or token state may feed detection, occupancy, mapping, velocity, tracking, prediction, and planning. Sharing saves repeated sensor encoding and lets tasks exchange scene context, but their losses have different units, label densities, and learning speeds. Each loss should first be normalized by a meaningful count. If one task still dominates shared layers, [uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) or [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) changes gradient magnitude. If gradients point in opposing directions, [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html), adapters, or an earlier architectural split addresses a different problem.

[![Animation comparing loss-scale weighting, GradNorm's training-rate targets, and PCGrad's projection of conflicting gradients](/assets/images/autonomous-perception-multitask-gradients.gif)](/assets/images/autonomous-perception-multitask-gradients.gif)
_Loss weighting changes gradient magnitude. GradNorm adjusts weights using relative training rates. PCGrad changes direction when task gradients conflict. These mechanisms solve different problems._

Multi-task learning is not a separate perception architecture. It is an optimization contract imposed on the shared representation. A task should share layers only while the shared features remain useful to it. “One model” is not a reason to force all tasks through the same bottleneck.

The broader principle is that end-to-end gradient flow and end-to-end runtime coupling are different decisions. Large teachers can use privileged sensors, longer history, future labels, simulation, language supervision, or expensive world models. A deployable student can inherit part of that knowledge while retaining a smaller and more testable runtime graph.

## From unified perception to a driving foundation model
The boundary between perception and planning is becoming less rigid. [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) connected detection, tracking, mapping, motion prediction, occupancy, and planning through unified query interfaces optimized toward the planning task. The significance was not simply that several heads occupied one repository. It made the downstream driving objective part of representation design.

Planning also cannot be reduced to one deterministic future. Traffic scenes contain genuine multimodality: yield or proceed, pass or wait, merge ahead or behind. [DiffusionDrive](/paper%20shorts/2024/11/22/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving.html) uses a truncated diffusion policy to generate diverse driving trajectories with a small number of denoising steps. [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) separates candidate generation from candidate scoring. These methods move the output contract from “predict one trajectory” toward “represent several plausible actions and evaluate them.”

World models add another layer. Instead of using the current scene state only to emit an action, they predict how road users, geometry, and sensor observations may evolve under candidate actions. The runtime question is whether that model must execute onboard. [WPT](/paper%20shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html) offers one answer: use a world model and learned reward model to train a teacher policy, then distill the resulting reasoning into a lightweight student. This preserves real-time deployability while allowing a richer model to shape training.

These lines of work lead to a system-level foundation model: a shared model family and world-state vocabulary across perception, prediction, planning, simulation, evaluation, and data generation. The difficult part is not attaching a VLM to a sensor encoder. It is deciding which component owns geometry, which component supplies semantics, how uncertainty is represented, and where independent validation remains possible.

## Waymo's public foundation-model architecture
Waymo has publicly described one version of this system-level design. In [Waymo Co-CEO Dmitri Dolgov's talk, “The Demo Is Only 1% Of The Work”](https://www.youtube.com/watch?v=Gp4zrV3-6N8) and Waymo's [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/), the Waymo Foundation Model contains a Sensor Fusion Encoder, a Driving VLM, and a World Decoder. Waymo also describes learned embeddings coexisting with compact materialized representations such as objects, semantic attributes, and roadgraph elements.

This is an architecture overview, not a complete specification of the deployed online graph. It establishes the public interfaces and training philosophy, but not the execution frequency, model sizes, exact state schema, or all safety checks.

![The Waymo Foundation Model diagram with sensor fusion, a driving VLM, and a generative world decoder](/assets/images/waymo-foundation-model-architecture.png)
_Waymo's diagram separates a fast sensor-fusion path from a slower semantic-reasoning path, then joins both inside a World Decoder. Source: [Dmitri Dolgov's Waymo talk](https://www.youtube.com/watch?v=Gp4zrV3-6N8); see Waymo's public [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) and earlier [foundation-model overview](https://waymo.com/blog/2024/10/ai-and-ml-at-waymo/)._

| Component | Publicly described input and state | Publicly described role |
| --- | --- | --- |
| Sensor Fusion Encoder | Camera, LiDAR, and radar over time → objects, semantics, and learned embeddings | Fast metric perception and reaction |
| Driving VLM | Rich camera data, driving data, and broader learned world knowledge → semantic signals | Reasoning about rare, novel, or semantically complex situations |
| World Decoder | Sensor-fusion and VLM representations | Predict road-user behavior, produce maps, generate vehicle trajectories, and provide trajectory-validation signals |
| Driver validation layer | Candidate trajectory and materialized state | Independently verify the generative trajectory onboard |
| Simulator and Critic | Shared foundation-model family and compact world state | Generate closed-loop worlds, evaluate behavior, identify failures, and produce training signals |

The architecture makes several choices that are consistent with the progression in this article.

First, the latency-critical path still preserves sensor measurements, establishes geometry, and updates state at driving frequency. A VLM does not replace calibration, depth, occupancy, motion, or tracking.

Second, semantics and geometry enter through different paths. A VLM can recognize that a burning vehicle, unusual hand signal, or temporary construction pattern should alter behavior even when free space appears geometrically open. That signal is useful because it changes the interpretation of the scene, not because language is a better range sensor.

Third, the world state is dual. Learned embeddings retain information that a fixed schema may omit. Materialized objects, semantics, and roadgraph elements support validation, simulation, and evaluation. This is neither a classical modular stack nor a single opaque policy.

Fourth, the model that teaches is larger than the model that runs. Waymo describes adapting large teacher models to the Driver, Simulator, and Critic, then distilling smaller students. The onboard Driver mirrors the foundation-model structure but remains paired with a separate validation layer.

Finally, the Driver, Simulator, and Critic share a world-model family. This creates a common state vocabulary for action generation, closed-loop scenario generation, evaluation, and data selection. The benefit is not only parameter reuse. It is the ability to turn a failure discovered by the Critic into a simulation, a training target, and a regression test without translating between unrelated representations at every step.

## How I would design a system in this family
The following is my design synthesis, not a claim about Waymo's exact implementation.

The architecture would have two online perception paths and one broader training ecosystem:

```text
camera / LiDAR / radar
        ↓
modality-specific encoders
        ↓
calibration, time alignment, ego-motion compensation, sensor health
        ↓
hybrid recurrent world state
        ├── materialized state: tracks, occupancy, roadgraph, semantics, uncertainty, age
        └── latent state: dense features and compact scene tokens

selected camera history + route context + rare-event trigger
        ↓
driving VLM
        ↓
grounded semantic hypotheses with confidence and expiry

materialized state + latent state + grounded semantic hypotheses
        ↓
world decoder
        ↓
multimodal agent futures + candidate ego trajectories
        ↓
independent scorer and validation layer
        ↓
control
```

### The fast path should own geometry and freshness
The sensor-fusion path should run at the highest control-relevant frequency. Camera, LiDAR, and radar should keep separate encoders until they occupy calibrated metric support. Timestamps, ego pose, point age, scan phase, and sensor-health signals should enter before or during temporal fusion, not be reconstructed afterward.

Its recurrent state should be hybrid. A dense or semi-dense field should preserve occupancy, free space, and weak background evidence. Sparse queries should preserve actor identity, kinematics, and roadgraph elements. Compact latent tokens should retain residual scene information for the world decoder. No one representation needs to carry every contract.

The path should produce two interfaces. The first is materialized state: tracked actors, semantic attributes, occupancy or traversability, roadgraph elements, traffic controls, uncertainty, provenance, and freshness. The second is a learned latent state with higher bandwidth than the schema. Planning consumes both. Validation should be able to operate on the first even when it cannot interpret every dimension of the second.

### The slow path should produce grounded hypotheses, not unbounded authority
The Driving VLM should operate at a lower frequency or be triggered by uncertainty and rare events. Its inputs can include selected camera views, short temporal clips, route context, and a compact summary of the fast path. Sending the entire raw sensor stream through a large language-conditioned model at control frequency is difficult to justify when most frames contain routine geometry.

Its outputs should be grounded semantic hypotheses, not free-form instructions. A useful output might say that a particular region contains a vehicle fire, that a person is likely directing traffic, or that temporary signage invalidates the nominal lane rule. Each hypothesis should identify its supporting region or entity, confidence, timestamp, and expiry condition.

The VLM should not directly overwrite metric state. It should modify costs, constraints, route preferences, or uncertainty only after grounding into the shared world representation. This prevents a stale semantic token from silently moving an object or declaring free space where the fast path sees an obstacle.

### The world decoder should represent several plausible futures
The decoder should predict a distribution over future world evolution, not one averaged future. For each relevant actor, it should preserve several plausible modes and their interactions with the ego plan. It should also predict scene-level evolution such as occupancy flow, traffic-control state, or roadgraph changes when those are decision-relevant.

Ego planning should similarly generate a compact set of diverse trajectories rather than one point estimate. Diffusion or autoregressive decoding is one mechanism, but the representation matters more than the generator. The candidate set must cover materially different choices, not minor perturbations of the same behavior.

### Generation, scoring, and validation should remain distinct contracts
A generative decoder is optimized to cover plausible actions. A scorer is optimized to rank them. A validation layer is optimized to reject unreasonable risk. These objectives overlap, but they are not identical.

The generate-then-score split in [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) is a useful learned pattern. Waymo's public description adds a separate onboard validation layer. I would preserve both distinctions. A learned scorer can evaluate safety, comfort, progress, compliance, and semantic appropriateness. A validation layer can apply independent checks using materialized geometry, vehicle dynamics, route rules, uncertainty bounds, sensor health, and fallback policy.

The validator should not be asked to prove that the entire neural network is correct. It should verify concrete properties of the proposed trajectory against the current world state. The model remains responsible for proposing capable behavior; the validator prevents a single generative failure from becoming control without another contract being violated.

### The training graph should be broader than the onboard graph
Offline teachers can use longer temporal context, privileged future labels, richer sensor inputs, large VLMs, expensive world models, and simulation rollouts. The onboard student should inherit the resulting representation and policy improvements without reproducing the entire teacher graph.

[WPT](/paper%20shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html) is a useful example of this separation. Its world model and learned reward guide a teacher policy, then policy and reward knowledge are distilled into a faster student. The broader lesson is that a world model can shape the decision boundary without remaining tightly coupled to runtime.

Distillation should occur at more than the final action. Matching intermediate world state, future distributions, trajectory rankings, and uncertainty can preserve more of the teacher's reasoning. The student should still be trained on the failure modes created by its own smaller capacity, because teacher-consistent outputs under clean data do not guarantee graceful behavior under sensor corruption or distribution shift.

### The Driver, Simulator, and Critic should share state semantics
A simulator does not need every internal Driver activation, but it benefits from the same materialized vocabulary: actors, geometry, roadgraph, traffic controls, uncertainty, and behavior modes. It can generate synthetic camera and LiDAR observations from that compact state, alter scene factors, and test counterfactual behavior.

The Critic should evaluate both trajectories and representation failures. It should flag not only a poor maneuver but also stale tracks, inconsistent occupancy, sensor disagreement, VLM hypotheses that outlive their evidence, or candidate sets that omit the safe mode. Those failures can then define targeted data mining, simulation perturbations, teacher supervision, and regression suites.

The closed loop begins with a real or simulated failure. The Critic diagnoses it; the system turns that diagnosis into a targeted scenario and labels, improves the teacher, distills the student, runs closed-loop regression, and reviews the result before deployment creates new evidence.

<div class="compact-flow-diagram"><a href="/assets/images/driving-failure-learning-loop.svg"><img src="/assets/images/driving-failure-learning-loop.svg" alt="Compact closed-loop diagram connecting a driving failure to Critic diagnosis, targeted data, teacher improvement, student distillation, regression, deployment review, and new evidence"></a></div>
_The loop matters only if the meaning of the original failure survives every handoff. Otherwise the system produces more training activity without producing a reliable fix._

A shared foundation-model family helps only if this loop preserves the semantics of the failure. A larger common encoder is not itself a learning flywheel.

### Evaluation should test the contracts, not only the final score
A system in this family needs matched evaluation across several levels:

- clean-sensor accuracy with controlled backbone, resolution, and training recipe,
- missing, corrupted, delayed, and miscalibrated sensors,
- state freshness after occlusion, pose drift, and long recurrence,
- query or token saturation in crowded scenes,
- coverage and calibration of multimodal agent and ego futures,
- stale or incorrectly grounded VLM semantics,
- disagreement between objects, occupancy, and roadgraph outputs,
- P95 and P99 component and end-to-end latency,
- closed-loop interventions by the scorer or validator,
- regression performance in scenarios discovered by the Critic.

The most informative failures occur where contracts disagree. If a track says an actor is absent but occupancy remains blocked, the system should expose uncertainty rather than average the representations into silent confidence. If the VLM says a lane is unsafe but its grounding has expired, the semantic cost should decay. If all cameras are degraded while LiDAR is healthy, the model should alter both its fusion weights and its uncertainty.

This is the practical reason to materialize selected state. It creates places where the system can detect disagreement before the final trajectory.

## The design rule
A useful way to read any autonomous-driving perception architecture is to ask six questions:

1. What measurement does each sensor contribute that the others do not?
2. Where does camera evidence acquire metric support?
3. Which modality controls admission into the shared representation?
4. What state survives through time, and how is it aged or reset?
5. Which world properties are materialized for downstream reasoning and validation, and which remain latent?
6. What information exists only during training, and what actually runs onboard?

These questions cut through many naming differences. A BEV model, query model, recurrent model, world model, and driving foundation model all make the same irreversible decisions. They decide what evidence to preserve, what to compress, and what the next component is allowed to know.

My current read is that the most credible architecture is hybrid at every important boundary. It uses sensor-specific encoders before shared geometry; dense fields for free space and occupancy together with sparse entities for actors and maps; materialized state together with learned embeddings; a fast metric path together with slower semantic reasoning; and large training-time teachers together with a smaller onboard student and an independent validation layer.

This thesis is falsifiable. It would be wrong if a substantially simpler unstructured policy consistently matched or exceeded the hybrid system under controlled closed-loop evaluation, sensor failures, tail latency, multimodal coverage, and validation interventions. The point is not to preserve modules for their own sake. It is to preserve information and contracts only where they improve capability, deployability, or evidence of safety.

A driving foundation model is therefore not defined by its parameter count or by the presence of a VLM. It is defined by whether one learned system can preserve measurement-specific evidence, build a calibrated temporal world state, represent several plausible futures, and expose enough structure for closed-loop validation. The model classes will keep changing. Those invariants will not.

[Vision-Language Models: A Reading Guide](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) covers the adjacent progression from image-text alignment to grounding, video, and action. Autonomous driving adds metric geometry, real-time recurrence, multimodal uncertainty, and the requirement that every useful capability survive contact with closed-loop validation.
