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
A distant cyclist at dusk may occupy only a few image pixels, two or three LiDAR returns, and one noisy radar detection with radial velocity. None of these measurements is the cyclist. Each sensor contributes partial evidence with its own sampling pattern, uncertainty, and failure mode.

It is tempting to call a model *unified* when it converts every sensor into one tensor as early as possible. That is usually the wrong goal. Early unification can erase the signal that made each sensor useful—image texture, LiDAR height structure, radar Doppler, measurement age, or sensor-specific confidence—and a larger downstream model cannot reconstruct evidence that the encoder has already averaged away.

The real design problem is therefore about interfaces. Where does each measurement acquire geometry? When can modalities interact without losing their distinct evidence? What state survives from one frame to the next, and which parts of that state must remain explicit for prediction, planning, simulation, and validation? The article follows those contracts in order:

<figure class="sensor-architecture-flow" aria-labelledby="sensor-architecture-flow-title">
  <p class="sensor-architecture-flow__eyebrow" id="sensor-architecture-flow-title">The architecture contracts, in order</p>
  <ol class="sensor-architecture-flow__stages">
    <li class="sensor-architecture-flow__stage">
      <span class="sensor-architecture-flow__number">01</span>
      <strong>Sensor-specific evidence</strong>
      <span>Camera · LiDAR · radar</span>
    </li>
    <li class="sensor-architecture-flow__stage">
      <span class="sensor-architecture-flow__number">02</span>
      <strong>Calibrated metric support</strong>
      <span>Shared place and time in 3D</span>
    </li>
    <li class="sensor-architecture-flow__stage">
      <span class="sensor-architecture-flow__number">03</span>
      <strong>Cross-modal interaction</strong>
      <span>Aligned evidence interacts</span>
    </li>
    <li class="sensor-architecture-flow__stage">
      <span class="sensor-architecture-flow__number">04</span>
      <strong>Temporal world state</strong>
      <span>Scene memory across frames</span>
    </li>
    <li class="sensor-architecture-flow__stage sensor-architecture-flow__stage--split">
      <span class="sensor-architecture-flow__number">05</span>
      <strong>Two state interfaces</strong>
      <span class="sensor-architecture-flow__split"><b>Materialized:</b> tracks · occupancy · roadgraph</span>
      <span class="sensor-architecture-flow__split"><b>Latent:</b> features · tokens</span>
    </li>
    <li class="sensor-architecture-flow__stage sensor-architecture-flow__stage--outcome">
      <span class="sensor-architecture-flow__number">06</span>
      <strong>Prediction and planning</strong>
      <span>Futures · scoring · validation</span>
    </li>
  </ol>
</figure>

This progression gives *unified* a stricter meaning. A system can share representations along several independent axes without forcing every component into the same architecture:

| Axis of unification | What becomes shared | What can remain specialized |
| --- | --- | --- |
| Sensors | A vehicle-centered geometric support and downstream state | Native encoders, uncertainty models, sampling patterns, and health signals |
| Time | A recurrent world state | Update, aging, birth, deletion, and reset rules for different state elements |
| Tasks | Sensor computation and scene context | Task heads, losses, label spaces, and validation contracts |
| System | A common model family or world-state vocabulary across the Driver, Simulator, and Critic | Model size, execution frequency, supervision, and deployment constraints |

These axes answer different questions. Sharing a BEV backbone across detection and mapping does not mean that camera and radar should share an encoder. Backpropagating through perception and planning does not require every learned intermediate to remain opaque at runtime. A common foundation-model family across driving and simulation does not require the same graph to run onboard and offline. The design rule running through all four axes is this: **preserve sensor-native evidence until geometry makes interaction meaningful, then preserve enough structured and latent state to make the next decision both capable and testable.**

## The system contract
The runtime path begins with encoders matched to each measurement process. Camera intrinsics map rays to pixels; sensor extrinsics place each camera, LiDAR, and radar relative to the vehicle; timestamps and ego poses bring their measurements to a common time. Only after these transformations can the model ask whether an image edge, a LiDAR return, and a radar detection could describe the same physical object.

Calibration does not add information; it defines the correspondence rule. When that rule is wrong, fusion creates structured errors rather than independent noise. A vehicle may inherit image evidence from an adjacent lane, a lane boundary may shift across the bird's-eye-view (BEV) grid, and a small angular error may become a large lateral displacement at long range. Timing errors create the same problem for moving actors: two correct measurements can disagree because they describe different moments.

Alignment makes interaction possible, but the model still needs a representation that can survive through time. A dense BEV field assigns a persistent cell to each ground-plane location, which suits occupancy, free space, lanes, and maps. Sparse queries allocate state to selected actors or map elements, which suits detection, tracking, motion prediction, and vectorized road structure. Learned latent tokens compress the scene further and ask the training objective to decide what is worth keeping. A practical system will usually combine these forms because each preserves a different part of the world.

[![Autonomous-driving perception pipeline from sensor-specific encoders through calibrated metric representations, dense or sparse temporal state, and task heads, with learned latent tokens and a separate training-only graph](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_The runtime path moves from sensor-specific features to calibrated metric support, then into a dense field, sparse query set, or learned latent state. The lower path shows supervision and teacher models that can exist during training without becoming deployed sensor inputs._

The word *fusion* often hides three separate operations:

1. **Alignment** establishes which measurements could refer to the same place and time.
2. **Interaction** determines how evidence from one modality changes features from another.
3. **Materialization** determines what shared state downstream tasks are allowed to consume.

Keeping these operations separate makes architecture comparisons more precise. Two models may use the same BEV coordinates yet allow very different cross-modal interaction. One may fuse features deeply while still exposing separate objects, occupancy, and roadgraph outputs; another may concatenate aligned tensors once and call the result unified even though one stream dominates the computation. Shared coordinates also do not imply shared certainty. Downstream state must retain, explicitly or implicitly, which sensors support a feature, how old that support is, and whether the relevant sensor is degraded. A confident prediction supported by current camera, LiDAR, and radar evidence is not equivalent to the same score resting on one stale stream.

## Sensor encoders preserve different evidence
The first modeling decision is not how to fuse the sensors. It is how to encode each sensor without erasing the measurement that makes it useful.

### Cameras: dense semantics at several scales
Cameras provide the richest appearance signal in the stack. They distinguish lane paint from a crack in the road, read lights and signs, recognize unusual objects, and preserve boundaries that occupy only a handful of pixels. Yet a pixel identifies only a ray through the camera center, not a distance along that ray. The camera encoder must therefore preserve semantics and spatial detail without pretending that either one resolves depth.

Convolutional backbones build local features with translation-equivariant kernels and map well to optimized inference libraries. Production-oriented systems such as [NVAutoNet](/paper%20shorts/2023/03/23/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving.html) use efficient CNNs for both image and BEV processing. Vision transformers instead use content-dependent attention to connect distant regions, which helps when recognition needs wider context but becomes expensive at full surround-camera resolution. Large pretrained image models can improve either backbone's appearance features; they still do not supply calibrated 3D geometry.

Feature pyramids remain important because apparent object size changes sharply with range. High-resolution levels preserve a distant pedestrian, traffic light, or narrow lane marking; lower-resolution levels gather broader context more cheaply for vehicles, road layout, and scene semantics. [EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) is a clear 2D example of learned multiscale fusion. [BEVFormer v2](/paper%20shorts/2022/11/18/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition.html) adds an optimization lesson: when supervision arrives only after BEV conversion, the image backbone receives a weak, indirect signal, so an auxiliary perspective-view loss can improve the features before they enter 3D.

For a 3D point $X$, camera $i$ uses intrinsics $K_i$ and extrinsics $(R_i, t_i)$ to compute

$$
u_i = \pi(K_i, R_i, t_i, X).
$$

The same point lands at a different pixel in each camera. At pyramid level $l$ with stride $s_l$, the model samples the feature map at $u_i/s_l$. This projection can retrieve a fine boundary only if the backbone preserved that boundary, and it can retrieve broad context only if the receptive field encoded it. Geometry chooses where to look; the encoder determines what is available there.

[![Animation showing one metric 3D point projected to different pixels in two calibrated cameras and sampled at stride-adjusted feature-pyramid coordinates](/assets/images/autonomous-perception-vision-encoder.gif)](/assets/images/autonomous-perception-vision-encoder.gif)
_A single 3D reference point lands at different pixels in different cameras and at different coordinates on each pyramid level. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) applies this operation to object queries; [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) applies it to BEV queries._

The camera branch is strongest where appearance matters, but its central 3D uncertainty remains depth. More image resolution or a larger backbone may improve recognition while leaving the same unanswered question: where along the camera ray does that evidence belong?

### LiDAR: sparse geometry before dense BEV
A LiDAR sweep answers the camera's depth question directly, but its measurements are irregular, sparse, and strongly range-dependent. Point encoders preserve individual returns at the cost of expensive neighborhood construction. Pillar encoders group points into vertical columns and collapse height early, while voxel encoders retain a 3D grid long enough to distinguish overpasses, trucks, poles, and other height-dependent structure. The LiDAR encoder therefore chooses where to exchange 3D detail for the speed and regularity of a 2D BEV backbone.

[PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) makes that trade early: it pools each occupied pillar, scatters the result into a dense pseudo-image, and performs nearly all later computation with 2D convolutions. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) keeps sparse 3D voxels through a middle encoder and densifies only after height has been compressed.

Sparse convolution shares local kernels over occupied cells, but coordinate-map construction and irregular memory access remain real system costs. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) changes how those cells communicate: it partitions variable-density voxels into bounded local sets, applies attention within each set, and rotates the partition between layers so information crosses set boundaries. This lets non-adjacent occupied cells interact more directly than they would through a small convolutional kernel while keeping each attention problem bounded. Actual latency still depends on sorting, padding, token density, height compression, and hardware support—not on FLOPs alone.

[![Animation comparing early densification in PointPillars, late BEV densification in SECOND and DSVT, and box prediction from active voxels in VoxelNeXt](/assets/images/autonomous-perception-lidar-encoder.gif)](/assets/images/autonomous-perception-lidar-encoder.gif)
_PointPillars compresses height before its dense 2D backbone. SECOND and DSVT retain sparse 3D cells for longer, then construct BEV. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) keeps the active-voxel representation through the prediction head._

The right trade depends on the operating envelope. Pillars are a strong latency baseline on mostly planar roads; sparse voxels are preferable when vertical separation, long range, or dense 3D structure matters. Fully sparse heads can avoid BEV work while occupied cells remain rare, although a large FLOP reduction may produce only a modest wall-clock gain on hardware with weak sparse-kernel support. Whichever representation the model uses, intensity, return type, timestamp, and point age should remain attached to the geometry. LiDAR range is direct, but a rotating sweep is not an instantaneous snapshot: a point measured near the beginning may already be stale when inference begins.

### Radar: range and motion under uncertainty
Radar contributes a different kind of evidence. A return may contain range, azimuth, elevation, radial velocity, Radar Cross Section (RCS), timestamp, and a sensor-specific confidence estimate. Doppler can reveal a moving actor before an image-only temporal model has enough baseline, and millimeter-wave sensing is insensitive to darkness and often more tolerant than cameras or LiDAR in rain and fog. The trade is weak angular localization, sparse semantic evidence, multipath, and ghost returns. Radar supplies motion and range constraints, not clean 3D boxes.

The encoder determines where that uncertain evidence can enter the rest of the model. A point encoder keeps individual returns for association around an object proposal. A polar or range-azimuth tensor preserves the native sampling pattern before Cartesian resampling. A radar-BEV encoder creates an independent metric field for later alignment with camera or LiDAR BEV. Rasterizing any of these too early into binary occupancy removes Doppler, RCS, confidence, and return-level ambiguity—the signals needed to separate motion from clutter.

[![Animation comparing camera-radar interaction at the proposal, depth, and BEV stages in CRAFT, CRN, and RCBEVDet](/assets/images/autonomous-perception-radar-encoder.gif)](/assets/images/autonomous-perception-radar-encoder.gif)
_[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates a soft set of radar returns with each camera proposal. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to refine camera depth before lifting and aligns both BEV maps with deformable attention. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) combines point and transformer radar paths before BEV fusion._

Recent camera-radar systems improve less by treating radar as an extra image channel than by choosing where range and Doppler alter the computation. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) found that retaining radar metadata, disabling an aggressive outlier filter, and accumulating aligned sweeps all affected performance. [Doppler-Aware LiDAR–Radar Fusion](/paper%20shorts/2025/10/23/doppler-aware-lidar-radar-fusion-for-weather-robust-3d-detection.html) processes radar power and Doppler as distinct signals during multimodal interaction, while [DinoRADE](/paper%20shorts/2026/04/09/dinorade-full-spectral-radar-camera-fusion.html) combines dense radar tensors with DINOv3 image features for adverse-weather detection. A useful comparison should therefore report adverse-weather and long-range accuracy together with false associations and velocity error. Higher average detection is not enough if a ghost return is attached to the wrong actor.

The encoders now preserve the right evidence. The next question is where that evidence acquires metric support.

## Where metric geometry enters
There is no geometry-free sensor fusion. A model may hide geometry inside attention, depth logits, positional encodings, or learned correspondences, but camera evidence must eventually acquire physical support before it can interact coherently with LiDAR, radar, maps, or temporal state. The main camera-to-3D families differ in where that metric hypothesis begins and in how much of the scene they represent.

### Push image evidence into 3D: LSS and BEVDepth
A pixel fixes a ray through the camera center, not a distance along that ray. [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a categorical depth distribution for each image location, copies the image feature along the candidate depth bins, and pools those lifted features into BEV:

image pixel → ray → depth distribution → 3D feature cloud → BEV grid

The depth distribution in the original LSS is latent: downstream BEV losses push it toward geometry that helps the task, but no depth label supervises it directly. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) adds projected LiDAR depth supervision, giving the camera branch a direct geometric signal as well as the downstream task loss. LiDAR can therefore shape training and disappear at inference. This is learning with privileged information; it becomes teacher-student or cross-modal distillation only when a separate teacher transfers knowledge to a student. Forward lifting preserves dense image evidence and naturally suits occupancy and maps, but its failure mode is equally direct: a depth error writes the feature into the wrong metric cell.

![Figure 1 from Lift, Splat, Shoot, showing multiview evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_Lift, Splat, Shoot predicts depth along every camera ray and pools the lifted features into a vehicle-centered BEV grid. Source: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

### Pull image evidence from metric space
Pull-based methods reverse the direction: they begin with a hypothesis in physical space and ask the images for supporting evidence. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) projects each 3D voxel into the cameras and bilinearly samples visible image features. In its controlled experiments, input resolution and effective batch size changed vehicle-segmentation IoU more than the lifting operator. The result is a useful warning for architecture comparisons: match the backbone, image resolution, training schedule, batch size, and sensor inputs before crediting a benchmark gain to the view transformer.

The physical hypothesis does not have to be dense. [DETR](/paper%20shorts/2020/05/26/end-to-end-object-detection-with-transformers.html) introduced object queries for 2D detection: a bounded set of learned vectors asks which objects should be represented, and a transformer decoder turns those queries into predictions. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) anchors each query to a 3D reference point, projects that point into every camera, and samples image evidence around the projections.

The difference from dense lifting is where the model spends its geometric budget. LSS represents every image location across candidate depth bins; DETR3D asks only about a bounded set of candidate objects. The latter can be efficient for detection, but it creates a recall dependency: if no query acquires support for an actor, later refinement cannot recover evidence that never entered the representation.

[PETR](/paper%20shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html) moves more of the geometry into the image features. Multi-view features otherwise encode appearance and camera identity without directly stating which region of physical 3D space they describe. PETR injects 3D positional information so attention can reason over image evidence with explicit geometric context. Put simply, DETR3D places much of the geometry in the query and reference-point mechanism, while PETR makes the features themselves geometry-aware.

[BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) changes the unit from an object query to a BEV query. A learned token for each bird's-eye-view location samples image evidence at several heights along the corresponding vertical pillar. This creates a dense spatial field without explicitly predicting a depth distribution for every pixel. The price is that BEV cells, cameras, heights, and attention samples can dominate computation.

[![Animation comparing depth along an image ray in LSS, a 3D object-query reference point in DETR3D, and vertical reference points from a BEV cell in BEVFormer](/assets/images/autonomous-perception-camera-lifting.gif)](/assets/images/autonomous-perception-camera-lifting.gif)
_The highlighted variable is the source of metric support: depth along an image ray in LSS, a 3D object reference point in DETR3D, or several heights above a fixed BEV cell in BEVFormer._

| 3D construction | Where the metric hypothesis begins | Best fit | Main cost | Characteristic failure |
| --- | --- | --- | --- | --- |
| Depth lift and splat | Every image location predicts depth | Dense occupancy, maps, and detection | Pixels × depth bins, followed by BEV processing | Wrong depth moves evidence to the wrong metric cell |
| Voxel-to-image sampling | Every metric voxel projects into visible cameras | Simple dense BEV baselines | Voxels × cameras | A sampled pixel may mix depths or lie behind an occluder |
| Object queries | A bounded set of 3D actor hypotheses | Detection, tracking, and motion | Queries × views × samples | An actor is missed when no query acquires support |
| BEV queries | Every ground-plane cell samples several heights | Dense scene state with selective image retrieval | BEV cells × heights × views | Weak projected support creates false or empty cells |

No construction dominates every task. Dense methods spend compute to represent the world; object queries spend it on candidate actors. The meaningful comparison is not which abstraction is newer, but which evidence it discards and whether a later component has any path to recover that evidence.

## Fusion is alignment, interaction, and routing
Once camera, LiDAR, and radar features have metric support, the model can decide how they should interact. Early fusion combines near-raw inputs even though pixels, points, and Doppler returns do not share a sampling pattern. Late fusion combines independent predictions, which is modular and easy to validate but gives up feature-level complementarity. Intermediate fusion offers a more useful default for heterogeneous sensors: each modality first preserves its own evidence, then meets the others after geometric alignment. The decisive choice is the granularity of that meeting.

### Point, query, and dense-field fusion
[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) attaches camera class scores to LiDAR points before voxelization. The operation is cheap and geometrically direct, but image evidence survives only where LiDAR produced a return; empty space and camera-only observations disappear. [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) moves the meeting point to a 3D object hypothesis, sampling camera, LiDAR, and radar features around the same reference point. This matches an object-centric output and spends computation selectively, but still does not retain a complete background field.

[BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) gives each modality an appropriate encoder, transforms the resulting features into a shared BEV grid, and fuses there:

- camera images → camera encoder → camera BEV
- LiDAR point cloud → point or voxel encoder → LiDAR BEV
- camera BEV + LiDAR BEV → fusion → task heads

BEV is a natural meeting place because a camera feature at $(x,y)$ and a LiDAR feature at $(x,y)$ refer to approximately the same physical support. The model need not make the camera behave like LiDAR or LiDAR behave like a camera. It preserves each sensor's inductive bias until physical alignment makes interaction meaningful.

[![Animation showing the same actor and lane evidence entering point, object-query, and dense-BEV fusion](/assets/images/autonomous-perception-fusion-granularity.gif)](/assets/images/autonomous-perception-fusion-granularity.gif)
_Point fusion retains camera features only at measured points. Query fusion gathers evidence around selected actors. Dense BEV fusion keeps actor evidence together with the surrounding lane, occupancy, and free-space field._

### Proposal recall is an architectural ceiling
Proposal-conditioned fusion spends compute selectively. [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) uses LiDAR proposals to retrieve image evidence around candidate objects instead of trusting one calibrated pixel. This can tolerate small correspondence errors, but it exposes a structural failure mode: if LiDAR misses an actor that the camera sees, a LiDAR-controlled proposal stage may never give the camera evidence a chance to recover it.

[![Animation comparing LiDAR-controlled proposals, merged proposals from multiple modalities, and shared-BEV fusion before detection](/assets/images/autonomous-perception-proposal-recall.gif)](/assets/images/autonomous-perception-proposal-recall.gif)
_Proposal-conditioned fusion is selective, but proposal recall can become a ceiling. Multi-proposal fusion restores a recovery path at the cost of matching and deduplication; shared-BEV fusion preserves both modality fields before detection._

One response is to let every modality generate proposals, then merge them before refinement. This removes the single-modality recall ceiling but introduces duplicates, conflicting confidence, inconsistent localization, and cross-modal matching. Dense shared fusion makes the opposite trade: it preserves both modality fields before generating detections, spending more computation to keep a recovery path when one sensor misses an actor. Both designs expose the same diagnostic question: **which modality controls admission into the shared representation?** If one sensor supplies the proposals, points, or confidence threshold, its recall becomes the system's ceiling unless another path explicitly bypasses it.

> **Deep insight:** In multimodal fusion, the hidden bottleneck is often admission rather than attention. The sensor that creates the proposals, points, or thresholds decides which evidence becomes eligible for downstream computation; without a bypass, its recall becomes the system's ceiling.

### Interaction can occur before either stream is finished
A simple design runs the camera and LiDAR encoders to completion, then applies one fusion block. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) instead lets the streams update one another across representation stages, while [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) applies shared transformer blocks after modality-specific tokenization. The important distinction is not the mere presence of cross-attention. It is whether one modality can change what another stream preserves before feature extraction is complete. Repeated interaction can improve complementarity, but it also spreads contamination: a corrupted stream may affect several layers instead of one terminal block. Deeper fusion therefore needs stronger health-aware routing and modality-specific diagnostics.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion keeps camera and LiDAR encoding separate until both modalities occupy the same BEV grid, then shares that grid across detection and map heads. Source: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

### Missing, degraded, and misaligned sensors
A fusion network trained only with all sensors often becomes dependent on its strongest stream. Simply zeroing LiDAR at inference does not turn that network into a competent camera-only model, because the remaining branch was never asked to carry the full task.

[UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) demonstrates the failure directly. In its reported ablation, a model trained only in fused mode collapses under camera-only inference, while modality dropout and normalization over the streams that remain produce a usable fallback path. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) trains full, camera-only, and LiDAR-only modes and lets BEV queries attend to whichever encoders are available.

[![Animation contrasting modality availability in UniBEV and MetaBEV with reliability gating in Grace-BEV](/assets/images/autonomous-perception-modality-dropout.gif)](/assets/images/autonomous-perception-modality-dropout.gif)
_Modality dropout covers discrete cases in which a stream is absent. Reliability gating covers the harder case in which a tensor is present but its evidence should receive less weight._

Sensor absence is only the easiest failure to represent. Blur, saturation, fog, reduced LiDAR beams, blocked fields of view, packet delay, interference, and calibration drift all leave a tensor present while making its evidence unreliable. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) adds reliability-aware gating, while [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) evaluates several corruptions in addition to missing streams.

A production model needs at least three mechanisms:

- modality dropout for supported sensor configurations,
- corruption training and observable health signals for partial degradation,
- calibration and timing perturbations as separate tests.

The third category needs its own tests because miscalibration can make every projected association confidently wrong in the same direction; it is not equivalent to random feature noise. More generally, tensor availability is not evidence quality. Reliability must survive fusion through modality support, timestamp or age, and a health estimate. Without those signals, the network may emit a confident fused prediction without exposing that it rests entirely on one degraded stream.

## Time: what should survive the next frame?
A single camera frame or LiDAR sweep cannot provide a complete motion state, preserve evidence through an occlusion, or stabilize an uncertain depth estimate. Radar contributes radial velocity, but not the full velocity of every actor. Temporal modeling fills these gaps by carrying selected evidence forward and comparing it with the next observation.

Reusing old state first requires putting it in the current coordinate frame. Ego-motion compensation can align roads and static structures, but independently moving actors need velocity, motion hypotheses, or learned updates. Timestamp error, pose error, rolling shutter, and scan motion then appear as residual displacement after alignment rather than as true scene motion.

A useful abstraction is

$$
S_t = f\left(\operatorname{Align}(S_{t-1}, \Delta T_t), X_t, \Delta t, H_t\right),
$$

where $S_{t-1}$ is the prior scene state, $X_t$ is current sensor evidence, $\Delta T_t$ is the ego-frame transformation, $\Delta t$ is elapsed time, and $H_t$ contains sensor-health information. The update is easy to write down. The architectural choice is what $S_t$ contains—and therefore what the model can remember or forget.

### Dense scene memory
Dense temporal memory stores a scene-level BEV field. [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) warps the previous BEV feature into the current ego frame, concatenates it with the current feature, and lets a BEV encoder learn displacement cues. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) performs the same broad job through temporal attention over a recurrent BEV representation.

[SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) makes the temporal budget explicit: a short high-resolution history supports fine stereo correspondence, while a longer low-resolution BEV history supports depth and velocity. Dense state retains roads, free space, and weak background evidence that may later support a new detection. That coverage is valuable, but its memory and warp costs grow with BEV area and history, and conservative updates can preserve stale background evidence along with useful context.

### Sparse entity memory
Sparse temporal memory stores selected actors or queries instead of a complete field. [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) retains a bounded queue of foreground queries, conditions them on ego pose, elapsed time, and velocity, and introduces fresh queries for newly visible actors. [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) transforms prior instances into the current frame and combines them with fresh anchors. This memory is lighter and naturally object-centric, but query birth, duplicate removal, aging, and deletion become learned parts of the state update. Free space, road structure, undetected actors, and context that has not yet become a confident query may disappear.

[![Animation comparing a warped dense BEV field, transformed recurrent instances with fresh anchors, and a bounded foreground-query queue](/assets/images/autonomous-perception-temporal-memory.gif)](/assets/images/autonomous-perception-temporal-memory.gif)
_Dense recurrence carries every BEV cell. Sparse4D v2 transforms recurrent object instances and adds fresh anchors. StreamPETR carries a bounded foreground-query queue and introduces new queries for actors not already in memory._

[![Figure 3 from StreamPETR, showing object queries propagated through a temporal memory queue](/assets/images/streampetr-paper-figure-3.png)](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR transforms selected object queries into the current frame, updates them from current images, and keeps the strongest foreground queries for the next step. Query selection saves memory but can discard weak evidence before an actor is confidently detected. Source: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

[SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) keeps sparse object support but retrieves it from several stored frames, so its cost still grows with history. This exposes an important ambiguity: *sparse temporal* may mean compressing history into a recurrent state, or it may mean retaining history and reading only selected locations. The two designs have different memory, latency, and error-accumulation behavior. StreamPETR remembers selected objects; SOLOFusion remembers a scene field.

Dense and sparse memory fail in opposite directions. Dense state preserves weak evidence but can carry clutter and stale features; sparse state limits cost but may delete evidence before it becomes important. A practical hybrid can retain a lower-resolution scene field for occupancy and topology, high-resolution queries for actors and map elements, and explicit age or confidence for both.

Sparsity should also be reported per component. A sparse LiDAR backbone avoids empty voxels; a sparse camera decoder avoids a full BEV field; a sparse temporal model avoids replaying every location in every frame. None of these choices removes the dense surround-camera backbone, and sparse operators still pay for indexing, sorting, padding, gathers, and irregular memory access. FLOPs are therefore an incomplete deployment measure. The evaluation should also report component latency, peak memory, P95 and P99 end-to-end latency, active-token count in crowded scenes, recall when the query budget saturates, and error after long occlusions or ego-pose drift.

## The world state is not one tensor
Dense BEV fields and sparse object queries are often presented as competing philosophies. A more useful view treats them as points on a compression ladder: the state may preserve every spatial cell, selected entities, a small set of learned latent tokens, or one pooled scene vector. Each step saves computation by asking the objective to discard more of the scene.

[![Animation comparing dense BEV memory, sparse object queries, learned latent tokens, and a single pooled embedding](/assets/images/autonomous-perception-latent-memory.gif)](/assets/images/autonomous-perception-latent-memory.gif)
_The scene remains fixed while the representation changes: every BEV cell, selected actors, a compact learned token set, or one pooled vector. Moving right saves computation but asks the learning objective to decide which spatial evidence can be discarded._

Let $X_t$ contain current sensor features and let $Z_{t-1}$ be a compact set of learned latent tokens. A recurrent latent state could be updated as

$$
Z_t = \operatorname{CrossAttention}(Z_{t-1}, X_t).
$$

The latent tokens form an information bottleneck, so compute scales with the number of retained tokens rather than with every BEV cell or stored observation. Perceiver-style bottlenecks, memory tokens, token compression, and latent world models all use versions of this trade: fixed compute in exchange for a learned decision about what survives.

Mean-pooling the entire map into one embedding is usually too aggressive for driving. One vector would need to preserve a pedestrian on the left, a stop sign ahead, lane curvature, a cyclist approaching from behind, an occluded vehicle, and free-space topology. That compression may answer “what kind of scene is this?” but is poorly matched to “where is the pedestrian relative to the ego vehicle?”

A small token set is more plausible because different tokens can specialize. [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) explores this direction for end-to-end driving: camera-aware register tokens compress multi-camera features into a compact scene representation, then lightweight decoders generate and score candidate trajectories. The result shows that targeted token compression can support planning without carrying every camera token downstream. It does not show that all metric structure should disappear.

[UniLION](/paper%20shorts/2025/11/03/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns.html), a 2025 preprint, pushes unification into the backbone itself. Its linear group RNN supports LiDAR-only, temporal LiDAR, multimodal, and multimodal-temporal variants across perception, prediction, and planning tasks. The useful claim is not that one operator has solved the stack; it is that sensor, time, and task unification can be posed as one sequence-modeling problem. Whether one architecture remains optimal across hardware, calibration, and failure constraints remains an empirical question.

The compression ladder is therefore:

dense scene field → sparse structured entities → learned latent registers → single pooled embedding

Moving right saves memory and compute while increasing the burden on the objective to preserve the right information. At every step, the design question remains the same: what is discarded, when is it discarded, and can any downstream component recover it?

### Materialized structure and latent state should coexist
Compression controls how much state survives; materialization controls what the rest of the system can inspect. A driving model can expose several complementary views of the same learned world state:

| Representation | What it preserves well | What it tends to miss | Natural consumers |
| --- | --- | --- | --- |
| Objects and tracks | Dynamic agents, identity, kinematics, compact interaction state | Unmodeled geometry, unusual objects, amorphous hazards | Prediction, interaction modeling, behavior planning, validation |
| Vector roadgraph and map elements | Lane boundaries, connectivity, stop lines, route topology | Non-map obstacles and uncertain geometry | Routing, rule reasoning, planning, simulation |
| Occupancy and free space | Detailed geometry, unknown obstacles, traversability | Stable instance identity and long-range semantics | Collision checking, mapping, simulation, planning |
| Dense BEV features | Spatially organized residual evidence | Expensive to store and difficult to validate directly | Shared perception heads and local planning features |
| Latent tokens | High-bandwidth information under a fixed compute budget | Explicit geometry and independent interpretability | World decoders, policies, generative models |

Bounding boxes are efficient, but they assume that the relevant world can be divided into known instances. Occupancy asks a more basic question: which parts of 3D space are occupied, free, or unobserved? [Occ3D](/paper%20shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html) formalized dense, visibility-aware semantic occupancy benchmarks; [PanoOcc](/paper%20shorts/2023/06/16/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation.html) treats occupancy as a unified representation for camera-based 3D panoptic understanding; and [OccAny](/paper%20shorts/2026/03/24/occany-generalized-unconstrained-urban-3d-occupancy.html) extends the research frontier toward metric occupancy in out-of-domain and even uncalibrated urban scenes. These results do not remove calibration from a production stack. They explain why occupancy is becoming a general scene interface rather than one auxiliary head.

Road structure has a different natural form. [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) represents online map elements as structured point sets rather than raster pixels, preserving connectivity and shape in a form that planning and simulation can consume directly. The strongest system contract is therefore not “structured outputs or learned embeddings.” It is structured outputs **and** learned embeddings. Materialized objects, roadgraph elements, occupancy, semantics, uncertainty, and timestamps provide compact interfaces for validation and simulation; latent features preserve residual information that those schemas cannot express.

This dual interface also resolves a common false choice. End-to-end learning does not require an unstructured runtime: a model can backpropagate through perception, prediction, and planning while materializing selected state for independent checks. Conversely, a stack may have named modules and still be difficult to validate when its interfaces carry poorly calibrated learned features.

## Training contracts shape the deployed model
The graph that learns can be much richer than the graph that runs. Architecture descriptions become misleading when they call both graphs “the model” without stating which sensors, labels, and teachers remain available at inference.

The role of LiDAR makes the distinction concrete. In [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), projected LiDAR supervises camera depth and disappears at inference. In [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), sparse depth remains a deployed input. In [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html), a camera-LiDAR teacher transfers features, relations, and outputs to a camera-radar student. Describing all three systems as “using LiDAR” hides the runtime sensor contract that actually determines deployment.

[![Animation separating LiDAR depth labels, runtime sparse-depth input, and a LiDAR-camera teacher](/assets/images/autonomous-perception-lidar-training-contracts.gif)](/assets/images/autonomous-perception-lidar-training-contracts.gif)
_LiDAR supplies labels to BEVDepth, remains a runtime input for Sparse-to-Dense, and belongs to the training-only teacher in CRKD. The deployed sensor contract differs in every column._

Pretraining has the same boundary. Image classification can teach appearance, but not calibration, metric depth, ego motion, or temporal persistence. [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) instead reconstructs masked camera and LiDAR inputs through a shared 3D volume, while [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) reconstructs masked LiDAR columns. Their objectives force the representation to retain sensor structure that an image label never asks for.

[UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html), [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html), and [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) extend this idea to future occupancy, point-cloud prediction, or dynamic state. Their shared premise is that a useful scene representation should explain not only the current observation but also how the world may evolve. Forecast horizon alone is not evidence of a better representation, however: distant futures become increasingly multimodal, and a poorly chosen objective may reward an averaged future that corresponds to no plausible scene.

The same BEV or token state may feed detection, occupancy, mapping, velocity, tracking, prediction, and planning, saving repeated sensor encoding and letting tasks exchange scene context. The optimization problem is that these losses have different units, label densities, and learning speeds. Each loss should first be normalized by a meaningful count. If one task still dominates the shared layers, [uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) or [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) can change gradient magnitude. If gradients point in opposing directions, [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html), adapters, or an earlier architectural split addresses the separate problem of gradient conflict.

[![Animation comparing loss-scale weighting, GradNorm's training-rate targets, and PCGrad's projection of conflicting gradients](/assets/images/autonomous-perception-multitask-gradients.gif)](/assets/images/autonomous-perception-multitask-gradients.gif)
_Loss weighting changes gradient magnitude. GradNorm adjusts weights using relative training rates. PCGrad changes direction when task gradients conflict. These mechanisms solve different problems._

Multi-task learning is therefore not a separate perception architecture. It is an optimization contract imposed on a shared representation, and a task should share layers only while those shared features remain useful to it. “One model” is not a reason to force every task through the same bottleneck. More broadly, end-to-end gradient flow and end-to-end runtime coupling are different decisions. Large teachers may use privileged sensors, longer history, future labels, simulation, language supervision, or expensive world models; a deployable student can inherit part of that knowledge while keeping a smaller and more testable runtime graph.

## From unified perception to a driving foundation model
The boundary between perception and planning becomes less rigid once both operate on the same recurrent state. [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) connected detection, tracking, mapping, motion prediction, occupancy, and planning through query interfaces optimized toward the planning task. Its significance was not that several heads occupied one repository; it made the downstream driving objective part of representation design.

That objective cannot be reduced to one deterministic future. Traffic contains genuine multimodality: yield or proceed, pass or wait, merge ahead or behind. [DiffusionDrive](/paper%20shorts/2024/11/22/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving.html) uses a truncated diffusion policy to generate diverse trajectories with a small number of denoising steps, while [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) separates candidate generation from candidate scoring. Together, these methods shift the output contract from “predict one trajectory” to “represent several plausible actions and evaluate them.”

World models extend the contract again. Rather than using the current scene state only to emit an action, they predict how road users, geometry, and sensor observations may evolve under candidate actions. The deployment question is whether that expensive model must execute onboard. [WPT](/paper%20shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html) offers one answer: use a world model and learned reward model to train a teacher policy, then distill the result into a lightweight student. A richer model shapes the decision boundary during training without dictating the runtime graph.

Taken together, these lines of work suggest a system-level foundation model: a shared model family and world-state vocabulary across perception, prediction, planning, simulation, evaluation, and data generation. The difficult work is not attaching a vision-language model (VLM) to a sensor encoder. It is assigning ownership: which component establishes geometry, which supplies semantics, how uncertainty survives their interaction, and where independent validation remains possible.

## Waymo's public foundation-model architecture
Waymo has publicly described one concrete version of this system-level design. In [Waymo Co-CEO Dmitri Dolgov's talk, “The Demo Is Only 1% Of The Work”](https://www.youtube.com/watch?v=Gp4zrV3-6N8) and Waymo's [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/), the Waymo Foundation Model contains a Sensor Fusion Encoder, a Driving VLM, and a World Decoder. Learned embeddings coexist with compact materialized representations such as objects, semantic attributes, and roadgraph elements. This public overview establishes the interfaces and training philosophy, but it is not a complete specification of the deployed online graph: execution frequency, model sizes, the exact state schema, and the full set of safety checks are not reported.

![The Waymo Foundation Model diagram with sensor fusion, a driving VLM, and a generative world decoder](/assets/images/waymo-foundation-model-architecture.png)
_Waymo's diagram separates a fast sensor-fusion path from a slower semantic-reasoning path, then joins both inside a World Decoder. Source: [Dmitri Dolgov's Waymo talk](https://www.youtube.com/watch?v=Gp4zrV3-6N8); see Waymo's public [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) and earlier [foundation-model overview](https://waymo.com/blog/2024/10/ai-and-ml-at-waymo/)._

| Component | Publicly described input and state | Publicly described role |
| --- | --- | --- |
| Sensor Fusion Encoder | Camera, LiDAR, and radar over time → objects, semantics, and learned embeddings | Fast metric perception and reaction |
| Driving VLM | Rich camera data, driving data, and broader learned world knowledge → semantic signals | Reasoning about rare, novel, or semantically complex situations |
| World Decoder | Sensor-fusion and VLM representations | Predict road-user behavior, produce maps, generate vehicle trajectories, and provide trajectory-validation signals |
| Driver validation layer | Candidate trajectory and materialized state | Independently verify the generative trajectory onboard |
| Simulator and Critic | Shared foundation-model family and compact world state | Generate closed-loop worlds, evaluate behavior, identify failures, and produce training signals |

The public architecture makes four choices that align with the progression in this article. First, its latency-critical path still preserves sensor measurements, establishes geometry, and updates state at driving frequency; the VLM does not replace calibration, depth, occupancy, motion, or tracking. Second, semantics and geometry enter through different paths. A VLM may recognize that a burning vehicle, an unusual hand signal, or temporary construction should alter behavior even when free space appears geometrically open. That signal matters because it changes the interpretation of the scene, not because language is a better range sensor.

Third, the world state is dual. Learned embeddings retain information that a fixed schema may omit, while materialized objects, semantics, and roadgraph elements support validation, simulation, and evaluation. The result is neither a classical modular stack nor a single opaque policy. Fourth, the model that teaches is larger than the model that runs: Waymo describes adapting large teacher models to the Driver, Simulator, and Critic, then distilling smaller students. The onboard Driver mirrors the foundation-model structure but remains paired with a separate validation layer.

The shared model family matters most as a learning loop. A common state vocabulary connects action generation, closed-loop scenario generation, evaluation, and data selection. The benefit is not merely parameter reuse; it is the ability to turn a failure discovered by the Critic into a simulation, a training target, and a regression test without translating between unrelated representations at every step.

## How I would design a system in this family
The evidence above points to a hybrid design with two online paths and a broader training ecosystem. This is my synthesis, not a claim about Waymo's exact implementation. A fast sensor-fusion path owns metric state; a slower semantic path handles selected ambiguous or rare events; a world decoder combines their evidence without collapsing generation, scoring, and validation into one operation.

### The fast path should own geometry and freshness
The sensor-fusion path should run at the highest control-relevant frequency. Camera, LiDAR, and radar should retain separate encoders until their features occupy calibrated metric support. Timestamps, ego pose, point age, scan phase, and sensor-health signals must enter before or during temporal fusion because the model cannot reliably reconstruct freshness after the evidence has been mixed.

Its recurrent state should also be hybrid. A dense or semi-dense field can preserve occupancy, free space, and weak background evidence; sparse queries can preserve actor identity, kinematics, and roadgraph elements; compact latent tokens can retain residual scene information for the world decoder. No single representation needs to carry every contract.

The path should expose two interfaces. The materialized interface contains tracked actors, semantic attributes, occupancy or traversability, roadgraph elements, traffic controls, uncertainty, provenance, and freshness. The learned latent interface carries more bandwidth than that schema can express. Planning consumes both, while validation remains able to inspect the materialized state even when it cannot interpret every dimension of the latent state.

### The slow path should produce grounded hypotheses, not unbounded authority
The Driving VLM should operate at a lower frequency or wake only for uncertainty and rare events. Selected camera views, short temporal clips, route context, and a compact summary of the fast path give it enough context without sending the entire raw sensor stream through a large language-conditioned model at control frequency. Most frames contain routine geometry; the slower path should spend its budget where semantics can change the decision.

Its outputs should be grounded semantic hypotheses rather than free-form driving instructions. A useful hypothesis might identify a vehicle fire in a particular region, a person likely directing traffic, or temporary signage that invalidates the nominal lane rule. Each claim should name its supporting region or entity, confidence, timestamp, and expiry condition so the fast path can decide whether the evidence still applies.

The VLM should never overwrite metric state directly. After grounding into the shared world representation, it may modify costs, constraints, route preferences, or uncertainty. This boundary prevents a stale semantic token from silently moving an object or declaring free space where the fast path sees an obstacle.

### The world decoder should represent several plausible futures
The world decoder should predict a distribution over future evolution rather than one averaged continuation. For each relevant actor, it should retain several plausible modes and their interactions with the ego plan. When they affect the decision, the same state should cover scene-level changes such as occupancy flow, traffic-control state, or roadgraph updates.

Ego planning needs the same multimodality. It should generate a compact set of diverse trajectories rather than one point estimate. Diffusion and autoregressive decoding are possible generators, but coverage matters more than the family name: the candidate set must represent materially different choices, not minor perturbations of the same behavior.

### Generation, scoring, and validation should remain distinct contracts
A generative decoder covers plausible actions, a scorer ranks them, and a validation layer rejects unreasonable risk. These objectives overlap, but they are not interchangeable. Combining them behind one score makes it harder to tell whether a failure came from missing the safe trajectory, ranking it poorly, or accepting an invalid choice.

The generate-then-score split in [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) is a useful learned pattern, and Waymo's public description adds a separate onboard validation layer. I would preserve both boundaries. A learned scorer can weigh safety, comfort, progress, compliance, and semantic appropriateness; a validator can apply independent checks using materialized geometry, vehicle dynamics, route rules, uncertainty bounds, sensor health, and fallback policy.

The validator cannot prove that the entire neural network is correct. Its narrower job is to verify concrete properties of the proposed trajectory against the current world state. The model remains responsible for capable behavior; the validator prevents one generative failure from reaching control without violating an independent contract.

### The training graph should be broader than the onboard graph
Offline teachers can use longer temporal context, privileged future labels, richer sensor inputs, large VLMs, expensive world models, and simulation rollouts. The onboard student should inherit the resulting improvements without reproducing the teacher's full graph or latency.

[WPT](/paper%20shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html) demonstrates this separation: its world model and learned reward guide a teacher policy, then policy and reward knowledge are distilled into a faster student. A world model can therefore shape the decision boundary without remaining coupled to runtime.

Distillation should reach beyond the final action. Matching intermediate world state, future distributions, trajectory rankings, and uncertainty can preserve more of the teacher's reasoning. The student still needs training on failures created by its own smaller capacity, because matching a teacher on clean data does not guarantee graceful behavior under sensor corruption or distribution shift.

### The Driver, Simulator, and Critic should share state semantics
A simulator does not need every internal Driver activation, but it benefits from the same materialized vocabulary of actors, geometry, roadgraph, traffic controls, uncertainty, and behavior modes. From that compact state, it can generate synthetic camera and LiDAR observations, alter individual scene factors, and test counterfactual behavior.

The Critic should evaluate both trajectories and representation failures. A poor maneuver is one failure; stale tracks, inconsistent occupancy, sensor disagreement, VLM hypotheses that outlive their evidence, and candidate sets that omit the safe mode are others. Each diagnosis can drive targeted data mining, simulation perturbations, teacher supervision, and regression suites.

The closed loop is:

real-world or simulated failure → Critic diagnosis → targeted scenario and labels → teacher improvement → student distillation → closed-loop regression → deployment review

A shared foundation-model family helps only when this loop preserves the semantics of the original failure. A larger common encoder is not, by itself, a learning flywheel.

### Evaluation should test the contracts, not only the final score
The architecture exposes several contracts, so its evaluation must test each one under matched conditions:

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

The most informative failures occur where contracts disagree. If a track says an actor is absent while occupancy remains blocked, the system should expose uncertainty rather than average the representations into silent confidence. If the VLM says a lane is unsafe after its grounding has expired, the semantic cost should decay. If every camera is degraded while LiDAR remains healthy, the model should change both its fusion weights and its uncertainty. Materializing selected state creates the interfaces where these disagreements can be detected before they become a trajectory.

## The design rule
The details of model classes will change, but six questions expose the irreversible decisions in any autonomous-driving perception architecture:

1. What measurement does each sensor contribute that the others do not?
2. Where does camera evidence acquire metric support?
3. Which modality controls admission into the shared representation?
4. What state survives through time, and how is it aged or reset?
5. Which world properties are materialized for downstream reasoning and validation, and which remain latent?
6. What information exists only during training, and what actually runs onboard?

These questions cut through naming differences because a BEV model, query model, recurrent model, world model, and driving foundation model all decide what evidence to preserve, what to compress, and what the next component is allowed to know.

My current read is that the most credible architecture is hybrid at every important boundary: sensor-specific encoders before shared geometry; dense fields for free space and occupancy alongside sparse entities for actors and maps; materialized state alongside learned embeddings; a fast metric path alongside slower semantic reasoning; and large training-time teachers paired with a smaller onboard student and an independent validation layer.

This thesis is falsifiable. A substantially simpler unstructured policy would disprove it by consistently matching or exceeding the hybrid system under controlled closed-loop evaluation, sensor failures, tail latency, multimodal coverage, and validation interventions. The point is not to preserve modules for their own sake. It is to preserve information and interfaces only where they improve capability, deployability, or evidence of safety.

A driving foundation model is not defined by parameter count or by the presence of a VLM. It earns the name only if one learned system can preserve measurement-specific evidence, build a calibrated temporal world state, represent several plausible futures, and expose enough structure for closed-loop validation. Model classes will keep changing. These contracts should not.

[Vision-Language Models: A Reading Guide](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) covers the adjacent progression from image-text alignment to grounding, video, and action. Autonomous driving adds metric geometry, real-time recurrence, multimodal uncertainty, and the requirement that every useful capability survive contact with closed-loop validation.
