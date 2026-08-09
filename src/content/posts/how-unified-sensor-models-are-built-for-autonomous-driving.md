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
summary: How modern driving systems encode each sensor, convert evidence into metric space, fuse it, optimize several tasks, use training-only LiDAR, and carry state through time.
---

# Perception for Autonomous Driving

Modern driving perception has three interfaces: **sensor-specific encoding, calibrated metric representation, and task-specific prediction**. Cameras measure appearance along rays, LiDAR measures sparse range and height, and radar measures range and radial velocity with noisy angle. Treating them as interchangeable tensors throws away the reason each sensor is useful. They become comparable only after calibration places their evidence in a shared vehicle frame.

The same representation decision controls fusion, temporal memory, and multi-task learning. A dense bird's-eye-view field can retain roads, free space, and weak background evidence; a sparse query set can spend computation on actors and tracks. LiDAR can be an onboard input, a source of training labels, or part of a teacher. Those are different systems, even when all three are described as “using LiDAR.” This survey follows the representation through the complete pipeline and links each architectural decision to the papers that introduced or sharpened it.

## Perception pipeline

The runtime graph below keeps each measurement in a sensor-native encoder until calibrated geometry converts it into a dense BEV field or sparse 3D queries. Temporal modules then carry either the field or selected objects across frames, and specialized heads turn that state into detection, occupancy, lanes, velocity, and planning outputs. The dashed lower path is a separate training graph: LiDAR labels, teacher features, and future frames can supervise the model without becoming runtime inputs.

[![Autonomous-driving perception pipeline from sensor-specific encoders through calibrated metric representations and temporal state to task heads, with a separate training-only graph](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_The shared interface is metric, not merely multimodal. Camera, LiDAR, and radar evidence can meet in BEV or query space only after their coordinate systems and timestamps are reconciled._

Four questions locate most of the important design choices: what information survives each encoder, which variable chooses the 3D location, what carrier holds fused evidence, and what state crosses the next frame boundary. The figures below answer those questions paper by paper.

## Sensor encoders

### Camera features

A surround-camera system can share one CNN or vision transformer across views, but its features do not share pixel coordinates. For a metric point $X$, camera $i$ uses its intrinsics and extrinsics to compute $u_i=\pi(K_i,R_i,t_i,X)$; a different camera generally receives a different pixel. If the backbone exposes a feature pyramid, sampling level $l$ also requires the stride-adjusted coordinate $u_i/s_l$. Feature pyramids predate 3D camera perception—[EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) is one efficient 2D example—so the 3D contribution is calibrated sampling from those inherited multiscale maps, not the existence of the pyramid itself.

[![Animation showing one metric 3D point projected to different pixels in two calibrated cameras and sampled at stride-adjusted feature-pyramid coordinates](/assets/images/autonomous-perception-vision-encoder.gif)](/assets/images/autonomous-perception-vision-encoder.gif)
_[DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) starts with a 3D object-query reference point, projects it into every visible camera, and samples multiscale image features there. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) applies the same calibration principle to reference points generated from BEV queries. The animation isolates that geometric step from the inherited image backbone._

The backbone still matters because projection cannot recover evidence that downsampling already erased. Small and distant actors need a sufficiently fine feature level; large objects and context benefit from coarser levels. The clean division of responsibility is therefore: the backbone preserves appearance at several receptive fields, while calibration tells the 3D module where each hypothesis should read those features.

### LiDAR sparsity

LiDAR's central encoder decision is when occupied 3D cells become a dense 2D map. [PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) groups points in vertical columns, pools each column, scatters the result into a dense pseudo-image, and performs most computation with a 2D CNN. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) instead retains a sparse 3D middle encoder before height compression. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) changes the sparse mixer—bounded attention replaces sparse convolution in that stage—but it preserves the same broader principle: do not pay for empty voxels.

[![Animation comparing early densification in PointPillars, late BEV densification in SECOND and DSVT, and box prediction from active voxels in VoxelNeXt](/assets/images/autonomous-perception-lidar-encoder.gif)](/assets/images/autonomous-perception-lidar-encoder.gif)
_All three columns receive the same stacked road and overpass returns. PointPillars collapses height first. SECOND and DSVT retain occupied 3D cells through more of the encoder and densify for a BEV head. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) moves the boundary again by predicting from active voxels instead of constructing a dense detection heatmap._

This lineage is not simply “pillars versus voxels.” It progressively delays or removes densification. Early height collapse is fast on mostly planar roads; later sparse 3D processing preserves stacked surfaces and height-dependent evidence; a fully sparse head also avoids allocating empty BEV locations at the output. Intensity, acquisition time, and point age should remain attached to the active cells because a rotating scan is assembled while both the ego vehicle and surrounding actors move.

### Radar range and velocity

Radar should not be encoded as low-resolution LiDAR. Its useful attributes—range, radial velocity, Radar Cross Section, timestamp, and uncertainty—arrive with poor angular resolution, ambiguous elevation, multipath, and ghost returns. The camera-radar literature is best understood by asking **where radar first changes the camera computation**.

[![Animation comparing camera-radar interaction at the proposal, depth, and BEV stages in CRAFT, CRN, and RCBEVDet](/assets/images/autonomous-perception-radar-encoder.gif)](/assets/images/autonomous-perception-radar-encoder.gif)
_[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) begins with a camera proposal and associates a soft set of compatible radar returns in polar coordinates. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) intervenes earlier in geometry: radar changes the camera depth distribution before lifting, then deformable attention aligns the camera and radar BEV features. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) gives radar a point path and a transformer path, constructs a radar-specific BEV, and then aligns it with camera BEV._

The three mechanisms use radar range at different decision points: selecting returns for an object, selecting depth along a camera ray, or building an independent metric field. Rasterizing returns before those attributes have influenced correspondence can erase Doppler and confidence. The radar encoder is therefore part of the fusion design, not a generic preprocessing block.

## Camera features to 3D

Cameras observe rays, while driving outputs require positions in meters. Camera-only 3D perception must therefore choose which variable resolves depth. [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) lets each image location predict a categorical distribution along its calibrated ray and pools the lifted features into BEV. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) keeps that construction but supervises the depth distribution with projected LiDAR.

![Figure 1 from Lift, Splat, Shoot, showing multiview evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_LSS converts perspective evidence from several cameras into one vehicle-centered metric grid. Source: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

Query methods reverse who makes the geometric proposal. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) lets an object query choose a metric $(x,y,z)$ reference point, projects it into the cameras, and reads image evidence around those pixels. BEVFormer gives every BEV cell an $(x,y)$ location, lifts that cell into several reference heights along a vertical pillar, and uses projected attention to retrieve visible image support. It is dense in stored metric state but sparse in image retrieval.

[![Animation comparing depth along an image ray in LSS, a 3D object-query reference point in DETR3D, and vertical reference points from a BEV cell in BEVFormer](/assets/images/autonomous-perception-camera-lifting.gif)](/assets/images/autonomous-perception-camera-lifting.gif)
_The moving marks show the variable each model chooses. LSS chooses a depth distribution along one image ray. DETR3D chooses a full 3D object location. BEVFormer fixes the BEV cell's $(x,y)$ and samples multiple $z$ references. These are different geometric commitments, not three versions of the same projection._

The commitment determines both cost and failure mode. LSS-style models spend work across pixels and depth bins; a misplaced depth distribution writes evidence into the wrong BEV cells. DETR3D spends work around a bounded set of object hypotheses; evidence outside their support is not retrieved. BEVFormer pays to maintain a dense BEV state while limiting how much image evidence each cell samples.

| Representation | Natural outputs | Dominant cost | Characteristic miss |
| --- | --- | --- | --- |
| Lifted dense BEV | Occupancy, roads, maps, detection | Image locations × depth bins, then BEV area | Wrong depth commits evidence to the wrong cell |
| Object queries | Detection, tracking, motion | Queries × visible views × feature samples | Missing or poorly initialized object hypothesis |
| BEV queries | Dense scene state with sparse retrieval | BEV cells × projected samples | Weak support at the cell's reference points |

## Point, query, and BEV fusion

Fusion is a choice of carrier. [PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) projects camera class scores onto LiDAR returns before voxelization, so only image semantics at LiDAR-hit pixels enter the fused representation. [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) uses a shared 3D reference point to sample camera, LiDAR, and radar features into an object query. [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) independently encodes camera and LiDAR, converts both to dense BEV, and fuses the aligned fields.

[![Animation showing the same actor and lane evidence entering point, object-query, and dense-BEV fusion](/assets/images/autonomous-perception-fusion-granularity.gif)](/assets/images/autonomous-perception-fusion-granularity.gif)
_The scene is identical in all three columns. PointPainting retains camera semantics only where LiDAR has a return, so a lane cue without points disappears. FUTR3D retains evidence gathered around an actor hypothesis, but it does not represent the whole background field. BEVFusion retains dense camera semantics and LiDAR geometry, so both the actor and lane field can reach downstream heads._

The carrier constrains recall before the task head sees anything. Point fusion is efficient when the output follows reliable returns; query fusion matches sparse actor outputs; BEV fusion supports dense scene tasks but allocates computation by area. [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) uses LiDAR proposals to focus camera attention, [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) preserves separate modality streams while they update one another, and [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) shares transformer blocks after modality-specific token construction. Each changes correspondence or provenance without making the sensors interchangeable.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion's paper diagram makes the separation explicit: sensor-specific tokenization, a shared metric field, and task-specific outputs. Source: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

## Training for missing and degraded sensors

A fused model does not automatically know how to operate with fewer sensors. [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) trains complete and missing-modality packets, then normalizes fusion over the streams that remain. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) lets BEV queries retrieve from the available encoders and uses modality-specific experts. Both methods condition on **availability**: did this stream arrive?

[![Animation contrasting modality availability in UniBEV and MetaBEV with reliability gating in Grace-BEV](/assets/images/autonomous-perception-modality-dropout.gif)](/assets/images/autonomous-perception-modality-dropout.gif)
_During the “camera degraded” phase, the camera is still marked present for UniBEV and MetaBEV; their availability mask is therefore the same as the healthy two-sensor case. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) adds a reliability estimate, so a present but corrupted camera can receive a low gate while healthy LiDAR remains trusted._

Missing and degraded sensors require different supervision. Modality dropout teaches the network the discrete cases it may encounter at runtime, including how to renormalize one surviving stream. It does not by itself identify blur, glare, miscalibration, packet corruption, or plausible but wrong returns. Those cases need corruption training, an observable health signal, and a fusion rule that can reduce trust without pretending the sensor is absent.

## Multi-task optimization

Detection, occupancy, lanes, velocity, and tracking can share image encoding and metric scene reasoning, but their losses differ in units, label density, and convergence speed. A shared trunk with specialized decoders is the common compromise; [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) uses this pattern, while [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) makes the dependency graph explicit by passing tracking and map queries into motion, occupancy, and planning. Sharing creates three separate optimization problems: numerical loss scale, relative training rate, and gradient direction.

[![Animation applying Kendall uncertainty weighting, GradNorm, and PCGrad to the same pair of raw task gradients](/assets/images/autonomous-perception-multitask-gradients.gif)](/assets/images/autonomous-perception-multitask-gradients.gif)
_[Kendall et al.](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) learn one uncertainty scale per task, changing its contribution through $\mathcal{L}_i/(2\sigma_i^2)+\log\sigma_i$. [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) changes task weights so shared-layer gradient norms track relative training rates. Both change magnitude without resolving a negative angle. [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html) acts on direction: when two gradients conflict, it projects away the opposing component._

The diagnostic should match the intervention. Normalize each loss by a meaningful count before learning weights; otherwise a dense grid can dominate simply because it contains more terms. Track task-wise shared-layer gradient norms to detect unequal training rates, and track cosine similarity to detect conflict. If direction remains unstable after weighting, the remedy may be gradient surgery, task adapters, alternating updates, or an earlier architectural split—not another scalar coefficient.

| Failure | Measurement | Intervention |
| --- | --- | --- |
| Loss scale | Normalized loss magnitude | Unit-aware normalization; uncertainty weighting |
| Training rate | Shared-layer gradient norm and relative loss decay | GradNorm |
| Gradient conflict | Pairwise gradient cosine similarity | PCGrad, adapters, or partial separation |

## Training-only LiDAR

“Using LiDAR for depth” can describe three incompatible deployment contracts. In [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), projected LiDAR provides camera-depth labels during training and disappears at inference. In [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), sparse depth is an input in both training and driving, so removing LiDAR changes the deployed model. In [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html), a camera-LiDAR teacher transfers feature, relation, and response knowledge to a camera-radar student; only the student is deployed.

[![Animation separating LiDAR depth labels, runtime sparse-depth input, and a LiDAR-camera teacher](/assets/images/autonomous-perception-lidar-training-contracts.gif)](/assets/images/autonomous-perception-lidar-training-contracts.gif)
_Read each column vertically from TRAIN to DRIVE. BEVDepth removes LiDAR after it supervises the camera depth distribution. Sparse-to-Dense keeps sparse depth in the runtime graph. CRKD removes the teacher but retains the student's camera-radar inputs. A training sensor, runtime sensor, and teacher are different contracts even when all improve depth._

The runtime graph and label-generation graph should therefore be documented separately. Offline supervision may use LiDAR, future frames, repeated passes, larger teacher models, and human review. Projected LiDAR labels still require visibility checks, pose interpolation, uncertainty, and ignore regions because calibration error, occlusion, timestamp mismatch, and actor motion can move a return across an image boundary.

## Temporal state

A temporal model is defined by what crosses the frame boundary. [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) warps the previous camera-BEV feature into the current ego frame and fuses it with the new feature; BEVFormer recurrently updates a dense field of BEV queries. Dense state preserves road layout, free space, background, and weak evidence that has not yet become an object, but its memory scales with BEV area and history. Ego-motion alignment repairs the static frame; independently moving actors still require learned motion correction.

[![Animation comparing a warped dense BEV field, transformed recurrent instances with fresh anchors, and a bounded foreground-query queue](/assets/images/autonomous-perception-temporal-memory.gif)](/assets/images/autonomous-perception-temporal-memory.gif)
_The left column carries every BEV cell. [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) transforms prior instances into the current frame and adds fresh anchors for new objects. [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) keeps a FIFO memory of top foreground queries, transforms their reference points with ego pose, discards background queries, and adds fresh queries. Sparse recurrence bounds memory, but it must explicitly manage births, aging, and stale state._

[SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) uses a different dense compromise: short high-resolution stereo for fine correspondence and a longer low-resolution BEV history for depth and velocity. [SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) keeps sparse object support but reopens several stored timestamps, so retrieval cost still grows with the number of frames. “Sparse temporal” can therefore mean recurrently compressing history into state or sparsely accessing a retained history; those have different memory and latency terms.

![Figure 3 from StreamPETR, showing object queries propagated through a temporal memory queue](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR carries selected foreground query state instead of a sequence of full scene grids. Source: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

Dense and sparse memory fail differently. A dense field retains discovery context but spends capacity on the full scene. A sparse state is efficient once an actor exists, but a discarded background cue cannot help until a fresh query is born. A deployed recurrent state needs explicit age, observation freshness, confidence, birth, and reset rules; the cited papers establish learned temporal mechanisms, not a complete sensor-outage policy.

## Sparse computation costs

Sparse LiDAR encoders avoid processing empty 3D space, and sparse camera detectors avoid updating every possible BEV cell. The savings apply to specific terms, not the complete stack. Surround-camera backbones still process every image at several scales, while sparse operators add indexing, sorting, padding, and irregular memory access whose cost depends on the accelerator and compiler.

The papers remove different dense terms. [VoTr](/paper%20shorts/2021/09/06/votr-voxel-transformer-for-3d-object-detection.html), [SST](/paper%20shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html), and DSVT bound interaction among occupied LiDAR voxels; VoxelNeXt removes the dense LiDAR head; Sparse4D v2 and StreamPETR bound recurrent object state; SparseBEV replaces a full camera-BEV field with learned pillar support points. FLOPs alone cannot show whether those savings survive synchronization and memory traffic, so profiling should report component latency, peak memory, P95/P99 end-to-end latency, and recall in dense scenes.

## Pretraining targets

Image-classification pretraining teaches appearance but not calibration, cross-view correspondence, metric depth, ego motion, or temporal persistence. [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) masks camera patches and LiDAR voxels, maps visible features into a shared 3D volume, and reconstructs both modalities. [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) masks vertical LiDAR columns and predicts normalized point coordinates and density, matching the anisotropic structure of outdoor scans.

[UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) predicts current and future 4D occupancy, [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html) predicts future point clouds from historical images, and [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) separates dynamic memory from propagated static state. Longer prediction is not automatically stronger supervision: motion, occlusion, pose error, and multimodal futures make distant targets increasingly ambiguous. The useful scaling variable is the diversity of geometric and temporal situations, verified by transfer across tasks and label budgets.

| Pretraining target | Learned relation | Main limitation |
| --- | --- | --- |
| Masked camera and LiDAR reconstruction | Cross-modal correspondence | Local correlated evidence can make reconstruction easy |
| Current occupancy | Metric scene structure | Requires geometric labels or a teacher |
| Future occupancy or point clouds | Motion and persistence | One target cannot represent every valid future |
| Teacher features and pseudo-labels | Task-specific abstractions | Inherits teacher errors and blind spots |

## The next step: a driving foundation model

The perception stack described above ends in specialized task heads. A natural next step is to make the shared representation and decoder reusable across more of the driving lifecycle. In [Waymo Co-CEO Dmitri Dolgov's talk, “The Demo Is Only 1% Of The Work”](https://www.youtube.com/watch?v=Gp4zrV3-6N8), Waymo presents a Waymo Foundation Model with a Sensor Fusion Encoder, a Driving VLM, and a Generative World Decoder. The boundary moves from sensor-specific encoders → metric scene representation → task heads toward multimodal evidence → shared world representation → actions and predictions. This is an architectural direction, not evidence that every block in the slide is the deployed online controller.

![The Waymo Foundation Model diagram with sensor fusion, a driving VLM, and a generative world decoder](/assets/images/waymo-foundation-model-architecture.png)
_Figure: The Waymo Foundation Model architecture shown in [Dmitri Dolgov's Waymo talk](https://www.youtube.com/watch?v=Gp4zrV3-6N8). Screenshot supplied for this post; see Waymo's public [architecture description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) and earlier [foundation-model overview](https://waymo.com/blog/2024/10/ai-and-ml-at-waymo/)._

At a high level, the three blocks make different information contracts:

| Path | Input → intermediate representation | Output role |
| --- | --- | --- |
| Sensor Fusion Encoder | Camera, LiDAR, radar → objects and sensor embeddings | Fast, metric evidence and reactions |
| Driving VLM | Text prompts, sensor data, and autonomous-driving history → VLM embeddings, semantics, rationales, and text tokens | Slower semantic reasoning for rare or complex situations |
| Generative World Decoder | Both representations → a shared world model | Driving actions, agent predictions, and other predictions |

The table's important point is not that the VLM replaces the sensor stack. The sensor-fusion encoder keeps the geometry and latency-sensitive evidence that the earlier sections treated as non-negotiable. The VLM adds a second route for language-conditioned semantics and broader world knowledge; the “thinking fast, thinking slow” annotation makes the timing contract explicit. The decoder then has to reconcile those representations instead of handing language directly to steering.

“Generative world decoder” also changes the target. Perception asks what is present now; a world decoder can be trained to predict how the scene and its agents may evolve, while also producing actions or signals used to validate them. Waymo's public description says its World Decoder predicts road-user behavior, generates maps and vehicle trajectories, and supplies trajectory-validation signals; it also describes adapting large teacher models to the Driver, Simulator, and Critic before distilling smaller students. Those details explain the slide's “versatile” and “multi-stage training” labels, but they do not specify the exact online graph or guarantee closed-loop safety.

This is the next step in the evolution traced by this post: preserve sensor-specific measurement and metric geometry, but expose them to a reusable multimodal model that can carry semantics, history, prediction, and action. It connects directly to [Vision-Language Models: A Reading Guide](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html), which follows the progression from image-text alignment to grounding, video, and action. The Waymo diagram instantiates those interfaces inside a driving system: the VLM contributes semantic context, sensor fusion contributes calibrated world evidence, and the shared decoder must turn both into future predictions and safe behavior.

The open question is therefore sharper than whether a VLM can describe a road scene. Can a shared world decoder remain grounded in metric sensor evidence while using slower language-level reasoning, predict counterfactual futures, and meet the latency and validation requirements of closed-loop driving?

## Design checklist

1. Preserve what each sensor uniquely measures before converting it: camera appearance, LiDAR geometry, and radar range, velocity, and uncertainty.
2. State which variable chooses 3D location: image depth, an object query, or a BEV cell with height samples.
3. Choose the fusion carrier from the output: points for point-aligned evidence, queries for actors, and dense BEV for scene fields.
4. Train every supported sensor configuration, and separate availability from reliability.
5. Diagnose loss scale, training rate, and gradient direction separately.
6. Document runtime inputs, label sources, and teacher models as different graphs.
7. Name what crosses the frame boundary and define birth, age, freshness, and reset behavior.
8. Profile the executed system; sparsity in one module does not make the full pipeline sparse.

The recurring pattern is sensor-specific evidence, calibrated metric conversion, and selective state. Most perception architectures become easier to compare once each paper is located at the boundary it actually changes.
