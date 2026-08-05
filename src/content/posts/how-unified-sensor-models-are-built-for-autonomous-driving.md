---
title: 'Machine Learning for Autonomous Driving Perception'
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
summary: How modern driving systems encode each sensor, fuse in metric space, train several tasks, use LiDAR supervision, and carry scene state through time.
---

# Machine Learning for Autonomous Driving Perception

Modern driving perception is built around a simple split: **sensor-specific encoders at the input, a shared metric representation in the middle, and task-specific outputs at the end**. Cameras, LiDAR, and radar should not enter the network as interchangeable tensors. They measure different quantities and fail differently. They become useful to one another only after the model has preserved those differences and placed the evidence in a common 3D frame.

The same principle applies to tasks and time. Detection, occupancy, lanes, velocity, and tracking can reuse expensive scene reasoning, but they do not need the same output resolution or loss. Dense BEV features are good short-term memory for roads and free space; sparse object queries are better long-term memory for actors. LiDAR can supervise geometry during training even when the deployed model uses cameras alone.

This post explains those design choices and links each mechanism to an individual paper note.

## The system in one graphic

At runtime, sensor-specific encoders feed metric fusion, persistent scene state, and task-specific outputs. The dashed path exists only during training.

[![Unified autonomous-driving perception system with sensor-specific encoders, shared metric geometry, dense and sparse temporal memory, task heads, and training-only supervision](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_A high-level architecture synthesized from the linked literature._

The rest of the architecture follows from four questions: what measurement must survive the encoder, where it becomes metric, what state persists through time, and which tasks can safely update the same parameters.

## Sensor encoders should preserve sensor physics

### Camera

A surround-camera stack normally shares one CNN or vision transformer across views. Intrinsics, extrinsics, distortion, camera identity, and timestamps then tell the geometric layers how those views differ. Sharing the visual backbone saves compute and discourages the model from memorizing one camera position; omitting calibration forces the network to treat viewpoint changes as unexplained appearance variation.

Resolution is part of the sensor model. A distant motorcycle may occupy only a few pixels. Once a high-stride backbone removes it, neither BEV fusion nor a larger decoder can recover it. Multiscale camera features therefore preserve two things: semantic evidence about what is visible and spatial detail about where small or distant evidence remains.

![Animation comparing a coarse-only vision encoder with a multiscale feature pyramid that preserves a small distant actor](/assets/images/autonomous-perception-vision-encoder.gif)
_[EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) repeatedly fuses adjacent feature scales; [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) projects each 3D query into those multiscale camera features. The distant actor must survive the shared backbone before either operation can use it._

### LiDAR

LiDAR supplies direct range and height but samples only a small part of 3D space. Dense 3D convolution wastes most computation on empty voxels, so the main lineage is about where to exploit sparsity and when to compress height.

[VoxelNet](/paper%20shorts/2017/11/17/voxelnet-end-to-end-point-cloud-3d-detection.html) learns features inside voxels. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) keeps the 3D middle encoder sparse before height compression, while [PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) collapses height early and does most work with fast 2D convolution. [VoTr](/paper%20shorts/2021/09/06/votr-voxel-transformer-for-3d-object-detection.html), [SST](/paper%20shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html), and [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) enlarge the receptive field with attention over occupied voxels. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) stays sparse through the detection head instead of constructing a dense BEV heatmap.

![Animation contrasting early pillar-to-BEV compression with a sparse 3D LiDAR encoder that retains height](/assets/images/autonomous-perception-lidar-encoder.gif)
_Watch the blue overpass returns and amber road returns. [PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) encodes each vertical column, collapses height, and runs a 2D CNN. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html), [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html), and [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) retain occupied 3D voxels longer, so stacked surfaces remain distinct before BEV compression._

The trade-off is straightforward. Early height compression is efficient for mostly planar roads. Retaining sparse 3D structure helps with overpasses, stacked geometry, and occupancy. A useful LiDAR token should also carry intensity, acquisition time, and point age because a rotating scan is collected while the ego vehicle and other actors move.

### Radar

Radar is not low-resolution LiDAR. Its useful measurements are range, radial velocity, Radar Cross Section, return time, and uncertainty. Its characteristic errors are poor angular resolution, ambiguous elevation, multipath, and ghost returns. Rasterizing radar into a generic occupancy map too early can erase the velocity and confidence information that justified the sensor.

[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates radar returns with camera proposals in polar coordinates. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) instead uses radar to guide camera lifting and repair alignment with deformable attention. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) gives radar both pointwise and transformer paths before BEV fusion. These choices preserve radar attributes until the model has used them to resolve camera depth or actor motion.

![Animation showing CRAFT polar proposal-return association, CRN radar-guided lifting, and RCBEVDet dual-stream radar encoding](/assets/images/autonomous-perception-radar-encoder.gif)
_[CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates a camera proposal with compatible radar returns in polar coordinates. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to sharpen camera lifting, then repairs BEV misalignment with deformable fusion. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) combines a point stream with a transformer stream to construct the radar BEV._

## Put the sensors in a common metric frame

Cameras observe rays; driving tasks reason in meters. The central camera-perception problem is therefore the view transform: where along each ray should an image feature be placed in 3D?

[Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a categorical depth distribution for each image location, lifts image features along the ray, transforms them with camera calibration, and pools them into bird's-eye view. [BEVDet](/paper%20shorts/2021/12/22/bevdet-high-performance-multicamera-3d-object-detection-in-bev.html) turns that representation into a detector. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) adds explicit LiDAR-derived depth supervision.

![Figure 1 from Lift, Splat, Shoot, showing multiview evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_LSS places perspective features in one vehicle-centered metric grid. Source: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

A dense BEV grid is useful because detection, occupancy, lanes, maps, and free space all need aligned geometry. Its cost, however, is fixed by range and resolution rather than scene content. Doubling both horizontal dimensions roughly quadruples the number of cells, and storing several frames multiplies the memory again.

Query-based models avoid materializing every cell. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) projects 3D object queries into multiscale image features. [PETR](/paper%20shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html) embeds possible 3D coordinates into perspective features before attention. [PETRv2](/paper%20shorts/2022/06/02/petrv2-unified-3d-perception-from-multicamera-images.html) aligns those features over time and gives detection, lanes, and segmentation different query geometries. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) sits between the two approaches: it stores a dense field of learned BEV queries but retrieves evidence through projected attention.

These methods place the irreversible step in different locations. LSS and BEVDepth assign image evidence to depth bins before BEV pooling; a wrong bin becomes a wrong cell. DETR3D and Sparse4D leave features in perspective space and use calibrated 3D queries to retrieve them, so geometric compute scales with queries and sampled points rather than BEV area—but evidence outside that support is never seen. BEVFormer pays for a dense metric state while keeping image retrieval sparse. The representation choice therefore determines both the compute term and where evidence can disappear.

![Animation contrasting lift-and-splat depth distributions, sparse object-query retrieval, and dense BEV queries with sparse image sampling](/assets/images/autonomous-perception-camera-lifting.gif)
_The three columns move geometric commitment to different operations: [LSS](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a depth distribution, expands a frustum, and pools it into BEV; [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) projects a bounded set of 3D object queries and samples nearby image features; [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) starts with a dense field of BEV queries but samples image support sparsely._

The choice should follow the output:

| Representation | Best suited to | Main cost | Main failure |
| --- | --- | --- | --- |
| Dense BEV | Occupancy, roads, maps, weak scene evidence | Area × resolution | Depth errors are committed to cells |
| Object queries | Detection, tracking, motion | Queries × sampled views | Missed or poorly initialized queries |
| Sensor-native features | Fine appearance or geometry | Repeated projection and sampling | Harder cross-sensor alignment |

Many strong systems use more than one representation: dense BEV proposes and describes the scene, while sparse queries carry actors.

## Fuse at the granularity required by the output

Sensor fusion differs mainly in where correspondence is imposed.

[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) projects camera class scores onto LiDAR points before voxelization. The operation is cheap and interpretable, but only pixels touched by LiDAR survive. A calibration error can also attach the wrong semantic score to a point.

[TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) uses LiDAR to propose object queries and lets each query attend to an image region. [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) projects a shared 3D reference point into camera, LiDAR, and radar feature spaces. Query fusion is a natural fit for actor detection because the output is already a sparse set.

[BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) independently encodes camera and LiDAR, converts both to dense BEV, fuses them once, and shares the result across detection and mapping heads. This retains dense image semantics for scene-level outputs. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) keeps the camera and LiDAR streams separate and lets them update one another, which preserves modality provenance. [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) shares transformer blocks after modality-specific token construction while retaining appropriate 2D and 3D neighborhoods.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion specializes tokenization, shares metric scene processing, and separates the output heads. Source: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

There is no single best fusion layer. Use point fusion when the association is reliable and the output follows points, query fusion for actors, and BEV fusion for dense scene structure. Keep a sensor-native path when downstream tasks still need precise appearance, height, Doppler, or uncertainty.

![Animation comparing point-level, object-query, and bird's-eye-view sensor fusion](/assets/images/autonomous-perception-fusion-granularity.gif)
_[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) projects pixel semantics onto LiDAR points before voxelization. [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) projects one 3D reference point into camera, LiDAR, and radar features for an actor query. [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) first converts camera and LiDAR features into aligned BEV maps, then fuses the two fields._

A fused model is not automatically a fallback model. [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) reports only 3.0 camera-only mAP when training always includes both sensors, versus 35.0 when modality dropout exposes the same network to missing-input modes and fusion is normalized over the streams that remain. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) similarly lets BEV queries select whichever modality is available. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) handles the harder case in which a stream is present but unreliable. Sensor availability and health must condition fusion; replacing a failed sensor with zeros is not a calibrated fallback.

![Animation comparing UniBEV modality dropout, MetaBEV available-modality attention, and Grace-BEV reliability gating](/assets/images/autonomous-perception-modality-dropout.gif)
_[UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) drops modalities during training and normalizes the sum over streams that remain. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) lets BEV queries retrieve from whichever encoder is available. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) estimates trust so a present-but-degraded stream can be downweighted rather than treated as healthy._

## Multi-task learning: share the trunk, control the gradients

Detection, occupancy, lanes, velocity, and tracking all need semantic features, metric geometry, and ego motion. A shared trunk avoids recomputing those features and can make the outputs geometrically consistent. The heads should remain specialized because the tasks differ in resolution, label density, sensor affinity, and error tolerance.

The practical default is a shared geometric trunk with task-specific decoders or adapters. [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) follows this pattern. [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) goes further: tracking and map queries become inputs to motion prediction, occupancy, and planning. That explicit task graph can improve consistency, but an upstream error now propagates into more consumers.

Multi-task optimization has three distinct failure modes:

| Problem | Symptom | Useful response |
| --- | --- | --- |
| Loss scale | Dense segmentation dominates sparse regression numerically | Normalize by valid cells, matched objects, lane points, or trajectories; then use learned weighting |
| Learning speed | One task converges while another stalls | Track shared-layer gradient norms; consider [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) |
| Gradient conflict | Tasks request opposing shared updates | Measure gradient cosine similarity; use adapters, partial splits, alternating updates, or [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html) |

[Homoscedastic uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) learns one scale per task:

$$
\mathcal{L}=\sum_i\left[\frac{1}{2\sigma_i^2}\mathcal{L}_i+\log\sigma_i\right].
$$

The inverse variance changes the task weight and the logarithm prevents every weight from collapsing to zero. This is useful after each loss has a meaningful unit. It does not solve conflicting gradients or per-scene sensor reliability. If weighting fails, the shared representation may need separate normalization, adapters, or an earlier split.

![Animation comparing Kendall uncertainty weighting, GradNorm gradient balancing, and PCGrad conflict projection](/assets/images/autonomous-perception-multitask-gradients.gif)
_[Kendall et al.](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) learn one uncertainty scale per task and use it to rescale that loss. [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) adjusts task weights so shared-layer gradient norms track relative training rates. [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html) changes direction instead of magnitude: when two task gradients conflict, it projects away the opposing component._

## Use LiDAR during training without requiring it at inference

“Using LiDAR for depth” describes three different deployment contracts:

| System | Runtime input | What LiDAR does |
| --- | --- | --- |
| Camera depth estimation | Cameras only | Supplies depth labels during training |
| Depth completion | Cameras plus sparse runtime depth | Remains an inference input |
| Teacher-student distillation | Cheaper sensor set | Supervises features, occupancy, or predictions offline |

[BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) is the clean first case. Projected LiDAR trains the camera depth distribution, but inference uses images and calibration. The camera model learns where image evidence should be lifted into BEV without carrying LiDAR hardware in the deployed graph.

Depth completion is different. [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), [DeepLiDAR](/paper%20shorts/2018/12/02/deeplidar-surface-normal-guided-depth-completion.html), [NLSPN](/paper%20shorts/2020/07/20/nlspn-non-local-spatial-propagation-network-for-depth-completion.html), and [GuideFormer](/paper%20shorts/2022/06/19/guideformer-transformers-for-image-guided-depth-completion.html) densify sparse depth that is still present at runtime. Removing that sensor changes the model input, not merely its supervision.

Distillation permits a larger sensor gap. [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html) transfers feature, relation, and response knowledge from a camera-LiDAR teacher to a camera-radar student. [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) uses image-LiDAR sequences to create 4D occupancy supervision, then removes the pretraining decoder for downstream camera tasks.

![Animation contrasting LiDAR depth labels, runtime depth completion, and teacher-student distillation](/assets/images/autonomous-perception-lidar-training-contracts.gif)
_The dashed boundary separates training from driving. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) projects LiDAR into camera depth labels and removes LiDAR at inference. [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html) consumes sparse runtime depth, so that sensor remains deployed. [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html) transfers knowledge from a camera-LiDAR teacher into a camera-radar student._

The deployed graph and the label-generation graph should be documented separately. Offline labeling can use LiDAR, future frames, repeated passes, large models, and human review. Onboard inference can remain camera-only or camera-radar. Tesla's public [AI material](https://www.tesla.com/AI) and [2021 AI Day presentation](https://www.youtube.com/watch?v=j0z4FweCy4M) support this general separation between expensive offline labeling and camera-based onboard inference; they do not establish that every deployed network is distilled from a LiDAR teacher.

Projected LiDAR is also imperfect supervision. Occlusion, timestamp mismatch, actor motion, and calibration error can move a return across an object boundary. Depth-label generation therefore needs visibility checks, pose interpolation, confidence, and ignore regions.

## Temporal modeling: choose what persists

A single frame does not provide stable velocity, identity through occlusion, or enough parallax for reliable depth. Temporal models differ in the state they retain.

### Dense scene memory

[BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) warps the previous camera-BEV feature into the current ego frame and fuses it with the current feature. BEVFormer carries recurrent BEV queries. [SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) uses short high-resolution stereo for fine correspondence and a longer low-resolution BEV history for depth and velocity.

Dense memory preserves roads, free space, background, and weak evidence that has not become an object. Its raw state grows approximately as $O(HWTD)$ for grid size $H\times W$, history $T$, and channel dimension $D$. Ego warping aligns static structure, but moving actors still need learned alignment.

### Sparse object memory

[Sparse4D](/paper%20shorts/2022/11/19/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion.html) represents actors as 3D anchors. Each anchor predicts keypoints, projects them into cameras and timestamps, samples nearby image features, and refines the object state. Fusion cost moves from every BEV cell toward a bounded number of hypotheses.

[Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) transforms previous instances into the current frame and reserves new anchors for births. Only the prior sparse state crosses the frame boundary, reducing temporal decoder scaling from $O(T)$ to $O(1)$ in history length. [Sparse4D v3](/paper%20shorts/2023/11/20/sparse4dv3-end-to-end-3d-detection-and-tracking.html) adds temporal denoising and separate box-quality estimation so propagated queries learn to recover geometric error instead of treating class confidence as localization quality.

[StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) keeps a FIFO memory of top foreground queries, transforms their reference points with ego pose, and discards background queries. [SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) instead reopens several timestamps through learned support points, so its retrieval cost still grows with the number of frames. Both are called sparse, but one compresses history into recurrent state while the other sparsifies access to stored observations.

![Animation contrasting dense BEV scene memory, sparse recurrent object memory, and a hybrid temporal state](/assets/images/autonomous-perception-temporal-memory.gif)
_[BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) and [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) carry dense scene context; [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) and [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) carry selected actor state. The hybrid column is a synthesis: keep dense context briefly for scene structure and query birth, then retain actors sparsely for longer._

![Figure 3 from StreamPETR, showing object queries propagated through a temporal memory queue](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR carries selected object state instead of a sequence of full scene grids. Source: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

Sparse recurrence makes a missing frame computationally cheap, not semantically harmless. The model can continue from memory, but an unobserved actor state becomes stale while discarded background evidence cannot be recovered until a fresh query is born. A deployed state therefore needs age, observation freshness, confidence, birth, and reset rules; the papers establish learned recurrence, not a complete fallback policy.

The useful default is hybrid: short dense memory for road structure, free space, and query birth; longer sparse memory for actors and vectorized map elements. [SparseDrive](/paper%20shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html) extends this object-and-map state into motion prediction and planning.

## Sparse transformers reduce specific costs, not the whole system

Sparse LiDAR encoders avoid processing empty 3D space. Sparse camera detectors avoid constructing or repeatedly updating every BEV cell. Both save work, but neither makes the full perception stack sparse.

Surround-camera backbones still process every image, often at several scales. Sparse attention also introduces indexing, sorting, padding, and irregular memory access. Its real benefit depends on the accelerator and compiler, not only FLOPs. A useful latency profile separates image encoding, geometry, fusion, temporal memory, heads, and postprocessing.

The key sparse-transformer papers change different terms. VoTr, SST, and DSVT bound attention among occupied LiDAR voxels. VoxelNeXt removes the dense LiDAR detection head. Sparse4D and StreamPETR bound object-level temporal memory. SparseBEV replaces a full camera-BEV grid with learned pillar support points. Those are complementary choices rather than one architecture family with a single efficiency claim.

Measure end-to-end P95 and P99 latency, peak memory, and recall under dense scenes. Average FPS can hide synchronization, transfer, sparse-kernel overhead, and the rare frame in which many queries or voxels become active.

## Pretraining should teach geometry and persistence

Image-classification pretraining teaches appearance but not calibration, cross-view correspondence, metric depth, ego motion, or temporal persistence. Driving pretraining should use synchronized sensor packets or clips and a target that requires those relationships.

[UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) masks camera patches and LiDAR voxels, maps visible features into a shared 3D volume, and reconstructs both modalities. [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) masks vertical LiDAR columns and predicts normalized point coordinates and density, matching the structure of outdoor scans. Its reported improvement is larger in low-label regimes and narrows with the full Waymo training set.

[UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) predicts current and future 4D occupancy. [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html) predicts future point clouds from historical images. [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) separates dynamic memory from propagated static state and adapts the result to downstream tasks.

| Pretraining target | What it teaches | Main limitation |
| --- | --- | --- |
| Masked camera and LiDAR reconstruction | Cross-modal correspondence | Local correlated evidence can make reconstruction easy |
| Current occupancy | Metric scene structure | Requires geometric labels or a teacher |
| Future occupancy or point clouds | Motion and persistence | A single future cannot represent every valid outcome |
| Teacher features and pseudo-labels | Task-specific abstractions | Inherits teacher errors and blind spots |

Longer pretraining targets are not automatically better. UniWorld reports that three target frames outperform five; multi-sweep occupancy becomes less reliable as motion, occlusion, pose error, and future ambiguity accumulate. Scale should therefore mean more distinct geometric and temporal situations, with transfer measured across tasks and label budgets, rather than a longer target horizon or a larger count of near-duplicate frames.

## Practical takeaways

1. Encode each sensor around what it uniquely measures: camera semantics, LiDAR geometry, and radar velocity and uncertainty.
2. Use vehicle-frame geometry as the shared interface. Choose dense BEV for scene fields and sparse queries for actors.
3. Fuse at the output's natural granularity instead of forcing every modality into one tensor.
4. Share expensive geometry across tasks, but keep task decoders, loss units, and safety thresholds explicit.
5. Treat loss scale, learning speed, and gradient conflict as different multi-task problems.
6. Separate the label-generation graph from runtime, and train every supported sensor configuration explicitly. Privileged LiDAR can supervise camera-only inference; depth completion cannot remove its sparse-depth input.
7. Keep short dense temporal state for scene discovery and longer sparse state for tracked entities, but age or reset state when observations go missing.
8. Profile the executed system. Sparse fusion does not remove the cost of camera backbones or irregular memory access.
9. Pretrain on geometry, correspondence, and future state, then verify transfer across tasks and data regimes.

The core pattern is specialized sensing, shared geometry, and selective state. Most modern perception papers can be understood by asking which of those three boundaries they move.
