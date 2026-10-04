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
summary: How modern autonomous-driving systems preserve sensor-specific evidence, align it in space and time, fuse it into a persistent world state, and expose that state to planning.
---
# Autonomous-Vehicle Perception, circa 2026

Autonomous-vehicle perception combines camera, LiDAR, and radar measurements into road geometry, actors, free space, and motion. The literature has moved fusion from projected points to object queries, dense BEV fields, and shared multimodal backbones. Temporal models make a related choice between storing a spatial field and carrying a bounded set of actor hypotheses.

Each representation preserves different evidence. Cameras provide texture and semantics but uncertain depth. LiDAR provides sparse metric geometry. Radar adds range and radial velocity with weak angular localization. Calibration and timing determine whether their features describe the same location and instant.

## Overall architecture

At runtime, the perception stack roughly follows this sequence: encode each sensor stream, align the resulting evidence in space and time, fuse the modalities, carry the scene state across frames, and pass that state to prediction and planning. Information can be lost at every step, and later stages cannot always recover it.

[![Autonomous-driving perception pipeline from sensor-specific encoders through coordinate alignment, dense or sparse temporal state, and task heads, with learned latent tokens and a separate training-only graph](/assets/images/autonomous-driving-perception-system.svg)](/assets/images/autonomous-driving-perception-system.svg)
_Sensor-specific encoders preserve native evidence before geometry and timing make it comparable. Fusion then writes into a dense field, sparse entities, latent tokens, or a mixture of the three. The deployed graph exposes selected state to downstream tasks; dashed paths add supervision only during training._

## Modality-specific encoders

Cameras, LiDAR, and radar produce measurements with different structure, uncertainty, and failure modes. Their encoders therefore need different inductive biases: image backbones preserve dense appearance and fine boundaries, point or voxel encoders preserve sparse 3D geometry, and radar encoders preserve range, Doppler, and return confidence.

### Camera encoders

Cameras capture color and texture at high spatial resolution. That makes them useful for reading traffic lights, distinguishing lane paint from road cracks, and tracing object boundaries. Their geometric weakness is depth: a pixel identifies a ray through the camera, not a distance along it.

The encoder must retain a distant cyclist without losing the context of the surrounding intersection. Convolutional backbones remain attractive because their local computation maps well to optimized inference libraries. [NVAutoNet](/paper%20shorts/2023/03/23/nvautonet-fast-and-accurate-360-3d-visual-perception-for-self-driving.html), for example, combines efficient CNN backbones with precomputed camera-to-BEV geometry and reports 53 FPS for the full multitask network on NVIDIA DRIVE Orin. Feature pyramids combine high-resolution maps that retain precise spatial detail with lower-resolution maps whose larger receptive fields capture more of the scene; [EfficientDet](/paper%20shorts/2020/04/01/efficientdet-scalable-and-efficient-object-detection.html) makes this exchange bidirectional and learns how strongly to weight each input.

[ConvNeXt](/paper%20shorts/2022/01/10/a-convnet-for-the-2020s.html) modernizes ResNet with a patchify stem, depthwise 7-by-7 spatial mixing, inverted bottlenecks, and LayerNorm. Its controlled path shows that a pure CNN can absorb many post-ViT design choices without attention.

Vision transformers can connect distant image regions directly, but full attention across every token from every surround camera is too expensive. [Deformable DETR](/paper%20shorts/2020/10/08/deformable-detr-deformable-transformers-for-end-to-end-object-detection.html) replaces dense attention over image features with a small set of learned samples around each query's reference point.

The camera backbone also has to learn which evidence will matter after projection. When supervision arrives only after camera features have been projected into 3D, that learning signal is indirect. [BEVFormer v2](/paper%20shorts/2022/11/18/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition.html) adds perspective-view supervision before projection, giving the backbone a direct reason to preserve image evidence that fusion cannot recover later.

### LiDAR encoders

LiDAR measures range directly, but its point cloud is sparse, irregular, and acquired over the duration of a sweep. A LiDAR encoder should retain each point's 3D position, return intensity, and acquisition time. It can then collapse the cloud into a dense bird's-eye-view grid early or preserve sparse 3D structure deeper into the network.

[PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) takes the aggressive route: pool points in vertical columns, scatter them into a dense pseudo-image, and let fast 2D convolutions do the rest. The explicit height axis disappears early, although the pooled pillar feature still encodes point heights. [SECOND](/paper%20shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html) delays that loss by keeping a sparse 3D voxel grid through the middle of the network. Sparsity avoids computation in empty space, but the occupied voxels still need to exchange context. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) groups them into fixed-size attention sets and alternates the grouping direction between layers, allowing context to cross set boundaries without global attention. [VoxelNeXt](/paper%20shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html) keeps the representation sparse through prediction: selected occupied voxels predict boxes directly, and sparse max pooling replaces peak extraction over a dense BEV heatmap.

PointPillars trades vertical detail for regular 2D computation. Sparse voxel models retain more of the 3D structure, but every layer must map active input voxels to active outputs, then gather and scatter their features through memory. That bookkeeping and irregular access can erase part of the arithmetic savings. The right point to become dense depends on the hardware and the outputs the model must support.

![PointPillars learns pillar features and scatters them into a 2D pseudo-image](/assets/images/pointpillars-fast-point-cloud-encoders-paper-figure.webp)
*PointPillars makes the height reduction explicit: the pillar feature network summarizes each vertical column before the 2D backbone. SECOND and later sparse-voxel models postpone or avoid that reduction. Source: [PointPillars, Figure 2](https://arxiv.org/abs/1812.05784).*

### Radar encoders

Radar contributes range and radial velocity and is especially useful when poor lighting weakens cameras or adverse weather degrades camera and LiDAR measurements. A return may also include azimuth, elevation, radar cross section, timestamp, and sensor confidence. Its weaknesses are poor angular localization, sparse semantics, multipath, and ghost returns. Radar supplies constraints, not clean objects.

The encoder should preserve the measurements that distinguish radar from LiDAR. A point encoder supports proposal-level association. A polar tensor respects the sensor's native range-azimuth sampling. A radar-BEV encoder builds an independent metric field for later fusion. Rasterizing returns into binary occupancy too early discards Doppler, confidence, and ambiguity—the very signals that make radar useful.

Radar has gained influence by intervening earlier. [CRAFT](/paper%20shorts/2022/09/14/craft-camera-radar-3d-object-detection-with-spatio-contextual-fusion-transformer.html) associates returns with camera proposals, which leaves the camera branch in control. [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) lets radar refine camera depth before the branches meet in BEV. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) builds an independent radar representation first. The intervention point determines how much radar evidence can survive camera errors—and how far radar errors can spread. Radial velocity also leaves a blind spot: a cyclist moving mostly across the line of sight can have little Doppler even while moving quickly. The temporal model still needs geometry and repeated observations to estimate the full motion.

A second branch changes the radar input representation: using more of what radar actually measures. [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html) showed that metadata retention, accumulated sweeps, and outlier filtering materially affect fusion; preprocessing is part of the model. [Doppler-Aware LiDAR–Radar Fusion](/paper%20shorts/2025/10/23/doppler-aware-lidar-radar-fusion-for-weather-robust-3d-detection.html) keeps power and Doppler separate during interaction. [DinoRADE](/paper%20shorts/2026/04/09/dinorade-full-spectral-radar-camera-fusion.html) moves toward dense spectral tensors rather than LiDAR-like points. Radar becomes more valuable as the encoder stops forcing it to imitate another sensor.

The encoders now contain complementary evidence, but their features are still tied to different coordinate systems and acquisition times.

## Aligning sensor evidence in space and time

Camera intrinsics map pixels to viewing rays. Sensor extrinsics express those rays, LiDAR points, and radar returns in the vehicle frame. Timestamps and ego-motion estimates then transform observations to a common time. Calibration tells us where measurements could correspond; it does not establish that two returns came from the same actor or add missing observations.

That distinction matters when the geometry is wrong. An extrinsic error shifts projected evidence across the bird's-eye-view (BEV) grid, with angular misalignment producing larger displacement at longer range. A clock error moves a dynamic actor because the sensors observed it at different moments. Ego-pose error shifts the whole transformed scene. Fusion can then combine individually valid measurements that do not describe the same place or time.

Camera features need an additional transformation because they begin in perspective-view coordinates. A pixel identifies a viewing ray, but not where along that ray the evidence lies. Modern methods take one of two directions: push image features into metric space, or start with a metric hypothesis and pull image evidence into it.

### Two ways to place camera features in 3D

[Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) predicts a depth distribution at each image location, copies the image feature across candidate depths, and pools the lifted features into BEV. The original depth distribution is latent: downstream BEV losses teach it only through the final task. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) adds projected LiDAR depth supervision during training. Both methods produce dense coverage, but a depth error writes evidence into the wrong metric cell.

![Lift Splat Shoot outer product of a depth distribution and a pixel feature](/assets/images/lift-splat-shoot-source-figure-3-lift.png)
*LSS multiplies one feature vector by a distribution over depth bins. Each candidate range receives a weighted copy before geometric pooling; the feature has not yet been assigned to one depth. Source: [Lift, Splat, Shoot, Figure 3](https://arxiv.org/abs/2008.05711).*

Query-based methods start with a 3D hypothesis and retrieve image evidence for it. [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) assigns each object query a 3D reference point, projects that point into the cameras, and samples the camera feature maps at the projected pixels. [PETR](/paper%20shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html) samples candidate points along each camera ray, transforms them into the ego frame using calibration, and encodes those coordinates as a positional embedding for the image feature at that pixel. Object queries then attend over image features that already carry 3D location hypotheses. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) starts from a dense grid of BEV queries and samples several heights along each vertical pillar. The first two preserve candidate actors; the third preserves a field that can support lanes, free space, and occupancy.

![BEVFormer spatial cross-attention and temporal self-attention](/assets/images/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers-paper-figure.png)
*BEVFormer reverses the retrieval direction: BEV queries sample projected image features and aligned history. Its dense grid survives the view transformation and can feed several scene-level heads. Source: [BEVFormer, Figure 2](https://arxiv.org/abs/2203.17270).*

Controlled comparisons complicate claims that one projection operator is intrinsically better. In [Simple-BEV](/paper%20shorts/2022/06/16/simple-bev-what-really-matters-for-multi-sensor-bev-perception.html), image resolution and effective batch size changed vehicle-segmentation performance more than the lifting choice in the tested setup. A meaningful comparison must hold the backbone, resolution, schedule, batch size, and sensor inputs fixed. Otherwise, the training recipe is being credited to the geometry module.

## Choosing where to fuse sensor evidence

Fusion can happen before encoding, between sensor encoders, or after each modality has produced task predictions. Raw measurements are difficult to combine directly because cameras, LiDAR, and radar sample the scene differently. Prediction-level fusion keeps each branch modular, but weak evidence discarded by one detector cannot be recovered by another. Most modern systems therefore fuse encoded features before the task heads.

[PointPainting](/paper%20shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html) can fuse camera evidence only where LiDAR returns exist. It attaches camera scores to those points, keeping computation low but making LiDAR sparsity a hard limit on camera contribution. [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) replaces that point-wise attachment with soft attention from object queries to image regions. Each query can search beyond a single projected pixel, which makes the association less brittle to small calibration errors; an image-guided initialization path can also add candidates that the LiDAR heatmap misses. Fusion is still organized around candidate objects rather than the full scene.

[FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) makes the object query modality-agnostic. Each query projects its 3D reference point into the cameras, samples LiDAR and radar features at the same BEV location, and combines the results to predict a box. [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) removes the object-query limit by fusing aligned camera and LiDAR features at every BEV cell before the task heads.

![BEVFusion camera and LiDAR branches meet in a shared BEV](/assets/images/bevfusion-multi-task-multi-sensor-unified-bev-source-figure-2.webp)
*BEVFusion keeps both sensor encoders and view transformations, then joins their dense BEV fields before detection and map segmentation. This preserves camera features at locations without LiDAR returns. Source: [BEVFusion, Figure 2](https://arxiv.org/abs/2205.13542).*

BEVFusion combines two completed BEV branches once. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) instead keeps separate image and LiDAR representations, exchanges features in both directions during encoding, and alternates object queries between the two streams during decoding. [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) shares the transformer weights themselves. Its image and LiDAR tokenizers remain separate, while 2D and 3D neighborhood partitions control which tokens interact. Moving fusion into the backbone lets camera semantics reshape LiDAR features and LiDAR geometry reshape camera features before detection. A degraded stream can now alter the other branch across several layers rather than at one final fusion block.

A missing sensor changes the input distribution and the scale of the fused features. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) trains camera-only, LiDAR-only, and fused modes; its BEV queries cross-attend to whichever sensor features are available, while modality-specific experts adapt the decoder to each mode. [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) also uses modality dropout, then normalizes channel-wise fusion weights over the streams that remain. The training coverage matters: in UniBEV's ablation, a model trained only with both sensors falls to 3.0 camera-only mAP even though the camera branch still runs.

Their dropout recipes explicitly cover which sensors are present. A degraded sensor passes that binary check while supplying unreliable features. [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) estimates a continuous trust score from the LiDAR features, uses it to balance a LiDAR-guided expert against a vision-only expert, and gates the fused BEV features. An absent stream can be masked; a degraded stream must first be detected and down-weighted.

## Building state across time

To combine observations across frames, the model first expresses earlier evidence in the current ego frame. Ego motion is enough for the static scene; moving actors require an additional motion estimate. Pose and timestamp errors, camera rolling shutter, and motion during a LiDAR scan limit the accuracy of this alignment.

A compact abstraction is

$$
S_t = f\left(\operatorname{Align}(S_{t-1}, \Delta T_t), X_t, \Delta t, H_t\right),
$$

The prior state $S_{t-1}$ is aligned using the ego-frame transform $\Delta T_t$, then updated with current evidence $X_t$, elapsed time $\Delta t$, and sensor health $H_t$. The methods below differ mainly in the representation of the updated state $S_t$ and its update rule.

In particular, warping a stored cyclist into the current ego frame only corrects the vehicle’s motion. It does not tell us how far the cyclist moved while occluded. That needs an actor-motion prediction, followed by correction when new measurements arrive.

Dense temporal models carry a BEV field from one frame to the next. [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) establishes the simple baseline: warp the previous field with ego motion, concatenate it with the current field, and let another BEV encoder learn from the displacement between them. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) changes the update rule. Each current BEV query uses temporal attention to retrieve features near its aligned location in the previous field. [SOLOFusion](/paper%20shorts/2022/10/05/solofusion-temporal-multiview-3d-object-detection.html) then separates the history by purpose: a few recent frames remain at high resolution for stereo matching, while a longer sequence of low-resolution BEV features provides wider temporal baselines for depth and motion.

Object-centric models carry a bounded set of 3D hypotheses instead of the full field. At each frame, [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) transforms the foreground queries retained in its queue into current coordinates, updates them with the latest image features, and introduces fresh queries that can detect new actors. [Sparse4D v2](/paper%20shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html) applies the same recurrent idea to 3D instance anchors: it transforms the previous anchors and features, adds proposals from the current frame, and refines both together. Because only the latest instance set crosses the frame boundary, decoder cost no longer grows with the nominal history length.

![StreamPETR transforms and updates a bounded queue of object queries](/assets/images/streampetr-paper-figure-3.png)
*StreamPETR carries selected object queries through time. Ego transformation aligns their coordinates; current image features and fresh queries update the state; top-scoring outputs replenish the queue. Source: [StreamPETR, Figure 3](https://arxiv.org/abs/2303.11926).*

[SparseBEV](/paper%20shorts/2023/08/18/sparsebev-high-performance-sparse-3d-object-detection.html) makes a different trade. It retains camera features from several frames, then lets each pillar query sample a small set of 3D locations across that history. This avoids a dense BEV memory without compressing the past into recurrent object state, although inference cost still grows with the number of stored frames.

A hybrid memory can retain a coarse BEV field for free space and uncertain geometry while tracking actors and map elements as explicit instances. The field and the instances should carry age and uncertainty. Compute is only part of the tradeoff; latency, memory, query saturation in crowded scenes, and accuracy after long occlusions matter too.

## Downstream interfaces

Different parts of the autonomy stack need different views of the scene. Dynamic actors are exposed as detections or tracks. A detection typically carries a class, 3D box, confidence, and geometric uncertainty; a track adds identity, velocity, and age. Semantic segmentation assigns a class to each image pixel or BEV cell, while occupancy represents space in 3D. Lane boundaries, centerlines, curbs, and stop lines are represented as polylines or polygons, with attributes such as type, direction, connectivity, confidence, and age.

Bounding boxes work well for discrete actors, but they do not describe free-form geometry or space that belongs to no object instance. Occupancy instead assigns a state to each voxel. Occupied, free, and unobserved space need different interpretations, although a model may use visibility masks rather than predict an explicit unknown class. [Occ3D](/paper%20shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html) provides semantic labels and visibility masks, allowing evaluation to distinguish errors in observed space from predictions in regions the cameras could not see. [PanoOcc](/paper%20shorts/2023/06/16/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation.html) adds instance identities to foreground voxels, so one volume can represent both surfaces and individual actors. [OccAny](/paper%20shorts/2026/03/24/occany-generalized-unconstrained-urban-3d-occupancy.html) asks whether metric occupancy can transfer to new datasets without known target-camera calibration. Its results suggest that geometry transfers more readily than fine-grained semantics.

Road structure needs a vector interface. A raster can mark every lane pixel without preserving which points form one boundary or which direction the lane runs. [MapTR](/paper%20shorts/2022/08/30/maptr-structured-modeling-and-learning-for-online-vectorized-hd-map-construction.html) instead predicts lane dividers, road boundaries, and pedestrian crossings as point sets, while its matching objective handles equivalent point orderings.

The vector loss is sparse: direct geometric supervision comes from matched map queries and sampled points along each element. [MapTRv2](/paper%20shorts/2023/08/10/maptrv2-an-end-to-end-framework-for-online-vectorized-hd-map-construction.html) adds one-to-many matching so more queries receive positive targets, then supervises the image and BEV features with auxiliary depth and segmentation losses. These branches provide denser training signals and are removed at inference; the deployed output remains a set of vectors.

[MGMap](/paper%20shorts/2024/04/01/mgmap-mask-guided-learning-for-online-vectorized-hd-map-construction.html) keeps the dense spatial cue inside the decoder. It predicts instance masks from multiscale BEV features, uses them to update map queries, and samples local mask patches to refine individual points. The final interface is still vectorized, but the mask-guided computation remains part of inference. That distinction has a measured cost: MGMap’s runtime ablation rises from 64.7 to 84.3 ms for its enhanced neck and local refinement. A dense training target can disappear after learning; a dense feature used to move a predicted point must still be computed on the vehicle.

No fixed schema captures every useful feature. The model can retain latent state internally while decoding tracks, occupancy, and vector map elements at its boundary. Explicit outputs give prediction, planning, simulation, and validation defined fields to consume and check; latent features retain context for learned downstream heads.

Explicit outputs are compatible with end-to-end training. A planning loss can propagate through perception and prediction while detections, occupancy, and map elements remain supervised and inspectable. A modular graph is easier to test only when the values passed between modules have defined semantics, units, uncertainty, and age.

In [UniLION](/paper%20shorts/2025/11/03/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns.html), cameras and LiDAR keep separate encoders while their sparse voxels share a linear group RNN backbone. Task-specific heads decode detection, tracking, map segmentation, occupancy, motion prediction, and planning from the resulting BEV feature.

The deployed model defines what must run on the vehicle; training can use additional sensors, labels, and teacher models.

## Training the representation

The deployed sensor set does not determine which signals can be used during training. [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html) consumes sparse depth at inference. [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) instead uses projected LiDAR returns only to supervise camera depth. [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html) moves LiDAR one step further away by using a camera-LiDAR teacher to train a camera-radar student.

![CRKD camera-LiDAR teacher supervises camera-radar student through several losses](/assets/images/crkd-source-figure-2-full-architecture.png)
*CRKD transfers radar objectness, gated camera features, fused features, spatial relations, and detection responses. The LiDAR branch belongs to the teacher; the student retains camera and radar at inference. Source: [CRKD, Figure 2](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.pdf).*

BEVDepth and CRKD remove LiDAR from the vehicle, not from data collection. If LiDAR is unavailable altogether, metric supervision must come from radar range, stereo or temporal correspondence with a known baseline, simulation, map or occupancy labels, or an external teacher. Monocular images without a metric reference determine geometry only up to scale. The replacement signal then becomes the source of scale, calibration, and domain error.

Once the geometric signal is chosen, the pretraining target determines what the encoder must preserve. [BEV-MAE](/paper%20shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html) masks entire columns in a LiDAR voxel grid and reconstructs their point coordinates and density, teaching the encoder local shape and range-dependent sampling. [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) masks camera patches and LiDAR voxels, aligns their visible features in a shared 3D volume, and reconstructs each sensor with a separate decoder. Its target requires the two modalities to exchange evidence, but both methods still reconstruct the current scene.

Reconstructing the present does not require the state to predict change. [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) therefore uses fused multi-frame LiDAR to generate 4D occupancy targets for a camera encoder. [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html) predicts future LiDAR returns from historical images while conditioning the update on ego motion. [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) predicts current and future occupancy and actions while maintaining separate updates for dynamic actors and static structure. These objectives train the transferred state to retain persistence and motion, though all three still depend on LiDAR-derived targets.

[WPT](/paper%20shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html) moves prediction from representation pretraining into policy training. Its world model rolls candidate actions forward through agents, occupancy, and road structure; a learned reward scores the resulting futures; query and reward distillation train a smaller policy. The student runs without the world model at 64 ms planning latency, compared with 312 ms for the teacher, but its reported collision rate rises from 0.11% to 0.24%. The world model can leave the vehicle graph, but its decision quality is not transferred perfectly.

## End-to-end planning

Here, end to end describes the training graph, not an opaque sensor-to-control model. [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) builds dense BEV features, then passes agent and map queries through tracking, motion prediction, occupancy, and ego-planning modules. Each task retains its own supervision, but the modules are trained as one planning-oriented pipeline rather than as independent products.

![UniAD passes tracking and map queries through prediction to planning](/assets/images/uniad-planning-oriented-autonomous-driving-paper-figure.png)
*UniAD exposes the dependencies between tasks. Track and map queries condition motion prediction; occupancy and predicted agents then inform the ego planner. Joint training retains supervised intermediate representations. Source: [UniAD, Figure 2](https://arxiv.org/abs/2212.10156).*

UniAD still relies on dense BEV and occupancy fields. [VAD](/paper%20shorts/2023/03/21/vad-vectorized-scene-representation-for-efficient-autonomous-driving.html) replaces that planning interface with vectors for agents, their motion, and map elements. The planner reasons over explicit instances and geometric constraints instead of dense raster features and hand-designed post-processing. VAD reports 2.5× faster inference for its base model than the previous best method in its comparison. The tradeoff is recall: if the vector extractor misses an actor or road boundary, the planner has no dense field in which that evidence can remain.

The next problem is that driving has more than one plausible action. [DiffusionDrive](/paper%20shorts/2024/11/22/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving.html) starts from clustered human-trajectory anchors and refines several candidates with two denoising steps, preserving distinct maneuvers without a long diffusion chain. [Driving on Registers](/paper%20shorts/2026/01/08/driving-on-registers.html) separates candidate generation from candidate scoring: one decoder proposes trajectories, while another predicts safety, comfort, and efficiency scores. Coverage and selection are different failure modes. A diverse generator is useless if the scorer chooses the wrong trajectory.

Deploying these systems is difficult to validate. Open-loop evaluation measures how closely a plan matches recorded driving, not how the system responds when its own action changes the scene. High-fidelity 3D simulation is expensive, and its sensor and agent models introduce their own errors. Joint training also couples the failure modes: a change to perception can alter prediction and planning even when the intermediate benchmark improves.

Joint training is still useful because detection, prediction, and planning otherwise optimize separate proxies. Planning loss gives the shared representation an action-level objective, while detection, map, occupancy, and motion losses keep its geometry explicit. The deployment problem is to preserve that coordination while retaining component tests, closed-loop simulation, sensor-corruption tests, and an independent check on the selected trajectory.

## Waymo's world model

Waymo's [December 2025 description](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) presents the Waymo Foundation Model as a world model shared across the Driver, Simulator, and Critic. Its interfaces contain both learned embeddings and structured state such as objects, semantic attributes, and roadgraph elements. The embeddings support end-to-end training; the structured outputs support onboard validation, simulation, and evaluation.

Two components feed the World Decoder. The Sensor Fusion Encoder combines camera, LiDAR, and radar observations over time and produces objects, semantics, and learned embeddings. The Driving VLM uses camera context and is trained using Gemini for rare or semantically difficult scenes. The World Decoder predicts road-user behavior, produces high-definition maps and vehicle trajectories, and supplies signals for trajectory validation. Waymo calls this a Think Fast and Think Slow architecture, but its public description does not specify execution rates, triggers, or the exact tensors passed between components.

<div class="source-explainer-comparison source-explainer-comparison--architecture">
  <figure>
    <div class="comparison-label">01 · Source architecture</div>
    <a href="/assets/images/waymo-foundation-model-architecture.png"><img src="/assets/images/waymo-foundation-model-architecture.png" alt="Waymo Foundation Model architecture with a sensor-fusion encoder and Driving VLM feeding a generative world decoder"></a>
    <figcaption>Waymo's published architecture shows the disclosed interfaces: sensor fusion and a Driving VLM feed a generative world decoder. Source: <a href="https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/">Waymo AI Team, 2025</a>.</figcaption>
  </figure>
</div>

Waymo adapts the foundation model into larger teacher models for the Driver, Simulator, and Critic, then distills smaller students for each role. The Driver student runs onboard with a separate trajectory-validation layer. Simulator students generate closed-loop worlds and synthetic sensor data at scale. Critic students scan driving logs and produce evaluation and training signals. The three systems share a model base and structured vocabulary, but they do not run the same graph or operate under the same compute budget.

Waymo's disclosed architecture combines several branches of the literature: sensor-specific encoding, shared temporal state, structured outputs, semantic conditioning, and teacher-student deployment. The Driver, Simulator, and Critic reuse a foundation model while retaining different runtime graphs and budgets.

The remaining question is whether sharing a world model improves the full learning loop: finding failures in logs, reproducing them in simulation, and transferring a correction back to the Driver. Public architectural descriptions establish the interfaces, but do not isolate the contribution of each shared component to closed-loop improvement.
