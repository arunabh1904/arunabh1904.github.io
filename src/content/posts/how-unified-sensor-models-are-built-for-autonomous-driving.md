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
summary: A mechanism-first guide to what modern driving models share across cameras, LiDAR, radar, time, and perception tasks—and what they deliberately keep separate.
---

# Machine Learning for Autonomous Driving Perception

A unified perception model is often drawn as three sensor arrows entering one box. That picture hides every expensive decision. Cameras produce dense semantics with uncertain depth. LiDAR produces sparse but accurate geometry. Radar adds range and radial velocity with different noise, multipath, and angular resolution. Detection wants compact object state; lane and free-space prediction need dense structure; tracking needs identity through time; planning cares about the consequence of an error, not whether every head improved its average benchmark.

The modern answer is therefore not “encode everything with one transformer.” State-of-the-art systems unify selected interfaces while preserving the distinctions that carry sensor physics or task contracts. They may share a vehicle-centered BEV, object queries, temporal memory, backbone weights, or a deployment graph. They usually retain modality-specific tokenization, calibration, uncertainty, and task heads. The architecture is defined by where that boundary sits.

The implementation questions follow from that boundary: why each sensor needs its own encoder, where geometry becomes shared, how task losses are balanced, which temporal state survives, how privileged sensors supervise a cheaper runtime model, what pretraining target can use fleet-scale unlabeled logs, and where sparsity actually reduces onboard cost. Those are the questions this guide answers.

This guide follows that boundary through the literature. The evidence cutoff is July 31, 2026. I focus on onboard multi-camera, LiDAR, and radar perception, with detection, mapping, segmentation, state estimation, and the handoff toward prediction and planning. I exclude cooperative vehicle-to-vehicle fusion and language-first driving agents except where they clarify the consumer interface. Reported results come from the cited papers; the taxonomy and recommendations are my synthesis.

## 1. “Unified” names five different bets

The word _unified_ can describe at least five architectural choices. Papers that make different choices should not be placed on one leaderboard as if they solved the same problem.

| Unification boundary | Shared object | Representative papers | Main question |
| --- | --- | --- | --- |
| Coordinate | Dense ego-frame BEV | [LSS](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) | Where should evidence from different views live? |
| Fusion | Fused BEV or object queries | [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html), [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) | At what granularity should sensors exchange evidence? |
| Representation | Interaction protocol or shared backbone | [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html), [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) | Which sensor-specific structure should survive sharing? |
| Task | Shared scene state plus specialized heads or queries | BEVFusion, [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html), [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) | Which outputs should shape one another? |
| Operating mode | One model across sensor configurations and failures | FUTR3D, MetaBEV, [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html), [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) | What should the model do when a modality is weak or absent? |

These boundaries form a dependency chain. Before sensors can share a representation, the model needs a spatial correspondence. Before tasks can share a trunk, that trunk must retain what each task needs. Before one network can replace fallback models, training and evaluation must cover degraded modes. “One model” is the end of the argument, not its starting point.

## 2. Sensor encoders should preserve measurement physics

A sensor encoder is not a generic adapter that makes tensors the same width. It decides which measurements survive long enough to become useful. Early fusion of raw channels is attractive because it appears maximally shared; in practice, camera pixels, LiDAR returns, and radar detections do not have compatible sampling, noise, units, or neighborhoods.

### Camera: preserve semantics, scale, and viewpoint

A surround-camera encoder is usually one CNN or vision transformer shared across views, followed by a multi-scale neck. Weight sharing is justified because every camera measures the same appearance statistics, while intrinsics, extrinsics, exposure, distortion, timestamp, and camera identity tell the geometric stages how that view differs. Separate encoders per camera multiply cost and can memorize a rig; one encoder without calibration treats different focal lengths and poses as unexplained domain shifts.

Resolution is a range allocation decision. Small distant actors may occupy a few pixels, so aggressive backbone stride creates an irreversible miss before fusion. Feature pyramids, high-resolution crops, or sparse high-resolution attention spend compute on those actors, but their benefit must be measured by distance and class. Camera encoders should emit both semantic context and the information needed to infer depth; a classification-pretrained feature can be semantically strong while discarding metric detail.

### LiDAR: preserve sparsity, height, and measurement age

LiDAR starts as an unordered set of returns with position, intensity, ring or beam identity, and timestamp. [VoxelNet](/paper%20shorts/2017/11/17/voxelnet-end-to-end-point-cloud-3d-detection.html) learns features inside 3D voxels; [PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html) collapses height early for speed; sparse convolutions process only occupied voxels; [SST](/paper%20shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html) and [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) use local sparse attention to enlarge receptive fields without densifying the full volume. [CenterPoint](/paper%20shorts/2020/06/19/centerpoint-center-based-3d-detection-and-tracking.html) then shows how a BEV center representation can support both boxes and tracking.

The encoder must also de-skew a scan collected while the vehicle moves. Treating all returns as simultaneous creates position error before fusion begins. Multi-sweep accumulation needs the age of each point, ego-pose interpolation, and a policy for dynamic objects that cannot be corrected by ego motion alone. Flattening to BEV is efficient for road scenes, but height-sensitive tasks should keep a 3D path or delay height compression.

### Radar: preserve Doppler, uncertainty, and return structure

Radar returns include range, azimuth, sometimes elevation, radial velocity, Radar Cross Section, and measurement uncertainty. A LiDAR voxelizer can ingest the coordinates, but it does not automatically exploit Doppler or model multipath and angular noise. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) uses pointwise and transformer branches plus RCS-aware scattering; [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to guide camera lifting. Both approaches make radar affect geometry before a generic BEV encoder can wash out its distinctive signal.

This leads to a practical rule: share weights after each modality has constructed a physically meaningful token. Camera patches, LiDAR voxels, and radar returns can enter a shared interaction block, as [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) demonstrates, but their tokenizers, normalization, positional metadata, and health signals should remain explicit.

## 3. BEV is a contract between sensing and autonomy

Surround cameras observe perspective views; planners reason about where actors, lanes, and free space sit around the ego vehicle. [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html) made this mismatch explicit. Each pixel predicts context and a depth distribution, is lifted into a camera frustum, transformed by calibration, and pooled into an ego-frame grid. Camera count and pose become inputs to the view transform rather than assumptions baked into the task head.

![Figure 1 from Lift, Splat, Shoot, showing multi-camera evidence represented in vehicle-centered BEV](/assets/images/lift-splat-shoot-paper-figure-1.png)
_LSS establishes the first durable interface: arbitrary camera views in, one metric scene map out. Source note: [Lift, Splat, Shoot](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), Figure 1._

The BEV grid earns its cost because several consumers need the same geometry. Object detection needs centers, extents, and orientation. Map segmentation needs lane and road layout. Occupancy needs free and occupied cells. Tracking and prediction need ego-motion-compensated history. A vehicle-centered grid makes these outputs comparable and lets convolution or attention move information across camera boundaries.

BEV also commits the model to a compression. Camera depth hypotheses and vertical structure are collapsed into cells; LiDAR voxels are often flattened along height; resolution and spatial extent set memory before the scene is known. Long range demands many mostly empty cells. A dense grid is therefore a strong default for scene-level tasks, not a universal answer.

[BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) replaces explicit depth splatting with learned BEV queries that sample projected image features, and it carries the grid through time with temporal self-attention. The contrast with LSS is useful: LSS places uncertainty in a depth distribution before pooling; BEVFormer places it in learned attention from metric queries. Both build the same downstream contract, but they spend compute and absorb calibration error differently.

## 4. Depth has three different jobs

“Use LiDAR for depth” can describe three systems with different runtime contracts.

| Depth problem | Runtime inputs | Output or training role | Representative work |
| --- | --- | --- | --- |
| Monocular depth estimation | RGB only | Dense or categorical depth from appearance | [CaDDN](/paper%20shorts/2021/03/01/caddn-categorical-depth-for-monocular-3d-detection.html), [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) at inference |
| Depth completion | RGB plus sparse runtime depth | Dense depth that retains measured ranges | [Sparse-to-Dense](/paper%20shorts/2017/09/21/sparse-to-dense-depth-prediction-from-sparse-depth-and-rgb.html), [DeepLiDAR](/paper%20shorts/2018/12/02/deeplidar-surface-normal-guided-depth-completion.html), [GuideFormer](/paper%20shorts/2022/06/19/guideformer-transformers-for-image-guided-depth-completion.html) |
| Privileged depth supervision | RGB at runtime; LiDAR or offline geometry only during training | Teaches lifting, occupancy, features, or pseudo-labels | BEVDepth, [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html), [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) |

Depth completion is sensor fusion. A failed runtime LiDAR removes the sparse measurements the completion model expects. Privileged depth supervision is different: projected LiDAR points define a loss during training, while the deployed network predicts depth from cameras. Confusing the two leads to a fallback design that silently depends on a sensor the production bill of materials removed.

[BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) provides the cleanest controlled case. It shows a large gap between learned and ground-truth depth inside a Lift-Splat detector, then supervises the categorical depth distribution with projected LiDAR, conditions the depth head on camera parameters, and refines the lifted volume. LiDAR supplies labels; the inference graph remains camera-only.

![Figure 4 from BEVDepth, showing explicit depth supervision during training and camera-only BEV inference](/assets/images/bevdepth-paper-figure-4.png)
_The red depth target can be generated by an instrumented training fleet without entering the deployed graph. Source note: [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), Figure 4._

Sparse LiDAR is not dense truth. Returns miss dark or distant surfaces, projected points can cross object boundaries when timestamps or calibration differ, and a rotating scanner samples dynamic objects at different times. Good target generation filters occlusions, models label confidence, keeps ignore regions, and separates direct measurements from offline densification. More labels do not help if the target-generation geometry is wrong.

## 5. Fusion moved from calibrated points to shared spaces and queries

Early camera-LiDAR fusion often painted image features onto projected LiDAR points. The association is cheap and geometrically legible, but it uses LiDAR sampling density to decide which image information survives. A small calibration error can attach the wrong pixel; background semantics that matter to mapping never enter the representation.

[TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html) loosens that association. LiDAR proposes object queries, then each query attends to an image region. Calibration narrows the search without dictating one correspondence. This works well when the prediction unit is an object and when LiDAR should remain a usable fallback.

[BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html) makes the opposite granularity choice. It independently converts camera and LiDAR features into dense BEV maps, concatenates them, and learns a BEV encoder before task heads. The shared grid preserves background semantics for segmentation and amortizes fusion across multiple tasks.

![Figure 2 from BEVFusion, showing modality-specific encoders converging on a shared BEV and task-specific heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
_BEVFusion shares after geometry is normalized, then specializes at the output. Source note: [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), Figure 2._

[FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) avoids requiring every modality to become one dense map. A 3D object query samples camera, LiDAR, and radar features in their native representations. The same detector can accept different sensor configurations because availability changes the set of features sampled, not the shape of the detection interface.

The choice follows the output contract:

- Use dense BEV fusion when several heads need complete scene context and the spatial envelope is bounded.
- Use query fusion when object-centric outputs dominate, most of the scene is empty, or sensors should remain independently usable.
- Use both when a dense scene memory serves mapping and occupancy while sparse queries carry actors, tracks, or plans.

This hybrid is increasingly common because “dense versus sparse” is not a philosophical choice. It is a capacity allocation decision by output type.

## 6. Unification can destroy the evidence it meant to share

A fused tensor is convenient, but convenience is not proof that the tensor retains sensor provenance or native structure. [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html) makes this failure visible. It keeps the image feature map and LiDAR BEV alive, updates them bidirectionally, and lets object queries alternate between them. Camera features retain dense perspective neighborhoods; LiDAR retains metric sparsity. The shared object is an interaction protocol, not one representation.

![DeepInteraction's comparison between feature collapse and retained modality streams](/assets/images/deepinteraction-paper-figure-1.png)
_DeepInteraction asks whether a fused latent has discarded exactly the modality-specific evidence the detector needs. Source note: [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html), Figure 1._

[UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html) pushes in the other direction: share the expensive transformer weights across image patches and LiDAR voxels. It avoids pretending the tokens are identical. Modality-specific tokenizers construct them; intra-modal partitions respect native neighborhoods; cross-modal blocks build mixed 2D and 3D local sets. Weight sharing removes serial duplicate encoders while coordinate structure tells shared attention what locality means.

These papers define the real test for a unified backbone. Parameter count and benchmark accuracy are insufficient. A matched study should ask:

- Does sharing reduce latency after tokenization and view transforms are included?
- Which rare classes or ranges improve, and which lose sensor-specific capacity?
- Does the model remain calibrated when one modality degrades?
- Can a consumer tell whether a state estimate was observed by camera, LiDAR, radar, or only propagated through time?
- Does a larger separate-backbone control recover the same accuracy at equal compute?

Without those controls, “shared representations improve collaboration” is a plausible interpretation, not an isolated result.

## 7. Multi-task modeling needs capacity and loss accounting

Detection, lanes, segmentation, velocity, and tracking do not merely require different labels. They use different spatial granularity and error tolerances. A detection head can tolerate smooth background features; lane topology cannot. Velocity may depend heavily on radar and time; semantic class may depend on camera texture. Sharing the trunk saves compute only if it preserves all of those contracts.

BEVFusion uses a shared BEV encoder and independent task heads. That is the safe baseline: share geometry and expensive scene features, then keep losses and outputs explicit. [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) adds mixtures of experts to reduce detection/segmentation interference. The experts are a capacity valve: shared attention handles common structure, while routing lets tasks or modality states avoid one set of parameters doing incompatible work.

[UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html) changes the task graph. Tracking, mapping, motion, occupancy, and planning are not parallel heads attached to one trunk. Their queries pass structured state forward, and planning is the organizing objective. This exposes a distinction that multi-task benchmarks often miss: two outputs can be individually accurate yet mutually inconsistent or poorly timed for planning.

The first optimization problem is scale. A box regression loss measured in meters, a cross-entropy segmentation loss summed over thousands of pixels, and a velocity loss do not arrive with comparable magnitudes. [Kendall, Gal, and Cipolla](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html) derive homoscedastic uncertainty weighting. For regression tasks, the useful form is

$$
\mathcal{L}_{\text{total}}=\sum_i \left(\frac{1}{2\sigma_i^2}\mathcal{L}_i+\log \sigma_i\right),
$$

where each learned $\sigma_i$ acts as a task-level noise scale. The inverse variance downweights a noisy or large-scale objective; the log term prevents the model from sending every weight to zero. In practice, implementations optimize $s_i=\log \sigma_i^2$ for numerical stability. The method is attractive because the weights are learned with the network instead of swept by hand.

Homoscedastic weighting solves neither of the other two multi-task problems. It does not measure whether two tasks ask shared parameters to move in opposite directions, and one global $\sigma_i$ cannot represent scenario-dependent reliability. [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html) targets unequal learning rates by adapting weights from gradient norms. [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html) projects away pairwise conflicting gradient components. Alternating batches, task-specific normalization, adapters, or mixture-of-experts allocate capacity when weighting alone cannot reconcile the objectives.

A practical debug sequence is therefore:

1. Normalize every loss by a meaningful unit: positives, valid pixels, anchors, or trajectories rather than raw tensor size.
2. Log per-task loss, gradient norm, and pairwise gradient cosine at the shared trunk.
3. Compare fixed weights, homoscedastic weighting, and one gradient-aware method under the same training budget.
4. Freeze or split progressively deeper blocks to locate where transfer turns negative.
5. Evaluate every head and the downstream planner; never select a checkpoint from the weighted scalar alone.

If a 2% mAP gain creates more planner interventions through jitter, velocity error, or lane inconsistency, the unified model regressed where it matters. Loss balance is an optimization tool; the consumer contract remains the deployment criterion.

## 8. Temporal modeling splits into dense BEV and sparse 4D state

Temporal fusion adds evidence that no single frame contains: velocity, occlusion recovery, track continuity, and improved depth through ego motion. It also adds a new failure mode. Historical evidence was observed under an older pose, sensor state, and world state. “More frames” is not the architectural choice; the choice is what state represents those frames.

[BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html) is the dense baseline: ego-warp the previous camera BEV, concatenate it with the current BEV, and learn temporal fusion. [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) uses temporal attention over the persistent grid. Dense recurrence preserves road, free-space, and background evidence, but history length multiplies a representation whose cost is fixed by spatial extent.

[Sparse4D](/paper%20shorts/2022/11/19/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion.html) follows 3D anchors instead. Multiple keypoints around each anchor are projected into every camera, scale, and timestamp; only those features are sampled and fused. [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html) stores a FIFO memory of foreground object queries, compensates them with ego pose, and updates them against the current images. [Sparse4D v3](/paper%20shorts/2023/11/20/sparse4dv3-end-to-end-3d-detection-and-tracking.html) adds temporal denoising and quality estimation, then uses recurrent instances directly as tracks.

![Figure 3 from StreamPETR, showing a recurrent queue of object queries updated by current multi-view images](/assets/images/streampetr-paper-figure-3.png)
_StreamPETR's top-k query memory makes long history cheap by discarding most scene state. Source note: [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), Figure 3._

| Temporal state | Compute scales with | Strongest use | Blind spot |
| --- | --- | --- | --- |
| Dense BEV memory | Spatial cells × history | Static context, lanes, occupancy, uncommitted evidence | Long-range empty space and long histories are expensive. |
| Multi-frame perspective features | Cameras × scales × history | Deferred geometry and rich image evidence | Repeated projection and sampling. |
| Sparse 4D anchors or queries | Instances × keypoints × history | Actors, tracking, long object histories | Misses state that never became a query. |
| Hybrid dense + sparse | Short BEV plus long instance memory | Full scene plus actors | Two memories must remain consistent. |

A production model often benefits from the hybrid: a short high-resolution BEV memory for static context and a longer compressed object memory for actors. [SparseDrive](/paper%20shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html) pushes the sparse state beyond detection into map and motion instances used by planning.

The maximum history should be selected by task and failure slice, not by available VRAM. Longer memory can recover an occluded vehicle but also preserve an actor that left the scene. Ego warping aligns static structure; it does not align independently moving objects. Pose error accumulates, and rolling-shutter or timestamp mismatch can make nominal calibration wrong for a moving platform.

A useful temporal evaluation therefore measures jitter, track continuity, stale false positives, latency-adjusted accuracy, and re-observation behavior. Frame-level mAP cannot tell whether the model is stable, fresh, or merely slow.

## 9. Missing sensors are training distributions, not zero tensors

Most fusion models are trained on healthy sensors and evaluated on healthy sensors. Zeroing a failed feature tensor at deployment does not create a principled fallback. Normalization statistics, attention weights, and task heads were optimized for a different joint distribution.

[MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html) samples full and missing modalities during training, uses dense queries that can attend to camera, LiDAR, or both, and routes through modality-specific experts. [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html) studies uniform BEV encoders, shared queries, and normalized weighted fusion so the detector can run on camera plus LiDAR, camera only, or LiDAR only without retraining. The 2026 [Grace-BEV](/paper%20shorts/2026/05/29/grace-bev-graceful-degradation-under-sensor-failures.html) goes beyond a binary modality mask and explicitly estimates reliability before recalibrating fusion.

![Figure 3 from MetaBEV, showing an evolving BEV query attending to available sensor features](/assets/images/metabev-paper-figure-3.png)
_MetaBEV turns modality availability into an explicit input to fusion and routing. Source note: [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html), Figure 3._

The progression is from availability to health. “Camera present” is not enough when glare saturates one view, radar multipath creates ghosts, LiDAR returns thin out in weather, or calibration drifts gradually. A deployable interface should carry sensor-health indicators, masks, timestamps, calibration confidence, and uncertainty. Training should include camera dropout, partial field-of-view loss, timestamp jitter, calibration perturbations, structured corruption, and transitions into and out of each mode.

One conditional model is not automatically safer than separate fallbacks. It reduces memory and versioning cost, but it expands the validation matrix. The decisive comparison is against specialist models under the same total onboard budget, including uncertainty calibration and worst-case latency.

## 10. Radar must keep the properties that cameras do not have

Radar is attractive because it supplies direct range and radial velocity, works at long range, and degrades differently from cameras. It is difficult because returns are sparse, angular resolution is weaker, elevation may be ambiguous, multipath creates false structure, and Radar Cross Section depends on object and aspect.

[CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html) uses radar to guide camera lifting and deformable attention to reconcile camera-radar misalignment. [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) gives radar a dual point/transformer encoder, RCS-aware scattering, and cross-attention alignment in BEV. [CRKD](/paper%20shorts/2024/06/17/crkd-camera-radar-distillation-from-lidar-camera.html) uses a camera-LiDAR teacher to improve a cheaper camera-radar student at training time. The 2026 [RPGFusion](/paper%20shorts/2026/06/01/rpgfusion-4d-radar-prior-guided-fusion.html) uses 4D-radar priors to guide image BEV queries and densify radar features.

The sequence shows two current recipes. One builds a strong radar representation and fuses it with camera BEV. The other uses radar geometry to decide how camera evidence is lifted or queried. Both retain radar attributes until after they have influenced geometry; simply rasterizing returns as occupancy throws away much of the reason to carry radar.

Weather robustness still needs direct evidence. A nuScenes aggregate gain or random sensor-dropout test is not a substitute for rain, fog, snow, glare, distance, actor class, and transition slices with calibrated confidence.

## 11. Training-time sensors can be richer than runtime sensors

Production hardware does not have to define the supervision ceiling. A LiDAR-equipped development fleet can train a camera-radar model through depth targets, occupancy labels, BEV feature distillation, pseudo-labels, or a privileged teacher. CRKD makes this design explicit: the camera-radar student learns from a stronger camera-LiDAR teacher while retaining its cheaper runtime sensor set.

[BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html) uses the same lifecycle for a camera-only student: projected LiDAR supervises depth during training, but images and calibration are the only inference inputs. [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) pretrains camera features by predicting 4D occupancy from large image-LiDAR pairs. A richer offline teacher can also combine future frames, multiple passes, expensive models, map priors, or human review; none needs to fit the onboard deadline if its output becomes a versioned target rather than a runtime dependency.

The distinction prevents a common architecture mistake. A LiDAR-derived target is privileged supervision; a LiDAR-dependent view transform is a runtime dependency. The former can disappear after training. The latter cannot. Every source of geometry should be labeled by lifecycle: needed to collect data, generate labels, train, calibrate, or run onboard.

Privileged sensing also creates risks. Teacher errors become labels, the student may inherit a representation it cannot reconstruct from its inputs, and offline gains may concentrate in clear scenes where the teacher is already strongest. Distillation should be evaluated against direct supervision under the same student architecture, with uncertainty and adverse-condition slices.

Tesla illustrates the deployment principle but should be described precisely. [Tesla's public AI material](https://www.tesla.com/AI) documents camera-based onboard inference, multi-camera video-to-BEV networks, and ground-truth generation that combines vehicle sensors across space and time; its [2021 AI Day presentation](https://www.youtube.com/watch?v=j0z4FweCy4M) also describes offline auto-labeling. Those sources do not establish the blanket claim that production Tesla networks are trained by a LiDAR teacher. The supported lesson is broader: expensive offline labeling and temporal reconstruction can supervise a cheaper camera-only inference graph. BEVDepth, CRKD, and UniWorld provide paper-level evidence for specific privileged-sensor versions of that idea.

## 12. Pretraining at scale should predict geometry and change

ImageNet initialization helps a camera encoder recognize objects, but it does not teach multi-camera geometry, sensor correspondence, ego motion, or which state persists. Driving pretraining must choose a target that makes unlabeled fleet logs carry those lessons.

[UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html) masks camera patches and LiDAR voxels, projects visible features into a shared 3D volume, and reconstructs both modalities. It learns cross-modal completion from synchronized pairs. [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html) predicts current and future 4D occupancy from image-LiDAR pairs. [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html) predicts future point clouds from historical images, forcing a camera encoder to represent semantics, 3D geometry, and dynamics. [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) pretrains a spatiotemporal latent with dynamic memory, static-scene propagation, and downstream task prompts.

![Figure 2 from UniM²AE, showing masked camera and LiDAR tokens interacting in a shared 3D volume](/assets/images/unim2ae-paper-figure-2.png)
_Masked reconstruction uses synchronized sensors to learn correspondence; future-geometry objectives add the requirement to model change. Source note: [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html), Figure 2._

The objectives teach different invariances:

| Pretraining target | What the encoder must retain | Shortcut or cost |
| --- | --- | --- |
| Masked image/LiDAR reconstruction | Cross-modal appearance and local geometry | Can copy correlated observations without learning dynamics. |
| Current occupancy | Metric scene completion | Requires geometric targets and may average uncertainty. |
| Future occupancy or point cloud | Geometry, motion, and persistence | Multi-modal futures make deterministic targets brittle. |
| Teacher features or pseudo-labels | Task-relevant semantics and geometry | Inherits teacher bias and confidence errors. |
| Ego-motion/photometric consistency | Video geometry without LiDAR labels | Breaks on dynamic objects, occlusion, and lighting change. |

Scaling data is not only adding clips. Mixtures change the gradient budget: common highway scenes can swamp rare construction, adverse weather, and near-collision events. A useful large-scale recipe deduplicates near-identical cruising, samples by scenario and sensor health, measures representation transfer by task, and retains held-out geographic and temporal slices. The pretraining unit—frame, synchronized sensor packet, clip, masked volume, or future interval—determines which correlations the model can learn.

The expensive ablation holds encoder size and total tokens fixed while comparing more scenes, longer clips, and richer sensor supervision. Otherwise “scale” may mean only more redundant frames or more LiDAR-equipped miles.

## 13. Sparse transformers move the budget from area to evidence

Sparsity appears at three different levels. Sparse LiDAR backbones process only occupied voxels. Sparse camera detectors sample image features around object hypotheses. Sparse end-to-end systems carry actor and map instances into prediction and planning. Each saves a different dimension of compute.

For LiDAR, [SST](/paper%20shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html) keeps a single high-resolution stride and applies attention inside non-empty windows, preserving small objects that downsampling can erase. [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html) partitions variable-density voxels into bounded local sets, rotates partitions between layers to exchange context, and avoids custom sparse-convolution operators in its transformer path. UniTR reuses DSVT-style blocks across camera and LiDAR tokens, making sparse local attention part of sensor unification.

For cameras, Sparse4D and StreamPETR do not make the image encoder sparse; they make fusion and memory sparse. The multi-view backbone still processes every image, often dominating latency. The saving comes after feature extraction: a bounded number of anchors or queries retrieve evidence instead of constructing and recurrently updating every BEV cell.

Sparse models therefore need four budgets reported separately: image-backbone FLOPs, number of active voxels or image tokens, number of queries/keypoints, and temporal memory size. A model can call itself sparse while carrying a dense high-resolution camera pyramid that dominates P99 latency.

Sparsity also creates recall risk. Empty voxels cannot represent unobserved free space; a query detector cannot sample an object it never proposed; top-k temporal memory can evict a low-confidence hazard. Dense and sparse representations are complementary when the dense path retains safety-relevant scene context and the sparse path allocates long-horizon capacity to actors.

## 14. SOTA is a recipe, not one winning block

Recent leaderboard papers often refine one axis: [IS-Fusion](/paper%20shorts/2024/06/17/is-fusion-instance-scene-multimodal-fusion.html) combines scene- and instance-level fusion; [GAFusion](/paper%20shorts/2024/06/17/gafusion-adaptive-lidar-camera-fusion.html) adds LiDAR-guided depth, occupancy, global interaction, multi-scale processing, and temporal fusion; Sparse4D v3 strengthens sparse recurrent detection and tracking; UniM²AE pretrains camera and LiDAR encoders through masked reconstruction in a shared 3D volume. These are useful components, but a production SOTA system is assembled under a hardware and safety contract.

The defensible current recipe is:

1. Use modality-specific front ends to preserve sensor physics and calibration.
2. Supervise or pretrain the camera geometry path explicitly; do not assume detection loss will discover reliable depth.
3. Normalize geometry into BEV when dense scene tasks need a common canvas.
4. Retain sparse object or track queries for actor-centric reasoning and long temporal horizons.
5. Share the expensive trunk only where matched ablations show positive task and modality affinity.
6. Normalize task losses, then measure gradient conflict before adding learned weights, surgery, adapters, or experts.
7. Train nominal, degraded, and missing-sensor modes explicitly; pass health and age metadata into the model.
8. Use privileged sensors and offline teachers when they improve a runtime-feasible student without becoming hidden dependencies.
9. Pretrain on synchronized clips with geometry- or future-aware targets, not only 2D image semantics.
10. Optimize the real execution graph—view transforms, token construction, memory movement, synchronization, and P99 latency—not FLOPs alone.
11. Evaluate range, weather, calibration, temporal stability, sensor state, uncertainty, and planner impact alongside aggregate accuracy.

No paper in this reading set demonstrates every item in one system. That gap matters. Academic comparisons usually hold the dataset fixed and vary a block; production must hold a behavior contract fixed while sensor health, weather, scene density, compute load, and task ownership change.

## 15. A cumulative reading path

Read the papers in an order that produces an architectural artifact after each layer.

| Layer | Read | Question | Artifact to produce |
| --- | --- | --- | --- |
| 1. Sensor encoding | [VoxelNet](/paper%20shorts/2017/11/17/voxelnet-end-to-end-point-cloud-3d-detection.html), [PointPillars](/paper%20shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html), [SST](/paper%20shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html), [DSVT](/paper%20shorts/2023/01/15/dsvt-dynamic-sparse-voxel-transformer.html), [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) | Which measurement attributes and neighborhoods must survive? | Specify one token schema and health contract per sensor. |
| 2. Coordinate and depth | [LSS](/paper%20shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html), [BEVDepth](/paper%20shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html), [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html) | How does uncertain perspective evidence enter metric space? | Draw the view transform and label train-only versus runtime geometry. |
| 3. Fusion granularity | [TransFusion](/paper%20shorts/2022/03/22/transfusion-robust-lidar-camera-fusion-with-transformers.html), [BEVFusion](/paper%20shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html), [FUTR3D](/paper%20shorts/2022/03/20/futr3d-unified-sensor-fusion-framework-for-3d-detection.html) | Should evidence meet at points, cells, or queries? | Choose the fusion unit for objects, maps, and occupancy separately. |
| 4. Sharing and tasks | [DeepInteraction](/paper%20shorts/2022/08/23/deepinteraction-3d-object-detection-via-modality-interaction.html), [UniTR](/paper%20shorts/2023/08/15/unitr-unified-efficient-multimodal-transformer-for-bev.html), [uncertainty weighting](/paper%20shorts/2017/05/19/multi-task-learning-using-homoscedastic-uncertainty.html), [GradNorm](/paper%20shorts/2017/11/07/gradnorm-adaptive-loss-balancing.html), [PCGrad](/paper%20shorts/2020/01/19/pcgrad-gradient-surgery-for-multi-task-learning.html) | What survives parameter sharing, and how are losses normalized? | Specify shared blocks, adapters, per-task loss units, and conflict diagnostics. |
| 5. Temporal state | [BEVDet4D](/paper%20shorts/2022/03/31/bevdet4d-temporal-cues-in-multicamera-3d-detection.html), [Sparse4D](/paper%20shorts/2022/11/19/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion.html), [StreamPETR](/paper%20shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html), [Sparse4D v3](/paper%20shorts/2023/11/20/sparse4dv3-end-to-end-3d-detection-and-tracking.html) | Which scene state must persist, and how does it age? | Budget dense cells, object queries, history length, and birth/death behavior. |
| 6. Failure and radar | [MetaBEV](/paper%20shorts/2023/04/19/metabev-solving-sensor-failures-for-bev-perception.html), [UniBEV](/paper%20shorts/2023/09/25/unibev-robust-multimodal-detection-with-uniform-bev-encoders.html), [CRN](/paper%20shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html), [RCBEVDet](/paper%20shorts/2024/03/25/rcbevdet-radar-camera-fusion-in-bev.html) | Does the model understand sensor availability and reliability? | Build a sensor-state matrix with required capability and uncertainty per mode. |
| 7. Pretraining | [UniM²AE](/paper%20shorts/2023/08/21/unim2ae-multimodal-masked-autoencoders-with-unified-3d-representation.html), [UniWorld](/paper%20shorts/2023/08/14/uniworld-autonomous-driving-pretraining-via-world-models.html), [ViDAR](/paper%20shorts/2023/12/29/vidar-visual-point-cloud-forecasting-for-autonomous-driving.html), [DriveWorld](/paper%20shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html) | Which unlabeled unit teaches geometry and dynamics? | Define the pretraining unit, target, data mixture, and matched transfer test. |
| 8. Task graph | [UniAD](/paper%20shorts/2022/12/20/uniad-planning-oriented-autonomous-driving.html), [SparseDrive](/paper%20shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html) | Are heads colocated, or does scene state serve planning? | Define consumer contracts and scenario-level regression gates. |

The final artifact should be a budgeted system proposal, not another architecture collage. For every shared block, name what it saves, which tasks and modalities update it, how failure is signaled, and which matched ablation would cause you to split it.

## 16. The research test that would change my mind

My current thesis is that the strongest unified driving model will be **shared in geometry and expensive scene computation, conditional in sensor availability, sparse in actor state, and specialized at physics-sensitive inputs and safety-sensitive outputs**. Full feature collapse is too lossy; fully separate stacks waste compute and create inconsistent state.

The thesis is falsifiable. Train three systems on the same data and augmentation: separate sensor/task specialists; a fully shared backbone and latent; and a structured hybrid with modality tokenizers, shared BEV or interaction blocks, task adapters, and explicit health conditioning. Match total parameters, training compute, onboard memory, and P99 latency. Evaluate nominal accuracy, calibration perturbation, partial and full sensor loss, weather, long range, temporal staleness, and planner interventions.

If the fully shared model matches the hybrid on every degraded slice and downstream metric while remaining easier to optimize and deploy, the extra structure is unnecessary. If specialists dominate within the same total budget, the shared scene representation is not carrying enough reusable information. Until that comparison exists, “unified” should describe an experimentally justified boundary—not a preference for fewer boxes in the diagram.
