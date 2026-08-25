---
postSlug: how-unified-sensor-models-are-built-for-autonomous-driving
sourceSha256: 84111d9f3da79c827efa6ce82bd2673df84ebdd5e7035af0e93b5888fe529e2f
---

# Autonomous-Vehicle Perception, circa 2026

The perception system on an autonomous vehicle turns camera, LiDAR, radar, and other sensor measurements into a world state that the autonomy stack can use. Calibration supplies the geometry needed to relate observations from different viewpoints. Fusion combines their partial evidence into a representation of the road, actors, free space, and motion. The entire process has to remain current and run in real time.

That description becomes difficult as soon as evidence conflicts, an actor is partly occluded, timestamps drift, or one sensor degrades. At long range, a cyclist may occupy only a small image region, produce sparse LiDAR returns, and register in radar as an estimate of range and radial velocity. Each sensor captures a different aspect of the same actor, with its own uncertainty and failure mode.

## Overall architecture

At runtime, the stack roughly follows five transitions. It encodes each sensor stream, aligns the evidence in space and time, fuses the modalities, carries the scene state across frames, and passes that state to prediction and planning. Information can be lost at every transition. The architecture is therefore defined as much by what it preserves as by what it computes.

## Modality-specific encoders

Cameras, LiDAR, and radar do not produce different views of the same tensor. Their sampling patterns and measurement errors differ, so their encoders need different inductive biases.

### Camera encoders

Cameras provide dense color, texture, and fine boundaries. They can read traffic lights, distinguish lane paint from road damage, and classify objects at long range. Their weakness is depth: a pixel identifies a viewing ray, not a distance along it. A camera backbone must preserve fine spatial detail for a distant actor while also retaining the wider context of the intersection. Feature pyramids combine high-resolution features with lower-resolution features that see more of the scene. Deformable attention makes multiscale transformers affordable by sampling a few learned locations around each reference point instead of attending to every image token. Perspective-view supervision can also teach the backbone before its features are projected into three dimensions, giving it a direct reason to retain evidence that later fusion cannot recover.

### LiDAR encoders

LiDAR measures range directly, but the point cloud is sparse, irregular, and acquired over the duration of a sweep. Its encoder should retain position, return intensity, and acquisition time. PointPillars pools points within vertical columns and moves quickly into a dense bird's-eye-view image, trading away height for efficient two-dimensional convolution. SECOND keeps a sparse three-dimensional voxel grid deeper into the network. DSVT lets occupied voxels exchange context through bounded attention sets, while VoxelNeXt stays sparse through the detection head. Sparse computation preserves more three-dimensional structure, but sorting, indexing, and irregular memory access can consume part of the theoretical savings. The right point to become dense depends on both the output and the hardware.

### Radar encoders

Radar contributes range and radial velocity, and remains useful under darkness and weather that weaken other sensors. Its angular localization and semantics are poorer, and multipath can create ghost returns. A radar encoder should preserve Doppler, power, timestamp, confidence, and radar cross section rather than reducing every return to binary occupancy. The field has gradually given radar more influence. CRAFT attaches it to camera proposals. CRN uses it earlier to refine camera depth. RCBEVDet builds an independent radar bird's-eye-view representation before fusion. Radar becomes more valuable when the encoder stops forcing it to imitate LiDAR.

## Aligning sensor evidence in space and time

The sensor features are useful, but they are still tied to different coordinate systems and acquisition times. Camera intrinsics map pixels to viewing rays. Sensor extrinsics express those rays, LiDAR points, and radar returns in the vehicle frame. Timestamps and ego-motion estimates then transform observations to a common time. Calibration establishes which measurements could correspond; it does not add evidence.

Errors in this geometry create systematic offsets. An extrinsic error shifts projected evidence across the bird's-eye-view grid, with a larger displacement at longer range. A clock error moves a dynamic actor because the sensors observed it at different moments. Ego-pose error shifts the transformed scene. Fusion may then combine valid measurements that do not describe the same place or time.

### Two ways to place camera features in 3D

Camera features need one more transformation because the pixel still has no depth. Lift, Splat, Shoot predicts a depth distribution along each ray and pools the lifted features into bird's-eye view. BEVDepth supervises that distribution with projected LiDAR during training. Query-based methods reverse the direction. DETR3D begins with an object query at a three-dimensional reference point and retrieves image evidence at its camera projections. PETR embeds candidate three-dimensional locations into the image features. BEVFormer begins with a dense grid of bird's-eye-view queries and samples several heights for each cell. Dense lifting provides broad scene coverage. Sparse object queries spend compute on selected actors. Dense bird's-eye-view queries sit between those extremes.

## Choosing where to fuse sensor evidence

Fusion decides where evidence from different sensors is allowed to interact. Raw measurements are hard to combine because the sensors sample the scene differently. Prediction-level fusion preserves modular branches, but it cannot recover weak evidence that an individual detector has already discarded. Most recent systems therefore combine encoded features before the task heads.

PointPainting attaches camera scores to LiDAR points. It is efficient and geometrically explicit, but camera evidence can enter only where LiDAR returned a point. TransFusion moves the interaction to object queries and uses soft attention to search a nearby image region, making association less brittle and allowing an image-guided path to add candidates that the LiDAR heatmap missed. FUTR3D gives each object query a shared three-dimensional reference and samples camera, LiDAR, and radar features around it. BEVFusion removes the object-query limit by aligning dense camera and LiDAR fields and combining every bird's-eye-view cell. Moving from measured points, to object hypotheses, to a dense field increases scene coverage and computation together.

Fusion can also move deeper into the backbone. DeepInteraction lets image and LiDAR representations update one another at several stages. UniTR keeps sensor-specific tokenizers but shares transformer weights after tokenization. Earlier interaction gives one modality more chances to repair another, but a corrupted stream can now contaminate more layers.

That makes sensor health part of the architecture. MetaBEV and UniBEV train camera-only, LiDAR-only, and fused modes so an absent stream is a supported input rather than a surprise. Presence is still a binary signal. A blurred camera or degraded LiDAR remains present while providing unreliable evidence. Grace-BEV estimates a continuous trust score and uses it to balance a LiDAR-guided path against a vision-only path. An absent sensor can be masked. A degraded sensor must first be recognized and down-weighted.

## Building state across time

Once the sensors have been fused, the next problem is carrying that evidence across time. Earlier state is first transformed into the current ego frame. Ego motion handles the static scene; moving actors need an additional motion estimate. Pose and timestamp error, rolling shutter, and motion during a LiDAR sweep limit the alignment.

Dense temporal methods carry a bird's-eye-view field. BEVDet4D warps the previous field and concatenates it with the current one. BEVFormer retrieves prior evidence selectively with temporal attention. SOLOFusion keeps recent history at high resolution for stereo cues and longer history at lower resolution for motion and depth.

Sparse methods remember entities instead of the full field. StreamPETR carries foreground queries forward and adds fresh queries for new actors. Sparse4D transforms previous instance anchors, combines them with current proposals, and refines both. SparseBEV retains several frames of camera features and samples them only where an object query asks. Sparse memory is cheaper, but birth, duplication, aging, and deletion become part of the model. A practical hybrid can retain a coarse field for free space and uncertain geometry while tracking actors and map elements as explicit instances. Both should carry age and uncertainty.

## Downstream interfaces

The world state must be exposed in forms the rest of autonomy can use. A detection carries a class, three-dimensional box, confidence, and geometric uncertainty. A track adds identity, velocity, and age. Semantic segmentation labels pixels or bird's-eye-view cells. Occupancy distinguishes occupied, free, and unobserved space. Lane boundaries, centerlines, curbs, and stop lines are naturally polylines or polygons with direction, connectivity, confidence, and age.

Boxes describe discrete actors but miss free-form geometry. Occ3D made visibility-aware semantic occupancy measurable. PanoOcc added foreground instance identities so one volume can represent both surfaces and actors. OccAny then tested whether occupancy geometry transfers to new datasets without known target-camera calibration.

Road structure needs a different interface. MapTR predicts map elements as ordered point sets because a planner needs lane shape and connectivity, not only a raster of lane pixels. That vector supervision is sparse. MapTR version two adds one-to-many matching and auxiliary depth and segmentation losses to train more queries and provide denser geometric feedback. Those auxiliary heads disappear at inference. MGMap instead uses predicted masks inside the decoder to guide map queries and refine individual points, so its dense cue remains part of runtime.

No fixed schema captures every useful feature. The model can retain latent state internally while decoding tracks, occupancy, and vector maps at its boundary. Explicit outputs give planning, simulation, and validation defined fields to inspect. Latent features preserve context that those schemas omit. End-to-end training and explicit runtime interfaces are compatible.

## Training the representation

The deployed sensor set does not determine which evidence can be used during training. Sparse-to-Dense consumes sparse depth at inference. BEVDepth uses projected LiDAR only to supervise camera depth. CRKD moves LiDAR further away by using a camera-LiDAR teacher to train a camera-radar student. BEVDepth and CRKD remove LiDAR from the vehicle, not from data collection. Without LiDAR anywhere in the pipeline, metric supervision must come from radar range, stereo or temporal correspondence with a known baseline, simulation, map or occupancy labels, or another teacher. The replacement signal becomes the source of scale, calibration, and domain error.

The pretraining target then determines what the encoder must preserve. BEV-MAE masks columns of a LiDAR voxel grid and reconstructs their point coordinates and density, teaching local shape and range-dependent sampling. UniM squared AE masks camera patches and LiDAR voxels, aligns their visible features in a shared three-dimensional volume, and reconstructs each sensor. Both reconstruct the present.

Future prediction adds motion. UniWorld uses multi-frame LiDAR to produce four-dimensional occupancy targets for a camera encoder. ViDAR predicts future LiDAR returns from historical images while conditioning on ego motion. DriveWorld predicts present and future occupancy and actions while updating dynamic actors separately from static structure. These objectives teach persistence and change, though each still depends on LiDAR-derived targets.

WPT moves the world model into policy training. It rolls candidate actions forward through agents, occupancy, and road structure, then uses a learned reward to score those futures. Query and reward distillation train a smaller policy that runs without the world model. The reported student plans in sixty-four milliseconds instead of the teacher's three hundred and twelve, but its collision rate rises from zero point one one percent to zero point two four percent. The expensive model can leave the vehicle, but its decision quality is not transferred perfectly.

## End-to-end planning

Here, end to end describes the training graph, not an opaque sensor-to-control model. UniAD builds dense bird's-eye-view features and passes agent and map queries through tracking, motion prediction, occupancy, and ego planning. Each task keeps its own supervision, while planning loss can shape the shared representation.

VAD replaces the dense planning interface with vectors for agents, motion, and map elements. The planner reasons over explicit instances and geometric constraints instead of dense raster features and hand-written post-processing. The base model reports two and a half times faster inference than the previous best method in its comparison. The tradeoff is recall: if the vector extractor misses an actor or boundary, no dense field remains to preserve the weak evidence.

Driving also has more than one plausible action. DiffusionDrive starts from clustered human-trajectory anchors and refines several candidates with two denoising steps. Driving on Registers separates candidate generation from candidate scoring. One decoder proposes trajectories; another estimates safety, comfort, and efficiency. Coverage and selection are different failure modes. A good trajectory can fail because it was never generated or because the scorer ranked it poorly.

These systems remain difficult to deploy. Open-loop evaluation measures similarity to recorded driving, not what happens when the model's own action changes the scene. High-fidelity simulation is expensive and imperfect. Joint training couples failures across perception, prediction, and planning. It is still useful because otherwise each task optimizes a separate proxy. The deployment challenge is to preserve that coordination while retaining component tests, sensor-corruption tests, closed-loop simulation, and an independent check on the selected trajectory.

## Waymo's world model

Waymo's public description presents its Foundation Model as a world model shared across the Driver, Simulator, and Critic. Its interfaces contain both learned embeddings and structured state such as objects, semantic attributes, and roadgraph elements. The embeddings support end-to-end learning; the structured outputs support validation, simulation, and evaluation.

Two components feed the World Decoder. The Sensor Fusion Encoder combines camera, LiDAR, and radar over time and produces objects, semantics, and learned embeddings. A Driving vision-language model uses camera context and is trained with Gemini for rare or semantically difficult scenes. The World Decoder predicts road-user behavior, high-definition maps, and vehicle trajectories, and supplies signals for trajectory validation. Waymo calls this a Think Fast and Think Slow architecture, but its public description does not disclose the execution rates, triggers, or exact interfaces.

Waymo adapts larger teacher models for the Driver, Simulator, and Critic, then distills smaller students for each role. The Driver student runs onboard with a separate trajectory-validation layer. Simulator students generate closed-loop worlds and synthetic sensor data. Critic students scan driving logs and produce evaluation and training signals. They share a model base and structured vocabulary without running the same graph or operating under the same compute budget.

The design I take from this is selective sharing. Each sensor keeps its own encoder, then contributes to a shared spatial and temporal state. Objects, occupancy, roadgraph elements, uncertainty, and age remain available for checks, while latent features carry context those schemas omit. A semantic model can add a grounded constraint; it should not overwrite measured position, motion, or free space.

The most exciting part is the learning loop. A Driver encounters a difficult scene. The Critic finds what went wrong. The Simulator turns that failure into a broader test, and the next Driver comes back better. If shared world models can accelerate that loop without making the runtime harder to inspect, the payoff is concrete: fewer mistakes on real roads and, ultimately, lives saved. That is the future I am most excited to help build.
