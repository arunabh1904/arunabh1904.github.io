---
postSlug: how-unified-sensor-models-are-built-for-autonomous-driving
sourceSha256: e62cce271a3ae13867d92bbfe37b168959ce9c0679d0938f66e1eba269dab1cc
---

# Autonomous-Vehicle Perception, circa 2026

The perception task on an autonomous vehicle turns raw camera, LiDAR, radar, and other sensor measurements into a world state that the rest of autonomy can use. Calibration fixes the geometric relationship among those sensors. Fusion combines their partial evidence into one coherent representation of the road, actors, free space, and motion. And all of it has to run in real time. It is an incredibly hard task.

That clean description breaks as soon as evidence conflicts, an actor is occluded, timestamps drift, or one sensor degrades. A distant cyclist at dusk may occupy a few image pixels, two or three LiDAR returns, and one noisy radar detection with radial velocity. None of those measurements is the cyclist. Each sensor contributes partial evidence with its own uncertainty and failure mode.

That observation gives us the main design rule. Preserve sensor-native evidence until geometry makes interaction meaningful. Then preserve enough structured and latent state for prediction, planning, simulation, and validation. A unified system does not need one encoder or one tensor. It needs shared interfaces that do not erase what each sensor knows.

## Start with the system contract
<!-- covers: The system contract: preserve, align, interact, persist, materialize -->

Camera intrinsics map rays to pixels. Sensor extrinsics place cameras, LiDAR, and radar in the vehicle frame. Timestamps and ego poses align their measurements in time. These operations establish where and when each piece of evidence belongs. Calibration does not add information; it defines which measurements are allowed to correspond.

This matters because alignment errors create structured mistakes. A small angular error can move evidence into the next lane at long range. Two correct measurements can disagree when they describe different moments. A robust shared state must therefore retain sensor support, measurement age, and health, rather than treating every aligned feature as equally current and reliable.

Three operations are often hidden inside the word fusion. Alignment establishes possible correspondence. Interaction lets one modality change another. Materialization decides what downstream tasks may consume. Separating those operations makes architectures easier to compare and failures easier to locate.

## Preserve what each sensor measures
<!-- covers: Sensor encoders preserve different evidence | Cameras: dense semantics at several scales | LiDAR: sparse geometry before dense BEV | Radar: range and motion under uncertainty -->

Cameras provide dense appearance. They recognize signs, lights, lane paint, unusual objects, and small boundaries. A camera encoder needs several scales because distant actors require spatial detail while road layout needs wider context. But a pixel still identifies a ray, not a distance along that ray. A larger image backbone can improve semantics without resolving depth.

LiDAR answers depth directly, but its points are sparse, irregular, and range dependent. Pillar encoders collapse height early and gain a regular two-dimensional bird's-eye-view backbone. Sparse voxel encoders retain vertical structure longer, which helps separate overpasses, trucks, poles, and other height-dependent geometry. The right choice depends on the operating envelope and hardware, because sparse FLOPs do not automatically become low wall-clock latency.

Radar contributes range and Doppler under darkness and adverse weather, but with weaker angular localization, multipath, and ghost returns. Its useful fields include velocity, power, confidence, timestamp, and radar cross section. Rasterizing those returns too early into binary occupancy removes the information needed to distinguish motion from clutter. Radar works best when the model chooses where its range and Doppler should alter camera or LiDAR computation.

The sensor encoders are therefore specialized by design. The common representation begins only after their evidence has physical support.

## Decide where image evidence acquires metric support
<!-- covers: Where metric geometry enters | Push image evidence into 3D: LSS and BEVDepth | Pull image evidence from metric space -->

There is no geometry-free sensor fusion. Geometry may appear as depth logits, attention reference points, positional encodings, or learned correspondences, but image evidence must acquire metric support before it can interact coherently with LiDAR, radar, maps, or temporal state.

Push-based methods begin with image pixels. Lift, Splat, Shoot predicts a depth distribution along each camera ray and pools the lifted features into bird's-eye view. BEVDepth adds projected LiDAR supervision, so LiDAR shapes camera geometry during training and disappears at inference. These methods preserve dense image evidence and suit occupancy and maps, but a depth error writes the feature into the wrong metric cell.

Pull-based methods begin with a hypothesis in physical space and ask the cameras for evidence. A dense method can project every voxel into visible views. DETR3D instead anchors a bounded set of object queries to three-dimensional reference points. BEVFormer gives each ground-plane cell a query and samples several heights. The trade is coverage. Dense fields spend compute on the scene; object queries spend it on candidate actors. If no query acquires support for an actor, later refinement has nothing to recover.

No construction dominates every task. The useful question is which evidence the construction discards and whether a downstream component retains a recovery path.

## Fusion is also an admission policy
<!-- covers: Fusion is alignment, interaction, and routing | Point, query, and dense-field fusion | Proposal recall is an architectural ceiling | Interaction can occur before either stream is finished | Missing, degraded, and misaligned sensors -->

Once the features have metric support, the system chooses where they meet. Point fusion is cheap and direct, but camera evidence survives only where a point exists. Query fusion gathers evidence around selected actors and matches an object-centric output. Dense bird's-eye-view fusion preserves actors together with lanes, occupancy, and free space, at a higher compute cost.

The hidden bottleneck is often admission rather than attention. If LiDAR creates every proposal, then a camera-only actor may never enter downstream computation. Letting every modality propose candidates restores recall but creates matching and deduplication work. Dense shared fusion keeps a recovery path by retaining each modality field before detection. The architectural question is which modality controls admission, because its recall becomes the system ceiling unless another route bypasses it.

Interaction can also happen before either stream is complete. Repeated cross-modal updates may improve complementarity, but they spread contamination when one sensor is corrupted. Deeper interaction therefore requires health-aware routing and modality-specific diagnostics.

Missing sensors are only the simplest failure. Blur, fog, delay, interference, reduced LiDAR beams, and calibration drift can leave a tensor present while making its evidence unreliable. Training needs modality dropout for supported configurations, corruption examples and observable health signals for degraded streams, and separate perturbation tests for timing and calibration. Tensor availability is not evidence quality.

## Memory must represent both the scene and its entities
<!-- covers: Time: what should survive the next frame? | Dense scene memory | Sparse entity memory | The world state is not one tensor | Materialized structure and latent state should coexist -->

A single frame cannot provide complete motion, preserve evidence through occlusion, or stabilize uncertain depth. Old state must first be transformed into the current ego frame. Moving actors then need their own motion hypotheses, while pose and timestamp errors must not be mistaken for scene motion.

Dense temporal memory carries a bird's-eye-view field. It preserves roads, free space, and weak background evidence, but costs memory and can retain stale clutter. Sparse memory carries selected actors or queries. It is lighter and naturally tracks identity, but birth, deletion, duplicate removal, and aging become learned decisions. Evidence that never becomes a confident query may disappear.

These failure modes point to a hybrid. A lower-resolution field can preserve occupancy and topology. High-resolution queries can preserve actors and map elements. Explicit age and confidence can govern both.

The same idea extends into a compression ladder: dense fields, sparse structured entities, learned latent tokens, and finally one pooled scene vector. Moving right saves compute while asking the objective to discard more spatial evidence. A single pooled vector is usually too aggressive for driving because location is part of the question.

Compression and materialization solve different problems. Objects and tracks preserve identity and kinematics. Vector roadgraphs preserve connectivity. Occupancy preserves free, occupied, and unknown space. Latent tokens preserve residual information under a fixed compute budget. The strongest interface exposes structured state and learned embeddings together. End-to-end learning does not require an opaque runtime.

## Training can be richer than deployment
<!-- covers: Training contracts shape the deployed model -->

Architecture descriptions become misleading when they do not separate the graph that learns from the graph that runs. LiDAR may supervise camera depth and disappear at inference. It may remain a deployed sparse-depth input. Or it may belong to a teacher that distills knowledge into a camera-radar student. All three systems use LiDAR during development, but they have different runtime sensor contracts.

Pretraining and world-model objectives can also use masked reconstruction, future occupancy, point-cloud forecasting, longer history, privileged labels, and larger teachers. The onboard student can inherit the result without copying the full training graph. Multi-task losses still need explicit normalization and conflict handling; one model is not a reason to force every task through the same bottleneck.

## A foundation model is a system of interfaces
<!-- covers: From unified perception to a driving foundation model | Waymo's public foundation-model architecture -->

The boundary between perception and planning becomes less rigid once both consume recurrent world state. A modern planner should represent several plausible futures and several materially different ego actions. Candidate generation, scoring, and validation remain distinct jobs: a safe trajectory can fail because it was never generated, because it was ranked poorly, or because an invalid candidate passed a weak check.

Waymo's public foundation-model description offers one concrete system-level design. A fast Sensor Fusion Encoder establishes metric perception. A Driving vision-language model supplies semantic signals for rare or ambiguous events. A World Decoder predicts behavior, maps, and candidate trajectories. Learned embeddings coexist with materialized objects, semantics, and roadgraph elements, and a separate validation layer checks the proposed trajectory.

The public description does not report every deployed model size, frequency, state field, or safety check. Its useful lesson is the ownership boundary. Language-conditioned semantics can change how a scene should be interpreted, but they do not replace calibration, depth, occupancy, motion, or tracking. Large teachers can improve a smaller Driver, Simulator, and Critic while the onboard graph remains fast and testable.

## How I would build the complete loop
<!-- covers: How I would design a system in this family | The fast path should own geometry and freshness | The slow path should produce grounded hypotheses, not unbounded authority | The world decoder should represent several plausible futures | Generation, scoring, and validation should remain distinct contracts | The training graph should be broader than the onboard graph | The Driver, Simulator, and Critic should share state semantics | Evaluation should test the contracts, not only the final score -->

I would give the fast path ownership of geometry and freshness. Camera, LiDAR, and radar would keep separate encoders until their features occupy calibrated support. The recurrent state would combine occupancy, entities, map elements, compact latent tokens, uncertainty, provenance, and age.

The slower semantic path would wake for uncertainty and rare events. It would produce grounded hypotheses tied to a region or entity, with confidence, timestamp, and expiry. It could modify costs or constraints after grounding into shared state, but it would never overwrite metric geometry directly.

The world decoder would generate several plausible actor futures and ego trajectories. A scorer would rank them. An independent validator would check concrete properties against materialized geometry, dynamics, route rules, sensor health, and fallback policy. Keeping these contracts separate makes a failure diagnosable.

Offline teachers could use longer history, future labels, large language-conditioned models, simulation, and expensive world models. Distillation would transfer intermediate state, future distributions, rankings, and uncertainty into the onboard student. The Driver, Simulator, and Critic would share state semantics so that a real failure can become a diagnosis, a targeted simulation, a training example, and a regression test without losing its meaning.

Evaluation would test the contracts, not only average detection. It would include missing and misaligned sensors, state freshness after occlusion, query saturation, future-mode coverage, stale semantic hypotheses, disagreement among objects and occupancy, tail latency, validator interventions, and regression scenarios found by the Critic.

## The design rule
<!-- covers: The design rule -->

Six questions expose the irreversible decisions. What unique measurement does each sensor contribute? Where does image evidence acquire metric support? Which modality controls admission? What state survives through time? Which world properties are materialized, and which remain latent? What information exists only during training?

My current answer is hybrid at every important boundary: specialized encoders before shared geometry; dense fields alongside sparse entities; materialized state alongside learned embeddings; a fast metric path alongside slower semantic reasoning; and large training-time teachers paired with a smaller onboard student and an independent validator.

A driving foundation model is not defined by parameter count or by the presence of a vision-language model. It earns the name when one learned system preserves measurement-specific evidence, builds a calibrated temporal world state, represents several plausible futures, and exposes enough structure for closed-loop validation.
