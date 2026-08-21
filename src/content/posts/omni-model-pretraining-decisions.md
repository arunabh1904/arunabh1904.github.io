---
title: 'Pre-Training for Robotics'
date: '2026-07-15T09:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: omni-model-pretraining-decisions
legacyPath: /blog/2026/07/15/omni-model-pretraining-decisions.html
tags:
  - Robotics
  - Pretraining
  - Multimodal AI
summary: A systems guide to how semantics, geometry, dynamics, and motor priors should transfer from internet, video, simulation, and robot data into a general robot policy—and how to test whether that transfer is real.
---
# Pre-Training for Robotics

_Updated August 21, 2026._

A robot can inherit the word “drawer” from the internet. It can learn what drawers look like, where handles usually are, and which instruction refers to which object. The internet does not tell it how a sticky drawer feels, how far a particular arm can reach, what force closes the gripper, or what to do after the object slips.

That gap is the reason robotics pretraining is both promising and easy to misunderstand. Internet-scale data supplies broad semantics. Human video supplies temporal and interaction structure. Robot trajectories supply actions, contact, embodiment, and recovery. These sources are useful precisely because they are different. Treating them as interchangeable tokens inside one large model does not make the differences disappear.

The central design problem is therefore not simply **how to pretrain a large multimodal model**. It is:

> **Which priors should transfer into control, through which parameters and interfaces, and what evidence would show that the transfer is real?**

A useful robot policy needs at least four kinds of prior:

1. **Semantic priors:** objects, language, tasks, relations, and commonsense structure.
2. **Geometric and temporal priors:** metric position, motion, persistence, occlusion, and state change.
3. **Dynamics and affordance priors:** what can move, what causes what, and how actions change the scene.
4. **Motor priors:** embodiment-specific control, contact, precision, timing, recovery, and constraints.

There is a hidden asymmetry in many VLA recipes: the visual and language pathways arrive with years of pretraining, while the action path starts from random weights. Calling the combined system “pretrained” can obscure that the most deployment-critical interface is still learning from the scarcest data.

The cheapest source can easily erase the most valuable signal. Text is abundant. Robot failures are not. A shared representation can enable transfer while a high-volume objective quietly trains the model to ignore precisely the metric and temporal information that control needs.

This is Part II of the series. [Part I: Tracing the VLM Progression](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) follows the visual interfaces that made language grounding possible. [Part III: Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) begins after deployment, when the policy creates its own state distribution and has to improve from the consequences.

The scope here is pretraining design. The papers establish what happened under particular recipes; the decision rules, experiment plans, and preferred architecture are my synthesis. That boundary matters. Robotics papers often change the dataset, embodiment mix, visual encoder, action representation, parameter count, and training budget together. A strong result can justify adopting a recipe without telling us which ingredient caused it.

## Start with the transfer contract

“Multimodal,” “generalist,” and “foundation model” describe model families. They are not deployment capabilities.

Before choosing an architecture, write the transfer contract as observable behavior:

| Question | Example answer |
| --- | --- |
| What must work without robot-specific data? | Recognize objects, parse instructions, identify plausible grasp regions |
| What may use a small target dataset? | Adapt to a new gripper, camera layout, or action convention |
| What must transfer across scenes? | Geometry, object state, and instruction grounding |
| What must transfer across embodiments? | Task structure and high-level intent, not necessarily raw joint commands |
| What control deadline is non-negotiable? | Replan at 10 Hz while the lower-level controller runs faster |
| What failure kills the program? | No few-shot gain over training from scratch; unstable contact; semantic regressions after robot pretraining |

The phrase **generalization** is too broad unless it names the held-out axis. A policy may generalize to new object instances while failing under a new camera pose. It may follow unseen language while failing when the same task needs a different gripper. It may succeed from clean initial states and collapse after its first small mistake.

The transfer contract should separate at least:

- new objects and visual appearances;
- new scenes and camera geometry;
- new compositions of known skills;
- new task instructions;
- new embodiments or controller conventions;
- perturbations, failures, and recovery states;
- longer horizons than those seen during training.

It should also specify **zero-shot performance, the slope of the few-shot adaptation curve, control latency, and recovery behavior**. A model that reaches 70% after 50 target demonstrations can be more valuable than one that reaches 75% only after 5,000. For pretraining, the most important metric is often not final success. It is **transfer per robot-hour**.

## The pretraining ladder

Robotics pretraining is not one dataset. It is a ladder of increasingly physical supervision.

| Data source | What it can teach | What it usually cannot identify | Representative work |
| --- | --- | --- | --- |
| Image–text and web documents | open-vocabulary semantics, task language, object relations, broad visual concepts | metric state, contact, action consequences | [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html), [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) |
| Human and egocentric video | temporal persistence, hands and objects, interaction sequences, affordance cues | robot action labels, controller dynamics, exact contact forces | [R3M](https://arxiv.org/abs/2203.12601), [Voltron](https://arxiv.org/abs/2302.12766), [VC-1](https://arxiv.org/abs/2303.18240) |
| Human manipulation captured through UMI, retargeting, or robot-format synthesis | broad task and scene diversity with action-like motion supervision | target robot dynamics, calibration, and contact forces | [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330), [Ego2Robot](https://arxiv.org/abs/2608.02580) |
| In-domain robot video without task labels | target viewpoints, robot morphology, scene dynamics, and—when controls are logged—action-conditioned transitions | task intent, success, and broad semantic coverage | [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) |
| Action-labeled robot trajectories | action-conditioned transitions, control conventions, task execution | broad semantic coverage; counterfactual actions not attempted | [RT-1](https://arxiv.org/abs/2212.06817), [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html), [DROID](https://arxiv.org/abs/2403.12945) |
| Failures, interventions, and recoveries | boundary states, corrective actions, what not to do | broad coverage unless deliberately collected | targeted recovery datasets and deployment logs |
| Simulation and synthetic trajectories | controllable diversity, rare events, privileged labels, counterfactuals | the full visual and contact distribution of the real system | sim-to-real and synthetic-data mixtures |

Each rung reduces a different uncertainty. Image–text data can tell the model that a mug is a container. Human video can show that mugs are lifted by handles or bodies. Robot video can expose the target camera and arm. Action-labeled trajectories can connect a command to motion. Recovery data can show what happens after the grasp is slightly wrong.

This leads to a practical rule:

> **Do not ask a dataset to supervise a variable it does not identify.**

A next-frame objective on passive video may learn motion regularities. It cannot, by itself, distinguish whether an object moved because of the robot, a human, gravity, or an unobserved event. A language caption may describe “opening the drawer.” It does not specify the coordinate frame, impedance, force, or timing needed to open this drawer with this robot.

The ladder is also not strictly sequential. A mature run may interleave all levels. But the distinction remains useful because it tells us what signal is scarce, which capability should improve, and what negative transfer to monitor.

A recent line of work tries to build the missing bridge from abundant human video to executable robot behavior. [LAWM](https://arxiv.org/abs/2509.18428) learns latent actions through world modeling, while [CLAP](https://arxiv.org/abs/2601.04061) aligns video transitions with a proprioceptive latent space and an executable codebook. The key question is not whether a latent action reconstructs motion. It is whether a small amount of robot data can consistently map that latent into commands across tasks and embodiments.

This intermediate rung is becoming increasingly important. [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) pretrains on UMI-captured trajectories labeled with descriptive state transitions, then uses cross-embodiment post-training to move from those descriptions and UMI actions to imperative robot instructions and robot control. The collection device, label ontology, and alignment stage are therefore one training design—not separable data-preparation details.

[Ego2Robot](https://arxiv.org/abs/2608.02580) takes a complementary route: retarget and render egocentric human manipulation into robot-format trajectories, curate the result at scale, and jointly pretrain with robot data. The useful test is not whether the synthesized clips look convincing. It is whether they improve transfer across held-out appearance, layout, embodiment, and task shifts—including on a real robot.

## Robot data is structured, correlated, and attached to hardware

Internet data is cheap to copy. Robot trajectories are expensive, correlated, and entangled with the collection system. An hour of teleoperation contains the operator’s habits, controller smoothing, camera calibration, reset procedure, safety limits, and the parts of state that happened to be logged.

“More trajectories” can mean several different things:

| Variation in the corpus | Transfer the model might learn | Shortcut that can look like transfer |
| --- | --- | --- |
| More tasks on one robot | instruction and object semantics | memorized scene or controller conventions |
| More scenes with one setup | visual robustness and geometry | regular reset locations or camera cues |
| More embodiments | shared task structure | averaging incompatible action units |
| More operators | timing and strategy diversity | operator-specific motion signatures |
| More failures and corrections | boundary states and recovery | intervention-device or labeling artifacts |
| More repetitions | lower estimation variance | near-duplicate temporal windows mistaken for diversity |

[Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html) made cross-robot pretraining a concrete program by pooling 22 robot embodiments. [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html) trained on 800,000 Open X trajectories and treated new sensors, action spaces, and embodiments as adaptation problems. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) combined internet visual-language priors with 970,000 robot demonstrations. [DROID](https://arxiv.org/abs/2403.12945) emphasized environmental diversity: 76,000 trajectories across 564 scenes and 84 tasks, collected by 50 operators.

[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) pushes scale along a different axis: more than 100,000 hours of real-world UMI manipulation trajectories, automatically labeled with language descriptions of state transitions, followed by more than 10,000 hours of cross-embodiment post-training. Those are not 100,000 robot hours. The useful lesson is that scalable human-operated capture, automatic semantic labeling, and embodiment alignment can form one data engine whose pretraining gains survive into robot evaluation.

[RoboCat](https://arxiv.org/abs/2306.11706) adds another important axis: the pretrained agent can become part of the data engine. It adapts to new tasks and robots with a small target set, collects additional experience with the adapted policy, and folds that experience back into the next training iteration. That is a different scaling loop from passively accumulating demonstrations; model quality changes the distribution and cost of the next dataset.

Later architectures make the embodiment boundary more explicit. [HPT](https://arxiv.org/abs/2409.20537) uses embodiment-specific stems and heads around a shared trunk. [CrossFormer](https://arxiv.org/abs/2408.11812) studies a shared policy across manipulation, navigation, locomotion, and aviation. These are not merely architecture tricks. They encode a claim: semantic and behavioral structure may transfer broadly, while raw observations and actuator interfaces often should not be forced into one undifferentiated space.

Cross-embodiment learning therefore needs an interface schema, not just padding and normalization. At minimum, log:

| Interface field | Why it matters |
| --- | --- |
| Coordinate frame | A Cartesian delta in the base frame is not the same target as one in the tool or camera frame |
| Units and scale | Radians, meters, velocities, torques, and normalized values carry different geometry |
| Controller mode | Position, velocity, torque, and impedance commands induce different closed-loop behavior |
| Control frequency | The same numeric delta at 5 Hz and 50 Hz does not mean the same motion |
| Action horizon | A one-step command and a 1-second chunk expose different feedback assumptions |
| Joint topology and limits | Embodiments do not share a natural per-axis correspondence |
| Gripper semantics | Binary open/close, width, force, and multi-finger commands are not interchangeable |
| Sensor availability | Missing tactile or proprioceptive input should be represented as missing, not as physical zero |

The most useful dataset table is therefore not a single episode count. It reports task entropy, scene entropy, embodiment coverage, operator coverage, success and recovery rates, control frequencies, action conventions, missing modalities, and effective unique decision windows after temporal overlap.

An episode is not an independent sample. Sliding a two-second window by one frame can turn one maneuver into hundreds of training examples without creating hundreds of new decisions.

## Representations must preserve what control needs

Visual pretraining for robotics sits between two opposing pressures.

Recognition rewards invariance. The representation should ignore lighting, texture, and background changes that do not alter the task. Control needs equivariance and metric detail. A small change in object pose, gripper offset, or contact state should produce a meaningful change in the representation.

The right representation is therefore not “maximally invariant.” It should be:

> **Invariant to nuisance appearance, but sensitive—or equivariant—to task-relevant geometry, motion, and state.**

This is why general image representations do not automatically dominate embodied tasks. [R3M](https://arxiv.org/abs/2203.12601) pretrains on Ego4D human video using temporal and language-alignment objectives, then uses the representation as a frozen perception module. [Voltron](https://arxiv.org/abs/2302.12766) similarly uses language-conditioned human video to preserve both semantic and low-level information. [VC-1](https://arxiv.org/abs/2303.18240) evaluates pretrained visual representations across 17 embodied tasks and finds no universally dominant encoder; task- or domain-specific adaptation remains important.

[Theia](https://arxiv.org/abs/2407.20179) shows another route: distill several visual foundation models with different biases into one compact spatial encoder for robot learning. More recently, [Patch Policy](https://arxiv.org/abs/2607.18236) isolates the cost of visual compression. In its studied tasks, frozen dense DINOv2 or WebSSL patch features substantially outperformed pooled features and were competitive with or better than a much larger fine-tuned VLA on precision-heavy manipulation. The result is not a universal model ranking. It is evidence that direct access to dense spatial tokens can matter more than carrying an entire language model into the control loop.

That result changes how the encoder should be evaluated. ImageNet or retrieval accuracy is not enough. Measure:

- object and instruction semantics;
- keypoint and pose sensitivity;
- temporal correspondence and object persistence;
- contact-state separability;
- robustness to lighting, clutter, viewpoint, and camera changes;
- downstream success under frozen, linear-probe, partial-tuning, and full-tuning regimes;
- few-shot transfer as target demonstrations increase.

A frozen encoder tests whether the prior already contains the right information. Full fine-tuning tests whether the initialization is useful. Those are different questions. A representation that is weak when frozen but adapts rapidly can still be the better pretraining choice.

There is also a compression question. A vision-language model can discard the exact edge, depth, and motion information needed for control while preserving enough semantics to answer a question. The connector cannot recover information the encoder removed. Before sweeping elaborate fusion modules, test resolution, visual-token count, temporal sampling, and encoder adaptation. The controlled studies in [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) are useful here: data composition and visual representation choices often have more leverage than endlessly modifying the connector.

## Representation, world-model, and policy pretraining are different targets

“Pretraining for robotics” often hides three distinct learning problems.

A **representation model** maps observations into useful state:

$$
z_t = f_\theta(o_{\leq t}, \ell).
$$

A **world model** predicts how state changes under an action:

$$
p_\phi(z_{t+1:t+H} \mid z_{\leq t}, a_{t:t+H-1}, \ell).
$$

A **policy** chooses actions from the current history and goal:

$$
p_\psi(a_{t:t+H-1} \mid o_{\leq t}, a_{<t}, \ell).
$$

These objectives can share a backbone, but they do not supervise the same conditional distribution.

### Video prediction is not automatically a world model

Video generation produces plausible futures. A world model must preserve **action-conditioned consequences**. A policy must choose the action that changes the world in the desired way.

A visually realistic model can still be useless for planning if it:

- ignores the action;
- changes the scene in ways that are plausible but causally wrong;
- loses object identity under occlusion;
- smooths over contact and irreversible state changes;
- accumulates error under repeated rollout;
- represents latent actions inconsistently across scenes.

[Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html) learns latent actions from unlabeled video and conditions an interactive dynamics model on them. [DINO-WM](https://arxiv.org/abs/2411.04983) predicts future pretrained visual features from observations and actions, then uses the model for zero-shot planning. [V-JEPA 2](https://arxiv.org/abs/2506.09985) first learns from large-scale internet video and then trains an action-conditioned world model with a comparatively small amount of robot video, demonstrating how broad temporal pretraining can be converted into planning-relevant prediction.

The evaluation contract must change when the model is called a world model. FID, reconstruction quality, or human preference are insufficient. Measure:

1. **Interventional consistency:** do different actions cause the corresponding different futures?
2. **Counterfactual ranking:** does the model prefer the action that actually reaches the goal?
3. **State persistence:** are object identity, pose, and irreversible changes preserved?
4. **Contact fidelity:** are grasps, collisions, and support relations represented correctly?
5. **Rollout stability:** does prediction remain useful over repeated planning steps?
6. **Planning value:** does model-based action selection outperform an equally sized policy or a non-action-conditioned predictor?

The clean ablation is not “better-looking videos.” It is whether action-conditioned pretraining improves downstream planning or policy learning at matched robot data and compute.

### World-action models expose where foresight enters control

The useful taxonomy is not whether a paper uses the phrase *world-action model*. It is where the predicted future enters the action path:

| Interface | Training signal | Inference path | Main failure |
| --- | --- | --- | --- |
| Imagine, then execute | future observations plus inverse dynamics | generate a visual subgoal, then ground it into actions | visual errors propagate into control |
| Predictive-feature conditioning | future-oriented latent features | action head reads the video model’s internal state | latent may be correlated with the task without being causal |
| Joint future-and-action modeling | future and action targets in one model | shared stream or interacting experts emit both | action head may ignore the predicted consequence |
| Auxiliary future prediction | prediction loss regularizes the policy | future decoder can disappear at deployment | gain may be representation shaping rather than online planning |

[From World Models to World Action Models](https://arxiv.org/abs/2607.00836) makes these interfaces explicit. The separation matters because the video model and inverse dynamics model can use different data, while a joint model can share more state but makes causal attribution harder.

A 2026 line of work begins to collapse the world-model and policy interfaces operationally while keeping their objectives legible. [DreamZero](https://arxiv.org/abs/2602.15922) jointly predicts future video and robot actions from a pretrained video backbone. [InternVLA-A1](https://arxiv.org/abs/2601.02456) uses interacting experts for scene understanding, visual foresight, and action. [Dream-Tac](https://arxiv.org/abs/2606.08737) extends joint future-and-action modeling to tactile dynamics for contact-rich manipulation.

This direction is promising because future prediction can regularize action learning with dense state-change supervision. It is also easy to overread. Jointly generating actions and futures does not prove that the action head uses the predicted consequence, that tactile predictions improve closed-loop correction, or that a video backbone has learned controllable dynamics rather than visual regularity. The matched test is to remove, stop-gradient, or scramble the future-prediction path and measure perturbation recovery, counterfactual action ranking, and target-data efficiency.

Tactile data sharpens the same lesson. Contact signals are sparse in time but decisive at slip, insertion, grasp closure, and release. [N0-TWAM](https://arxiv.org/abs/2607.23783) treats touch both as a future-prediction target during pretraining and as observed feedback for action generation. Those roles should be ablated separately: predicting contact may shape the state representation, while conditioning on contact may improve closed-loop correction. Sampling and loss normalization should be conditioned on contact events, not only wall-clock duration.

### Policies and world models scale differently

Behavior cloning learns the behavior present in the dataset. A world model learns observed transitions and can, in principle, evaluate actions that a planner proposes. Their bottlenecks differ. A policy is limited by action coverage and demonstration quality. A world model is limited by state-action coverage and the accuracy of long-horizon consequences.

This matters for scaling. A single power-law exponent should not be assumed to transfer from language modeling to policy loss, from policy loss to world-model error, or from either loss to real-robot success. [Scaling Laws for Pre-training Agents and World Models](https://arxiv.org/abs/2411.04434) explicitly treats behavior cloning and world modeling as different objectives and finds that their scaling behavior depends on the architecture and target.

## Share semantics; specialize interfaces

The phrase “one model” can describe very different parameter-sharing decisions. Tokenizers, encoders, attention blocks, normalization, experts, action heads, losses, and optimizer states can be shared independently.

For robotics, four recurring architecture patterns are useful:

| Pattern | Representative systems | What is shared | What is specialized | Main risk |
| --- | --- | --- | --- | --- |
| One token stream and autoregressive decoder | [Gato](https://arxiv.org/abs/2205.06175), [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) | sequence model and next-token interface | modality tokenizers | quantization, long action sequences, text-rate decoding |
| Shared trunk with embodiment stems and heads | [HPT](https://arxiv.org/abs/2409.20537), [CrossFormer](https://arxiv.org/abs/2408.11812) | task and behavioral representation | sensor adapters and actuator heads | trunk may learn an over-averaged representation |
| Pretrained VLM plus continuous action expert | [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), [GR00T N1](https://arxiv.org/abs/2503.14734) | semantic context and visual-language reasoning | high-rate continuous action generation | coordination and serving complexity |
| Embodied reasoning layered with control | [PaLM-E](https://arxiv.org/abs/2303.03378), [Gemini Robotics](https://arxiv.org/abs/2503.20020), hierarchical VLA systems | language grounding, task context, and planning | reactive action generation and embodiment interfaces | errors or latency at the reasoning–control boundary |

![The Pi0 paper's architecture, with a pretrained vision-language trunk conditioning a flow-based action expert](/assets/images/pi0-vision-language-action-flow-model-for-general-robot-control-paper-figure.jpeg)
_Pi0 keeps semantic context in the pretrained VLM path while a separate action expert models continuous action chunks. The split is a concrete example of selective sharing rather than forced token unification. Source: [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)._

The architecture rule I find most useful is:

> **Share the parameters that should learn common invariances. Specialize interfaces whose units, bandwidth, geometry, or deadlines differ.**

Language is an excellent interface for task semantics and a poor native clock for high-frequency control. A Cartesian action has metric structure that a word token does not. Tactile streams and camera streams have different rates. A humanoid and a mobile manipulator may share “pick up the cup” while sharing no obvious joint-space target.

That does not mean language-shaped supervision has no role in motor pretraining. [LAP](https://arxiv.org/abs/2602.10556) represents the net effect of low-level end-effector actions in structured natural language to encourage cross-embodiment transfer, then pairs the resulting VLM representation with a lightweight continuous action expert for efficient execution. The distinction is important: **the representation used to align actions across robots need not be the interface used to serve actions at control rate.**

Selective sharing is not free. A separate action expert adds parameters and a serving path. An embodiment-specific head may hide whether the trunk learned transferable structure. Every split should therefore be tested under matched active parameters, sequence length, training compute, and inference latency.

### The shared variable matters more than the shared decoder

Normalizing every action dimension into $[-1,1]$ does not create shared physical semantics. Cross-embodiment pretraining needs a bridge variable that is invariant enough to pool data and predictive enough to guide control.

Candidate bridges include camera-frame tool-center-point motion, end-effector traces, object-centric motion, language descriptions of motion, and learned latent actions. [Unify Robot Actions in Camera Frame](https://arxiv.org/abs/2511.17001) converts heterogeneous datasets into standardized camera-frame TCP actions so that the output shares the observation’s geometry. [Cross-Embodiment Transfer via Behavior-Aligned Representations](https://arxiv.org/abs/2607.27549) studies language motions, bounding boxes, and end-effector traces; in its benchmark, end-effector traces were especially useful and became more valuable as the prior dataset grew.

These approaches make different bets. Camera-frame actions preserve metric executability but require calibration. End-effector traces align visible behavior while leaving inverse kinematics to a later stage. Language motion is broad and easy to co-train with a VLM but imprecise. Latent actions can be compact but may entangle appearance or embodiment. The bridge used during pretraining and the native command served at execution need not be the same.

### Pretrain the motor path, not only the semantic trunk

A common VLA recipe starts with a pretrained VLM and a randomly initialized action module. Early optimization must simultaneously discover valid motion, align that motion with visual-language context, and adapt to the target controller. That is an avoidable three-way burden.

[Learning Action Priors for Cross-embodiment Robot Manipulation](https://arxiv.org/abs/2606.26095) pretrains an action encoder-decoder on unconditioned trajectories before cross-modal VLA training, then reuses and distills that motion prior during alignment. In its experiments, this improves convergence and data-scarce transfer. The broader design rule is not that every action head should be pretrained blindly. It is that motion structure deserves the same explicit prior and ablation that vision and language receive.

Compare random initialization, within-embodiment action pretraining, cross-embodiment action pretraining, and frozen versus jointly tuned motor modules under matched robot data. If action-only pretraining lowers reconstruction loss but does not improve closed-loop success, perturbation response, or few-shot adaptation, it learned smoothness rather than a useful control prior.

### Measure gradient agreement instead of debating fusion abstractly

The useful question is not whether fusion is “early” or “late.” It is whether objectives produce compatible updates in the modules they share.

Build an interaction matrix:

| Add this objective ↓ / measure this capability → | Semantics | Metric state | Temporal prediction | Action | Recovery |
| --- | --- | --- | --- | --- | --- |
| Text/image–text pretraining | — | ? | ? | ? | ? |
| Human-video pretraining | ? | — | ? | ? | ? |
| Action-conditioned world modeling | ? | ? | — | ? | ? |
| Robot behavior cloning | ? | ? | ? | — | ? |
| Failure/recovery training | ? | ? | ? | ? | — |

Each cell should report transfer at matched compute along with per-objective gradient norm, cosine similarity, update norm by module, and downstream capability. A lower joint loss does not imply that any robot capability improved.

## A dataloader share is not a learning share

Before choosing transformer depth, decide what counts as one training unit.

Text arrives as compressed symbolic tokens. Images become patches or visual features. Video adds frame sampling and temporal compression. Robot trajectories become overlapping observation windows and action targets. Actions may be per-axis bins, chunks, frequency coefficients, diffusion trajectories, or flow fields.

The unit determines sequence length. Sequence length determines attention cost, context competition, batch composition, and loss balance. Counting one image, one clip, and one robot trajectory as one “example” hides how many targets and how much compute actually reach the shared model.

The figure makes this accounting problem concrete using an illustrative equal-example mixture. The percentages are not a reported recipe; they are held fixed so that the later imbalances become visible.

[![Animation showing equal text, image, video, and action example shares expanding into unequal training units, compute, and shared-parameter updates](/assets/images/blog-multimodal-gradient-budget.gif)](/assets/images/blog-multimodal-gradient-budget.gif)
*Equal sampled-example shares need not produce equal predicted units, FLOPs, or update norms. Video and action sequences expand differently before they reach shared parameters, so one percentage cannot describe the mixture. Custom explanatory synthesis informed by [MM1](https://arxiv.org/abs/2403.09611), [Scaling Laws for Generative Mixed-Modal Language Models](https://arxiv.org/abs/2301.03728), [Scaling Laws for Optimal Data Mixtures](https://arxiv.org/abs/2507.09404), and [Pi0](https://arxiv.org/abs/2410.24164). Values are illustrative, not paper-reported measurements.*

> **Deep insight:** A dataloader share is not a learning share. Sequence expansion sets the target count, the model path sets compute, and gradient alignment decides who moves the shared trunk.

Keep four ledgers side by side:

1. sampled examples or episodes;
2. predicted units—tokens, patches, frames, latent targets, or action dimensions;
3. consumed FLOPs and wall-clock time;
4. update norm by objective and module.

Then add a fifth robotics-specific ledger: **effective independent decisions**. Overlapping windows from one trajectory can contribute thousands of action dimensions while containing one maneuver. The repeated supervision may still be useful, but it should not be mistaken for behavioral diversity.

Normalize losses at several levels:

- per predicted unit, so target dimensionality is visible;
- per example or trajectory, so long sequences do not silently dominate;
- per consumed FLOP, so expensive modalities justify their allocation;
- per independent scene-task-embodiment unit, so duplication is visible.

A mixture is balanced only relative to a capability objective. It is not balanced because its percentages sum to 100.

## The action interface is four decisions, not one

“Discrete versus continuous actions” compresses several orthogonal design choices into one phrase.

### 1. Coordinate system

The policy may predict joint positions, joint deltas, velocities, torques, end-effector poses, end-effector deltas, gripper commands, or a parameterization consumed by another controller. This choice determines what can transfer across embodiments and which physical structure the model must learn.

End-effector deltas can align different arms at the task level, but only if frame conventions, gains, control rates, and low-level controllers are compatible. Joint-space actions preserve embodiment fidelity but expose little direct correspondence across robots.

### 2. Temporal basis

The model may predict one action, a fixed chunk, a variable-duration skill, or coefficients that reconstruct a trajectory.

[ACT](https://arxiv.org/abs/2304.13705) models action chunks to improve precise manipulation. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses high-frequency action chunks using a discrete cosine transform before tokenization. Chunking reduces autoregressive latency and can capture coherent motion. It also creates an open-loop commitment. The longer the chunk, the more the policy assumes that the world will evolve as expected.

That creates a three-way tradeoff:

- short horizons provide feedback but increase inference and communication cost;
- long horizons improve temporal coherence but accumulate model and environment error;
- receding-horizon execution predicts a long chunk but executes only a prefix, paying extra inference to regain feedback.

Control rate and executed-prefix length belong in every action-model result. “One action” is incomplete without its duration.

### 3. Conditional action distribution

The policy may use direct regression, discretized autoregression, a mixture model, diffusion, or flow matching.

| Distribution | Strength | Main cost | Typical failure |
| --- | --- | --- | --- |
| Direct L1/L2 regression | simple, fast, stable | weak multimodality | averages incompatible strategies |
| Discrete autoregression | explicit likelihood, shared token machinery | sequential decoding and quantization | slow high-frequency control |
| Discrete mode plus continuous offset | captures modes while preserving precision | more supervision and interface design | unstable or arbitrary clusters |
| Diffusion | expressive multimodal sequence generation | iterative sampling | latency and sampler sensitivity |
| Flow matching | continuous trajectories with efficient generation | separate objective and serving path | semantic/action coordination |

[Behavior Transformer](https://arxiv.org/abs/2206.11251) combines discretized action modes with continuous offsets. [Diffusion Policy](https://arxiv.org/abs/2303.04137) treats visuomotor control as conditional action-sequence diffusion and uses receding-horizon execution. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) uses a flow-based action expert. [RDT-1B](https://arxiv.org/abs/2410.07864) scales a diffusion transformer for bimanual action generation.

A lower behavior-cloning loss does not settle this choice. Compare success, multimodal coverage, contact-heavy error, response to perturbation, inference latency, and the amount of target data needed to adapt.

### 4. Who owns action generation?

Actions can be emitted by the language decoder, a dedicated head, an embodiment expert, a diffusion or flow module, or a lower-level controller conditioned by a high-level planner.

This is independent of tokenization. A dedicated action expert can output discrete tokens. A language-shaped trunk can condition continuous flow. The ownership decision determines which parameters absorb the motor prior and whether semantic reasoning and control compete at every update.

The action-interface memo should therefore state:

| Axis | Decision |
| --- | --- |
| Physical space | joints, Cartesian pose, velocity, torque, impedance target, skill |
| Reference frame and units | base/tool/camera; metric or normalized; controller convention |
| Temporal representation | single step, chunk, variable skill, frequency coefficients |
| Conditional distribution | regression, categorical, mixture, diffusion, flow |
| Generator | shared decoder, action head, expert, or controller hierarchy |
| Execution | open loop, receding horizon, event-triggered replan |
| Deadline | target control and end-to-end inference rates |

Without these fields, two “continuous-action VLAs” may be solving different problems.

## Data mixtures and curricula

A robotics corpus is not a pile of datasets. It is a sampling policy over sources, embodiments, tasks, quality levels, sequence lengths, success states, and training stages.

There is no context-free optimal percentage of text, image–text, video, and robot data. A mixture optimized for language grounding may be wrong for metric state. A mixture optimized for imitation loss may be wrong for recovery. A mixture optimized for one embodiment may wash out a smaller embodiment with different control statistics.

A useful curriculum often follows the causal ladder:

1. **Semantics:** image–text, language, and broad visual understanding.
2. **Temporal state:** human and egocentric video, correspondence, and object persistence.
3. **Embodied dynamics:** robot video and action-conditioned prediction.
4. **Behavior:** heterogeneous action-labeled trajectories.
5. **Target embodiment:** a small high-quality adaptation set, including failures and recoveries.

But “first A, then B” is not automatically better than interleaving. Staging can reduce optimization difficulty. It can also cause catastrophic forgetting. After every transition, rerun retention evaluations for semantics, geometry, world-model consistency, and prior embodiments. Compare the staged curriculum against an equal-compute interleaved mixture.

Rehearsal should be capability-driven. If robot behavior cloning degrades open-vocabulary grounding, retain enough semantic data to stop the regression. If web pretraining makes the state representation insensitive to small pose changes, increase temporal, metric, or robot objectives rather than merely changing a scalar loss weight.

Mixture weights should be selected through proxy runs that vary:

- model size and total compute;
- total examples and effective unique decisions;
- source and embodiment weights;
- annotation ontology and prompt form—state-transition descriptions, imperative instructions, subtask labels, and success labels;
- image and video compression;
- frame rate, context length, and action horizon;
- shared versus expert capacity;
- curriculum order and learning-rate resets;
- replay or retention allocation.

Prompt form is a curriculum variable. Xiaomi-Robotics-1 separates descriptive state-transition labels during pretraining from imperative instructions during post-training. The former is scalable to automatic annotation; the latter matches how a deployed robot is commanded. Mixing them without an explicit task or prompt-type signal can teach the model to treat a description of what happened as equivalent to a command for what should happen next.

Report uncertainty on the **ranking** of candidate mixtures at target scale. The expensive decision is not whether a fitted loss curve is smooth. It is whether candidate A is likely to beat candidate B when the budget increases.

## Scaling robotics data

Robotics has several scales, and they should not be collapsed:

- number of episodes;
- number of control steps;
- number of unique scenes, objects, and tasks;
- number of embodiments and operators;
- coverage of failures and recoveries;
- model parameters and active parameters;
- training FLOPs;
- real-robot collection hours;
- target-embodiment adaptation hours.

The scarce axis is often diversity, not raw step count. [Data Scaling Laws in Imitation Learning for Robotic Manipulation](https://arxiv.org/abs/2410.18647) studies more than 40,000 demonstrations and 15,000 real-robot rollouts. In its studied single-task regimes, adding environment or object diversity can be more valuable than repeatedly demonstrating the same variation after a threshold. That finding should not be universalized to every robot or task, but it is a warning against using episode count as the only x-axis.

[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) reports that increasing both pretraining data and model size improves action prediction and that stronger pretrained models retain an advantage after cross-embodiment post-training. That is stronger evidence than a source loss curve alone, but its x-axis still bundles UMI capture, automatic language labeling, task diversity, architecture, and embodiment alignment. The result supports the recipe; it does not identify one universal robotics exponent.

The common currency across human video, UMI data, simulation, and robot trajectories is therefore not hours. It is the change in zero-shot behavior and in the few-shot adaptation curve on a held-out robot.

Every scaling claim should name:

1. the fitted range of model size, data, and compute;
2. the architecture, representation, and action interface held fixed;
3. the target loss or capability measured;
4. the diversity and quality assumptions;
5. the residuals and confidence interval;
6. the extrapolation from proxy to target scale;
7. the experiment that would reverse the recommendation.

The bottleneck can move. At small scale, model capacity may dominate. At larger scale, unique robot data, action-interface distortion, long-horizon feedback, or control latency may take over. A modality interaction found with one visual encoder may disappear after changing resolution. A favorable token scaling law may reverse when the action decoder misses its real-time deadline.

For every candidate architecture, write a kill criterion before the proxy run:

- **One autoregressive token stream:** kill if action latency or quantization error grows faster than transfer benefit.
- **Shared trunk across embodiments:** kill if smaller embodiments regress or target adaptation is no better than separate training.
- **Continuous action expert:** kill if the extra path buys no perturbation robustness or multimodal coverage at matched compute.
- **Action-conditioned world model:** kill if interventions do not produce consistent causal changes or planning does not improve.
- **Long action chunks:** kill if open-loop drift outweighs the throughput gain.
- **Large internet mixture:** kill or reduce if semantic gains come with worse metric state or slower robot adaptation.

That habit turns paper reading into capital allocation.

## Proxy runs before the large run

A useful proxy program does not attempt to reproduce the final model at miniature scale. It tests the decisions most likely to reverse the architecture.

Run four experiment layers:

### 1. Representation sweep

Compare visual encoders, dense patch features versus pooled or compressed tokens, resolution, temporal context, frozen versus adapted perception, and the location of metric or 3D state. Include downstream probes for semantics, pose sensitivity, temporal correspondence, and contact-relevant state.

### 2. Action-interface sweep

Reconstruct identical trajectories with candidate coordinate systems, horizons, tokenizers, and distributions. Compare randomly initialized and action-pretrained motor modules. Measure spectral error, abrupt corrections, contact-heavy segments, wall-clock latency, and closed-loop success.

### 3. Interaction sweep

Vary data mixture, annotation and prompt ontology, loss normalization, curriculum order, shared capacity, and embodiment routing. Measure objective gradients and capability retention, not only aggregate loss.

### 4. Scaling sweep

Train several model and data sizes with held-out real-robot evaluations. Fit target-specific curves for semantic grounding, few-shot transfer, world-model error, policy success, and latency. Do not assume one curve predicts all five.

The deliverables should be decision artifacts:

- an architecture memo with matched-budget alternatives;
- a data-interface schema for every embodiment;
- an objective-interaction matrix;
- a mixture and scaling model with uncertainty;
- a throughput and memory model by modality;
- a capability dashboard with kill criteria;
- a failure playbook and restart objective.

The best proxy metric is the one that predicts the expensive real-robot decision. Cheap simulator success is useful only after showing that its model ranking tracks real performance.

## Evaluation must match the transfer claim

A pretraining paper is only as strong as the split that defines “unseen.” Random trajectory splits often leak scene, object, operator, and temporal near-duplicates across train and test.

Use explicit held-out cells:

| Axis | Example holdout |
| --- | --- |
| Visual | unseen lighting, backgrounds, camera poses, distractors |
| Semantic | new object instances, attributes, referring expressions |
| Task | unseen compositions or instruction paraphrases |
| Behavioral | perturbations, alternate strategies, recovery states |
| Embodiment | new robot, gripper, joint topology, or controller |
| Temporal | longer horizon or delayed consequence |
| Geographic/system | new lab, collection team, calibration, hardware revision |

[STAR-Gen](https://arxiv.org/abs/2503.01238) is useful because it distinguishes visual, semantic, and behavioral generalization rather than reporting a single “generalization” number. [SIMPLER](https://arxiv.org/abs/2405.05941) asks whether paired simulation can preserve real-world policy rankings, which is the right question for a scalable proxy. [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233) also shows how easily manipulation results can depend on shortcut solvability, data source, and statistical noise; the split and confidence interval are part of the result, not appendix details.

For each held-out axis, report:

- zero-shot success;
- success as a function of target demonstrations;
- performance after perturbation, not only from clean starts;
- action and end-to-end latency;
- variance across seeds, scenes, and hardware runs;
- regression on previously supported tasks and embodiments.

Few-shot curves are especially revealing. They separate a broad prior from a merely large policy. A useful pretraining run should either improve zero-shot behavior, reduce the data needed to reach a target success rate, or improve the asymptote under the same target data. If it does none of those, the pretraining did not transfer where claimed.

## The training system is part of the experiment

Once architecture and mixture are chosen, the question becomes operational: can the training stack preserve the intended experiment for weeks or months?

[TorchTitan](/paper%20shorts/2024/10/09/torchtitan-one-stop-pytorch-native-solution-for-production-ready-llm-pre-training.html) is useful as a systems reference because it treats parallelism, checkpointing, compilation, and logging as composable parts of one stack. For multimodal robotics, restart time and diagnostic quality matter at least as much as peak throughput.

The runbook should cover:

| Failure | Detection | Automatic response | Evidence retained |
| --- | --- | --- | --- |
| Sudden loss spike | per-source loss and gradient outlier | skip or roll back; quarantine batch | data IDs, optimizer state, activation stats |
| Slow capability degradation | eval residual versus proxy prediction | pause curriculum transition | mixture, learning rate, norm and update trends |
| Modality domination | update share by objective and module | resample, reweight, or route | gradient norms and cosine similarity |
| Embodiment domination | per-embodiment loss and update share | rebalance or cap repeated windows | embodiment/task/source lineage |
| Corrupt video or action shard | decode, timestamp, and continuity checks | quarantine shard | raw source and preprocessing version |
| Unit or frame mismatch | action-range and kinematic validation | block batch or dataset | coordinate schema and conversion commit |
| Dataloader or network stall | step-time decomposition | replace worker or rank | host, shard, topology, retry logs |
| Eval regression | capability dashboard | block checkpoint promotion | all changes since last accepted checkpoint |

Every checkpoint should bind model state to optimizer, scheduler, data cursor, mixture policy, tokenizers, action schemas, code commit, and evaluation configuration. A weight file without that lineage is not a recoverable experiment.

## A practical reading path

There is no single correct paper order. Read according to the decision being made and produce an artifact after each layer.

**Where do semantic, spatial, and temporal priors come from?** Read [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [R3M](https://arxiv.org/abs/2203.12601), [VC-1](https://arxiv.org/abs/2303.18240), [Theia](https://arxiv.org/abs/2407.20179), [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html), and [Patch Policy](https://arxiv.org/abs/2607.18236). Produce an information-flow diagram showing where metric, spatial, and temporal information can be lost.

**What transfers across robots?** Read [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html), [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html), [HPT](https://arxiv.org/abs/2409.20537), [Unify Robot Actions in Camera Frame](https://arxiv.org/abs/2511.17001), [LAP](https://arxiv.org/abs/2602.10556), and [Cross-Embodiment Transfer via Behavior-Aligned Representations](https://arxiv.org/abs/2607.27549). Produce a dataset-diversity table, an embodiment-interface schema, and a memo naming the shared bridge variable.

**What action prior should the model learn?** Read [Behavior Transformer](https://arxiv.org/abs/2206.11251), [ACT](https://arxiv.org/abs/2304.13705), [Diffusion Policy](https://arxiv.org/abs/2303.04137), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), [RDT-1B](https://arxiv.org/abs/2410.07864), and [Learning Action Priors for Cross-embodiment Robot Manipulation](https://arxiv.org/abs/2606.26095). Produce the action-interface memo, an initialization ablation, and a closed-loop latency benchmark.

**Should the model predict the world or the action?** Read [Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html), [DINO-WM](https://arxiv.org/abs/2411.04983), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [World Action Models Are Zero-shot Policies](https://arxiv.org/abs/2602.15922), and [From World Models to World Action Models](https://arxiv.org/abs/2607.00836). Produce an interventional world-model evaluation and a matched policy-versus-planning comparison.

**Which result deserves the large run?** Read [Data Scaling Laws in Imitation Learning](https://arxiv.org/abs/2410.18647), [Scaling Laws for Optimal Data Mixtures](/paper%20shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html), [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330), [SIMPLER](https://arxiv.org/abs/2405.05941), and [STAR-Gen](https://arxiv.org/abs/2503.01238). Produce a go/no-go memo with confidence intervals and precommitted kill criteria.

Then move to [Part III: Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html). Pretraining decides what the policy can represent and which behaviors are nearby. Deployment reveals which nearby behaviors are actually useful.

## A testable thesis

The strongest robot model will probably not make every modality or embodiment identical. It will share the parameters that benefit from transfer and specialize the interfaces where geometry, bandwidth, time, and control impose different requirements.

My preferred starting hypothesis is:

- a strong vision-language trunk for semantics and task context;
- metric, temporally persistent state representations that are not compressed away by language supervision;
- explicit action-conditioned dynamics when planning is part of the contract;
- a behavior-aligned bridge variable for cross-embodiment transfer rather than blind per-axis normalization;
- embodiment-aware input adapters and continuous action experts;
- a pretrained motor path—or explicit evidence that random initialization is not the bottleneck;
- objective accounting in examples, targets, FLOPs, independent decisions, and parameter updates;
- a mixture chosen through proxy scaling and retention tests rather than intuition;
- evaluation built around zero-shot behavior, few-shot slopes, perturbation recovery, and real-time execution.

The VLM can know what a drawer is. The temporal model can track that it moved. The world model must preserve what the robot’s action caused. The policy must choose and execute the action under a deadline. Pretraining succeeds only when those capabilities transfer into one another without erasing the distinctions that physical control requires.

The practical standard is demanding but clear: a literature review should end as an experiment plan. It should say what one training unit is, where information is compressed, how data becomes gradient, which prior is expected to transfer, what matched control is missing, and what result would reverse the architecture decision.

Anything less is a tour of papers, not a pretraining strategy.

## References

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)
- [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601)
- [Language-Driven Representation Learning for Robotics](https://arxiv.org/abs/2302.12766)
- [Where Are We in the Search for an Artificial Visual Cortex for Embodied Intelligence?](https://arxiv.org/abs/2303.18240)
- [Theia: Distilling Diverse Vision Foundation Models for Robot Learning](https://arxiv.org/abs/2407.20179)
- [Patch Policy: Efficient Embodied Control via Dense Visual Representations](https://arxiv.org/abs/2607.18236)
- [Gato: A Generalist Agent](https://arxiv.org/abs/2205.06175)
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)
- [RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/abs/2306.11706)
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)
- [DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945)
- [Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories](https://arxiv.org/abs/2607.15330)
- [Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data](https://arxiv.org/abs/2608.02580)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers](https://arxiv.org/abs/2409.20537)
- [Scaling Cross-Embodied Learning](https://arxiv.org/abs/2408.11812)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [Behavior Transformers](https://arxiv.org/abs/2206.11251)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [Pi0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)
- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
- [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020)
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning](https://arxiv.org/abs/2411.04983)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- [Latent Action Pretraining Through World Modeling](https://arxiv.org/abs/2509.18428)
- [CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos](https://arxiv.org/abs/2601.04061)
- [InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation](https://arxiv.org/abs/2601.02456)
- [World Action Models Are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836)
- [N0-TWAM: Scaling Tactile-Native World Action Model for Contact-Rich Manipulation](https://arxiv.org/abs/2607.23783)
- [LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer](https://arxiv.org/abs/2602.10556)
- [Learning Action Priors for Cross-embodiment Robot Manipulation](https://arxiv.org/abs/2606.26095)
- [Unify Robot Actions in Camera Frame](https://arxiv.org/abs/2511.17001)
- [Cross-Embodiment Transfer via Behavior-Aligned Representations](https://arxiv.org/abs/2607.27549)
- [Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation](https://arxiv.org/abs/2606.08737)
- [Scaling Laws for Pre-training Agents and World Models](https://arxiv.org/abs/2411.04434)
- [Data Scaling Laws in Imitation Learning for Robotic Manipulation](https://arxiv.org/abs/2410.18647)
- [SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941)
- [A Taxonomy for Evaluating Generalist Robot Manipulation Policies (STAR-Gen)](https://arxiv.org/abs/2503.01238)
- [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233)
- [Scaling Laws for Generative Mixed-Modal Language Models](https://arxiv.org/abs/2301.03728)
- [Scaling Laws for Optimal Data Mixtures](https://arxiv.org/abs/2507.09404)
