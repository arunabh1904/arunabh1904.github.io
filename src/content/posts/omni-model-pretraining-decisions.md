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
summary: How robot pretraining moved from web semantics and action tokens to cross-embodiment data, action chunks, continuous experts, and action-conditioned world models.
---
# Pre-Training for Robotics

_Updated August 22, 2026._

Robot data is expensive. Pretraining asks how much a policy can learn before we collect enough experience on this robot, this task, and this deployment. Internet data can teach the word *drawer*, what drawers look like, and which instruction refers to which handle. It cannot teach how a sticky drawer feels, how far this arm can reach, or what to do after the gripper slips.

That is why robot pretraining is more than adding action tokens to a VLM. Web data supplies semantics. Human video supplies time and interaction. Robot trajectories supply contact, embodiment, intervention, and recovery. We need all three precisely because they teach different things.

A simple history would list RT-1, PaLM-E, RT-2, OpenVLA, and Pi0 in order. The more useful history follows what changed. One branch turned robot actions into tokens. Another let a language model read continuous sensor state. RT-2 joined the two. Cross-embodiment datasets then mixed incompatible robots. Action chunks changed the target from one command to a short trajectory, while world models predicted how an action would change the state.

Each step brings something useful and leaves something behind. Action tokens reuse a pretrained decoder but waste sequence length on smooth motion. Pooled vision can recognize the peg while losing the millimeters needed to insert it. Video can predict plausible movement without learning which robot command caused it.

This post traces which visual, temporal, and motor priors transfer into a robot policy and how they enter the model. It also asks what evidence would show that the transfer affects closed-loop control.

This is Part II of the series. [Part I: Tracing the VLM Progression](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) follows the visual interfaces that made language grounding possible. [Part III: Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) begins after deployment, when the policy creates its own data.

## Putting sensor state and actions into the language model

One early approach expressed robot control as sequence modeling. [RT-1](/paper%20shorts/2022/12/13/rt-1-robotics-transformer-for-real-world-control-at-scale.html) turns images and instructions into tokens, quantizes each action dimension into one of 256 bins, and predicts the next action token. This allowed many tasks to share the same categorical training objective.

[PaLM-E](/paper%20shorts/2023/03/06/palm-e-embodied-multimodal-language-model.html) made the complementary change on the input side. It interleaves visual and continuous sensor embeddings with text, allowing the language model to answer embodied questions and produce plans. Low-level control still remained outside the decoder.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) combines both directions. The VLM conditions on images and instructions, robot actions are represented as output tokens, and web vision-language examples remain in the training mixture. The same autoregressive decoder can therefore produce a textual answer or a robot command.

![RT-2 co-fine-tunes web vision-language examples and robot trajectories through one token interface](/assets/images/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control-paper-figure.png)

*RT-2 turns actions into text-shaped targets, so web knowledge and robot behavior can update one decoder. Source: [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html).*

Action tokenization made robot demonstrations compatible with a decoder already pretrained on images and language. This allowed a robot command such as *move the gripper left* to reuse semantic representations rather than learning the policy entirely from robot data. The representation is convenient for transfer, but it does not remove the structure of continuous control. Per-dimension bins quantize metric motion, and autoregressive decoding adds one serial step for every action token. The shared training objective therefore introduces quantization and control-latency costs.

## Cross-embodiment data exposed hidden robot assumptions

A policy trained on one robot can absorb its camera pose, controller, gripper, and reset procedure as fixed properties of the task. Pooling data across robots makes those assumptions inconsistent.

[Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html) pooled data from 22 embodiments and trained RT-X across them. [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html) treated a new sensor or action space as an adaptation problem. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) combined a pretrained vision-language backbone with 970,000 demonstrations from the same corpus. At this scale, the action schema becomes part of the model. A joint delta, a camera-frame end-effector delta, and a torque command cannot be treated as interchangeable labels.

![Open X-Embodiment pools tasks, scenes, and robot morphologies into a shared training corpus](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-paper-figure.png)

*Cross-embodiment training made the dataset itself an architectural decision. The shared model still needs a schema that says what each robot observation and action means. Source: [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html).*

Normalization is not enough. Mapping every action dimension into $[-1,1]$ does not make a joint delta, a camera-frame end-effector delta, and a torque command physically equivalent. A usable cross-robot corpus must record coordinate frame, units, control mode, frequency, horizon, joint topology, gripper semantics, sensor availability, and calibration.

Cross-embodiment training also changes what *more data* means. Repeating the same task on the same table lowers variance. New scenes, operators, tasks, failures, and robots expand the states and decisions represented in the corpus. Sliding a two-second window forward by one frame may create hundreds of examples without creating hundreds of independent experiences.

Episode count should therefore be reported alongside coverage across tasks, scenes, robots, and failure conditions. The adaptation curve on a held-out robot provides a more direct measure of whether cross-embodiment pretraining transferred.

## Action chunks changed the unit of prediction

Single-step behavior cloning predicts a new action at every control step. This keeps the feedback loop short, but makes it harder to represent a coherent movement over a longer horizon. Each error also changes the state on which the next prediction is conditioned.

[ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html) changes the target from one command to a short sequence of future actions. Each prediction represents a coherent movement rather than a single instant, while temporal ensembling smooths overlapping chunks. The policy therefore makes fewer independent high-level decisions across the same physical trajectory.

![ACT predicts coherent action chunks instead of one control target at a time](/assets/images/action-chunking-with-transformers-act-paper-figure.png)

*Action chunking changes one training target from a scalar command into a short trajectory. Source: [ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html).*

Chunking alone does not handle several valid futures. If the robot can pass an obstacle on the left or the right, ordinary regression may average both paths into a collision. [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html) instead denoises a complete continuous trajectory, preserving multiple possible action chunks.

![Diffusion Policy denoises a continuous action trajectory under visual conditioning](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-paper-figure.png)

*Diffusion preserves a multimodal distribution over continuous action chunks, but it introduces an iterative sampling path. Source: [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html).*

Longer chunks can improve temporal coherence and reduce the number of decoder calls. They also commit the robot further before incorporating a new observation. Receding-horizon control predicts a longer chunk and executes only its prefix, trading additional inference for a shorter open-loop commitment.

Action results should therefore report the predicted horizon, the executed prefix, and the control rate. Without these three values, the duration and feedback frequency of one model output remain ambiguous.

## The action tokenizer became a model decision

Action chunks made the target longer. A naive tokenizer assigns one bin to every action dimension at every timestep, creating a long sequence of nearly repeated values. The language decoder spends the same autoregressive bandwidth on those values that it spends on words.

[FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses the trajectory as a time series before presenting it to the language model. A discrete cosine transform separates broad motion from high-frequency corrections, quantization converts the coefficients into integers, and byte-pair encoding compresses recurring patterns. The resulting action sequence is substantially shorter than per-dimension tokenization over every timestep.

![FAST converts an action chunk into frequency coefficients and compact autoregressive tokens](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-paper-figure.jpg)

*FAST spends tokens on the shape of a trajectory rather than every value at every timestep. Source: [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html).*

Low-frequency coefficients describe the broad motion, while higher frequencies capture abrupt corrections. The autoregressive policy therefore predicts the overall trajectory shape before its finer details. This ordering introduces a smoothness prior. Common low-frequency motion is represented compactly, while a rare high-frequency correction may require more tokens or be attenuated by compression.

The tokenizer therefore affects the policy beyond output formatting. It determines which temporal details are compact and which physical errors are nearby in token space. It also determines the autoregressive sequence length and the likelihood optimized during post-training.

## Continuous action experts separated semantics from control

A separate branch retained the pretrained VLM for images and instructions while moving motor generation outside the language vocabulary. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) adds a continuous action expert trained with flow matching. The shared trunk provides semantic context, while the action expert produces continuous chunks at a bandwidth suited to control.

![Pi0 uses a pretrained vision-language trunk to condition a flow-based action expert](/assets/images/pi0-vision-language-action-flow-model-for-general-robot-control-paper-figure.jpeg)

*Pi0 uses a pretrained vision-language trunk to condition a separate continuous action expert trained with flow matching. Source: [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html).*

Separating semantic processing from motor generation also allows the action path to change with the platform. [GR00T N1](/paper%20shorts/2025/03/18/groot-n1-open-foundation-model-for-humanoid-robots.html) uses a related fast-slow design for humanoids. [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) replaces the original autoregressive token head during adaptation. In its experiments, parallel continuous chunks trained with an L1 loss improve both inference speed and task success.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) uses both representations at different stages. FAST tokens allow web and robot tasks to share a discrete pretraining objective. A continuous expert added during post-training provides finer control and faster inference. This separates the representation used to scale heterogeneous training from the representation used to execute actions.

![Pi0.5 combines tokenized high-level outputs with a continuous low-level action expert](/assets/images/pi0-5-vision-language-action-model-with-open-world-generalization-paper-figure.png)

*Pi0.5 uses FAST tokens during mixture pretraining and adds a continuous action expert during post-training. Source: [Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html).*

I would retain this separation when the action space has physical units, control bandwidth, geometry, or latency requirements that differ from language. The semantic trunk can remain shared, while the motor path is specialized for execution.

## Human video supplied time without robot actions

Robot demonstrations are expensive, while human video provides much broader coverage of objects, scenes, and interactions. Before receiving robot action labels, a video model can learn object persistence through occlusion, hand-object interaction, motion, and the temporal order of a task.

Passive video does not identify the robot command that caused an observed change. [Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html) infers latent actions from unlabeled video and uses them to condition an interactive world model. The latent variables organize transitions in the video, but they are not directly executable on a robot.

One approach separates abundant video pretraining from scarce action-conditioned training. [V-JEPA 2](/paper%20shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html) first learns to predict representations from video without action labels. A smaller second stage then connects robot commands with future latent states. Internet video supplies broad temporal structure, while robot data identifies which state changes are controllable.

A second approach collects human manipulation through an interface closer to robot operation. [Xiaomi-Robotics-1](/paper%20shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html) records UMI trajectories and labels the state change in each sequence. A later cross-embodiment stage aligns those behaviors with robot controls. The collection stage increases task and scene diversity, while the robot stage maps that behavior into executable actions.

![Xiaomi-Robotics-1 separates scalable human-operated capture from robot embodiment alignment](/assets/images/xiaomi-robotics-1-paper-figure-1.png)

*Xiaomi-Robotics-1 collects human manipulation through UMI, labels the observed state change, and later aligns those trajectories with robot commands. Source: [Xiaomi-Robotics-1](/paper%20shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html).*

The transfer claim should therefore be evaluated by the amount of robot data required for adaptation. That adaptation should produce executable commands across new tasks, scenes, and embodiments. Video reconstruction alone does not establish the transfer.

## World models predict what an action changes

A representation model maps observations into state:

$$
z_t=f_\theta(o_{\leq t},\ell).
$$

A policy predicts an action:

$$
p_\psi(a_{t:t+H-1}\mid o_{\leq t},\ell).
$$

An action-conditioned world model predicts what that action changes:

$$
p_\phi(z_{t+1:t+H}\mid z_{\leq t},a_{t:t+H-1},\ell).
$$

These three models may share weights, but they learn different conditional distributions. A video predictor can generate a plausible future while ignoring the proposed action. A behavior-cloning policy can imitate demonstrations without representing the consequences of alternative actions.

An action-conditioned world model includes the intervention in the prediction target. Different proposed actions should produce different predicted futures, allowing a planner to rank the action whose future state reaches the goal.

Pixel reconstruction alone is insufficient evidence for this claim. The predicted state should preserve object identity, pose, contact, irreversible changes, and stability across a long rollout. Planning through the world model should also outperform an equally sized policy or a non-action-conditioned predictor under matched robot data and compute. Video pretraining models what tends to happen; action-conditioned training must identify how the robot's action changes that future.

## Equal examples do not mean equal training

Equal example percentages do not create equal training pressure across text, images, video, and robot episodes. Text becomes a relatively short token sequence, while video expands into frames and patches. One robot episode also produces many overlapping observation windows and action chunks. Each source therefore consumes different compute and contributes a different number of prediction targets.

[![Animation showing equal text, image, video, and action example shares expanding into unequal training units, compute, and shared-parameter updates](/assets/images/blog-multimodal-gradient-budget.gif)](/assets/images/blog-multimodal-gradient-budget.gif)

*Equal example shares do not create equal target counts, FLOPs, or parameter updates. The values are illustrative. The accounting follows the mixed-modal and action interfaces described across [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), and [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html).*

Keep five ledgers:

1. sampled examples or episodes;
2. predicted tokens, patches, latent targets, or action dimensions;
3. FLOPs and wall-clock time;
4. update norm by objective and module;
5. effective independent decisions after temporal overlap.

Increasing the web mixture can improve semantics while weakening metric state. Increasing one robot source can overfit its controller and reduce broader visual-language performance. I would track capability retention alongside gradient norm and gradient cosine similarity in the shared modules. Sampling percentage alone does not describe how strongly each objective updates the shared model.

Pretraining scale needs the same accounting. Episode count, control steps, unique scenes, task diversity, embodiments, failures, robot hours, model size, and compute describe different axes. The relevant outcome is transfer, measured through zero-shot behavior and the rate of adaptation on a held-out robot.

## How to read a robot pretraining paper

I start by identifying what the paper claims to transfer. It may be object knowledge, spatial detail, temporal structure, a motor skill, or a model of action consequences. The next step is to trace where that knowledge enters the deployed policy.

These questions usually expose what changed:

| Question | What it reveals |
| --- | --- |
| What is one training target? | A word, patch, future latent, scalar action, chunk, or trajectory |
| Which parameters already have a prior? | Whether the motor path is truly pretrained or starts from random weights |
| What is shared across embodiments? | Semantics, visible motion, end-effector geometry, or raw controller units |
| Where is information compressed? | Pooled vision, fixed visual queries, action bins, frequency coefficients, or latent actions |
| Does the objective see interventions? | Whether the model learns correlation, behavior, or action-conditioned consequence |
| What is held out? | Objects, scenes, tasks, embodiments, perturbations, or longer horizons |
| What wins at equal budget? | Whether gains survive matched data, compute, active parameters, and latency |

A frozen encoder tests whether the pretrained features already contain what the robot needs. Full fine-tuning tests whether those weights are a useful starting point. World-model reconstruction tests prediction. Closed-loop planning tests whether that prediction helps the robot. These claims should not be collapsed into one transfer score.

## What pretraining must preserve

| Leap | What the model learned to predict | What it bought | What remained unresolved |
| --- | --- | --- | --- |
| Embodied language | sensor-conditioned text | semantic planning and grounded answers | control remained outside the decoder |
| Action tokens | discrete commands in the language vocabulary | one scalable autoregressive objective | quantization and serial control latency |
| Cross-embodiment pretraining | actions across many robots | broader task and robot transfer | incompatible units and controller semantics |
| Action chunks and diffusion | coherent continuous trajectories | temporal consistency and multimodality | open-loop commitment and sampling cost |
| FAST tokenization | compressed frequency-domain trajectories | shorter autoregressive action sequences | sharp corrections can be expensive or lost |
| Continuous action experts | flow or regression chunks conditioned on VLM features | high-rate control without text-shaped outputs | more complex coordination and training |
| Video and latent-action pretraining | temporal change before robot labels | cheap interaction and motion priors | latent change is not yet executable action |
| Action-conditioned world models | consequences under proposed actions | planning and counterfactual ranking | causal fidelity over long rollouts |

Every new target asks the model to preserve more: first semantics, then geometry, motion, actions, and consequences. The action representation decides how much of that knowledge reaches the controller. I would start with a strong vision-language model for semantics, dense visual features that persist through time, and a continuous action expert for control. Discrete action tokens are useful when they make web and robot data easier to train together. I would not assume those same tokens should run the robot. The motor path should be pretrained too, unless an ablation shows that random initialization is good enough.

If the model claims to plan, train it to predict action-conditioned futures and check that a changed action changes the future. If it claims cross-robot transfer, define the shared physical quantity before pooling controls. If it claims scale, report independent decisions and few-shot transfer, not only token count and training loss.

The VLM can know what a drawer is. The temporal model can track that it moved. The world model must preserve what the robot caused. The policy must choose and execute the action before the deadline. Pretraining succeeds when those components transfer useful structure without removing the geometric, temporal, and control distinctions required during execution.
