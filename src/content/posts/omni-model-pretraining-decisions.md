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

Robot pretraining began with two complementary interfaces. RT-1 represented actions as discrete targets; PaLM-E let a language model consume visual and continuous state. RT-2 joined web-trained visual semantics to tokenized robot control. Subsequent work changed the training corpus, action horizon, and decoder because semantic transfer alone did not solve control bandwidth or cross-robot adaptation.

The resulting branches remain distinct. Open X-Embodiment expands the robots and tasks represented in training. ACT and Diffusion Policy predict action chunks. FAST compresses those chunks for autoregressive decoding, while Pi0 uses a continuous action expert. Video pretraining supplies motion structure before action-conditioned data connects that structure to robot commands.

This is Part II of the series. [Part I: Tracing the VLM Progression](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) follows the visual interfaces that made language grounding possible. [Part III: Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) begins after deployment, when the policy creates its own data.

## Putting sensor state and actions into the language model

One early approach expressed robot control as sequence modeling. [RT-1](/paper%20shorts/2022/12/13/rt-1-robotics-transformer-for-real-world-control-at-scale.html) turns images and instructions into tokens, quantizes each action dimension into one of 256 bins, and predicts the resulting categorical action values. This allowed many tasks to share the same categorical training objective.

[PaLM-E](/paper%20shorts/2023/03/06/palm-e-embodied-multimodal-language-model.html) made the complementary change on the input side. It interleaves visual and continuous sensor embeddings with text, allowing the language model to answer embodied questions and produce plans. Low-level control still remained outside the decoder.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) combines both directions. The VLM conditions on images and instructions, robot actions are represented as output tokens, and web vision-language examples remain in the training mixture. The same autoregressive decoder can therefore produce a textual answer or a robot command.

![RT-2 co-fine-tunes web vision-language examples and robot trajectories through one token interface](/assets/images/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control-paper-figure.png)
*RT-2 turns actions into text-shaped targets, so web knowledge and robot behavior can update one decoder. source: [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html)*

Action tokenization made robot demonstrations compatible with a decoder already pretrained on images and language. This allowed a robot command such as *move the gripper left* to reuse semantic representations rather than learning the policy entirely from robot data. The representation is convenient for transfer, but it does not remove the structure of continuous control. Per-dimension bins quantize metric motion. A token head that decodes those values serially also adds one inference step per action token; RT-1's released design avoids that extra action-token feedback, while later autoregressive VLAs make the latency trade explicit. The shared training objective therefore introduces quantization and, for serial heads, control-latency costs.

## Cross-embodiment data exposed hidden robot assumptions

A policy trained on one robot can absorb its camera pose, controller, gripper, and reset procedure as fixed properties of the task. Pooling data across robots makes those assumptions inconsistent.

[Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html) pooled data from 22 embodiments and trained RT-X across them. [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html) treated a new sensor or action space as an adaptation problem. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) combined a pretrained vision-language backbone with 970,000 demonstrations from the same corpus. At this scale, the action schema becomes part of the model. A joint delta, a camera-frame end-effector delta, and a torque command cannot be treated as interchangeable labels.

![Open X-Embodiment pools tasks, scenes, and robot morphologies into a shared training corpus](/assets/images/open-x-embodiment-robotic-learning-datasets-and-rt-x-models-paper-figure.png)
*Cross-embodiment training made the dataset itself an architectural decision. The shared model still needs a schema that says what each robot observation and action means. source: [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html)*

Scaling each action dimension into a common normalized range $[-1,1]$ preserves neither units nor controller semantics. Cross-robot training therefore depends on the coordinate frame, control mode, frequency, horizon, and embodiment recorded with each trajectory.

Cross-embodiment training also changes what *more data* means. Repeating the same task on the same table lowers variance. New scenes, operators, tasks, failures, and robots expand the states and decisions represented in the corpus. Sliding a two-second window forward by one frame may create hundreds of examples without creating hundreds of independent experiences.

Episode count should therefore be reported alongside coverage across tasks, scenes, robots, and failure conditions. The adaptation curve on a held-out robot provides a more direct measure of whether cross-embodiment pretraining transferred.

## Action chunks changed the unit of prediction

Single-step behavior cloning predicts a new action at every control step. This keeps the feedback loop short, but makes it harder to represent a coherent movement over a longer horizon. Each error also changes the state on which the next prediction is conditioned.

[ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html) changes the target from one command to a short sequence of future actions. Each prediction represents a coherent movement rather than a single instant, while temporal ensembling smooths overlapping chunks. The policy therefore makes fewer independent high-level decisions across the same physical trajectory.

![ACT predicts coherent action chunks instead of one control target at a time](/assets/images/action-chunking-with-transformers-act-paper-figure.png)
*Action chunking changes one training target from a scalar command into a short trajectory. source: [ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html)*

Chunking alone does not handle several valid futures. If the robot can pass an obstacle on the left or the right, ordinary regression may average both paths into a collision. [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html) instead denoises a complete continuous trajectory, preserving multiple possible action chunks.

![Diffusion Policy denoises a continuous action trajectory under visual conditioning](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-paper-figure.png)
*Diffusion preserves a multimodal distribution over continuous action chunks, but it introduces an iterative sampling path. source: [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html)*

Longer chunks can improve temporal coherence and reduce the number of decoder calls. They also commit the robot further before incorporating a new observation. Receding-horizon control predicts a longer chunk and executes only its prefix, trading additional inference for a shorter open-loop commitment.

Action results should therefore report the predicted horizon, the executed prefix, and the control rate. Without these three values, the duration and feedback frequency of one model output remain ambiguous.

## The action tokenizer became a model decision

Action chunks made the target longer. A naive tokenizer assigns one bin to every action dimension at every timestep, creating a long sequence of nearly repeated values. The language decoder spends the same autoregressive bandwidth on those values that it spends on words.

[FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses the trajectory as a time series before presenting it to the language model. A discrete cosine transform separates broad motion from high-frequency corrections, quantization converts the coefficients into integers, and byte-pair encoding compresses recurring patterns. The resulting action sequence is substantially shorter than per-dimension tokenization over every timestep.

![FAST converts an action chunk into frequency coefficients and compact autoregressive tokens](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-paper-figure.jpg)
*FAST spends tokens on the shape of a trajectory rather than every value at every timestep. source: [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html)*

Low-frequency coefficients describe the broad motion, while higher frequencies capture abrupt corrections. The autoregressive policy therefore predicts the overall trajectory shape before its finer details. This ordering introduces a smoothness prior. Common low-frequency motion is represented compactly, while a rare high-frequency correction may require more tokens or be attenuated by compression.

FAST retains autoregressive learning, so compression reduces the number of decoder steps without eliminating sequential inference. This is the constraint that continuous action heads address next.

## Continuous action experts separated semantics from control

A separate branch retained the pretrained VLM for images and instructions while moving motor generation outside the language vocabulary. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) adds a continuous action expert trained with flow matching. The shared trunk provides semantic context, while the action expert produces continuous chunks at a bandwidth suited to control.

![Pi0 uses a pretrained vision-language trunk to condition a flow-based action expert](/assets/images/pi0-vision-language-action-flow-model-for-general-robot-control-paper-figure.jpeg)
*Pi0 uses a pretrained vision-language trunk to condition a separate continuous action expert trained with flow matching. source: [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html)*

Separating semantic processing from motor generation also allows the action path to change with the platform. [GR00T N1](/paper%20shorts/2025/03/18/groot-n1-open-foundation-model-for-humanoid-robots.html) uses a related fast-slow design for humanoids. [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) replaces the original autoregressive token head during adaptation. In its experiments, parallel continuous chunks trained with an L1 loss improve both inference speed and task success.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) uses both representations at different stages. FAST tokens allow web and robot tasks to share a discrete pretraining objective. A continuous expert added during post-training provides finer control and faster inference. This separates the representation used to scale heterogeneous training from the representation used to execute actions.

![Pi0.5 combines tokenized high-level outputs with a continuous low-level action expert](/assets/images/pi0-5-vision-language-action-model-with-open-world-generalization-paper-figure.png)
*Pi0.5 uses FAST tokens during mixture pretraining and adds a continuous action expert during post-training. source: [Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html)*

These papers separate two decisions that RT-2 combined: how robot and web examples share a training objective, and how the deployed model produces motor commands. Pi0.5 changes representation between stages; OpenVLA-OFT changes it during adaptation.

## Human video supplied time without robot actions

Robot demonstrations are expensive, while human video provides much broader coverage of objects, scenes, and interactions. Before receiving robot action labels, a video model can learn object persistence through occlusion, hand-object interaction, motion, and the temporal order of a task.

Passive video does not identify the robot command that caused an observed change. [Genie](/paper%20shorts/2024/02/23/genie-generative-interactive-environments.html) infers latent actions from unlabeled video and uses them to condition an interactive world model. The latent variables organize transitions in the video, but they are not directly executable on a robot.

One approach separates abundant video pretraining from scarce action-conditioned training. [V-JEPA 2](/paper%20shorts/2025/06/11/v-jepa-2-self-supervised-video-models.html) first learns to predict representations from video without action labels. A smaller second stage then connects robot commands with future latent states. Internet video supplies broad temporal structure, while robot data identifies which state changes are controllable.

A second approach collects human manipulation through an interface closer to robot operation. [Xiaomi-Robotics-1](/paper%20shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html) records UMI trajectories and labels the state change in each sequence. A later cross-embodiment stage aligns those behaviors with robot controls. The collection stage increases task and scene diversity, while the robot stage maps that behavior into executable actions.

![Xiaomi-Robotics-1 separates scalable human-operated capture from robot embodiment alignment](/assets/images/xiaomi-robotics-1-paper-figure-1.png)
*Xiaomi-Robotics-1 collects human manipulation through UMI, labels the observed state change, and later aligns those trajectories with robot commands. source: [Xiaomi-Robotics-1](/paper%20shorts/2026/07/16/xiaomi-robotics-1-scaling-vla-with-real-world-trajectories.html)*

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

Example counts therefore need target counts, compute, and temporal overlap to be interpretable. A batch of adjacent windows from one episode contains fewer independent decisions than the same number of diverse episodes.

Mixture weights also determine which objectives update the shared modules. Gradient norms and alignment can diagnose interference, but the relevant result is retained capability and adaptation on held-out robots. Sampling percentage alone does not measure either.

Pretraining scale needs the same accounting. Episode count, control steps, unique scenes, task diversity, embodiments, failures, robot hours, model size, and compute describe different axes. The relevant outcome is transfer, measured through zero-shot behavior and the rate of adaptation on a held-out robot.

## The changes in training target

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

The main progression is a separation of semantic transfer from motor generation. RT-2 shares the output vocabulary; FAST makes that vocabulary more efficient for trajectories; Pi0 and OpenVLA-OFT move control into continuous outputs. Pi0.5 uses both at different training stages. These are alternative allocations of learning and inference work, not successive proofs that one action representation dominates.

Video-based world models add a separate requirement: the predicted future must depend on the proposed action. V-JEPA 2 makes that connection in an action-conditioned stage after video pretraining. The remaining test is whether the resulting predictor improves closed-loop planning and reduces the robot data needed for transfer.
