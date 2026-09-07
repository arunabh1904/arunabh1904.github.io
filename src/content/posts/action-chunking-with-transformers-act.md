---
title: 'Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)'
date: '2023-04-23T00:00:00.000Z'
section: paper-shorts
postSlug: action-chunking-with-transformers-act
legacyPath: /paper shorts/2023/04/23/action-chunking-with-transformers-act.html
tags:
  - Robotics
  - Imitation Learning
field: 'Vision-Language-Action & Robotics'
summary: "2023 – Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)"
---

## 2023 – Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

**arXiv:** [2304.13705](https://arxiv.org/abs/2304.13705)

**Project:** [ALOHA](https://tonyzhaozh.github.io/aloha/)

## Summary

> ACT makes imitation learning behave like a short-horizon sequence model. It predicts a chunk of future joint targets from several camera views, queries that policy continuously, and temporally ensembles the overlapping predictions. A conditional VAE makes the chunk distribution flexible enough for stochastic human demonstrations.

## Core Insights

### The real problem is compounding error

A one-step behavior-cloning policy is trained on expert states but is deployed on its own states. A small error can move the robot into a state that never appeared in the demonstrations; the next prediction is then worse, and the error compounds. Fine manipulation makes this failure visible: the robot may need millimeter-level corrections while two arms coordinate around a transparent bag, a small cable loop, or a tight shoe.

ACT changes the unit of prediction from one action to a future sequence:

$$
\pi_\theta(a_{t:t+k}\mid s_t)
$$

If each chunk is executed before the next observation, $k=1$ gives ordinary one-step closed-loop control and an episode-length chunk gives open-loop control after the first observation. ACT later adds temporal ensembling, which queries the policy every step even when the predicted chunk is long. The useful regime is between those endpoints: the policy learns the local rhythm of a manipulation while still receiving new visual evidence.

![ACT CVAE architecture with multi-view inputs, a transformer action decoder, and a latent style variable](/assets/images/action-chunking-with-transformers-act-source-figure-4.png)
*Fig 1: The training encoder compresses joints and a demonstrated action sequence into a latent style variable; the deployment decoder combines that variable with four camera views and joints to predict a 14-dimensional action sequence. | source: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, Figure 4](https://arxiv.org/abs/2304.13705)*

The left side of the source figure exists only during training. A BERT-like encoder reads the current joints and the demonstrated future actions, uses a learned CLS representation to parameterize a diagonal Gaussian, and produces $z$. The policy on the right sees images, joints, and $z$; at test time the encoder is discarded and $z$ is set to the zero mean of its prior. This is a practical CVAE design: the latent helps the decoder learn the range of human solutions without requiring a latent sample at deployment.

The decoder processes four $480\times640$ RGB images with ResNet-18 encoders. Each view becomes $300\times512$ spatial features, giving 1,200 visual tokens after concatenation; projected joint state and $z$ add two more tokens. A transformer encoder fuses those 1,202 tokens, and a transformer decoder emits $k\times14$ targets for six arm joints and one gripper on each side. The paper reports about 80M parameters, roughly five hours of training on one RTX 2080 Ti, and about 0.01 seconds per inference on that machine. It also reports better precision with L1 reconstruction and absolute target joint positions than with delta joint positions.

### Temporal ensembling keeps the loop closed

ACT does not execute a chunk and wait $k$ steps before looking again. It queries the policy at every timestep. That creates several predictions for the same target timestep, so the system stores them in FIFO buffers and averages the overlapping predictions:

$$
A_t=\frac{\sum_i w_i\,\hat a_t^{(i)}}{\sum_i w_i},
\qquad
w_i=\exp(-m i).
$$

Here $i=0$ is the oldest prediction. A smaller $m$ lets newer observations take over faster. The average is taken over predictions of the same action time, rather than neighboring actions, which avoids the bias of ordinary temporal smoothing.

![ACT action chunks and temporal ensemble for a shared target timestep](/assets/images/action-chunking-with-transformers-act-source-figure-5.png)
*Fig 2: Overlapping chunks provide multiple estimates for the same future action; exponential weights combine those estimates so new observations can correct an older plan without an abrupt switch. | source: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, Figure 5](https://arxiv.org/abs/2304.13705)*

The figure's colored squares are not separate controllers. The first row shows a chunk predicted at one observation; the lower grid shows later chunks overlapping it. The outlined column is one target timestep, and the weighted sum turns several forecasts into the command actually sent to the low-level controller. This is why temporal ensembling can smooth motion while preserving feedback.

### Hardware and demonstrations set the operating point

ALOHA uses two off-the-shelf ViperX six-degree-of-freedom follower arms, two WidowX leader arms for joint-space teleoperation, custom see-through grippers, and four Logitech webcams: front, top, and one on each wrist. Images and teleoperation are recorded at 50 Hz. Joint-space mapping avoids inverse-kinematics failures near singularities and lets the difference between leader and follower positions implicitly carry the force applied by the PID controller.

![ALOHA bimanual workspace, camera viewpoints, gripper mechanism, and ViperX specifications](/assets/images/action-chunking-with-transformers-act-source-figure-1.webp)
*Fig 3: The low-cost ALOHA setup combines two ViperX follower arms, four camera viewpoints, and custom grippers so demonstrations expose the visual and coordinated motion needed by fine bimanual tasks. | source: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, Figure 3](https://arxiv.org/abs/2304.13705)*

The real-world dataset contains 50 demonstrations for each task, except Thread Velcro with 100. Episodes last 8–14 seconds, or 400–700 control steps, so each task has about 10–20 minutes of recorded motion but 30–60 minutes of wall-clock collection after resets and operator mistakes. The policy predicts the leader's joint positions as future targets while observing the follower's current positions. That choice preserves the force relationship that the follower's low-level controller will realize.

The eight-task evaluation contains two simulated tasks and six real tasks. In simulation, ACT is trained with either 50 scripted or 50 human demonstrations and evaluated over three seeds with 50 trials each. Real tasks use one seed and 25 evaluations. The real tasks are not interchangeable pick-and-place examples: opening a translucent condiment cup requires tipping it into the other gripper before prying the lid; Slot Battery requires one arm to hold a remote still while the other pushes against a spring; Thread Velcro requires a mid-air handover and insertion through a 3 mm by 25 mm loop.

### What the ablations actually isolate

The chunk-length experiment disables temporal ensembling and trains separate policies for each $k$. Averaged over the four simulated settings, success rises from about 1% at $k=1$ to 44% at $k=100$, then tapers for $k=200$ and $400$. Because this experiment executes chunks without temporal ensembling, longer chunks delay the next observation. The authors attribute the decline to both reduced reactivity and the difficulty of modeling longer action sequences; the experiment does not isolate those two costs.

Adding temporal ensembling separately tuned for each setting gives ACT a 3.3-point gain. That smaller effect is informative. Chunking changes the prediction target and reduces the effective horizon; ensembling mainly reconciles the errors left by overlapping predictions. The CVAE ablation is conditional on demonstration type: removing its objective barely changes scripted-data performance, but human-data success falls from 35.3% to 2%. Pauses and multiple valid handovers are part of the human distribution, so a single deterministic action sequence is a poor fit.

The final real-task numbers show both the promise and the boundary. ACT reaches 88% on Slide Ziploc and 96% on Slot Battery in Table I; Table II reports 84% on Open Cup, 20% on Thread Velcro, 64% on Prep Tape, and 92% on Put On Shoe. Thread Velcro falls from 92% after the lift stage to 40% after the grasp and 20% after insertion, with low contrast and small projected area making the failure hard to correct. The abstract's 80–90% headline refers to opening the translucent cup and slotting a battery from roughly ten minutes of demonstrations; the detailed table makes clear that performance varies sharply with the physical bottleneck.

### Where the method stops

ACT still imitates the coverage and limitations of its demonstrations. Even after 50 candy-unwrapping demonstrations, the system picks up candy in all ten initial trials, pulls both wrapper ends in eight, and unwraps none. The authors point to the hard-to-see wrapper seam and limited data. Allowing ten attempts on each of five candies changes the outcome: three of five are unwrapped. Recovery opportunities therefore change the measured result, while the one-attempt failure still exposes a perception bottleneck. ALOHA also has no force-torque sensor and parallel grippers, which limits tasks involving high force, multiple fingers, or fine fingertip contact.

The decision lesson is specific: choose a chunk horizon from the disturbance timescale, then measure recovery after a state leaves the demonstration distribution. Temporal averaging cannot supply missing sensing, and a CVAE cannot turn inconsistent demonstrations into a physical model. ACT earns its gains by making short sequences and dense feedback work together.

## High-Level Takeaways

- Predicting a short action sequence reduces the imitation-learning horizon; querying every step and ensembling overlapping forecasts preserves visual feedback.
- The CVAE is valuable when people demonstrate several valid, noisy ways to complete a task. It is almost irrelevant on deterministic scripted trajectories.
- In the tested suite, $k=100$ is a useful middle point, while very long chunks lose reactivity.
- ALOHA's inexpensive joint-space teleoperation is part of the method: it records the coordinated, high-frequency targets that ACT later models.
- Fine manipulation remains limited by sensing, contact coverage, and demonstration support.
