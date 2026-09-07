---
title: 'π0.7: A Steerable Generalist Robotic Foundation Model'
date: '2026-04-16T00:00:00.000Z'
section: paper-shorts
postSlug: pi0-7-steerable-generalist-robotic-foundation-model
legacyPath: /paper shorts/2026/04/16/pi0-7-steerable-generalist-robotic-foundation-model.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2026 – π0.7: A Steerable Generalist Robotic Foundation Model'
---
## 2026 – π0.7: A Steerable Generalist Robotic Foundation Model

**arXiv:** [2604.15483](https://arxiv.org/abs/2604.15483)

## Summary

> π0.7 treats context as part of the control interface. The VLA receives a task instruction plus an optional subtask, multi-view subgoal images, episode metadata, and a control-mode tag. Training with these signals lets one 5B policy use mixed-quality robot data, human video, and autonomous experience while remaining steerable at test time. The paper evaluates out-of-the-box dexterity, language following, cross-embodiment transfer, coaching, and data-diversity ablations.

## Core Insights

### “What to do” is only one part of a robot prompt

A task instruction such as “fold the shirt” leaves out the strategy, the desired intermediate state, and whether a demonstration was fast or error-prone. π0.7 makes those variables explicit. Its context can contain the overall task, a semantic subtask, one image per camera showing a desired near-future state, episode metadata, and a control-mode token selecting joint or end-effector actions.

The model is about 5B parameters: a Gemma 3 4B VLM backbone with the MEM-style history encoder and an 860M flow-matching action expert. The vision encoder compresses up to six history frames per camera to a fixed token budget; the action expert processes 50 tokens for a 50-step action chunk. Training also simulates 0–12 timesteps of inference delay, up to 240 ms at 50 Hz, so the policy learns to produce smoother actions under asynchronous execution.

![Figure 2 from π0.7 showing the VLA, high-level policy, world model, and prompt context](/assets/images/pi0-7-steerable-generalist-robotic-foundation-model-source-figure-2.webp)
*Fig 1: π0.7 combines history-aware visual control with subtask language, generated subgoal images, episode metadata, and optional high-level policy or world-model context. | source: [π0.7: A Steerable Generalist Robotic Foundation Model, Figure 2](https://arxiv.org/abs/2604.15483)*

Read Fig. 1 as a deployment graph rather than a single transformer block. The high-level policy or a human supplies the next semantic subtask. A lightweight world model, initialized from BAGEL-14B, turns that subtask and the current observation into visual goals. The main VLA then conditions its action expert on the latest context. Those auxiliary models run asynchronously, so they are part of the system’s runtime budget even though they are not inside the low-level action expert.

### Subgoal images make the desired state visible

A language subtask says “open the fridge,” but it does not say which grasp or final arm configuration will make progress. π0.7 uses multi-view subgoal images to show a plausible near-future scene. The images can specify both the environment-level outcome in a base view and the gripper-level outcome in a wrist view. During training, 25% of examples receive subgoal images. Among those examples, the subtask text is dropped 30% of the time, letting the image stand in for a richer description.

The model can also read episode metadata: speed is a binned episode length between 1,750 and 2,250 steps, quality is a human score from 1 to 5, and mistake marks an error in the segment. Metadata is dropped entirely on 15% of examples, and each field is dropped independently with 5% probability. Control mode is not dropped. At runtime, the reported recipe sets quality to 5, mistake to false, and speed near the task’s 15th-percentile episode length. Subgoals refresh when the semantic subtask changes or after four seconds, whichever comes first. This turns “be fast,” “avoid mistakes,” and “reach this intermediate state” into conditions the policy can be trained to interpret rather than informal operator wishes.

The prompt dropout matters because the same policy must work with subsets of context. It can run with language only, language plus metadata, or generated goals and metadata. Classifier-free guidance can further emphasize a prompt component, such as speed, at inference. The flexibility is earned by deliberately training the model to survive missing context; it is not a consequence of simply appending more tokens.

### Results separate dexterity from composition

For out-of-the-box dexterity, π0.7 is compared with task-specific π*0.6 specialists on laundry, espresso, and box building, plus specialist baselines on additional tasks. The paper reports success and normalized throughput for the former tasks and task progress for tasks such as making a peanut-butter sandwich, turning a shirt inside out, slicing zucchini, peeling vegetables, and taking out a trash bag. The same generalist can match the specialists and exceeds them in normalized throughput on the difficult diverse-laundry and box-building tasks.

The architecture’s cross-embodiment claim is concrete. The model sees no shirt-folding data on the target bimanual UR5e; most folding data comes from a smaller static bimanual robot. Human teleoperators with no prior UR5e folding experience average 90.9% task progress and 80.6% success on their first attempts. π0.7 reaches 85.6% progress and 80.0% success. It also changes strategy: the source robot encourages a tilted grasp, while the longer UR5e receives a more vertical grasp that better fits its geometry. Generated subgoal images improve this transfer by offering a visual analogy for the target robot’s reachable state.

![Figure 5 from π0.7 showing selected long-horizon evaluation tasks](/assets/images/pi0-7-steerable-generalist-robotic-foundation-model-source-figure-5.webp)
*Fig 2: Selected evaluations contrast a coarse “take out the trash” instruction with step-by-step coaching for an unseen toaster task. | source: [π0.7: A Steerable Generalist Robotic Foundation Model, Figure 5](https://arxiv.org/abs/2604.15483)*

Fig. 2 captures the paper’s distinction between direct composition and coached composition. π0.7 follows 3–6 open-ended instructions in 14 scenarios across four unseen kitchens and two unseen bedrooms, and handles unusual referential instructions better than π0.5 and π0.6. On the reverse-fridge task, generated subgoal images are critical because the world model converts an unusual language request into a visual target that breaks the dataset’s usual direction of motion.

Short-horizon tasks can be prompted directly even without task-specific robot data: pressing a French-press plunger, scooping rice, wiping office objects, and spinning articulated items are examples. Longer tasks such as loading or unloading an air fryer and toasting a bagel need step-by-step coaching. The coaching episodes contain language and low-level action data from the policy rollout, but no action-level training episodes for those particular tasks. A high-level policy learns to issue the subtask sequence, producing an autonomous version that approaches the coached performance without new low-level teleoperation.

### Metadata is what lets diversity help

The controlled data study trains models on the top 30%, 50%, 80%, or all laundry data by quality and speed. With metadata, performance keeps improving as the dataset grows even though average quality falls. Without metadata, adding lower-quality data can degrade performance. A separate matched experiment removes either the most task-diverse 20% or a random 20%; removing the diverse slice hurts unseen short-horizon tasks more. The message is precise: context annotations let the policy distinguish useful behavioral modes, while task diversity supplies combinations that can be remixed later.

The evidence has a natural limit. The training mixture is broad enough that a nominally “unseen” object or task may appear under another label or as part of another behavior. The paper itself frames compositional generalization as difficult to isolate in a corpus of this scale. The high-level policy, world model, metadata quality, and asynchronous refresh schedule also add dependencies to a deployment that a low-level VLA benchmark alone would not reveal.

## High-Level Takeaways

- π0.7 turns strategy, episode quality, and desired intermediate state into explicit control context.
- Prompt dropout makes the policy usable with different context subsets, but the auxiliary high-level policy and world model remain part of the runtime system.
- Cross-embodiment shirt folding shows adaptation of grasp strategy, not just copying source-robot motions.
- Mixed-quality data helps when metadata disambiguates behavior; broad training mixtures also make strict “unseen” claims difficult to establish.
