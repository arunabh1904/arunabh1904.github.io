---
title: 'GR00T N1: An Open Foundation Model for Generalist Humanoid Robots'
date: '2025-03-18T00:00:00.000Z'
section: paper-shorts
postSlug: groot-n1-open-foundation-model-for-humanoid-robots
legacyPath: /paper shorts/2025/03/18/groot-n1-open-foundation-model-for-humanoid-robots.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2025 – GR00T N1: An Open Foundation Model for Generalist Humanoid Robots'
---
## 2025 – GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

**arXiv:** [2503.14734](https://arxiv.org/abs/2503.14734)

## Summary

> GR00T N1 combines a vision-language reasoning module with a diffusion-transformer action module and trains the same policy across real robot data, human videos, simulation, and neural-generated trajectories. The released GR00T-N1-2B uses a 2.2B-parameter model to produce 16-action chunks, then adapts to individual embodiments through post-training. Its strongest evidence is data efficiency on simulated and GR-1 humanoid tasks, alongside a revealing example where narrow post-training erases a pretrained handover behavior.

## Core Insights

### Two rates, one action interface

GR00T N1 is built around a division of labor. System 2 is the Eagle-2 vision-language model: it reads images and the instruction, interprets the task, and runs at 10 Hz on an NVIDIA L40. System 1 is a diffusion transformer trained with flow matching: it cross-attends to the VLM tokens and generates continuous motor commands at a higher rate. The model couples the two modules during training instead of asking a language model to serialize low-level motor commands.

The public N1-2B checkpoint has 2.2B parameters, including 1.34B in the VLM. It predicts a chunk of 16 actions in 63.9 ms on an L40 in bf16; the paper describes the action module as supporting closed-loop motor generation at 120 Hz. State and action encoders are embodiment-specific MLPs that project different robot dimensions into a shared DiT space, while the final action decoder maps back to each robot’s command space. The action flow-matching loss starts from a noisy action chunk and learns the denoising vector field. Four Euler denoising steps are used at inference.

The backbone does not need to expose a final-layer semantic summary for control. For N1-2B, the authors use the Eagle-2 LLM’s 12th-layer representations, reporting both faster inference and higher downstream success than final-layer features. That detail explains why the architecture can keep semantic context while giving the action module a representation that remains useful for motor prediction.

![Figure 1 from GR00T N1 showing the data pyramid for robot foundation model training](/assets/images/groot-n1-open-foundation-model-for-humanoid-robots-source-figure-1.webp)
*Fig 1: The data pyramid places web and human videos at the broad base, synthetic and neural trajectories in the middle, and scarce real-robot data at the embodiment-specific top. | source: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, Figure 1](https://arxiv.org/abs/2503.14734)*

Fig. 1 is a data allocation picture, not a quality ranking. Human video supplies visual and behavioral breadth without robot actions. Simulation supplies controllable state-action trajectories. Neural video generation expands the number of counterfactual robot-like sequences. Real robot data grounds the policy in the target hardware. The upper layers are more faithful to the deployment embodiment, but much smaller; the point of the recipe is to make the lower layers useful without letting them replace that grounding.

### Give action-less video a common label space

Human videos and neural trajectories do not come with motor commands. GR00T N1 trains a VQ-VAE on consecutive frames and uses the continuous pre-quantized embedding as a latent action label. For robot data it can use ground-truth actions and latent actions; for neural trajectories it combines latent labels with inverse-dynamics-model predictions. These labels are treated as distinct embodiments within one flow-matching interface. A latent action can therefore make “move the right arm left” comparable across a human video, a simulated robot, and a real robot without pretending their torques are identical.

The scale comes from several different operations. The pretraining table totals 592.9 million frames and 8,375.7 hours: 262.3M real-robot frames / 3,288.8 hours, 181.3M human frames / 2,517.0 hours, 125.5M simulation frames / 1,742.6 hours, and 23.8M neural-video frames / 827.3 hours. Video generation expands 88 hours of in-house teleoperation data to about 827 hours, roughly 10×, using new language prompts and counterfactual object placements. A multimodal judge filters clips that do not follow the instruction, then the retained videos receive captions. DexMimicGen expands a small set of demonstrations into 780,000 simulation trajectories, equivalent to 6,500 hours, in 11 hours of generation.

The training cost is correspondingly substantial: GR00T-N1-2B uses roughly 50,000 H100 GPU-hours for pretraining, and the neural-video generation experiment reports about 105,000 L40 GPU-hours. During post-training, the language component of the VLM is frozen while the rest of the model adapts to a single embodiment. For neural-trajectory post-training, the authors co-train real and generated trajectories at a 1:1 sampling ratio and use inverse-dynamics labels for the GR-1 real-robot setting.

### What the benchmark controls actually show

The simulation evaluation uses 24 RoboCasa tasks, 9 DexMimicGen cross-embodiment tasks, and 24 GR-1 tabletop tasks. At 100 demonstrations per task, GR00T-N1-2B reaches 32.1% on RoboCasa, 66.5% on DexMimicGen, and 50.0% on GR-1, averaging 45.0%. Diffusion Policy reaches 25.6%, 56.1%, and 32.7%, for a 33.4% average. The GR-1 gap is 17.3 points, which is the clearest evidence that broad pretraining helps when the target embodiment and task family are demanding. The paper evaluates 100 trials per simulation task and reports the best of the last five checkpoints, so these are not single-rollout anecdotes.

![Figure 7 from GR00T N1 showing the simulation task families](/assets/images/groot-n1-open-foundation-model-for-humanoid-robots-source-figure-7.webp)
*Fig 2: The simulation suite spans RoboCasa kitchen tasks, DexMimicGen cross-embodiment manipulation, and GR-1 tabletop tasks, linking the benchmark to the model’s heterogeneous training data. | source: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, Figure 7](https://arxiv.org/abs/2503.14734)*

Read Fig. 2 as a coverage check. The rows do not show one homogeneous manipulation problem: they move from kitchen semantics to bimanual dexterity and humanoid tabletop control. That makes the average useful for measuring breadth, but it also means the average can conceal which embodiment is doing the work.

On the real GR-1, the full post-training set gives 76.8% average success across pick-and-place, articulated, industrial, and coordination tasks, versus 46.4% for Diffusion Policy trained on all data. With only 10% of demonstrations, GR00T reaches 42.6%, close to the full-data diffusion baseline at a 3.8-point gap. The pretrained checkpoint also reaches 76.6% on a coordinated left-to-right handover and 73.3% when placing novel objects into unseen containers. Neural-trajectory co-training adds 4.2, 8.8, and 6.8 points to RoboCasa averages at 30, 100, and 300 demonstrations, and 5.8 points on average across eight GR-1 tasks.

The most instructive failure is qualitative. The pretrained model can use its left hand to grasp an apple placed outside the right hand’s reach, hand it over, and place it in a basket. Post-training data uses only right-hand behavior, and the adapted checkpoint loses that handover capability. The result shows both the value of broad pretraining and the risk of embodiment-specific specialization: more target data can improve the measured task while narrowing the behavior prior.

The paper’s stated boundary is short-horizon tabletop manipulation. Neural generation still struggles with physical validity and diverse counterfactuals, and simulation does not settle long-horizon loco-manipulation. GR00T N1 makes the data pyramid testable through an open checkpoint and benchmarks, but it does not make latent, synthetic, and real actions interchangeable in contact-rich settings.

## High-Level Takeaways

- The VLM/DiT split preserves semantic interpretation while giving continuous actions their own high-rate generator.
- Latent actions, inverse-dynamics labels, simulation, and real trajectories let one model consume heterogeneous data, with embodiment-specific post-training providing the final grounding.
- The 10%-data real-robot result and neural-trajectory ablations support data efficiency; the lost handover behavior shows that narrow post-training can erase general skills.
- The evidence is strongest for short-horizon tabletop manipulation, where synthetic physics and generated videos remain useful but imperfect substitutes for real dynamics.
