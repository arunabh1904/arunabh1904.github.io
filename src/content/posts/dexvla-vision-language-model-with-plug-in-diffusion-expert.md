---
title: 'DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control'
date: '2025-02-09T00:00:00.000Z'
section: paper-shorts
postSlug: dexvla-vision-language-model-with-plug-in-diffusion-expert
legacyPath: /paper shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html
tags:
  - Other
field: 'Vision-Language-Action & Robotics'
summary: "2025 – DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control"
---

## 2025 – DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control

**arXiv:** [2502.05855](https://arxiv.org/abs/2502.05855)

**Project:** [dex-vla.github.io](https://dex-vla.github.io/)

## Summary

> DexVLA treats the action expert as a first-class scaling target. A Qwen2-VL backbone produces reasoning and action tokens; a billion-parameter, multi-head diffusion expert turns those representations into continuous motor commands. A three-stage curriculum then moves from cross-embodiment motor skills to embodiment alignment and task-specific dexterity.

## Why add a large action expert?

The paper separates semantic grounding from motor generation. Qwen2-VL encodes images and instructions, then emits two streams. A two-linear-layer projection with LayerNorm maps action tokens into the diffusion expert's space. Reasoning tokens are injected through FiLM layers that scale and shift the expert's projection features, so the model's generated substep description can condition action generation rather than merely decorate it.

The action module is Scale Diffusion Policy, a transformer-based diffusion policy scaled to 1B parameters. To train on different robot morphologies, the expert has multiple output heads, with one head per robot configuration. The overall objective is

$$\mathcal{L}=\mathcal{L}_{\mathrm{diff}}+\alpha\mathcal{L}_{\mathrm{ntp}}, \qquad \alpha=1.$$

The diffusion loss teaches continuous action denoising; next-token prediction teaches the VLM to generate language and intermediate reasoning. This is a different capacity allocation from simply making the language backbone larger: the paper asks whether action-specific parameters can absorb cross-embodiment motor variation.

## The embodied curriculum

![DexVLA connects Qwen2-VL reasoning and action tokens to a multi-head billion-parameter diffusion expert](/assets/images/dexvla-vision-language-model-with-plug-in-diffusion-expert-paper-figure.png)
*Fig 1: Stage 1 trains the diffusion expert alone on cross-embodiment data; Stages 2 and 3 connect it to the VLM for embodiment alignment and task adaptation, while separate output heads handle robot configurations. | source: [DexVLA, Figure 2](https://arxiv.org/abs/2502.05855)*

Figure 1 shows the critical asymmetry. Stage 1 discards the VLM during diffusion-expert pretraining, so the large expert learns motor patterns without competing with language alignment. Stage 2 brings back the VLM, projection, and expert on embodiment-specific data while freezing the VLM visual encoder. Stage 3 uses task-specific demonstrations for difficult skills. The authors train on about 100 hours of demonstrations and report 60 Hz inference on one Nvidia A6000, but the three stages still require distinct data and optimization runs.

Substep reasoning is the bridge to long horizons. Instead of using a single instruction such as “fold the shirt” for an entire trajectory, demonstrations are annotated with intermediate descriptions such as “smooth wrinkles,” “align sleeves,” and “secure folds,” typically about every five seconds. These descriptions are trained as intermediate language outputs, so the model learns to generate a substep and use it to guide the diffusion expert. It is an implicit high-level policy inside the VLA, rather than an external SayCan call.

## Evidence without task-specific adaptation

![DexVLA generates implicit substeps for laundry folding, dryer unloading, and sorting from direct prompts](/assets/images/dexvla-vision-language-model-with-plug-in-diffusion-expert-source-figure-2.webp)
*Fig 2: Direct prompting turns one long instruction into a sequence of generated substeps for laundry, dryer, and sorting tasks; success requires both dexterous control and state-dependent decomposition. | source: [DexVLA, Figure 3](https://arxiv.org/abs/2502.05855)*

The no-Stage-3 test uses one set of parameters across bin picking, shirt folding, and easy table bussing. Every method is fine-tuned on the same data and for the same number of epochs. Across ten trials per task, DexVLA scores 0.92 on shirt folding, while Diffusion Policy, Octo, and OpenVLA complete essentially none of the shirt-folding steps. This is a normalized task score, not a binary success rate, so the number means substantial partial completion.

On two embodiments absent from Stage 1 and Stage 2—Franka with a dexterous hand for drink pouring and bimanual UR5e for packing—the model is fine-tuned with 100 demonstrations per novel task. Averaged over ten trials, DexVLA scores 0.90 across the two tasks and outperforms the same OpenVLA, Octo, and Diffusion Policy comparisons. The control is meaningful because the baselines receive the same per-task training epochs, but the tasks remain carefully selected and the scoring rubrics are supplied in the appendix.

![DexVLA evaluates single-arm, bimanual, mobile, and dexterous-hand embodiments](/assets/images/dexvla-vision-language-model-with-plug-in-diffusion-expert-source-figure-4.webp)
*Fig 3: The experiments span bimanual UR5e, Franka, mobile AgileX, and Franka with a dexterous hand, exposing the action expert to different kinematics and camera configurations. | source: [DexVLA, Figure 4](https://arxiv.org/abs/2502.05855)*

On the LIBERO simulation benchmark, Table 5 reports 97.2% on Spatial, 99.1% on Object, 95.6% on Goal, and 97.3% averaged for DexVLA. The comparisons include Diffusion Policy at 79.7 average, OpenVLA at 84.1, π0-FAST at 93.9, and π0 at 97.1. These are useful as a controlled VLA comparison, but the simulated tasks do not settle visual or contact robustness in a new home.

## Ablations expose the dependency chain

The three-stage ablation in Table 2 is unusually direct. Training only Stage 1 or only Stage 2 yields 0.0 on shirt folding and laundry folding. Stages 1+2 without task-specific Stage 3 yield 0.92 on shirt folding but 0 on laundry folding; all three stages yield 0.92 and 0.4. The result says that the expert warm-up is necessary for learning meaningful actions and that long-horizon task mastery still needs specialized data. The paper presents the optimization explanation—that the large expert is difficult to train from scratch—as a hypothesis rather than a separately isolated causal proof.

Action-expert capacity also matters. On shirt folding, a 93M UNet obtains 0.17, a 410M diffusion expert 0.63, and the 1B expert 0.92 (Table 3). The authors observe oscillatory movement with the UNet and attribute the gap to interference between actions in a smaller parameter space; the ranking is evidence for capacity, while the interference explanation remains their interpretation.

The substep ablation gives the most revealing control. Training the diffusion expert with only direct task prompts reduces shirt-folding score from 0.92 to 0.07; removing substep reasoning from both Stage 1 and Stage 2 gives 0.0 (Table 7). Replacing the implicit substeps with SayCan scores 0.58 on hard table bussing versus 0.70 for implicit substep reasoning (Table 8). The advantage may reflect adaptive segmentation: SayCan updates at a fixed two-second interval, while DexVLA's language outputs can change with the task state.

Finally, zero-shot cross-embodiment transfer swaps the trained Robotiq gripper for an Inspire five-finger hand and constrains the hand to one degree of freedom to match the gripper. On 30 novel bin-picking objects, success is 60%, compared with 67% for the original gripper. The gap comes with changed hand appearance, wrist-camera viewpoint, and grasp height; it is encouraging representation transfer, not full dexterous control.

## High-Level Takeaways

- DexVLA gives the action model its own billion-parameter capacity and uses multi-head outputs for cross-embodiment data.
- The curriculum matters: expert pretraining warms up motor skills, embodiment alignment grounds them, and task data handles difficult long horizons.
- Substep reasoning is the key bridge from direct language to long-horizon action, with a 0.92-to-0.07 shirt-folding ablation when it is removed from expert training.
- DexVLA reaches 97.3% average on LIBERO and 60% zero-shot transfer on 30 novel objects under a one-degree-of-freedom hand constraint.
- The gain comes with expert size, staged data requirements, and training cost; matched action-head capacity and real-world recovery remain open tests.
