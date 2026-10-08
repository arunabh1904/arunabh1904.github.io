---
title: 'ViRA: Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?'
date: '2026-10-07T00:00:00.000Z'
section: paper-shorts
postSlug: vira-do-better-visual-representations-always-lead-to-better-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/10/07/vira-do-better-visual-representations-always-lead-to-better-end-to-end-autonomous-driving.html
tags: [Autonomous Driving, Representation Learning]
field: 'Autonomous Driving: VLA & Planning'
summary: '2026 – Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?'
---

## 2026 – ViRA: Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?

**Paper:** [arXiv:2610.09695](https://arxiv.org/abs/2610.09695) · [Full text, v1](https://arxiv.org/html/2610.09695v1) · [Official repository](https://github.com/OpenDriveLab/ViRA)

## Summary

> ViRA transfers a frozen vision foundation model's representations into a camera-only driving planner during training. The teacher and alignment heads disappear at inference. With a DVGT teacher, NAVSIM v2 EPDMS rises from 83.6 to 89.0 for TransFuser, 84.5 to 91.9 for DiffusionDrive, and 72.8 to 83.7 for the authors' Rap reimplementation. The less obvious result is that auxiliary perception supervision changes which teacher works best. A teacher's value depends on what the planner already learns, and aggregate gains can coexist with worse comfort or difficult-scenario performance.

## Core Insights

### Transfer the representation without deploying the teacher

Replacing a small driving encoder with a large vision foundation model increases inference cost. ViRA instead uses the foundation model to supervise the existing encoder. Both models receive the scene images during training. The planner predicts a trajectory through its original head, while two alignment losses teach its intermediate visual features to preserve relationships found in the frozen teacher.

These are non-language driving planners. ViRA does not add a language model, construct a prompt, or project features into an LLM token stream. Its projection heads exist to compare two visual representations in a shared latent space. The teacher's features are supervision targets; they are not additional inputs to the deployed planning head.

Where does the teacher enter this computation? Source Figure S1 separates the ordinary planning path from the two training branches. The spatial branch compares dependencies between feature locations. The scene branch compares representations across examples. Removing these branches leaves the original planner architecture.

![ViRA source Figure S1: a frozen vision foundation model supervises spatial dependency and scene semantic alignment during planner training](/assets/images/vira-source-figure-s1.jpg)
*Fig 1: Spatial and scene-level losses transfer a frozen teacher's representation into the planner encoder. The teacher and alignment heads are removed at inference. | source: [ViRA, Figure S1](https://arxiv.org/abs/2610.09695)*

Figure reproduced without modification under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Open the full-size figure](/assets/images/vira-source-figure-s1.jpg).

### Align relationships at two scales

Spatial Dependency Alignment, or SDA, starts with intermediate feature maps. The student map is resized to the teacher's spatial resolution, and $1\times1$ convolutional heads project their channels into a shared space. For the default DVGT teacher, learnable weights combine four aggregator levels. The method does not require the raw student and teacher channels to have the same meaning.

SDA compares attention-like distributions over locations. Student queries and teacher queries each use the same teacher keys:

$$
A_s=\operatorname{softmax}(Q_sK_t^\top/\gamma_{\mathrm{sda}}),
\qquad
A_t=\operatorname{softmax}(Q_tK_t^\top/\gamma_{\mathrm{sda}}).
$$

The KL-divergence loss asks the student distribution to match the teacher distribution, with temperature $\gamma_{\mathrm{sda}}=0.07$. Sharing teacher keys gives both distributions the same reference locations. The supervision therefore concerns which positions relate to which other positions, rather than forcing each raw feature channel to copy a teacher channel.

Scene Semantic Alignment, or SSA, adds a broader constraint. It aggregates the projected feature maps into scene representations, then applies a bidirectional contrastive loss over the batch. The student and teacher representations of the same scene form the positive pair; other scenes in the batch supply negatives. Its temperature is 1.0. This branch teaches scene identity and global discrimination that local dependencies alone may not preserve.

The complete objective is the native planner loss plus the two alignment terms:

$$
\mathcal L=\mathcal L_{\mathrm{planner}}+1.0\mathcal L_{\mathrm{SDA}}+0.1\mathcal L_{\mathrm{SSA}}.
$$

The encoder, planner, and alignment projections train; the foundation model remains frozen. Existing detection and segmentation losses remain part of the planner objective except in explicit ablations that remove them. Table S2 tests the value of the combined supervision: TransFuser reaches 84.3 EPDMS with KL alignment alone and 84.6 with contrastive alignment alone, versus 89.0 with both. Merely attaching an arbitrary feature loss does not reproduce the result.

### Keep inputs and optimization comparable

The experiments adapt TransFuser, DiffusionDrive, and Rap to camera-only planning. TransFuser and DiffusionDrive use three cameras, a reported image resolution of $2048\times512$, and a four-second trajectory sampled at 2 Hz. Rap uses four $768\times448$ images and a five-second trajectory at 2 Hz. This is waypoint spacing, not a claim that the deployed controller runs at two updates per second.

All three use ResNet-34 visual backbones in the main comparison. Rap is marked Rap* because the authors replace its original backbone and omit its original augmentation. Its absolute score must therefore be read as a reimplementation result. The matched before-and-after comparisons are more informative than treating it as the official Rap configuration.

Training uses NAVSIM's navtrain logs and eight H100 GPUs. TransFuser trains for 100 epochs with batch size 192, Adam, and a constant $10^{-4}$ learning rate. DiffusionDrive uses the same epochs and batch size, with AdamW and a cosine schedule starting at $6\times10^{-4}$. Rap* trains for 30 epochs with batch size 48, Adam, and a cosine schedule starting at $10^{-4}$.

This preserves the distinction from [DiffusionDrive's original decoder contribution](/paper%20shorts/2024/11/22/diffusiondrive-truncated-diffusion-model-for-end-to-end-autonomous-driving.html). ViRA changes visual supervision while keeping the planner's decoding mechanism. The original DiffusionDrive note also uses a different NAVSIM protocol, so its PDMS values should not be compared directly with these camera-only NAVSIM v2 results.

### Auxiliary tasks change the teacher comparison

The paper tests DVGT, VGGT, Depth Anything 3, DINOv3, and SAM3 as representation targets. Their value depends on the student's existing supervision. With detection and segmentation losses, TransFuser's aligned scores occupy a narrow 88.8–89.3 EPDMS range. Without those auxiliary tasks, the range widens to 86.4–89.1. The best target remains effective, while weaker targets lose more ground.

Source Figure 6 connects the planning gains to BEV perception quality. Its gain bars use different baselines: 83.6 EPDMS with auxiliary supervision and 82.6 without it. A larger improvement over the lower baseline does not automatically mean a higher final score.

![ViRA source Figure 6: BEV segmentation quality and EPDMS gains across teacher choices, with and without auxiliary perception supervision](/assets/images/vira-source-figure-6.jpg)
*Fig 2: Teacher comparisons change when detection and segmentation supervision is removed. Planning gains use separate baselines of 83.6 with auxiliary tasks and 82.6 without them. | source: [ViRA, Figure 6](https://arxiv.org/abs/2610.09695)*

Figure reproduced without modification under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Open the full-size figure](/assets/images/vira-source-figure-6.jpg).

The result narrows the title's question. “Better representation” cannot be ranked independently of the planner's losses. A teacher can supply information that auxiliary tasks already teach, or supply information they miss. A randomly initialized teacher lowers TransFuser from 83.6 to 82.0 EPDMS and reduces BEV mIoU from 36.4 to 30.7. This control supports transfer of learned visual information over a generic benefit from adding regularization.

The final ViRA-Diffusion variant uses a DINOv3 teacher without auxiliary perception tasks. It reaches 92.3 EPDMS on NAVSIM v2 navtest, compared with 90.4 for Discrete-WAM in the paper's comparison. That result is specific to the benchmark and configuration; it does not establish a universal teacher ranking or real-world driving performance.

### Gains survive several protocols, but not every component improves

NAVSIM v2 navtest evaluates 12,146 real-world scenarios with non-reactive simulation. Its EPDMS combines multiplicative penalties for failures such as collision and leaving the drivable area with weighted progress, time-to-collision, lane, and comfort terms. Its human-reference filtering also affects scoring. EPDMS is therefore a composite benchmark score, not a measured accident probability.

The navhard protocol differs. Its appendix describes 244 difficult real-world initial scenarios and 4,164 synthetic follow-up scenarios, generated with reconstructed scenes and reactive traffic. HUGSIM adds photorealistic closed-loop evaluation in Gaussian-reconstructed scenes. The models transfer from NAVSIM training to HUGSIM without fine-tuning; the reported HDScore combines route completion with safety and comfort penalties.

The default DVGT alignment improves the aggregate results across these settings:

| Planner | navtest EPDMS | navhard EPDMS | HUGSIM HDScore |
| --- | --- | --- | --- |
| TransFuser | 83.6 → 89.0 | 27.6 → 31.7 | 22.1 → 24.8 |
| DiffusionDrive | 84.5 → 91.9 | 30.5 → 35.8 | 22.2 → 28.6 |
| Rap* | 72.8 → 83.7 | 32.8 → 49.2 | 7.3 → 10.3 |

The component results qualify that pattern. Rap*'s navtest extended-comfort score falls from 56.7 to 48.1 while its aggregate EPDMS rises. TransFuser's HUGSIM extreme-scenario score falls from 10.9 to 9.9 despite its higher overall HDScore. Representation transfer improves the tested averages without guaranteeing improvement in every behavior or difficulty level.

The inference-cost claim is more direct. Rap* retains 69 million parameters and a reported 20 ms latency on one H100 with ViRA, because its training teacher is removed. Replacing its backbone with DINOv3 instead uses 351 million parameters and 65 ms, for a similar navtest score in that comparison. Teacher computation still costs training resources; the paper does not provide a complete wall-clock account that makes that extra cost free.

As checked on 8 October 2026, the official repository provides results but lists code and checkpoint release as pending. The published recipe is detailed enough to evaluate the design, while exact reproduction still awaits those artifacts.

## High-Level Takeaways

- ViRA distils spatial dependencies and scene identity into an existing driving encoder. Its teacher and projection heads are training components, so they add no deployed model branch.
- KL alignment and contrastive alignment work together: the TransFuser ablation reaches 89.0 EPDMS jointly, versus 84.3 and 84.6 separately. The loss combination is part of the method.
- Auxiliary perception losses change the value of a teacher. Comparing teachers without holding those losses fixed can confuse complementary supervision with a better representation in general.
- Higher aggregate planning scores can conceal worse comfort or extreme-scenario behavior. Those components remain necessary checks before choosing a teacher for deployment.
- My synthesis: compare teacher choices under the same total training budget, then report both aggregate and difficult-scenario outcomes. An unchanged inference graph is attractive, but an expensive teacher needs to justify its training cost and transferred failure modes.
