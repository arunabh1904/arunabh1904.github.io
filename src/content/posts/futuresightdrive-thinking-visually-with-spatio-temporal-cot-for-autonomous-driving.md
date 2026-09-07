---
title: 'FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving'
date: '2025-05-23T09:55:32.000Z'
section: paper-shorts
postSlug: futuresightdrive-thinking-visually-with-spatio-temporal-cot-for-autonomous-driving
legacyPath: /paper shorts/2025/05/23/futuresightdrive-thinking-visually-with-spatio-temporal-cot-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving"
---
## 2025 – FutureSightDrive (FSDrive)

**arXiv:** [2505.17685](https://arxiv.org/abs/2505.17685)

**Code:** [MIV-XJTU/FSDrive](https://github.com/MIV-XJTU/FSDrive)

## Summary

> FutureSightDrive makes its chain of thought a predicted visual scene rather than a text trace. A world-model path generates a future frame with background, future lane dividers, and 3D boxes; an inverse-dynamics VLA then plans a trajectory from the current observation and that visual spatio-temporal CoT. The paper reports improved trajectory accuracy and fewer collisions on nuScenes and NAVSIM, plus competitive video-generation FID and DriveLM understanding results. The paper does not report planning latency or a matched control that replaces the imagined frame with an equally informative nonvisual state.

## Core Insights

The representation choice is deliberate. Textual reasoning can discard geometry and temporal relations before planning, whereas a predicted future scene can carry lanes, actors, and motion in one visual object. FSDrive expands the vocabulary with visual tokens and jointly trains understanding, future-frame prediction, and planning. Its progressive curriculum first predicts lane dividers and 3D boxes, then renders the full frame, making physical structure part of the generation target rather than an after-the-fact caption.

### The intermediate is a future scene, not a verbal rationale

At planning time, the model no longer generates the lane-divider and 3D-box images separately. It emits one unified future frame containing the ordinary rendered scene plus red lane dividers and 3D detection boxes. The frame carries spatial relationships through those overlays and temporal relationships through the predicted appearance of the scene. The VLA then conditions its waypoint distribution on the current surround images, optional navigation command or ego status, and this visual $Q_{CoT}$ intermediate. In the paper’s formulation, it is acting as an inverse-dynamics model: infer the ego motion that explains a plausible future state.

The pipeline is easiest to read from left to right. During pre-training, the model learns to complete a road skeleton and object layout before it has to generate detailed pixels. During fine-tuning, the right-hand branch collapses those learned steps into one frame and passes that frame to the trajectory head. If a predicted lane bends incorrectly or a moving box is misplaced, the error is exposed in an inspectable image but can also propagate directly into the plan. The same intermediate is therefore the paper’s interpretability advantage and its principal error channel.

![FutureSightDrive visual CoT pipeline](/assets/images/futuresightdrive-source-figure-2-visual-cot.png)
*Fig 1: FSDrive first learns future lanes, boxes, and frames, then uses one unified future image as the visual reasoning step before predicting a trajectory. | source: [FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving, Figure 2](https://arxiv.org/abs/2505.17685)*

### Progressive pre-training supplies physical scaffolding

FSDrive starts from the MoVQGAN image tokenizer only to obtain a visual codebook; it does not replace Qwen2-VL’s ViT-based image encoder. The authors expand the language model’s vocabulary with the image tokens and use the tokenizer’s detokenizer to turn predicted tokens back into pixels. This preserves the original multimodal architecture while giving the next-token model a second output space. The model is initialized from Qwen2-VL-2B, fully fine-tuning the language model while freezing the encoders. The reported recipe uses 32 pre-training epochs; its 12-epoch fine-tuning run uses eight RTX A6000 GPUs.

Stage 1 mixes DriveLM-style understanding from OmniDrive-nuScenes, unlabeled nuScenes samples for future-frame prediction, and annotated lane-divider and 3D-box targets. The progressive order matters: lanes define the drivable skeleton, boxes constrain the motion of important objects, and the detailed future frame fills in appearance. Stage 2 combines DriveLM GVQA scene understanding with nuScenes trajectory planning. The tasks share one model but use task prompts at inference, so visual generation is a training target and a planning intermediate rather than an always-on simulator.

### Results keep the claims separate

On nuScenes, Table 1 reports average ST-P3 L2/collision of 0.53 m/0.17% without ego status and 0.28 m/0.10% with it for Qwen2-VL-2B. Under the separate UniAD metric implementation, the corresponding averages are 0.96 m/0.40% and 0.45 m/0.16%. The distinction matters because ST-P3 and UniAD aggregate displacement and collision over time differently; the table is not one common score. FSDrive also has a 0.60 m/0.19% ST-P3 result with LLaVA-7B without ego status, showing that the visual-CoT recipe is not tied to one backbone size.

On camera-only NAVSIM, Table 2 gives FSDrive 85.1 PDMS, ahead of LAW at 84.6 and DiffusionDrive-Cam at 83.6. This is a pseudo-closed-loop benchmark with images only, so it is stronger evidence about camera-only planning than a pure open-loop displacement number, while still falling short of reactive vehicle interaction. Table 3 reports FID 10.1 for 128×192 future frames, competitive with larger specialized generators. Table 4 reports a 0.57 DriveLM GVQA final score, combining accuracy and language metrics. These are three interfaces—trajectory, rendered future, and language QA—and should not be collapsed into one claim that the model “reasons” equally well in all of them.

### Ablations show where the gain comes from

Table 5 separates the pre-training ingredients. The no-auxiliary baseline is 1.22 m average L2 and 0.67% collision. Future-frame prediction by itself reaches 1.19 m/0.65%; 3D detection alone reaches 1.02 m/0.60%; lane-divider prediction alone reaches 1.06 m/0.61%. Combining VQA, future frames, 3D detection, and lanes reaches 0.98 m/0.58%. The result says that explicit geometric scaffolds help, but it does not prove that the detailed future pixels are the only source of the improvement.

Table 6 is more directly about the reasoning interface. No CoT gives 0.98 m/0.58% average L2/collision; text CoT gives 0.97/0.53; image-text CoT gives 0.98/0.50; unified spatio-temporal image CoT gives 0.96/0.40. The collision gap is larger than the displacement gap, which fits the mechanism: a lane and object overlay can make a future conflict explicit even when average waypoint error barely changes. In the qualitative test, the authors give a wrong navigation input to the planning step but a correct instruction while constructing the visual CoT; the correct instruction is an additional channel that guides the imagined scene. The lower deviation therefore cannot be attributed to the generated image alone. That experiment is suggestive of observation-based correction, not a substitute for systematic corrupted-instruction evaluation.

The generation ablation in Table 7 adds another boundary. FID improves from 29.4 with no future-frame pre-training to 16.2 with about 100K samples and 12.7 with about 200K; adding the progressive lane-and-box curriculum reaches 10.1. More image data helps, but the structural ordering supplies another improvement at the same reported 200K scale.

### The cost is forecast error and generation time

The planner consumes a generated scene, so errors in lanes, boxes, or background can become action errors even when the current camera view is clear. The authors also limit the generated future to the front view for efficiency, although the input is surround-view; a blind side or rear interaction is therefore not represented in the visual intermediate. The paper does not report the planning latency, image-token budget, loss weights, or a teacher-forced imagined-scene comparison. A useful next control would inject forecast errors independently into road geometry, moving actors, and texture, then compare the resulting action degradation with a latent or text-CoT control at equal compute.

## High-Level Takeaways

- FSDrive turns future lanes, boxes, and scene appearance into a visual intermediate that an inverse-dynamics planner can inspect and use.
- The full pre-training recipe reaches 0.98 m/0.58% average L2/collision in Table 5, while spatio-temporal image CoT reaches 0.96 m/0.40% in Table 6 and camera-only NAVSIM reaches 85.1 PDMS.
- The intermediate image is also an error channel: front-view-only forecasting, generation cost, and the absence of a matched latent-state control remain central deployment tests.
