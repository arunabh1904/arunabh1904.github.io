---
title: 'DriveMM: All-in-One Large Multimodal Model for Autonomous Driving'
date: '2024-12-10T00:00:00.000Z'
section: paper-shorts
postSlug: drivemm-all-in-one-large-multimodal-model-for-autonomous-driving
legacyPath: /paper shorts/2024/12/01/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2024 – DriveMM: All-in-One Large Multimodal Model for Autonomous Driving"
---

**arXiv:** [Original DriveMM report](https://arxiv.org/abs/2412.07689v1) · [Revised RoboTron-Drive report](https://arxiv.org/abs/2412.07689v5) · **Project:** [RoboTron-Drive](https://zhijian11.github.io/RoboTron-Drive)

## Summary

> Originally released as DriveMM and titled RoboTron-Drive in its fifth revision, this paper studies a generalist driving VLM trained across six datasets and thirteen task categories. Its main contribution is making heterogeneous supervision compatible: identify camera views explicitly, normalize object references, diversify question formats, and train from simple image alignment toward multi-view driving tasks. The results below use the revised report, which adds broader transfer and offline planning evaluations.

## Core Insights

### A shared model needs a shared meaning for its input and output tokens

The architecture uses a SigLIP vision encoder at 384×384, a two-layer MLP projector, and Llama-3.1 8B. It accepts images, multiple images, videos, and multiple videos through a common language-model interface. Video features receive 2×2 spatial pooling before becoming visual tokens.

The perspective-aware prompt labels each input with its format and sensor/view identity, including projected LiDAR inputs. Without that context, the same patch could describe an object in front of the vehicle or behind it, and the model would have to infer that distinction from visual content alone. The prompt supplies a naming convention; it does not introduce calibrated 3D geometry or a dedicated point-cloud encoder. LiDAR-derived visual representations can appear among the inputs, as in MAPLM's BEV images.

![RoboTron-Drive uses a shared vision-language architecture for multiple views and driving task prompts](/assets/images/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving-source-figure-2.webp)
*Fig 1: Camera and input-type labels accompany visual tokens before they enter the shared language model. The surrounding tasks change the prompt and expected answer, while retaining the same visual-language interface. | source: [RoboTron-Drive, Figure 2](https://arxiv.org/abs/2412.07689v5)*

Output conventions matter just as much. The source datasets encode object references differently, sometimes using camera names and sometimes numeric camera identifiers, with coordinates in different image scales. The authors standardize these into explicit camera labels and coordinates normalized to 0–100. A shared model can then learn one interpretation of a box rather than silently switching coordinate systems with the dataset.

### The curriculum separates alignment, perception, and driving specialization

The first stage trains only the projector on 558K image-text pairs. The second updates the whole model on single-image data and some language data. The third introduces general visual instructions and grounding examples across single and multiple views, images, and videos. The fourth fine-tunes all parameters on roughly 1.5 million examples from CODA-LM, MAPLM, DriveLM, LingoQA, OmniDrive, and NuInstruct.

This progression gives driving supervision a useful starting point: the model already knows how to associate a region with a category and how to consume several visual inputs. It also prevents “generalist” from being confused with zero-shot adaptation from an untouched language model. The final model has substantial multimodal pretraining and driving-specific fine-tuning.

![The curriculum progresses from image-language alignment through multi-capability pretraining to six standardized driving datasets](/assets/images/drivemm-source-figure-3.png)
*Fig 2: The right-hand stage combines varied driving inputs only after the left-hand stages establish image comprehension and perception. The augmentation and standardization examples show that answer formats are part of the curriculum. | source: [RoboTron-Drive, Figure 3](https://arxiv.org/abs/2412.07689v5)*

### Format cleanup produces gains before changing the architecture

GPT-4o-mini paraphrases question-answer pairs while preserving their intended meaning, and some open questions are converted into multiple choice. This is especially relevant to datasets with very few question templates: a model can otherwise learn their surface form without learning to handle equivalent requests.

In the QA-enhancement ablation, augmentation raises DriveLM's reported aggregate from 47.41 to 59.67. Standardization then raises it to 60.54, while NuInstruct improves from 34.77 to 42.44. The effect is uneven, as expected: coordinate and object-reference normalization matters most where answers contain localized entities.

The separate multi-dataset ablation uses models trained with the same overall recipe. Mixed training improves all six reported dataset aggregates over their individually trained counterparts, including MAPLM from 74.02 to 76.67 and LingoQA from 67.40 to 69.20. These within-dataset comparisons support transfer from shared training. Averaging the six aggregates produces the paper's summary score, but that average mixes different evaluators and units; it is not a single accuracy over all examples.

### Broad improvement does not mean winning every row

The revised model reports MAPLM 76.67, DriveLM 61.30, OmniDrive 50.25, and NuInstruct 46.30 under their respective aggregate metrics. Its strongest practical comparison is Drive-OV, a LLaVA-OneVision model given the same perspective prompt and augmented driving data. RoboTron-Drive improves on Drive-OV in several grounding-heavy datasets, but LingoQA favors Drive-OV, 70.10 versus 69.20, and DriveLM's accuracy component favors Drive-OV, 79.38 versus 76.09. The generalist result is broad competence, not uniform dominance on every submetric.

Transfer is tested on BDD-X, DRAMA, and DriveBench outside the six driving fine-tuning datasets. The generalist reaches reported scores of 43.10, 53.32, and 61.06, above the six single-dataset specialists in that table. These evaluations largely compare generated answers to reference answers using language-based evaluators; they should not be interpreted as closed-loop driving success rates or proof that the pretraining corpus contains no related visual material.

### Offline planning is an additional fine-tuning experiment

The authors further fine-tune on nuScenes planning data. Mean L2 error is 0.33 meters and mean predicted collision rate is 0.26%, compared with VAD-Base at 0.37 meters and 0.33% in the same table. The horizons are 1, 2, and 3 seconds, evaluated against logged trajectories. This is a separate planning adaptation, not a capability demonstrated by every QA checkpoint without additional training.

The paper does not report a closed-loop driving study or a real-time latency result that establishes deployment readiness. Its stronger lesson is about supervision: shared training becomes more useful when the meaning of a view, object reference, and answer format is consistent across datasets.

## High-Level Takeaways

- Camera labels and normalized object references make heterogeneous datasets compatible with a shared model; they are substantive parts of the method.
- Curriculum, QA augmentation, and standardization have separate ablations. The largest gains need not come from a novel backbone.
- Mixed training improves the matched specialist comparisons, while stronger external baselines still win some individual metrics.
- Keep QA transfer, additional offline planning fine-tuning, and closed-loop driving as separate capability claims.
