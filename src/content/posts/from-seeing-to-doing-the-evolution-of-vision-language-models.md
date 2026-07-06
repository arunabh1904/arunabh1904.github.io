---
title: 'From Seeing to Doing: The Evolution of Vision-Language Models'
date: '2026-07-05T20:00:00.000Z'
section: blog
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
legacyPath: /blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html
tags:
  - Research
summary: A linked reading map for how VLMs moved from image-text retrieval to visual assistants, driving stacks, benchmarks, and robot policies.
---

# From Seeing to Doing: The Evolution of Vision-Language Models

The useful history of vision-language models is not just "models got bigger." It is that each generation asked the visual representation to carry more responsibility.

At first, an image representation only had to meet text in a shared space. Then it had to support conversation. Then it was asked to reason about traffic scenes, explain decisions, survive benchmarks, and eventually produce actions. That is the real arc: vision-language models keep moving from seeing to doing.

## 1. Retrieval made vision open-vocabulary

[CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) made the first clean move. It put images and text in the same embedding space, then used language as the classifier. Vision stopped being limited to a fixed label set; a prompt could define the task.

[SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) kept that broad shape but improved the objective. By replacing batch-level softmax contrastive training with independent sigmoid losses, it made image-text matching less dependent on very large synchronized batches. The important idea stayed the same: language was no longer an annotation after the fact. It became part of the model's interface to the visual world.

## 2. Chat changed the interface

[LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) changed what a VLM felt like to use. Once a visual encoder was connected to an instruction-tuned language model, the system could answer questions, describe images, and participate in dialogue. The representation stopped being only searchable. It became conversational.

The next wave made that interface less brittle. [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html) studied the visual side carefully. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) and [DeepSeek-VL2](/paper%20shorts/2024/12/01/deepseek-vl2-mixture-of-experts-vision-language-models.html) pushed on flexible high-resolution perception. [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html) emphasized open, high-quality data. [InternVL 2.5](/paper%20shorts/2024/12/01/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models.html), [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html), and [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html) show that frontier VLM quality is not one trick. It is a recipe: visual tokenization, data curation, post-training, reasoning behavior, and video compression all matter.

## 3. Driving exposed the cost of pretending language is control

Autonomous-driving work asks a sharper question: can language help with decisions where mistakes have physical consequences?

[GPT-Driver](/paper%20shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html), [Driving with LLMs](/paper%20shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html), [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html), [AsyncDriver](/paper%20shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html), [EMMA](/paper%20shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html), [DriveMM](/paper%20shorts/2024/12/01/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving.html), [DIMA](/paper%20shorts/2025/01/01/distilling-multimodal-large-language-models-for-autonomous-driving.html), and [SENNA](/paper%20shorts/2024/10/01/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving.html) explore different placements for language. Some use language as the planner, some as an explanation layer, some as an offline teacher, and some as the shared interface for many driving tasks.

The pattern I find most convincing is hybrid. VLMs are useful for semantics, intent, and scene-level reasoning, but driving still needs explicit geometry, latency budgets, safety checks, and control structure. A fluent rationale is not a trajectory.

## 4. Benchmarks pulled the field back to ground

The benchmark papers are the necessary cold shower. [DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), [IDKB](/paper%20shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html), [TOD3Cap](/paper%20shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html) all say the same thing from different angles: a plausible answer is not enough.

A useful driving VLM has to ground its answer in the image, follow traffic rules, describe the scene spatially, and fail safely under corruptions, privacy probes, and adversarial inputs. The important question is no longer "can the model answer?" It is "did the model use the right evidence and obey the right constraints?"

## 5. Robotics turns words into state changes

Robotics is the natural endpoint of this arc. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), and [DexVLA](/paper%20shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html) move VLMs from describing the world to changing it.

That shift forces a different set of problems: action representations, continuous control, embodiment data, and closed-loop behavior. A robot does not get partial credit for a fluent caption if the gripper misses.

## Reading path

Start with [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) and [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) for the image-text objective.

Then read [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html), [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html), [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html), [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html), [InternVL 2.5](/paper%20shorts/2024/12/01/internvl-2-5-expanding-performance-boundaries-of-open-source-multimodal-models.html), [DeepSeek-VL2](/paper%20shorts/2024/12/01/deepseek-vl2-mixture-of-experts-vision-language-models.html), [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html), and [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) for the modern visual-chat stack.

For driving, follow [GPT-Driver](/paper%20shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html), [Driving with LLMs](/paper%20shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html), [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html), [AsyncDriver](/paper%20shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html), [EMMA](/paper%20shorts/2024/10/01/emma-end-to-end-multimodal-model-for-autonomous-driving.html), [DriveMM](/paper%20shorts/2024/12/01/drivemm-all-in-one-large-multimodal-model-for-autonomous-driving.html), [DIMA](/paper%20shorts/2025/01/01/distilling-multimodal-large-language-models-for-autonomous-driving.html), and [SENNA](/paper%20shorts/2024/10/01/senna-bridging-large-vision-language-models-and-end-to-end-autonomous-driving.html).

For evaluation, read [DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), [IDKB](/paper%20shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html), [TOD3Cap](/paper%20shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html).

For robotics, read [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), and [DexVLA](/paper%20shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html).

## Takeaways

The field moved through four interfaces: image-text retrieval, visual chat, decision support, and embodied action.

The bottleneck keeps moving too. First it was representation learning. Then instruction data. Then visual resolution, grounding, latency, and action representation.

For driving and robotics, the most credible systems are hybrids. Language models help with semantics and intent, but precise perception, planning, and control still need domain-specific structure.

The long-term direction is clear: VLMs are becoming world-interface models. The hard part is making that interface grounded enough to trust.
