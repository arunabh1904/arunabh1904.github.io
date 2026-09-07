---
title: 'OpenVLA: An Open-Source Vision-Language-Action Model'
date: '2024-06-13T00:00:00.000Z'
section: paper-shorts
postSlug: openvla-open-source-vision-language-action-model
legacyPath: /paper shorts/2024/06/01/openvla-open-source-vision-language-action-model.html
tags:
  - Other
field: 'Vision-Language-Action & Robotics'
summary: "2024 – OpenVLA: An Open-Source Vision-Language-Action Model"
---

## 2024 – OpenVLA

**arXiv:** [2406.09246](https://arxiv.org/abs/2406.09246)

**Project:** [openvla.github.io](https://openvla.github.io/)

## Summary

> OpenVLA turns a pretrained vision-language model into a generalist robot policy with a deliberately simple action interface: continuous controls become language-model tokens, and the model predicts those tokens with next-token cross-entropy. Its open 7B checkpoint transfers across robot tasks, but its results also show where the recipe depends on robot-specific visual fine-tuning and a large, carefully curated demonstration mixture.

## Core Insights

OpenVLA is best understood as an interface and data study built around a 7B Prismatic VLM. The backbone combines a 600M visual encoder with a two-layer projector and a Llama 2 7B language model. The visual encoder concatenates SigLIP features, which provide semantic image-text alignment, with DINOv2 features, which contribute finer spatial structure. The model then reuses the language model’s sequence machinery to emit actions. That choice makes the policy easy to inspect and fine-tune, while exposing the cost of asking an autoregressive language model to be a controller.

### Turning robot controls into a language-model sequence

OpenVLA discretizes each action dimension into 256 bins. The bin interval is the 1st–99th percentile range in the training data, so a few outlier motions do not consume most of the available resolution. Because the Llama tokenizer reserves too few new-token slots, the training recipe overwrites its 256 least-used vocabulary entries with action tokens. A seven-dimensional robot action therefore becomes a seven-token suffix, and cross-entropy is evaluated on that suffix rather than on the visual or instruction prefix.

The interface preserves the strengths of a VLM—language conditioning, a mature transformer, and a shared representation across tasks—but it does not make control continuous. Quantization and one-token-at-a-time decoding remain part of the policy’s behavior. This is why the release is useful as a baseline: a downstream result can be attributed to the data mixture, visual features, or fine-tuning recipe without first implementing a new action decoder.

![OpenVLA architecture and tokenized action interface](/assets/images/openvla-open-source-vision-language-action-model-paper-figure.png)
*Fig 1: Author schematic based on the paper’s architecture. The image and instruction enter the fused DINOv2–SigLIP encoder and projector, the Llama 2 7B backbone predicts an action-token suffix, and an action de-tokenizer maps those tokens back to the 7D robot control. | source: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)*

### Why the data mixture matters as much as the backbone

The full Open X-Embodiment collection contained more than 2M trajectories from over 70 robot datasets when the paper was written. OpenVLA curates 970k of them into a common setting: manipulation data with at least one third-person camera and single-arm end-effector control. Octo’s mixture weights are reused to favor diverse tasks and scenes, and DROID is first included conservatively at 10% before being removed for the final third of training because its action-token accuracy remained low. The resulting model sees many embodiments, but the common action and camera assumptions narrow what “out of the box” means.

The design sweeps explain two less obvious choices. At 224×224 pixels, OpenVLA performs as well as the tested 384×384 variant while taking roughly one-third the training time. Fine-tuning the visual encoder is crucial for control even though frozen encoders often help ordinary VLM transfer: robot actions need spatial detail around contacts, object orientation, and grasp geometry. The final run makes 27 passes through the robot data, reaching over 95% action-token accuracy; the authors found that the usual one or two language-model epochs were insufficient for real-robot performance.

### What the evaluations actually separate

On the 17-task BridgeData V2 WidowX suite, OpenVLA reaches 70.6% average success, ahead of RT-2-X at 50.6%, Octo at 20.0%, and RT-1-X at 18.5%. The category breakdown explains the average: OpenVLA reaches 87.0% on visual generalization, 60.0% on motion, 76.7% on physical generalization, and 90.0% on language grounding. Semantic generalization is the exception—36.3% for OpenVLA versus 38.8% for RT-2-X—consistent with RT-2-X retaining more internet-language co-training. The evaluation uses 170 rollouts, so the result is a broad task comparison rather than a claim that every individual manipulation is solved.

![BridgeData V2 results across generalization categories](/assets/images/openvla-open-source-vision-language-action-model-source-figure-2.webp)
*Fig 2: BridgeData V2 WidowX robot evaluation tasks and results. We evaluate OpenVLA and prior state-of-the-art generalist robot policies on a comprehensive suite of tasks covering several axes of generalization, as well as tasks that specifically assess language conditioning ability. OpenVLA achieves highest overall performance and even outperforms closed-source model RT-2-X in all categories except for semantic generalization. Average success rates ± StdErr are computed across 170 total rollouts per approach. See Table 4 for detailed results. | source: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)*

The Google mobile-manipulator evaluation checks whether that advantage survives a different robot and task distribution. OpenVLA averages 85.0%, compared with 78.3% for RT-2-X, 33.3% for RT-1-X, and 26.7% for Octo. The in-distribution scores are 88.0% and 72.0% for OpenVLA and RT-2-X; on out-of-distribution tasks, both reach 82.9%. Read together with BridgeData, the plots support a narrower conclusion than “the 7B model is universally better”: the open model transfers strongly when the task semantics and physical control conventions remain close enough to the curated robot data, while RT-2-X retains an edge on internet-style semantic knowledge in the BridgeData categories.

![Google robot results for in-distribution and out-of-distribution tasks](/assets/images/openvla-open-source-vision-language-action-model-source-figure-3.webp)
*Fig 3: Google robot evaluation results. We evaluate generalist robot policies on in-distribution and out-of-distribution (OOD) tasks on the mobile manipulator used in RT-1 and RT-2 evaluations [2, 7]. We find that OpenVLA and RT-2-X attain comparable performance and significantly outperform RT-1-X and Octo overall. Average success rates ± StdErr are computed across 60 total rollouts per approach. See Table 6 for detailed results. | source: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)*

### Fine-tuning and cost are part of the result

OpenVLA is also a starting point for adaptation. Across diverse and narrow fine-tuning tasks, it is the only tested approach to maintain at least 50% success on every task, although Diffusion Policy produces smoother trajectories on narrow, dexterous single-instruction tasks. The parameter-efficient sweep gives a practical boundary: full fine-tuning reaches 69.7 ± 7.2% mean success, LoRA reaches 68.2 ± 7.5% while training only 1.4% of the parameters, and last-layer-only tuning falls to 30.3 ± 6.1%. Freezing the visual encoder also hurts, at 47.0 ± 6.9%, supporting the claim that visual features must adapt to the target robot and scene. LoRA rank 32 and 64 perform the same within error bars, and the paper reports 10–15 hours on one A100 for a new task, an eightfold compute reduction relative to full fine-tuning.

The open checkpoint does not make the training bill disappear. The final model uses 64 A100 GPUs for 14 days, totaling 21,500 A100-hours. Inference occupies about 15 GB in bfloat16 and runs at roughly 6 Hz on an RTX 4090 without compilation or speculative decoding. Quantization reduces memory without compromising the paper’s reported real-robot performance, but the uncompiled speed and the common single-arm setup remain meaningful constraints for deployment.

## High-Level Takeaways

- OpenVLA’s central engineering move is to reuse a VLM sequence model for control by quantizing each action dimension into 256 data-derived bins and predicting an action-token suffix.
- The fused SigLIP–DINOv2 encoder and 970k-trajectory Open X mixture are as consequential as the 7B Llama backbone: spatially precise robot control requires visual fine-tuning and curated embodiment overlap.
- OpenVLA reaches 70.6% on BridgeData V2 and 85.0% on the Google robot; LoRA retains 68.2% versus 69.7% for full fine-tuning while using 1.4% of the parameters.
- The release is a strong open baseline, with 21,500 A100-hours of pretraining and roughly 6 Hz uncompiled inference; closed-loop, multi-arm, and longer-horizon tests define the next boundary.
