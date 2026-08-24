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

### Method and reported result

OpenVLA is a 7B-parameter vision-language-action model trained on real robot demonstrations from Open X-Embodiment. It fuses SigLIP and DINOv2 visual features, maps them into a Llama-style language model, and trains the model to emit robot actions instead of text.

## Summary

> The open release matters: checkpoints, code, and fine-tuning recipes make generalist robot policies easier to study and adapt.

## Core Insights

OpenVLA adapts a pretrained vision-language model into an action-generating robot policy and releases the result as an open baseline. The model takes robot observations and language instructions, then predicts actions for manipulation tasks. Its contribution is partly technical and partly ecosystem-oriented: provide an inspectable VLA recipe rather than leaving robot foundation policies closed. The evidence tests generalization across robot tasks and datasets. The caveat is action representation: a VLM backbone helps semantic grounding, but precise continuous control and embodiment transfer still require robot-specific data and validation.

![Figure 1: OpenVLA model architecture from OpenVLA: An Open-Source Vision-Language-Action Model](/assets/images/openvla-open-source-vision-language-action-model-paper-figure.png)
*Figure 1: OpenVLA model architecture. From the [OpenVLA: An Open-Source Vision-Language-Action Model paper](https://arxiv.org/abs/2406.09246). source: [OpenVLA: An Open-Source Vision-Language-Action Model paper](https://arxiv.org/abs/2406.09246)*

![Figure 2 from OpenVLA: An Open-Source Vision-Language-Action Model](/assets/images/openvla-open-source-vision-language-action-model-source-figure-2.webp)
*Figure 2 BridgeData V2 WidowX robot evaluation tasks and results. We evaluate OpenVLA and prior state-of-the-art generalist robot policies on a comprehensive suite of tasks covering several axes of generalization, as well as tasks that specifically assess language conditioning ability. OpenVLA achieves highest overall performance and even outperforms closed-source model RT-2-X in all categories except for semantic generalization. Average success rates StdErr are computed across 170 total rollouts per approach. source: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)*

![Figure 3 from OpenVLA: An Open-Source Vision-Language-Action Model](/assets/images/openvla-open-source-vision-language-action-model-source-figure-3.webp)
*Figure 3 Google robot evaluation results. We evaluate generalist robot policies on in-distribution and out-of-distribution (OOD) tasks on the mobile manipulator used in RT-1 and RT-2 evaluations [ 2 , 7 ] . We find that OpenVLA and RT-2-X attain comparable performance and significantly outperform RT-1-X and Octo overall. Average success rates StdErr are computed across 60 total rollouts per approach. See Table 6 for detailed results. source: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)*


**What to look at:**
- The visual encoder fuses SigLIP and DINOv2 features.
- The policy is trained on Open X-Embodiment robot demonstrations.
- The open checkpoints and fine-tuning notebooks are core artifacts.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Scale | 7B VLA | Large enough to reuse language-model priors. |
| Data | 970k robot demonstrations | Gives the model cross-embodiment behavior. |
| Artifact | Open code/checkpoints | Makes generalist robot policies inspectable. |

## High-Level Takeaways

- OpenVLA informs whether an inspectable 7B vision-language backbone trained across many robot datasets is a useful default policy before building a proprietary architecture. The policy fuses SigLIP's semantic features with DINOv2's spatial features, then autoregressively emits discretized actions from language-conditioned observations. Open X-Embodiment's 970,000 demonstrations supply the cross-robot curriculum, making data interoperability as central as model scale.
- The release establishes a strong open baseline and shows broad transfer, but it does not separate the contribution of internet-scale VLM priors from robot-data scale and dual visual encoders. A compute-matched policy trained from scratch and a single-encoder ablation across unseen embodiments would answer that. At ten times the embodiments, inconsistent action spaces and dataset imbalance could overwhelm shared tokens. The foundation-policy thesis would fail if per-robot specialists initialized from the same encoders adapt faster and achieve higher closed-loop reliability with equal demonstrations.
- OpenVLA made the VLA recipe concrete and public. It also showed that Internet-scale vision-language pretraining can combine with robot demonstration data to produce transferable manipulation policies.
- OpenVLA is the CLIP-to-actions moment: visual-language representations become a starting point for robot control.
