---
title: 'OpenVLA: An Open-Source Vision-Language-Action Model'
date: '2024-06-01T04:00:00.000Z'
section: paper-shorts
postSlug: openvla-open-source-vision-language-action-model
legacyPath: /paper shorts/2024/06/01/openvla-open-source-vision-language-action-model.html
tags:
  - Other
field: Robotics
summary: OpenVLA released a 7B open robot policy trained on 970k real robot demonstrations.
---
## 2024 - OpenVLA

**arXiv:** [2406.09246](https://arxiv.org/abs/2406.09246)

**Project:** [openvla.github.io](https://openvla.github.io/)

**Summary:** OpenVLA is a 7B-parameter vision-language-action model trained on real robot demonstrations from Open X-Embodiment. It fuses SigLIP and DINOv2 visual features, maps them into a Llama-style language model, and trains the model to emit robot actions instead of text.

The open release matters: checkpoints, code, and fine-tuning recipes make generalist robot policies easier to study and adapt.

## Paper Insights

OpenVLA adapts a pretrained vision-language model into an action-generating robot policy and releases the result as an open baseline. The model takes robot observations and language instructions, then predicts actions for manipulation tasks. Its contribution is partly technical and partly ecosystem-oriented: provide an inspectable VLA recipe rather than leaving robot foundation policies closed. The evidence tests generalization across robot tasks and datasets. The caveat is action representation: a VLM backbone helps semantic grounding, but precise continuous control and embodiment transfer still require robot-specific data and validation.

![Figure 1: OpenVLA model architecture from OpenVLA: An Open-Source Vision-Language-Action Model](/assets/images/openvla-open-source-vision-language-action-model-paper-figure.png)
_Figure 1: OpenVLA model architecture. From the [OpenVLA: An Open-Source Vision-Language-Action Model paper](https://arxiv.org/abs/2406.09246), via arXiv HTML._

**What to look at:**
- The visual encoder fuses SigLIP and DINOv2 features.
- The policy is trained on Open X-Embodiment robot demonstrations.
- The open checkpoints and fine-tuning notebooks are core artifacts.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Scale | 7B VLA | Large enough to reuse language-model priors. |
| Data | 970k robot demonstrations | Gives the model cross-embodiment behavior. |
| Artifact | Open code/checkpoints | Makes generalist robot policies inspectable. |

**Context:** OpenVLA made the VLA recipe concrete and public. It also showed that Internet-scale vision-language pretraining can combine with robot demonstration data to produce transferable manipulation policies.

**Takeaway:** OpenVLA is the CLIP-to-actions moment: visual-language representations become a starting point for robot control.
