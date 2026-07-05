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

**Plain-language summary:** OpenVLA is a 7B-parameter vision-language-action model trained on real robot demonstrations from Open X-Embodiment. It fuses SigLIP and DINOv2 visual features, maps them into a Llama-style language model, and trains the model to emit robot actions instead of text.

The open release matters: checkpoints, code, and fine-tuning recipes make generalist robot policies easier to study and adapt.

**Why it mattered:** OpenVLA made the VLA recipe concrete and public. It also showed that Internet-scale vision-language pretraining can combine with robot demonstration data to produce transferable manipulation policies.

**Take-home message:** OpenVLA is the CLIP-to-actions moment: visual-language representations become a starting point for robot control.
