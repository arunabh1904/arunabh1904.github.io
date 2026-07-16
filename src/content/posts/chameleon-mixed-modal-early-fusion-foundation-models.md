---
title: 'Chameleon: Mixed-Modal Early-Fusion Foundation Models'
date: '2024-05-16T09:00:00.000Z'
section: paper-shorts
postSlug: chameleon-mixed-modal-early-fusion-foundation-models
legacyPath: /paper shorts/2024/05/16/chameleon-mixed-modal-early-fusion-foundation-models.html
tags: [Multimodal AI]
field: Omni-Models
summary: '2024 – Chameleon: early fusion over mixed image-and-text token sequences.'
---

## 2024 – Chameleon: Mixed-Modal Early-Fusion Foundation Models

**arXiv:** [2405.09818](https://arxiv.org/abs/2405.09818)  
**Conference:** Technical report

**Summary:** Chameleon trains token-based, early-fusion models that can understand and generate images and text in arbitrary order within one sequence. The paper contributes a training and alignment recipe for keeping this unified setting stable.

## Paper Insights

Early fusion makes a direct architectural claim: one transformer can model mixed documents instead of bolting a vision encoder onto a language model after pre-training. The evidence spans visual question answering, captioning, text generation, image generation, and long-form mixed-modal generation.

| Decision | Chameleon's answer | Tradeoff |
| --- | --- | --- |
| Representation | Discrete image and text tokens in one stream | Visual fidelity depends on tokenization and sequence budget. |
| Sharing | Early fusion throughout the transformer | Shared capacity may create modality interference. |
| Objective | Next-token prediction | A simple unified objective, but expensive for long visual sequences. |

**Limits:** Strong unified generation does not establish that a fully shared representation is best for every visual understanding or action task.

**Takeaway:** Chameleon is the clean baseline for asking whether early fusion is worth its sequence-length and interference costs.

