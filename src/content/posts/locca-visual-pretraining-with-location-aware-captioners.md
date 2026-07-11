---
title: 'LocCa: Visual Pretraining with Location-aware Captioners'
date: '2024-03-28T04:00:00.000Z'
section: paper-shorts
postSlug: locca-visual-pretraining-with-location-aware-captioners
legacyPath: /paper shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html
tags:
  - Other
field: Vision-Language Models
summary: LocCa adds location-aware captioning tasks to visual pretraining, improving region grounding while preserving broad VLM transfer.
---
## 2024 - LocCa

**arXiv:** [2403.19596](https://arxiv.org/abs/2403.19596)

**Summary:** LocCa asks whether a captioning-style pretraining model can learn localization without becoming a special-purpose detector. The answer is yes: keep the encoder-decoder captioner interface, but add tasks where the decoder must talk about regions and predict their coordinates.

The result is a visual encoder that still transfers to holistic tasks like classification, captioning, OCR, and VQA, while becoming much more sensitive to object-level location.

## Paper Insights

Standard captioning pretraining gives a model a useful global visual representation, but it does not force the representation to know where described objects live. LocCa adds two location-aware proxy tasks next to normal captioning: automatic referring expression (AREF) and grounded captioning (GCAP). The same decoder predicts both text and bounding-box coordinates, with task prefixes telling the model which interface to use.

The important design choice is that localization enters during visual pretraining rather than only during later instruction tuning. This keeps the model simple: one vision transformer, one transformer decoder, one generative objective. The authors also call out a benchmark hygiene issue on RefCOCO: the splits overlap heavily, so they remove validation and test images from the combined training set.

![Figure 1 from LocCa showing captioning, automatic referring expression, and grounded captioning pretraining tasks](/assets/images/locca-visual-pretraining-with-location-aware-captioners-paper-figure.png)
_Figure 1 shows the LocCa task interface: normal captioning, automatic referring expression, and grounded captioning all share the same encoder-decoder model. From the [LocCa paper](https://arxiv.org/abs/2403.19596), via arXiv HTML._

**What to look at:**
- AREF asks the model to produce a region description and a box.
- GCAP asks the model to caption grounded regions, not only the whole image.
- The same model keeps the inference speed of standard caption-pretrained models.
- The RefCOCO cleanup matters because combined splits can otherwise leak test images.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Pretraining tasks | Cap, AREF, GCAP | Adds localization while preserving a language-like interface. |
| Localization transfer | RefCOCO, RefCOCO+, RefCOCOg | Tests whether the encoder actually learns region grounding. |
| Holistic transfer | CLS, CAP, OCR-VQA, VQA, GQA | Checks that localization did not break general visual understanding. |

**Compact result slice:**

| Finding | Evidence |
| ------- | -------- |
| Strong RefCOCO results | The paper reports state-of-the-art referring expression comprehension across RefCOCO variants. |
| Better frozen encoder | LocCa's vision encoder substantially outperforms CapPa and SigLIP-style baselines on localization transfer. |
| Clean evaluation | The authors remove overlapping validation/test images from the combined RefCOCO training set. |
| VLM transfer | A PaLI-3 model using the LocCa encoder improves over strong SigLIP encoder baselines, especially on object-sensitive tasks. |

**Context:** LocCa is a neat counterexample to the idea that grounding needs a separate detection-heavy architecture. A captioner can become location-aware if the pretraining interface makes coordinates part of the language game.

**Takeaway:** For VLM pretraining, the task interface is supervision. If captions include where as well as what, the encoder learns a more useful visual representation.
