---
title: 'GLIP: Grounded Language-Image Pre-training'
date: '2021-12-07T00:00:00.000Z'
section: paper-shorts
postSlug: glip-grounded-language-image-pretraining
legacyPath: /paper shorts/2021/12/07/glip-grounded-language-image-pretraining.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2021 – GLIP: Grounded Language-Image Pre-training'
---

## 2021 – GLIP: Grounded Language-Image Pre-training

**arXiv:** [2112.03857](https://arxiv.org/abs/2112.03857)

**Code:** [microsoft/GLIP](https://github.com/microsoft/GLIP)

## Summary

> GLIP turns detection into phrase grounding: the detector receives a text prompt, and each region is scored against contextual language features rather than a fixed classifier matrix. Deep image-language fusion makes the region features themselves prompt-aware, while teacher-generated boxes turn web image-text pairs into grounding data. GLIP-L is pretrained on 27M grounding examples—3M human-annotated and 24M web pairs—and reports 49.8 AP on zero-shot COCO and 26.9 AP on zero-shot LVIS. After COCO fine-tuning it reaches 60.8 AP on val2017, but its strongest story is transfer efficiency: one prompt-conditioned model can adapt to new categories and domains with little task-specific data.

## Core Insights

### Detection becomes matching regions to a prompt

A conventional detector asks, “Which of my fixed classes does this region belong to?” GLIP asks, “Which phrase in this prompt does this region describe?” The difference is small in the output layer but large in what can be trained. Region features $O$ are aligned with contextual token features $P$ through $S_{ground}=OP^\top$; the same loss can consume detection boxes, phrase-grounding annotations, and pseudo boxes generated from captions.

![GLIP prompt-conditioned zero-shot detections](/assets/images/glip-paper-figure-1.png)
*Fig 1: The user writes the categories or description into a prompt, and GLIP returns boxes for cars, pistols, holes, raccoons, or other mentioned concepts without a task-specific class head. | source: [GLIP, Figure 1](https://arxiv.org/abs/2112.03857)*

The paper adds a second ingredient because a late dot product is not enough. In the deep-fusion variant, cross-modality multi-head attention passes information between visual regions and language tokens in the last encoder layers. The visual representation therefore depends on the prompt before the final matching step. That is why prompt wording can carry domain knowledge: “stingray” is a weak visual cue, while “stingray, which is flat and round” supplies a useful shape prior.

### Self-training supplies boxes for a much larger vocabulary

The training scale comes from self-training, but the semantics come from language. A gold GLIP teacher parses noun phrases in web captions and predicts boxes for them. The student trains on the 3M human-annotated grounding examples plus 24M web image-text pairs, yielding 78.1M high-confidence pseudo annotations and 58.4M unique noun phrases in the reported pretraining pool. The teacher can make an educated guess for phrases outside the detection label set—for example, localizing “vaccine” through a vial or “turquoise” through a Caribbean sea description. That guess becomes a noisy target for the student; it is not equivalent to a human box annotation.

![GLIP zero-shot transfer across ODinW datasets](/assets/images/glip-grounded-language-image-pretraining-source-figure-9.webp)
*Fig 2: Zero-shot AP50 on five Object Detection in the Wild datasets. The first three contain categories not present in the Objects365 vocabulary, where grounding data is especially helpful; the last two are covered by Objects365. | source: [GLIP, Figure 5](https://arxiv.org/abs/2112.03857)*

The transfer result is clearest on categories absent from Objects365. Adding gold grounding data from GLIP-T (B) to GLIP-T (C) raises Pothole from 3.6 to 17.0 AP50, EgoHands from 0.5 to 49.1, and Raccoon from 47.0 to 51.5. The covered Cottontail Rabbits category instead moves slightly down, from 71.6 to 71.1. Grounding data expands missing semantic coverage; it does not improve every category uniformly. The larger GLIP-L reaches 25.7 on Pothole and 45.5 on EgoHands, so greater scale is not a monotonic improvement on every small downstream dataset either.

![GLIP manual prompt tuning](/assets/images/glip-grounded-language-image-pretraining-source-figure-10.webp)
*Fig 3: Adding the attributes “flat and round” to the stingray prompt improves AP50 from 4.6 to 9.7 on Aquarium, with no model update or annotated examples. The gain is a useful demonstration of language-conditioned localization, not a guarantee that descriptive prompts always help. | source: [GLIP, Figure 6](https://arxiv.org/abs/2112.03857)*

### Deep fusion makes prompt tuning effective

The model also makes adaptation cheaper. For one task, the prompt embedding can be cached and tuned while the language backbone and grounding model remain fixed. Prompt tuning nearly matches full-model tuning for GLIP-T and GLIP-L; a shallow GLIP-T variant without language-aware deep fusion behaves more like ordinary linear probing and leaves a much larger gap. This isolates the point of deep fusion: if the visual features are conditioned on language, a small task-specific prompt can steer the detector's internal representation.

The boundary is computational and statistical. The authors report that deep fusion adds less than one baseline model’s computation, but its effect is not uniformly positive: on LVIS, it can hurt common-category performance when the model is trained only on Objects365, while grounding data recovers rare-category transfer. Prompt design also becomes part of the evaluation protocol. A prompt that names all categories may exceed the language encoder's length limit, so GLIP chunks large vocabularies and queries multiple times. The open vocabulary is therefore flexible, but not free.

## High-Level Takeaways

- GLIP's central abstraction is a prompt-conditioned detector: detection is context-free grounding, and grounding is contextualized detection.
- Deep fusion matters because it lets language change the visual features before matching; prompt tuning is effective only when that conditioning reaches the representation.
- The 27M grounding pool expands semantic coverage through teacher-generated boxes, yet those boxes remain noisy guesses and should be treated differently from human localization labels.
- Zero-shot transfer to novel domains is the strongest evidence: gold grounding data moves EgoHands from 0.5 to 49.1 AP50 in the GLIP-T (B)/(C) comparison, while prompt attributes can double Aquarium stingray AP50 from 4.6 to 9.7.
- Prompt length, phrasing, and deep-fusion cost are part of deployment. “Open vocabulary” describes the interface; it does not remove the need to manage queries and domain shift.
