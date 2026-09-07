---
title: "Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles"
date: '2026-08-09T00:00:00.000Z'
section: paper-shorts
postSlug: distilling-vision-language-models-for-robust-traffic-sign-perception-in-autonomous-vehicles
legacyPath: /paper shorts/2026/08/09/distilling-vision-language-models-for-robust-traffic-sign-perception-in-autonomous-vehicles.html
tags:
  - Autonomous Driving
  - Robustness
  - Vision-Language Models
field: 'Autonomous Driving: VLMs & Evaluation'
summary: "2026 – Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles"
---

## 2026 – Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles

**arXiv:** [2608.08815](https://arxiv.org/abs/2608.08815)<br />
**Code:** [LAMDA](https://github.com/pedram-mohajer/LAMDA)

## Summary

> LAMDA transfers language-grounded structure into an image-only traffic-sign classifier. It builds frozen prototype banks from VLM-generated sign descriptions and class names, then uses two auxiliary alignment losses during training; the text encoder and prototypes are discarded at inference. Across GTSRB and LISA, four backbones, and three physical attack types, the paper reports consistent robustness gains while usually preserving clean accuracy.

## Core Insights

Traffic-sign defenses often specialize to one perturbation or sacrifice clean accuracy. LAMDA keeps the standard classifier objective but adds two directions of language supervision: descriptive prototypes provide richer visual semantics, while short class-name prototypes regularize the class geometry. The student therefore learns an image representation that is pulled toward stable language-defined directions without running a VLM on the vehicle.

The evaluation uses clean data for training and tests shadow, natural-light, and printable RP2 attacks. The reported maximum gains are +12.5 percentage points under shadow attacks on GTSRB and +13.2 points under natural-light attacks on LISA. In the physical RP2 experiment, the baseline correctly classifies 6 of 16 images, compared with 12 of 16 for LAMDA. The two losses are complementary: language replacement with irrelevant prototypes removes much of the robustness gain.

The training diagram is easiest to interpret as a teacher path that disappears. A VLM supplies frozen embeddings for a class name and a richer sign description; the image student is trained with the ordinary class loss plus two prototype-alignment losses, then the text encoder and prototype banks are removed. The remaining deployed model is just the image backbone and classifier. The other panels test whether that semantic anchor survives perturbations: the RP2 sequence varies viewing distance, while the clean natural-light sign shows the visual domain before the attack is applied.

![LAMDA training diagram with frozen language prototypes supervising an image-only traffic-sign classifier](/assets/images/lamda-training-paper-figure.png)
*Fig 1: The text encoder and prototype banks are used as fixed train-time teachers and removed before deployment. | source: [LAMDA, Figure 2](https://arxiv.org/abs/2608.08815)*

![Figure 4 from Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles](/assets/images/distilling-vision-language-models-for-robust-traffic-sign-perception-in-autonomous-vehicles-source-figure-4.webp)
*Fig 2: An RP2 adversarial patch is evaluated on a Speed Limit 35 sign at four viewing distances, exposing how recognition robustness changes with distance. | source: [LAMDA, Figure 4](https://arxiv.org/abs/2608.08815)*

![Figure 1 from Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles](/assets/images/distilling-vision-language-models-for-robust-traffic-sign-perception-in-autonomous-vehicles-source-figure-1.webp)
*Fig 3: A naturally lit Speed Limit 60 sign provides one clean-domain traffic-sign example used alongside adverse-light and attack conditions. | source: [LAMDA, Figure 1](https://arxiv.org/abs/2608.08815)*


The full protocol is unusually controlled. GTSRB uses 43 classes and LISA uses its 16-class subset; all methods train only on benign data, and the evaluation spans ResNet-18, ResNet-34, Swin-T, and ViT-B/16 against shadow, natural-light, and printable RP2 perturbations. The authors sweep the two loss weights over eight nonzero settings plus the CE-only baseline, producing 72 trained models across datasets, backbones, and splits. With λ = µ = 1, LAMDA improves every adversarial split for all eight dataset–backbone combinations and improves clean accuracy in seven of eight. The largest gains are +12.50 percentage points on GTSRB shadow for ResNet-18, +13.20 on LISA natural-light for ResNet-34, and +11.30 on that split for Swin-T. No attack examples enter training, and the prototype banks are built only from training-set images.

The important boundary is the attacker. Shadows and natural-light masks use black-box optimization, while RP2 is a printable patch optimized against a victim dataset and transformed for physical viewing; the paper keeps those target models fixed during evaluation. An adaptive white-box attacker against the final student, or a broader set of sign classes and weather conditions, remains untested. The result is evidence for cheap train-time semantic anchoring, not a complete adversarial-defense guarantee.

## High-Level Takeaways

- LAMDA informs whether a traffic-sign model can borrow VLM semantics during training without carrying a language model into deployment.
- The training unit is an image/class pair supervised by classification plus two frozen prototype targets; inference is an ordinary vision backbone and classifier.
- The strongest evidence is cross-backbone, cross-dataset, and cross-attack consistency, including a small physical patch test.
- The decisive test is adaptive white-box evaluation with matched clean-data budgets. The conclusion would weaken if the prototype losses fail against attackers optimized for the deployed student.
