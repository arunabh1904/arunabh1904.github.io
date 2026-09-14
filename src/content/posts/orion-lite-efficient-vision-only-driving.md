---
title: "Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models"
date: "2026-04-09T00:00:00.000Z"
section: paper-shorts
postSlug: orion-lite-efficient-vision-only-driving
legacyPath: /paper shorts/2026/04/09/orion-lite-efficient-vision-only-driving.html
tags: ["Distillation", "Autonomous Driving"]
field: "Autonomous Driving: VLA & Planning"
summary: "2026 – Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models"
---

## 2026 – Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models

**Paper:** [arXiv:2604.08266](https://arxiv.org/abs/2604.08266) · [PDF](https://arxiv.org/pdf/2604.08266)

## Summary

> Orion-Lite replaces ORION's 7B language module with a roughly 0.1B decoder and improves Bench2Drive Driving Score from 77.7 to 80.6. Overall latency falls from 806 to 267 ms on an RTX A6000. The transfer matches latent planning features and uses ground-truth trajectories; it does not show that teacher-generated language alone can train an equally capable small driver.

## Core Insights

### Distill the interface the planner actually consumes

[ORION](/paper%20shorts/2025/03/25/orion-a-holistic-end-to-end-autonomous-driving-framework-by-vision-language-instructed-action-generation.html) produces latent planning tokens that feed a VAE planner. Orion-Lite preserves the teacher-trained vision encoder and temporal QT-Former, freezes both, and replaces the language module with a six-layer transformer decoder. A learned planning query attends to projected scene tokens. An output projection restores the dimensionality expected by the planner.

The student removes text prompts from this path but retains driving commands and ego-state information with the visual context. “Vision-only” here describes the absence of an inference-time language model and LiDAR; it does not mean the planner receives no route or vehicle-state conditioning. The quoted 0.1B and 7B sizes refer to the replaced reasoning modules, not total system parameter counts.

The overview shows two simultaneous targets. The student planning token matches the frozen teacher's token, and the planner is trained against driving outputs. Both paths matter in the ablation.

![Latent planning-token distillation combined with trajectory supervision; source Figure 1](/assets/images/orion-lite-source-figure-1.webp)
*Fig 1: Orion-Lite retains the teacher-trained visual and temporal representation, replacing the language module with a shallow decoder. Joint feature matching and trajectory supervision train the student planning path. | source: [Paper, Figure 1](https://arxiv.org/abs/2604.08266)*

[View full-size figure](/assets/images/orion-lite-source-figure-1.webp)

The feature loss is an L1 distance averaged over batch and planning-token channels. The decoder and VAE planner are updated together; the inherited visual modules stay frozen. The paper trains on the standard Bench2Drive base set of 1,000 clips for twenty epochs, taking about twenty hours on one RTX A6000. Teacher training and the inherited perception representation are additional costs.

### The strongest result combines two training signals

| Student supervision | Driving Score | Success rate | Mean multi-ability score |
| --- | ---: | ---: | ---: |
| Trajectory ground truth only | 73.9 | 50.0% | 47.7% |
| Latent mimic loss only | 76.0 | 50.7% | 53.3% |
| Both | 80.6 | 55.5% | 60.5% |
| ORION teacher | 77.7 | 54.6% | 54.7% |

Tables 2 and 3 show the combination outperforming either student loss alone. Table 1 also reports a regression: average open-loop L2 error rises from the teacher's 0.68 m to 0.79 m despite the better closed-loop score. The student is not uniformly superior on every metric. A four-layer decoder obtains a higher Driving Score than six layers in the depth ablation, but six layers offer better mean multi-ability performance and become the default.

The reasoning module is reported as 150 times faster, but the complete system is only about three times faster because perception remains expensive. GPU memory falls from 31 GB to 8 GB. Those are useful hardware-specific measurements, not evidence of real-time suitability on an onboard accelerator.

Table 6 separately studies encoder initialization and shows much stronger results with teacher-trained features than a frozen generic EVA-02-L encoder. Its trajectory-only ORION initialization score differs from the trajectory-only row in Table 3, so the two ablations should not be merged into one controlled comparison. The paper's validation is confined to Bench2Drive; general language understanding, long-tail novelty, and physical deployment remain outside the demonstrated result.

## High-Level Takeaways

- Distill the representation consumed by the action head and retain task supervision; “use a stronger teacher” is not a complete training recipe.
- Charge the inherited visual encoder and teacher preparation to the method, even when only a small decoder is trained in the student stage.
- Evaluate latency for the complete pipeline and report metric regressions. A faster reasoning module can leave perception as the dominant deployment cost.
