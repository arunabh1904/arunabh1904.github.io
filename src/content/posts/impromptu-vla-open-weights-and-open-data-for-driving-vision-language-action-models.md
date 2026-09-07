---
title: 'Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models'
date: '2025-05-29T17:59:46.000Z'
section: paper-shorts
postSlug: impromptu-vla-open-weights-and-open-data-for-driving-vision-language-action-models
legacyPath: /paper shorts/2025/05/29/impromptu-vla-open-weights-and-open-data-for-driving-vision-language-action-models.html
tags:
  - Other
field: 'Autonomous Driving: VLA & Planning'
summary: "2025 – Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models"
---
## 2025 – Impromptu VLA

**arXiv:** [2505.23757](https://arxiv.org/abs/2505.23757)

**Code, data, and models:** [ahydchh/Impromptu-VLA](https://github.com/ahydchh/Impromptu-VLA)

## Summary

> Impromptu VLA treats rare driving failures as a data-distribution problem. It filters more than two million clips from eight public datasets into about 80,000 verified clips covering four kinds of unstructured road scene, then fine-tunes Qwen2.5-VL with scene, traffic, intent, and trajectory supervision. On the paper's NeuroNCAP simulation, the 3B model's average score rises from 1.77 to 2.15 out of 5 and collision rate falls from 72.5% to 65.5% relative to nuScenes-only fine-tuning. The open-loop nuScenes gain is smaller and the work does not establish real-vehicle improvement.

## Core Insights

### The dataset is a targeted distribution

The paper's central move is selection. The authors start with more than two million clips occupying over 10 TB from Mapillary, ONCE, NAVSIM, nuScenes, Waymo, Argoverse-V2, KITTI, and IDD. The released set keeps about 80,000 clips, with the largest contributions from Mapillary (22,062), NAVSIM (18,600), ONCE (18,093), and nuScenes (11,370). The four labels are roads with unclear boundaries, temporary traffic-rule changes, unconventional dynamic obstacles, and challenging road conditions. This gives the model a reason to spend capacity on scenes where the normal road prior is unreliable.

The curation also makes the sources comparable. Clips are aligned to 2 Hz, keep 1.5 seconds of past context and 5 seconds of future context, and can first be assembled into local packs of up to 15 seconds before keyclip selection. Each retained example carries scene description, traffic-signal state, vulnerable-road-user identification, motion intention, a meta-action plan, a planning explanation, and a future trajectory. That combination matters: the dataset is not only a rare-image collection, and the language labels are tied to the action that follows.

The taxonomy is generated with Qwen2.5-VL-72B and then reviewed by people. On 200 nuScenes images, the reported classification F1 is 0.90 for temporary rules, 0.81 for unconventional obstacles, and 0.91 for challenging road conditions; unclear boundaries are too rare in that check for a useful F1. All generated annotations receive accept/reject review or minor correction. The validation is evidence that the labels can be made usable, while the dependence on a VLM for the first pass leaves room for model-specific blind spots.

![Visual abstract of the Impromptu VLA dataset and its four corner-case categories](/assets/images/impromptu-vla-open-weights-and-open-data-for-driving-vision-language-action-models-source-figure-1.webp)
*Fig 1: The paper's visual abstract connects the four scene categories to the 80K-clip dataset and the reported NeuroNCAP and trajectory results. | source: [Impromptu VLA, Figure 1](https://arxiv.org/abs/2505.23757)*

The visual abstract is useful because it shows the intended causal chain: category definition changes the data mixture, the mixture changes what the VLA sees during adaptation, and the evaluation checks both safety-like behavior and geometric prediction. It does not show a new architecture. The proposed intervention is the training distribution and its annotations.

### What the closed-loop comparison isolates

For the main safety test, the authors fine-tune Qwen2.5-VL-3B either directly on nuScenes or first on Impromptu VLA and then on nuScenes. On NeuroNCAP's closed-loop nuScenes simulation, the Impromptu sequence raises average NeuroNCAP score from 1.77 to 2.15, while collision rate drops from 72.5% to 65.5%. The improvement is not uniform: frontal collision rate falls from 73.0% to 59.0%, while static-scene collision rate moves from 68.0% to 70.0%. That pattern is consistent with the dataset helping the model handle unusual interacting road users and rules, rather than simply making every scenario safer.

The open-loop result is more modest but helps locate the trade-off. For the 3B model, average nuScenes L2 error changes from 0.34 m with nuScenes-only fine-tuning to 0.30 m after Impromptu pretraining; for 7B it changes from 0.32 m to 0.30 m. The 3B result is close to the listed EMMA+ average of 0.29 m. These are logged-trajectory comparisons, so they support better prediction under the dataset's test distribution; the NeuroNCAP result tests what happens after the policy's action changes the simulated state.

![Dataset source counts, category distributions, and trajectory coverage](/assets/images/impromptu-vla-open-weights-and-open-data-for-driving-vision-language-action-models-source-figure-2.webp)
*Fig 2: The source paper compares how the eight datasets contribute clips, how the four categories are distributed, and how the retained trajectories cover driving behavior. | source: [Impromptu VLA, Figure 2](https://arxiv.org/abs/2505.23757)*

Figure 2 makes the selection decision inspectable. The source datasets do not contribute equally, and a large clip count alone does not guarantee coverage of the four corner cases. The distribution plots are therefore part of the result: they show which gaps the benchmark is trying to fill before any model is trained.

### The boundary is still data attribution

The paper shows that a rare-scene curriculum can improve a VLM-based planner, but it does not isolate which ingredient is responsible. There is no matched experiment that separately removes the taxonomy, the human review, the language explanations, and the trajectory labels while keeping the selected clips fixed. The NeuroNCAP test is simulated and nuScenes-based; the open-loop test cannot measure recovery after an incorrect action. The authors also acknowledge possible bias from Qwen2.5-VL-generated classifications and annotations.

The practical decision is therefore narrow: use targeted, multi-task corner-case data when the failure mode is underrepresented, and verify it with a closed-loop test. The evidence would weaken that decision if a category-balanced image-only set, or a random 80K subset with the same training budget, matched the gains. That comparison is the missing control.

## High-Level Takeaways

- Each training example is a 2 Hz keyclip with past context, a future trajectory, and several language labels; selection is the intervention.
- Human review and category stratification turn an 80K subset into a targeted safety curriculum, but the paper does not separate their effects.
- NeuroNCAP improves more clearly than open-loop L2, which is evidence for a behavior change under simulation rather than a universal accuracy gain.
- The missing control is an equal-size random or taxonomy-matched subset; until it is run, curation, annotation, and source overlap remain coupled.
