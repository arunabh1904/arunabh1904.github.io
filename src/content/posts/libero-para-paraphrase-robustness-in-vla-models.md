---
title: 'LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models'
date: '2026-03-30T09:00:00.000Z'
section: paper-shorts
postSlug: libero-para-paraphrase-robustness-in-vla-models
legacyPath: /paper shorts/2026/03/30/libero-para-paraphrase-robustness-in-vla-models.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2026 – LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models"
---

## 2026 – LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models

**arXiv:** [2603.28301](https://arxiv.org/abs/2603.28301)

**GitHub:** [cau-hai-lab/LIBERO-Para](https://github.com/cau-hai-lab/LIBERO-Para)

LIBERO-Para changes the instruction while holding the intended task fixed. It varies action expressions and object references independently, then measures whether a VLA's apparent language grounding survives phrasing that was absent from downstream fine-tuning.

## Paper Insights

Across seven VLA configurations from 0.6B to 7.5B parameters, the paper reports 22–52 percentage-point drops under paraphrasing. Object-level lexical substitutions drive much of the degradation, and 80–96% of failures arise from planning-level trajectory divergence rather than low-level execution. The policy often identifies the wrong task before motor control begins.

The benchmark also introduces PRIDE, which models paraphrase difficulty using semantic and syntactic factors. That matters because average success can be dominated by easy rewordings and hide inconsistent grounding.

| Variation | Diagnostic target |
| --- | --- |
| Action expression | Stable skill selection across “turn on,” “fire up,” and “activate” |
| Object reference | Preservation of object identity under synonyms |
| PRIDE difficulty | Success stability as linguistic distance grows |

## Decision Lens

LIBERO-Para informs whether a post-trained VLA learned task semantics or memorized the fine-tuning instruction surface. Its unit is a set of paraphrases mapped to one closed-loop task. Visual state and controller remain fixed so linguistic variation is the causal intervention.

The results establish a large robustness gap in current configurations, but generated paraphrases may not match how real users speak. At ten times the language diversity, ambiguity and legitimate task reinterpretation complicate the oracle. The benchmark's claim would weaken if human-equivalent commands are not semantically interchangeable or if instruction augmentation closes LIBERO-Para without improving natural user interactions.

**Context:** LIBERO-Para reveals a failure hidden by the original LIBERO protocol's identical train/eval instructions.

**Limits:** Language robustness is only one axis; success can still hide visual or control shortcuts.

**Takeaway:** A VLA has not grounded an instruction if a harmless paraphrase changes the plan.
