---
title: 'LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models'
date: '2026-03-30T00:00:00.000Z'
section: paper-shorts
postSlug: libero-para-paraphrase-robustness-in-vla-models
legacyPath: /paper shorts/2026/03/30/libero-para-paraphrase-robustness-in-vla-models.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2026 – LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models"
---


**arXiv:** [2603.28301](https://arxiv.org/abs/2603.28301)

**GitHub:** [cau-hai-lab/LIBERO-Para](https://github.com/cau-hai-lab/LIBERO-Para)

## Summary

> LIBERO-Para changes the instruction while holding the intended task fixed. It varies action expressions and object references independently, then measures whether a VLA's apparent language grounding survives phrasing that was absent from downstream fine-tuning.

## Core Insights

![LIBERO-Para controlled evaluation design varying action and object paraphrases between fine-tuning and testing](/assets/images/libero-para-paraphrase-robustness-in-vla-models-paper-figure.png)
*Fig 1: Shows why the benchmark is diagnostic: action wording and object wording vary on separate axes, letting a failure be localized to linguistic generalization rather than task execution alone. | source: [LIBERO-Para](https://arxiv.org/abs/2603.28301)*

![Figure 1 from LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models](/assets/images/libero-para-paraphrase-robustness-in-vla-models-source-figure-1.webp)
*Fig 2: Illustration of paraphrase robustness gap under data-scarce fine-tuning: VLA models can overfit to seen instruction phrasings during fine-tuning and fail to generalize to paraphrased variants at deployment. | source: [LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models](https://arxiv.org/abs/2603.28301)*

![Figure 3 from LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models](/assets/images/libero-para-paraphrase-robustness-in-vla-models-source-figure-3.webp)
*Fig 3: Examples of axis-specific paraphrases. Object variations modify target object references (e.g., same-polarity substitution, addition), while action variations cover lexical, structural, and pragmatic realizations grounded in established taxonomies. | source: [LIBERO-Para: A Diagnostic Benchmark for Paraphrase Robustness in VLA Models](https://arxiv.org/abs/2603.28301)*


### The benchmark makes language the intervention

LIBERO-Para evaluates seven VLA configurations from 0.6B to 7.5B parameters, spanning four architecture families. It generates 4,092 paraphrases across 43 action and object variation types, then keeps the scene, controller, and intended task fixed. Each success rate is averaged over five seeds for a task–paraphrase configuration; the appendix does not report standard deviations. That setup turns “language understanding” into a testable change: if a policy fails, the visual scene and motor target have not moved with the wording.

The drop is large even when the original LIBERO instruction is solved reliably. On LIBERO-Goal, OpenVLA-OFT falls from 97.9% on the original wording to 64.7% on LIBERO-Para; Xiaomi-Robotics-0 drops from 98.8% to 76.0%. The strongest and weakest models differ, but none is invariant to the intervention.

| Variation | Diagnostic target |
| --- | --- |
| Action expression | Stable skill selection across “turn on,” “fire up,” and “activate” |
| Object reference | Preservation of object identity under synonyms |
| PRIDE difficulty | Success stability as linguistic distance grows |

### The failure is usually a plan change

The paper attributes 80–96% of failures to planning-level trajectory divergence rather than low-level execution. That distinction matters: a robot can move smoothly while pursuing the wrong object or action. The largest reported original-to-paraphrase drops range from 22.8 percentage points for Xiaomi-Robotics-0 to 51.9 for VLA-Adapter, so parameter count alone does not explain the gap.

PRIDE makes the aggregate success number more diagnostic. With the default $\alpha=0.5$, it scores a paraphrase using semantic and syntactic factors, $PD = 1 - [\alpha S_K + (1-\alpha)S_T]$, and assigns zero to a failed sample. The paper reports 8.4–22.0% success overestimation when paraphrases are treated as an undifferentiated average, with absolute Pearson correlation $|r|=0.802$ between PRIDE difficulty and degradation. In the compositional ablation, original instructions reach 95.0% average success, while action-only, object-only, and combined paraphrases reach 82.1%, 78.6%, and 52.4%. The combined loss is 42.6 points, larger than either isolated intervention.

The benchmark therefore separates a controller that can execute a task from one that preserves the task identity under language variation. Its limitation is equally concrete: generated paraphrases are judged equivalent by the benchmark construction, while real users may introduce ambiguity or a legitimate change in intent.

## High-Level Takeaways

- LIBERO-Para informs whether a post-trained VLA learned task semantics or memorized the fine-tuning instruction surface. Its unit is a set of paraphrases mapped to one closed-loop task. Visual state and controller remain fixed so linguistic variation is the causal intervention.
- The original LIBERO protocol can hide instruction memorization because train and evaluation wording is identical; LIBERO-Para makes that shortcut visible.
- The 42.6-point compositional drop is more revealing than a single average: action and object variation interact when the policy must preserve both task identity and referenced entities.
- PRIDE is useful when deciding which paraphrase families to add to evaluation, but it remains a benchmark score rather than evidence that a command is natural or unambiguous for a human.
- A VLA has not robustly grounded an instruction if a semantically equivalent rewrite changes the plan before motor execution begins.
