---
title: "Can LVLMs Obtain a Driver's License?"
date: '2024-09-01T04:00:00.000Z'
section: paper-shorts
postSlug: can-lvlms-obtain-a-drivers-license-idkb
legacyPath: /paper shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html
tags:
  - Other
field: 'Autonomous Driving: VLMs & Evaluation'
summary: IDKB tested whether vision-language models know explicit driving rules, not just visual scene facts.
---
## 2024 - Can LVLMs Obtain a Driver's License?

**arXiv:** [2409.02914](https://arxiv.org/abs/2409.02914)

**Summary:** This paper introduces IDKB, an Interactive Driving Knowledge Base for evaluating whether LVLMs understand traffic rules and driving theory. It includes official handbook knowledge, exam-style questions, and applied road scenarios.

The result is a useful warning: a model may recognize cars and pedestrians but still fail rule-based reasoning that every licensed human driver is expected to know.

## Paper Insights

IDKB tests whether LVLMs know driving rules and applied traffic knowledge. The benchmark covers signs, laws, exam-style questions, and scenario reasoning, then evaluates 15 representative LVLMs. Its central claim is that visual-language capability is not the same as driving competence: a model may recognize a road scene but still choose an illegal or unsafe action. The benchmark is valuable because autonomous driving needs rule knowledge, not only perception. The caveat is scope: passing IDKB would show specialized knowledge, but it would not prove planning, control, or closed-loop safety.

![Figure 1: Performance of 15 representative Large Vision-Language Models on IDKB, evaluated by three driving knowledge understanding metrics from Can LVLMs Obtain a Driver's License?](/assets/images/can-lvlms-obtain-a-drivers-license-idkb-paper-figure.png)
_Figure 1: Performance of 15 representative Large Vision-Language Models on IDKB, evaluated by three driving knowledge understanding metrics. From the [Can LVLMs Obtain a Driver's License? paper](https://arxiv.org/abs/2409.02914), via arXiv HTML._

**What to look at:**
- IDKB focuses on explicit driving rules and written-test knowledge.
- The benchmark includes handbooks, exams, and scenario QA.
- Fine-tuning gains show that general VLM pretraining does not guarantee domain rules.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Dataset | 1M+ driving knowledge items | Covers rules, exams, and applied scenarios. |
| Evaluation | 15 LVLMs | Tests whether general models know driving theory. |
| Main failure | Missing specialized rule knowledge | Perception alone is not driving competence. |

## Decision Lens

IDKB informs whether a driving LVLM needs explicit traffic-rule knowledge in addition to visual scene description. Its atomic item is a rule-grounded image-question pair that tests recognition, regulation recall, and application of the rule to a scene.

The benchmark distinguishes legal knowledge from generic visual fluency, but written-test competence is not closed-loop driving competence. The missing comparison controls for text-only rule memorization by using counterfactual scenes and jurisdiction changes. At 10× rule coverage, contradictory local regulations and rare signage make annotation and retrieval the bottlenecks. The licensing analogy would fail if high IDKB scores did not predict correct decisions on unseen rule-scene combinations.

**Context:** Driving competence is not only perception. It is perception plus rule knowledge plus judgment under context.

**Takeaway:** Autonomous driving VLMs need a written test as well as a road test.
