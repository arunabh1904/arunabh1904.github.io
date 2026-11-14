---
title: 'Are Emergent Abilities of Large Language Models a Mirage?'
date: '2023-04-28T00:00:00.000Z'
section: paper-shorts
postSlug: are-emergent-abilities-of-large-language-models-a-mirage
legacyPath: /paper shorts/2023/04/28/are-emergent-abilities-of-large-language-models-a-mirage.html
tags: [Scaling Laws, Evaluation]
field: 'Language Models'
summary: '2023 – Are Emergent Abilities of Large Language Models a Mirage?'
---

## 2023 – Are Emergent Abilities of Large Language Models a Mirage?

**Paper:** [Schaeffer et al., arXiv:2304.15004v2](https://arxiv.org/abs/2304.15004v2)

## Summary

> A sharp benchmark curve can be produced by the scoring rule even when model outputs improve smoothly. Schaeffer et al. test this explanation by rescoring fixed outputs, increasing evaluation resolution, and constructing analogous effects in vision models. The result challenges the interpretation of several published emergence claims; it does not prove that models can never acquire qualitatively new behavior.

## Core Insights

### Complete success compounds small errors

[Wei et al.](/paper%20shorts/2022/06/15/emergent-abilities-of-large-language-models.html) define emergence through a jump from near-chance to useful task performance. This paper separates the task from the metric used to score it. Under an illustrative independent-token approximation, if each of $L$ tokens is correct with probability $p$, exact-sequence accuracy is $p^L$. A smooth increase in $p$ can therefore produce a steep-looking rise in complete-answer success, especially for long targets.

Token edit distance instead gives graded credit and changes approximately with the number of errors. The independence and constant-error assumptions are simplifications; the appendix does not model realistic dependencies or output-length errors fully. The value of the toy model is that it predicts interventions on real measurements, not that it exactly describes autoregressive generation.

### Keep the outputs fixed and change the measurement

The arithmetic experiment uses four InstructGPT/GPT-3 sizes, from 350M to 175B parameters, on two-shot multiplication of two-digit integers and addition of four-digit integers. Exact-match scores look abrupt for longer answers. Rescoring the same outputs with token edit distance reveals gradual improvement. Increasing the test set also resolves small models' nonzero above-chance accuracy, showing that a measured zero can reflect insufficient sample resolution.

The LaMDA comparison makes the intervention visible on classification tasks.

![Source Figure 6 left panel shows near-flat multiple-choice grades before their final rise.](/assets/images/emergent-abilities-mirage-source-figure-6a.png)
*Fig 1: The original multiple-choice grade discards probability improvement until the correct option becomes the highest-scoring answer. These are the left-hand plots of source Figure 6, shown separately for readable comparison. | source: [Paper, Figure 6](https://arxiv.org/html/2304.15004v2#S4.F6)*

![Source Figure 6 right panel shows gradual progress when the same tasks use negative Brier score.](/assets/images/emergent-abilities-mirage-source-figure-6b.png)
*Fig 2: Negative Brier score exposes graded changes in the predicted probabilities on the same tasks and model family. Changing the metric changes the apparent scaling curve without changing the evaluated model outputs. | source: [Paper, Figure 6](https://arxiv.org/html/2304.15004v2#S4.F6)*

[Open the grade plots](/assets/images/emergent-abilities-mirage-source-figure-6a.png) or [the Brier-score plots](/assets/images/emergent-abilities-mirage-source-figure-6b.png) at full size.

### The broader evidence is about measurement sensitivity

In the BIG-Bench meta-analysis, more than 92% of the hand-annotated emergence cases use exact string match or multiple-choice grade. That concentration is consistent with the hypothesis, although it is weaker evidence than rescoring fixed outputs: the benchmark's task and metric choices are themselves selected, and not every original model's outputs are accessible.

The authors also manufacture abrupt-looking curves in vision. A reconstruction-error threshold makes smooth autoencoder improvement look like a sudden capability; requiring every item in a group to be classified correctly creates the same effect for image classifiers. These controls show that the visual pattern is insufficient to identify a new internal mechanism.

Exact success can still be the right deployment metric. A nearly correct program or arithmetic answer may be unusable. My reading is that we need both views: complete success for the application, and a graded measure to diagnose how the model is improving. The paper explicitly leaves open genuine emergent behavior; it rejects treating a thresholded, sparsely sampled score curve as proof of it.

## High-Level Takeaways

- Score transformations can turn smooth progress into sharp-looking curves.
- Rescoring fixed outputs is a stronger control than comparing different model families.
- Rare successes require enough evaluation samples to distinguish low probability from zero.
- A task's operational success criterion and a scientific measure of learning can serve different purposes.
