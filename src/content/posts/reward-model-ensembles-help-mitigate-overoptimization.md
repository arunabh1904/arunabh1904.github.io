---
title: 'Reward Model Ensembles Help Mitigate Overoptimization'
date: '2023-10-04T00:00:00.000Z'
section: paper-shorts
postSlug: reward-model-ensembles-help-mitigate-overoptimization
legacyPath: /paper shorts/2023/10/04/reward-model-ensembles-help-mitigate-overoptimization.html
tags:
  - Alignment
  - Reward Models
field: 'Alignment & Post-Training'
summary: "2023 – Reward Model Ensembles Help Mitigate Overoptimization"
---

## 2023 – Reward Model Ensembles Help Mitigate Overoptimization

**arXiv:** [2310.02743](https://arxiv.org/abs/2310.02743)

**Conference:** ICLR 2024

## Summary

> This paper asks whether uncertainty across reward models can identify the regions where policy optimization is exploiting a proxy. It trains ensembles and optimizes either the worst predicted reward or an uncertainty-penalized reward instead of trusting a single mean score.

## Core Insights

![RLHF pipeline comparing a single proxy reward model with an ensemble used during policy optimization](/assets/images/reward-model-ensembles-help-mitigate-overoptimization-paper-figure.png)
*Fig 1: A schematic of the intervention: keep the ordinary SFT and preference-data stages, then optimize the policy against an ensemble of proxy reward models instead of one proxy. | source: [Reward Model Ensembles, Figure 1](https://arxiv.org/abs/2310.02743)*

![Figure 2 from Reward Model Ensembles Help Mitigate Overoptimization](/assets/images/reward-model-ensembles-help-mitigate-overoptimization-source-figure-2.webp)
*Fig 2: As optimization moves farther from the initial policy in KL divergence, the proxy score keeps rising after the gold score has begun to fall—the characteristic overoptimization pattern. | source: [Reward Model Ensembles, Figure 2](https://arxiv.org/abs/2310.02743)*

![Figure 3 from Reward Model Ensembles Help Mitigate Overoptimization](/assets/images/reward-model-ensembles-help-mitigate-overoptimization-source-figure-3.webp)
*Fig 3: In best-of-$n$ sampling, ensemble objectives sustain higher gold reward than a single reward model as policy KL grows, while the proxy reward continues upward and exposes overoptimization. | source: [Reward Model Ensembles, Figure 3](https://arxiv.org/abs/2310.02743)*


The evaluation extends the synthetic gold-reward setup used by the reward-overoptimization scaling paper and adds 25% label noise. For best-of-$n$, conservative ensemble objectives nearly eliminate overoptimization in the reported setting and improve performance by as much as 70%. For PPO, ensembles consistently reduce overoptimization; combining them with a small KL penalty prevents it in the studied runs without sacrificing performance.

The ensemble is useful because disagreement provides a local warning about extrapolation. It is not a guarantee: models trained on the same data and architecture can share the same blind spot. In robot learning, useful diversity may require different sensor views, label sources, architectures, or structured state rather than random seeds alone.

| Objective | Behavior |
| --- | --- |
| Single reward model | Cheap, but easy to exploit outside its labeled support |
| Worst-case ensemble | Conservative where any member predicts low reward |
| Uncertainty-weighted ensemble | Trades predicted reward against disagreement |

### Decision test and boundary

The design decision is whether extra critic capacity should buy one larger reward model or an ensemble whose disagreement constrains optimization. The synthetic gold-reward study reports that conservative ensemble objectives reduce overoptimization under 25% label noise and can improve best-of-$n$ performance by as much as 70%; PPO benefits when ensemble uncertainty is combined with a small KL penalty. Those results are orthogonal to simply scaling one critic, but members trained on the same data can share the same blind spot. The decisive test uses independently seeded critics with different label sources or representations, then checks whether disagreement ranks real failures and whether a held-out human or ground-truth metric improves while the proxy rises. In robotics, synthetic language rewards understate shared physical-perception errors. An ensemble helps only when its disagreement tracks the failures that matter.
