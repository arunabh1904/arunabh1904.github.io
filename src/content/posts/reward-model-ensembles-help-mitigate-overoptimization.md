---
title: 'Reward Model Ensembles Help Mitigate Overoptimization'
date: '2023-10-04T09:00:00.000Z'
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

This paper asks whether uncertainty across reward models can identify the regions where policy optimization is exploiting a proxy. It trains ensembles and optimizes either the worst predicted reward or an uncertainty-penalized reward instead of trusting a single mean score.

## Paper Insights

The evaluation extends the synthetic gold-reward setup used by the reward-overoptimization scaling paper and adds 25% label noise. For best-of-$n$, conservative ensemble objectives nearly eliminate overoptimization in the reported setting and improve performance by as much as 70%. For PPO, ensembles consistently reduce overoptimization; combining them with a small KL penalty prevents it in the studied runs without sacrificing performance.

The ensemble is useful because disagreement provides a local warning about extrapolation. It is not a guarantee: models trained on the same data and architecture can share the same blind spot. In robot learning, useful diversity may require different sensor views, label sources, architectures, or structured state rather than random seeds alone.

| Objective | Behavior |
| --- | --- |
| Single reward model | Cheap, but easy to exploit outside its labeled support |
| Worst-case ensemble | Conservative where any member predicts low reward |
| Uncertainty-weighted ensemble | Trades predicted reward against disagreement |

## Decision Lens

This paper informs whether extra critic capacity should buy a larger single model or an ensemble that exposes epistemic uncertainty. The unit is a response scored by several reward models; the policy objective changes how those scores are normalized and aggregated. The measured gains are orthogonal to simply scaling one reward model in the synthetic study.

The missing control is diversity: compare independently seeded replicas with critics trained from different label sources and representations. At ten times the deployment shift, correlated errors can make the ensemble confidently wrong. The approach is falsified if ensemble disagreement fails to rank real robot failures or if a held-out human/ground-truth metric regresses while the conservative objective rises.

**Context:** Ensembles turn uncertainty into an optimization constraint rather than a dashboard metric.

**Limits:** Synthetic gold rewards and language outputs understate shared physical-perception failures.

**Takeaway:** An ensemble helps only when its members disagree for reasons related to the failures that matter.
