---
title: 'Cross-View Sequential Visual Localization with Spatio-Temporal Context Modeling for Autonomous Driving'
date: '2026-08-11T08:44:42.000Z'
section: paper-shorts
postSlug: cross-view-sequential-visual-localization-with-spatio-temporal-context-modeling-for-autonomous-driving
legacyPath: /paper shorts/2026/08/11/cross-view-sequential-visual-localization-with-spatio-temporal-context-modeling-for-autonomous-driving.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2026 – recurrent temporal context sharpens satellite candidates before cross-view localization refinement'
---

## 2026 – Cross-View Sequential Visual Localization with Spatio-Temporal Context Modeling for Autonomous Driving

**arXiv:** [2608.10660](https://arxiv.org/abs/2608.10660)

## Summary

> This work moves temporal context before cross-view matching rather than aggregating only a final fused feature. A recurrent module enhances the current ground-view feature with the previous state; hierarchical features then classify satellite-map candidate regions and refine the local offset. On CVIS, the reported mean error falls from 3.80 m for the strongest listed baseline to 1.57 m, but the contribution is tied to six-frame sequences, a particular map-crop protocol, and weak performance in several difficult real-road settings.

## Core Insights

### History sharpens the coarse candidate distribution

The model consumes a sequence of ground images and one satellite map. The current ground feature is the query, while the recurrent previous state supplies keys and values. The resulting context-enhanced coarse feature scores satellite-grid candidates; multi-level features provide local structure for a second-stage offset regressor. The ordering matters: temporal evidence reduces ambiguity before the fine regressor is restricted to selected candidate cells.

![Cross-view sequential localization pipeline with temporal context, coarse satellite-grid matching, and fine offset refinement](/assets/images/cross-view-sequential-visual-localization-paper-figure.webp)
*History first sharpens the coarse satellite-grid distribution; only the retained candidates reach fine localization. source: [Cross-View Sequential Visual Localization](https://arxiv.org/abs/2608.10660)*

The CVIS ablation traces that claim. A DINOv2 matching baseline reports 12.25 m mean error; adding multi-level features reduces it to 5.92 m, a position-aware update to 4.96 m, and full temporal context to 1.57 m. For the top-64 candidate mask, the full model covers 99.98% of ground-truth cells while retaining 17.73% of the 19-by-19 search grid. That is evidence for better candidate recall, not merely trajectory smoothing after localization.

### The transfer and field tests expose the remaining ambiguity

On CVIS test data, the full model reports 1.57 m mean error, 1.21 m median error, and 40.22% R@1 m. Direct transfer to KITTI-CVL reports 2.61 m mean error, improving to 2.27 m after target-domain fine-tuning. A real-vehicle evaluation without model updates reports 2.84 m mean error and 96.86% R@5 m across its scenarios. The same paper identifies intersections, elevated roads, and slopes as harder: elevated roads reach 3.67 m mean error, while uphill segments reach 4.21 m.

| Stage | Function | Main failure risk |
| --- | --- | --- |
| Recurrent temporal context | Resolves ambiguous current-frame cues | Carries past-state errors or fails under long interruptions. |
| Coarse satellite-grid classification | Keeps plausible map regions | A missed true cell prevents downstream recovery. |
| Fine offset regression | Refines within retained cells | Cannot correct a wrong coarse candidate. |
| Hierarchical features | Separates global semantics from local textures | Adds coupled design choices beyond temporal context alone. |

## High-Level Takeaways

- The central decision is where temporal context enters the pipeline. Here it improves the candidate distribution before cross-view fusion and offset regression, which is more diagnostic than treating time as a final smoothing pass.
- The CVIS ablation supports both hierarchical features and temporal context, but it does not fully isolate their interaction, sequence length, map-crop uncertainty, and backbone choice under one matched budget.
- The direct KITTI-CVL transfer and real-vehicle results are promising deployment checks, yet target-domain fine-tuning improves the former and difficult road topologies remain a clear boundary.
- A stronger deployment test would vary outages, seasonal imagery, long-horizon drift, and GNSS-prior quality, then compare temporal recurrence with explicit motion constraints at fixed latency.
- Cross-view localization is most fragile when a single frame has many map lookalikes; temporal context earns its cost when it keeps the true map region alive for refinement.
