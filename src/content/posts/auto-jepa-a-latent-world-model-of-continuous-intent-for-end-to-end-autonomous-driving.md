---
title: 'Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving'
date: '2026-07-31T09:00:00.000Z'
section: paper-shorts
postSlug: auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/07/31/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - World Models
  - Planning
field: 'Motion Forecasting & Planning'
topics:
  - autonomy
  - learning
summary: '2026 – Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving'
---

## Summary

> Auto-JEPA predicts a latent representation of future ego motion instead of reconstructing the whole future scene. Four front-camera frames, four historical ego positions, and a route command feed a frozen V-JEPA 2 encoder and a Transformer predictor that emits eight future-intent tokens. The predicted latent retrieves 300 executable trajectories from a ground-truth-only memory; a scene scorer ranks them and a learned drivable-area gate filters failures. The result is a planning-oriented predictive state, not a general simulator of other agents.

## Core Insights

### The prediction target is future ego intent, not a complete world state

Auto-JEPA first trains a trajectory autoencoder on eight future `(x,y)` waypoints covering a four-second horizon at 0.5-second intervals. Its trajectory encoder maps each future into eight tokens of dimension 1,024; after the decoder learns to reconstruct the path, the decoder is discarded and the encoder is frozen. The driving model then predicts the same latent space from 256×256 front-camera frames, ego-motion history, and a route command. The visual encoder, trajectory encoder, and trajectory memory stay fixed while the history encoder, command encoder, and JEPA predictor learn the alignment.

This target is narrower than a dense world model but operationally sharper. The predictor does not need to preserve every visible object or forecast every agent. It only needs to encode the future ego maneuver well enough that a nearest-neighbor search returns useful geometry. That makes intent a planner interface: perception is judged by whether it changes the retrieved plan, not by whether it reconstructs the scene.

![Auto-JEPA predicts a future-trajectory latent and uses it to retrieve and select an executable path](/assets/images/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving-source-figure-3.webp)
*Fig 1: Training aligns the predicted intent with a frozen trajectory target; inference retrieves 300 candidates, then applies a scene scorer and drivable-area gate. | source: [Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving, Figure 3](https://arxiv.org/abs/2607.29031)*

### Retrieval supplies geometry that the latent predictor does not have to generate

The trajectory memory contains 110,335 ground-truth trajectory–latent pairs encoded by the same frozen trajectory encoder. Flat-cosine retrieval selects the top 300 candidates. A fine-tuned scene-conditioned scorer uses collision, drivable-area, time-to-collision, comfort, and ego-progress labels to rank them, while an independent gate rejects proposals that fail the drivable-area threshold. The division of labor is useful: the JEPA predictor says what kind of motion is appropriate, memory lookup instantiates a path that exists in the logged distribution, and selection chooses the safest available member of that local set.

The ablation makes the interface load-bearing. Replacing the predicted intent with a fixed latent medoid drops NAVSIM v1 PDMS from 91.3 to 52.6. Intent retrieval plus the gate reaches 87.6; adding the scorer reaches 91.3. Removing the gate changes PDMS only to 91.0 but lowers DAC from 98.3 to 97.9. Increasing the candidate pool from 1 to 200 to 300 raises PDMS from 87.6 to 91.1 to 91.3, suggesting that diversity matters until the memory begins to saturate.

### Selective visual dependence is measured, but the benchmark remains non-reactive

Auto-JEPA reaches 91.3 PDMS on NAVSIM v1 with one front camera, 98.4 no-at-fault collision, 98.3 drivable-area compliance, and 100.0 comfort. On NAVSIM v2 it reports 89.1 EPDMS under the updated official evaluator with human-behavior filtering; the original evaluator gives 85.6 EPDMS, so the scores should be kept separate.

The source Figure 1 illustrates the intended behavior. Occluding a non-interacting adjacent vehicle leaves the retrieved path essentially unchanged, while occluding the interacting lead vehicle moves the selected trajectory by 1.76 m in the shown scene. Across the full validation split, masking dynamic-agent regions changes the intent by 2.97 times as much as equal-area random masks and produces a larger change in 71.1% of scenes. This is stronger than an attention heatmap: it measures an intervention on both the predicted representation and the selected trajectory. It still does not prove causal safety, and it does not test whether a newly required maneuver exists in the fixed memory.

![Occluding an interacting lead vehicle changes the selected trajectory more than occluding a non-interacting vehicle](/assets/images/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 2: The qualitative intervention contrasts a non-interacting vehicle, which leaves the plan unchanged, with an interacting lead vehicle, whose occlusion shifts the plan. | source: [Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving, Figure 1](https://arxiv.org/abs/2607.29031)*

The evaluator is NAVSIM, a data-driven non-reactive benchmark. Auto-JEPA therefore demonstrates a strong camera-only open-loop planning interface. Its limitations are structural: it cannot synthesize a maneuver absent from 110,335 logged trajectories, it does not predict other agents, and the scorer and gate must remain calibrated as route and traffic distributions change.

## High-Level Takeaways

- Auto-JEPA is a good fit when a planner needs a strong proposal from a bounded logged-motion vocabulary and does not need a full scene simulator.
- The retrieval memory is part of the model's capability: candidate coverage and rare-maneuver recall matter as much as latent prediction quality.
- The dynamic-agent occlusion study shows planning-relevant visual dependence, while the non-reactive NAVSIM protocol leaves closed-loop interaction untested.
- A fair next comparison would hold visual backbone, data, scorer capacity, and candidate budget fixed across waypoint regression, intent retrieval, and intent-conditioned generation.
- Its central bet is that an action-relevant future latent can discard much of the world while retaining enough structure to choose a safe ego trajectory.
