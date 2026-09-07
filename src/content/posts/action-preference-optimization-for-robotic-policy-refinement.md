---
title: 'Human-Assisted Robotic Policy Refinement via Action Preference Optimization'
date: '2025-06-08T00:00:00.000Z'
section: paper-shorts
postSlug: action-preference-optimization-for-robotic-policy-refinement
legacyPath: /paper shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html
tags:
  - Robotics
  - Preference Optimization
field: 'Robot Post-Training & Evaluation'
summary: "2025 – Human-Assisted Robotic Policy Refinement via Action Preference Optimization"
---

## 2025 – Human-Assisted Robotic Policy Refinement via Action Preference Optimization

**arXiv:** [2506.07127](https://arxiv.org/abs/2506.07127)

**Project:** [Action Preference Optimization](https://gewu-lab.github.io/action_preference_optimization/)

## Summary

> Action Preference Optimization (APO) turns a human takeover into a local preference signal for a VLA. The robot is allowed to act, a SpaceMouse operator corrects it when needed, and the trajectory records automatic actions, the failure window, and the corrective actions. APO then combines a binary preference objective with continuous-action error weights, so the VLA learns which physical decisions were wrong without requiring a matched rejected trajectory from the same state.

## Core Insights

![Action Preference Optimization pipeline from human-assisted deployment and interventions to adaptively weighted VLA fine-tuning](/assets/images/action-preference-optimization-for-robotic-policy-refinement-paper-figure.png)
*Fig 1: APO closes the deployment loop: intervention makes the task safe enough to collect data, and the labels plus decoded action errors determine the next update. | source: [Action Preference Optimization](https://arxiv.org/abs/2506.07127)*

### A takeover is evidence about a local action, not a clean preference pair

Real intervention data is temporally asymmetric. When the operator takes over at step $t$, the preceding $K=10$ actions are annotated as undesirable, the human-corrected actions as desirable, and the autonomous actions outside that window remain ordinary interaction data. The paper uses a binary desirability objective related to KTO and regularizes the updated policy against the reference checkpoint. This is a better fit for an irreversible robot than fabricating a noisy rejected action: once the robot has moved the object, the counterfactual “what would have happened from exactly the same state?” is usually unavailable.

The training mixture is intentionally balanced: 50% expert actions, 25% human-intervention actions, and 25% failure actions in each batch. That keeps the model anchored to the original task distribution while exposing it to the states where the deployed policy needs help. APO is therefore closer to targeted refinement than to learning from every action in a rescued episode.

### Action-token probability is not physical action error

An autoregressive VLA discretizes a continuous control vector into action tokens. Cross-entropy can say that one token is unlikely, but it cannot say whether its decoded end-effector motion misses by a millimeter or sends the gripper far away. APO first computes an L1 error in continuous action space, normalizes it, and uses the result to reweight the preference loss. High-error desirable samples receive more weight because the policy needs to learn that correction; undesirable samples near the failure action are emphasized so that the model learns to move away from the specific mistake.

This is the paper’s most VLA-specific idea. The preference label says which side of the intervention is better, while decoded action distance says how much the token-level update matters. The reference-policy constraint supplies the other half: preserving the broad pretrained behavior prevents the intervention subset from becoming a new, narrow behavior distribution.

![Figure 4 from APO: success and intervention frequency during lifelong refinement](/assets/images/action-preference-optimization-for-robotic-policy-refinement-source-figure-4.webp)
*Fig 2: APO’s Coffee_D0 and StackThree_D0 success rates rise over rollout iterations while human intervention frequency falls. | source: [APO, Figure 4](https://arxiv.org/abs/2506.07127)*

### The controlled results separate preference learning from generic replay

The main RoboMimic comparison fine-tunes OpenVLA from 300 expert demonstrations, collects 50 interaction trajectories per task, and evaluates 50 trials under three unseen seeds. APO reaches 60%, 54%, 46%, and 32% on Coffee_D0, StackThree_D0, ThreePieceAssembly_D0, and Square_D0, respectively, for a 48.0% mean. The base policy averages 40.5%; KTO averages 43.5%; and trajectory-preference TPO averages 41.5%. Dagger and weighted behavior cloning do not improve the mean. These numbers support the claim that the intervention labels and adaptive weighting together help; they do not show that any preference method is universally superior.

The disruption experiment is a useful boundary check. With only 20 interaction trajectories plus 20 original expert demonstrations, APO reaches a 28.0% mean across randomized stick position, gray background, and wooden-block texture, versus 21.3% for the base policy. On the original tasks after this disruption update, APO averages 45.3% versus 39.3% for the base, while ordinary behavior-cloning variants suffer larger forgetting. In the lifelong experiment the model is updated every 20 interaction rollouts; the curves in Figure 2 show success rising at the same time that the operator intervenes less often.

![Figure 6 from APO: learned recovery behaviors during rollout](/assets/images/action-preference-optimization-for-robotic-policy-refinement-source-figure-6.webp)
*Fig 3: The learned policy retries a failed grasp and makes successive gripper adjustments when an insertion is obstructed. | source: [APO, Figure 6](https://arxiv.org/abs/2506.07127)*

The real-world test makes the recovery story concrete. For square insertion, the base policy scores 65% in-distribution, 25% under position disruption, 10% under background disruption, and 25% under texture disruption; APO reaches 85%, 55%, 30%, and 55%. The paper also reports gains on the π0-FAST model, where APO reaches 76% Coffee_D0, 74% StackThree_D0, and 95% Insert Square. The work remains limited to autoregressive VLAs and manually operated interventions; the authors do not claim the objective transfers unchanged to diffusion or regression policies.

## High-Level Takeaways

- APO treats intervention as a precise data-collection event: the robot exposes a failure state, the human supplies a correction, and the label records only the local preference the event can support.
- The adaptive weight is the bridge between a discrete action-token loss and continuous control. It makes a physically large mistake matter more than a merely unlikely token.
- On four RoboMimic tasks, APO raises the mean from 40.5% for the base policy to 48.0%; in real square insertion it raises the in-distribution score from 65% to 85% and improves all three tested disruptions.
- The gains depend on preserving the pretrained distribution. A preference update trained only on rescued states can learn the rescue context rather than the earlier cause of failure.
- The evaluation covers autoregressive VLAs, four simulated tasks, and a small real-world intervention set. It leaves open how to label delayed causes, operator disagreement, and policies whose actions are not naturally tokenized.
