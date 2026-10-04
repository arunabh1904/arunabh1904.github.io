---
title: 'Post-Training for Robotics'
date: '2026-07-16T10:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: post-training-vision-language-action-models-zero-to-hero
legacyPath: /blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html
tags:
  - Robotics
  - Post-Training
  - Vision-Language-Action
  - Reinforcement Learning
summary: How robot post-training moved from behavior cloning and interventions to action-aware preference learning, process critics, interactive RL, and deployment-scale policy improvement.
---

# Post-Training for Robotics

_Updated August 22, 2026._

Robot post-training has developed along three branches: adapting the action decoder with demonstrations, collecting corrections in states reached by the policy, and optimizing rewards from fresh rollouts. They address different limits of behavior cloning: a slow output interface, missing recovery states, and the inability to improve beyond the collected actions.

The feedback determines which branch is available. DAgger queries an expert on learner states. Preference methods use judgments about recorded behavior. Process critics estimate progress within a trajectory. Interactive RL collects new attempts after each update. Their comparisons depend on how accurately the feedback identifies the action or transition responsible for the outcome.

This is Part III of the series. [Part I](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) asks what visual evidence the model preserves. [Part II](/blog/2026/07/15/omni-model-pretraining-decisions.html) asks how semantics, dynamics, and motor priors enter the policy. This part starts when the pretrained policy reaches deployment and begins creating its own data.

## Behavior cloning established the baseline

Supervised fine-tuning, or behavior cloning, provides the baseline. Given an observation and instruction, the policy maximizes the likelihood of the expert action:

$$
\mathcal{L}_{\text{BC}}(\theta)
=-\mathbb{E}_{(o,\ell,a)\sim D_E}
\log \pi_\theta(a\mid o,\ell).
$$

The action $a$ may be one discrete bin, a sequence of FAST tokens, a continuous chunk, or a diffusion denoising target. This choice determines the loss, inference path, control rate, and level at which a later correction can assign credit.

The action head can change while retaining the pretrained visual-language model. [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) replaces serial action-token decoding with parallel chunks and continuous L1 regression. In its single-view LIBERO comparison, success rises from 76.5% for the reported OpenVLA baseline to 90.2% with parallel decoding and chunking, then 95.3% with continuous L1 outputs. The often-quoted 97.1% result also adds inputs and changes the comparison setting. This is evidence that adaptation can alter the control interface without rebuilding pretraining.

![OpenVLA-OFT contrasts serial token decoding with parallel continuous action chunks](/assets/images/openvla-oft-optimizing-speed-and-success-paper-figure.png)
*OpenVLA-OFT separates decoding order from action parameterization. Parallel chunks reduce repeated language-model calls; regression changes the target from discrete bins to continuous actions. Source: [OpenVLA-OFT, Figure 2](https://arxiv.org/abs/2502.19645).*

SFT remains limited to the states represented in its training data. It can learn a better action for those states, but does not determine which policy-induced states should enter the next dataset.

## Post-training inherits the action tokenizer

Language post-training usually assumes a token sequence with a tractable log probability. Robot actions have physical units, temporal correlation, and often several valid trajectories. Choosing DPO, PPO, or a critic therefore requires first defining the policy output and its likelihood.

| Action interface | Likelihood exposed to post-training | Main constraint |
| --- | --- | --- |
| Per-dimension tokens | categorical likelihood per action value | quantization and long sequences |
| FAST tokens | categorical likelihood over compressed trajectory coefficients | compression prior and autoregressive latency |
| Regression chunk | deterministic or simple parametric loss | can average distinct valid behaviors |
| Diffusion trajectory | likelihood through the denoising process | iterative sampling and specialized policy gradients |
| Flow action expert | continuous vector field over an action chunk | separate expert and integration path |

The action representation is part of the policy. Change it, and the same physical correction may go from one token edit to a coordinated change across an entire trajectory.

### FAST compresses the trajectory before it becomes language

FAST compresses the temporal redundancy in a trajectory before tokenization. It transforms a continuous action chunk into frequency coefficients and quantizes them. Broad low-frequency motion appears before finer corrections, after which byte-pair encoding compresses recurring patterns. Smooth trajectories therefore become shorter token sequences for an autoregressive VLM.

![FAST action tokenization transforms a trajectory into frequency coefficients and compact tokens](/assets/images/fast-efficient-action-tokenization-for-vision-language-action-models-paper-figure.jpg)
*FAST transforms continuous action chunks into frequency coefficients and compresses the resulting discrete sequence before autoregressive prediction. source: [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html)*

FAST exposes a categorical token likelihood that fits directly into SFT and preference objectives. Its token order also affects credit assignment: early low-frequency tokens define the broad motion, while later tokens refine it. A sequence-level preference can therefore penalize an otherwise valid approach because one high-frequency correction is wrong.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) uses a different action representation at each stage. FAST tokens allow web and robot tasks to share a discrete pretraining objective. A continuous expert added during post-training provides finer control and faster inference. The representation used for heterogeneous pretraining is therefore separated from the representation used during execution.

### Alpamayo keeps reasoning tokenized and trajectories continuous

Driving provides a different division between discrete and continuous outputs. [Alpamayo-R1](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html) generates a tokenized Chain of Causation that names the relevant actors, causal factors, and decision. A diffusion decoder then uses that state to produce a continuous, dynamically feasible trajectory.

![Alpamayo-R1 separates tokenized causal reasoning from continuous diffusion trajectory prediction](/assets/images/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail-paper-figure.webp)
*Alpamayo-R1 generates a tokenized Chain of Causation and conditions a diffusion decoder that produces the continuous driving trajectory. source: [Alpamayo-R1](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html)*

Post-training must align both outputs. SFT teaches the causal trace; RL then adds rewards for reasoning quality, consistency between the trace and action, and trajectory behavior. The trace must describe the scene correctly, and the diffusion decoder must produce a feasible plan. The reward must also detect disagreement between them.

FAST tokenizes the trajectory so one decoder can predict actions autoregressively. Alpamayo tokenizes the explanation while leaving the trajectory continuous. The relevant design choice is which output benefits from discrete language supervision and which must preserve metric continuity.

## Interactive imitation moved supervision onto policy states

The closed-loop problem predates VLAs. A supervised policy trains on expert states, then deploys under the state distribution created by its own actions. One mistake changes the next observation and can compound across the remaining horizon. [DAgger](/paper%20shorts/2011/04/11/dagger-reduction-of-imitation-learning-to-no-regret-online-learning.html) addresses this shift through iterative data collection.

DAgger runs the learner, queries the expert in the states the learner reaches, and adds those corrected actions to the dataset. Human takeovers, joystick corrections, recovery demonstrations, and successful reruns are modern forms of the same loop.

Corrections are most informative near the policy's competence boundary. Repeated easy successes add little new supervision, while catastrophic failures may be unsafe or too far outside the recoverable region. Near misses, ambiguous objects, perturbations, and recoverable contact errors identify states where a different local action can change the outcome.

Correction SFT remains the baseline for these data. Preference optimization or RL should be compared against it under the same robot-hour and human-effort budget.

## Preference learning exposed the counterfactual problem

Language preference data often provide two answers to the same prompt and a label indicating which one is preferred. [DPO](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html) directly increases the likelihood of the preferred answer relative to the rejected answer:

$$
\mathcal{L}_{\text{DPO}}
=-\mathbb{E}\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}
\right]
\right).
$$

Robot rollouts rarely provide the same matched comparison. A human correction begins after the original action has already changed the state. Two physical attempts may also differ in friction, camera pose, initialization, or object position. Treating those trajectories as if only the policy action changed can assign the preference to the wrong cause.

Unpaired rollout feedback requires an objective that does not assume a matched counterfactual. [KTO](/paper%20shorts/2024/02/02/kto-model-alignment-as-prospect-theoretic-optimization.html) learns from separately desirable and undesirable examples. [Action Preference Optimization](/paper%20shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html) applies related logic to robot interventions and weights token updates by the error in the decoded continuous action.

The method should follow the evidence:

| Deployment evidence | Defensible update | Claim to avoid |
| --- | --- | --- |
| Corrective action in the reached state | local correction SFT | the whole prefix was wrong |
| Matched alternatives from the same reset | paired preference objective | hidden physical state was identical |
| Independent successful or failed rollouts | binary desirable/undesirable objective | one action caused the terminal label |
| Human takeover | failure window near the intervention | every previous action deserves rejection |
| Safety violation | explicit constraint label | one scalar captures severity and task success |

## Process supervision localized the failure

Suppose the gripper misses the handle at step 42 and a human takes over at step 47. The terminal bit says the episode failed. The intervention says behavior was unacceptable by step 47. Neither tells us that every earlier action was wrong.

A process critic replaces an episode-level failure label with an estimate of progress at intermediate states. [VisualPRM](/paper%20shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html) provides the general recipe: label intermediate errors, train a critic, and validate it against held-out human judgment before optimization. [VLAC](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html) applies this idea to robotics by predicting signed progress and completion between two observations.

![VLAC generates action and reward tokens with a value head for PPO](/assets/images/vlac-vision-language-action-critic-for-real-world-rl-source-figure-3.webp)
*VLAC makes the learning signal part of the forward pass: action tokens select behavior, reward tokens estimate progress, and a value head supports PPO. Critic errors can therefore influence subsequent policy updates. Source: [VLAC, Figure 3](https://arxiv.org/abs/2509.15937).*

A single scalar can obscure disagreements among progress, completion, safety, uncertainty, and failure type. Pixels may also omit contact, controller lag, or the state of an occluded gripper. A robot critic may therefore need tracked objects, geometry, proprioception, and controller state in addition to images.

Credit should remain as local as the evidence allows. The dataset can preserve the prefix that still made progress and mark the first defensible failure window. It should also store the reached state and the corrective continuation. When an alternative cannot be replayed from the same state, the two trajectories should not be represented as a matched preference pair.

## Interactive RL put rollout collection inside optimization

Preference learning updates a policy from a fixed feedback dataset. Interactive RL updates the policy, then collects fresh rollouts from the new version. The optimizer is now changing its own training distribution.

For a standard stochastic policy, PPO clips the policy ratio to limit each update:

$$
\mathcal{L}_{\text{clip}}(\theta)
=\mathbb{E}_t\left[
\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

Clipping removes the incentive to move sampled action ratios farther in the rewarded direction beyond the clip interval. It does not enforce a global bound on the policy change or validate the reward.

The policy gradient also has to match the action generator. For a diffusion actor, [DPPO](/paper%20shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html) treats the denoising steps themselves as the stochastic policy. A denoised trajectory does not have the same likelihood as one categorical token or Gaussian action. Using the wrong likelihood assigns credit to the wrong part of generation.

Binary success can still provide a useful reward when the rollout system creates comparable groups. [RIPT-VLA](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html) and [SimpleVLA-RL](/paper%20shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html) run multiple attempts and learn from their relative outcomes. A group in which every attempt succeeds or every attempt fails contains no ranking signal, making task sampling part of the learning algorithm.

![RIPT-VLA and SFT success on LIBERO-LONG across demonstration counts](/assets/images/ript-vla-interactive-post-training-for-vision-language-action-models-source-figure-2.webp)
*RIPT-VLA compares interactive post-training with SFT under one to ten demonstrations on LIBERO-LONG. This is evidence about the value of policy-generated experience in the reported simulator setup. Source: [RIPT-VLA, Figure 2](https://arxiv.org/abs/2505.17016).*

Group-relative learning therefore couples optimization to task selection. As the policy improves, the sampler must find tasks with informative outcome variation; otherwise collection cost grows without a corresponding learning signal.

## Specialist reinforcement learning followed by distillation

Direct RL can improve one task while degrading capabilities shared across the generalist policy. [RLDG](/paper%20shorts/2024/12/13/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning.html) instead trains task-specific RL specialists, collects their improved trajectories, and distills those trajectories back into the general policy.

This places reinforcement learning upstream of the generalist. The specialist first improves the trajectory distribution, after which distillation transfers the behavior while attempting to retain other tasks, instructions, and visual concepts.

The approach is especially relevant when a simulator provides dense rewards for one task but the deployed policy must remain broad. A controlled comparison should match environment interactions and final model size between direct RL on the generalist and specialist RL followed by distillation.

## Average success can hide a brittle policy

Average task success can improve while robustness declines. [LIBERO-Para](/paper%20shorts/2026/03/30/libero-para-paraphrase-robustness-in-vla-models.html) reports large drops under instruction paraphrases, often because the high-level plan changes even when the low-level controller remains capable. [RobustVLA](/paper%20shorts/2025/11/03/robustvla-robustness-aware-reinforcement-post-training.html) adds observation sensitivity and action smoothness to the RL objective.

Training and evaluation should cover the same classes of perturbation. These include paraphrased instructions, camera shifts, occlusion, calibration error, latency, actuation noise, object substitutions, and recoverable contact mistakes. Results should report which failure type improved or regressed, not only the average.

The evaluation ladder should move from cheap diagnosis to physical evidence:

| Level | What it can establish | What it cannot establish |
| --- | --- | --- |
| Offline action and critic metrics | target fit, critic accuracy, regression bugs | closed-loop recovery |
| Closed-loop simulation | policy-induced states and controlled perturbations | real contact and hardware timing |
| Real-to-sim correlation | whether simulation preserves policy rankings | performance on an unseen hardware stack |
| Reproducible real trials | physical success, latency, contact, interventions | fleet-scale natural variation |
| Canary deployment | long-tail behavior under real use | safe unrestricted rollout by itself |

Each evaluation level should be validated against the more expensive level that follows it. [SIMPLER](/paper%20shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html) tests whether simulation preserves the ranking of real policies, rather than whether simulated success appears plausible in isolation. [VLA-REPLICA](/paper%20shorts/2026/05/20/vla-replica-low-cost-reproducible-real-world-evaluation.html) extends this progression toward reproducible physical trials. A cheaper metric is useful when it predicts the robot result used for the deployment decision.

## What changes between updates

Interactive methods change both the policy and the data distribution. A rollout from an older fleet policy, a relabeled reward from a new critic, or an action sequence produced by a different tokenizer is a different training object. Policy, evaluator, action-interface, and controller versions are needed to reconstruct the update and compare it with correction-only training.

## From demonstrations to policy-generated data

| Leap | New feedback unit | What it changed | Remaining risk |
| --- | --- | --- | --- |
| Behavior cloning | expert action in an expert state | gave the policy a stable task baseline | covariate shift |
| Action-aware adaptation | token, chunk, diffusion, or flow target | aligned the optimizer with the served action interface | likelihood and latency mismatch |
| FAST | compressed action-token sequence | made trajectories compatible with autoregressive post-training | compression can hide sharp corrections |
| Alpamayo | tokenized causal trace plus continuous trajectory | separated reasoning supervision from metric planning | reasoning can disagree with action |
| Interactive imitation | correction in a policy-created state | collected recovery behavior where the policy fails | intervention-state bias |
| Preference learning | chosen, rejected, or binary-labeled behavior | used deployment judgments without dense rewards | false counterfactuals |
| Process critics | progress inside a rollout | localized credit before terminal success | critic shortcuts and missing physical state |
| Interactive RL | fresh rollout group and environment reward | improved the data distribution while learning | reward exploitation and rollout cost |
| Specialist distillation | improved specialist trajectory | protected the generalist from direct RL instability | loss of specialist behavior during distillation |

These branches do not form a simple ranking by feedback quality. An expert correction supplies a target action in one reached state. A preference supplies a comparison. A process critic supplies a learned progress estimate, and RL supplies outcomes under the current policy. Each can add information unavailable to demonstrations, but each also introduces a different attribution error.

The open comparison is how much physical improvement each feedback source buys at equal interaction and human effort. That requires tracking recovery success, retained skills, and control latency alongside task completion. A higher simulated success rate alone cannot resolve the choice between correction SFT, preference learning, process rewards, and direct RL.
