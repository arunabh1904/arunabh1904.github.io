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

Post-training sounds like the final step after pretraining. In robotics, it is a repeated loop. The policy is deployed, a failure is attributed, the model is updated, and the change returns to the robot for evaluation.

A robot fails while closing a drawer. The outcome tells us that the rollout failed. It does not tell us whether the camera missed the handle, the planner chose a bad approach, the trajectory drifted, the gripper slipped, the controller lagged, or the success detector fired too early.

That attribution gap is the part I find most interesting. A robot action changes the next observation, so one mistake can create an unfamiliar state or an irreversible contact. The optimization step is usually straightforward once the target is defined. The difficult part is deciding which action, state, or model component the failure provides evidence about.

The progression follows how precisely the feedback locates the mistake. Behavior cloning learns expert actions in expert states. Intervention data adds corrections in states created by the policy. Preferences compare behaviors, while process critics assign progress inside a rollout. Interactive RL closes the loop by collecting the next batch from the updated policy.

The policy-improvement system looks like this:

<div class="compact-flow-diagram"><a href="/assets/images/robot-post-training-loop.svg"><img src="/assets/images/robot-post-training-loop.svg" alt="Robot post-training loop from a pretrained policy through adaptation, deployment, failure mining, conservative optimization, safety evaluation, canary deployment, and new evidence"></a></div>

*A candidate update returns to deployment only after safety and regression evaluation, and the resulting rollout becomes evidence for the next iteration.*

This is Part III of the series. [Part I](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) asks what visual evidence the model preserves. [Part II](/blog/2026/07/15/omni-model-pretraining-decisions.html) asks how semantics, dynamics, and motor priors enter the policy. This part starts when the pretrained policy reaches deployment and begins creating its own data.

## Behavior cloning established the baseline

Supervised fine-tuning, or behavior cloning, provides the baseline. Given an observation and instruction, the policy maximizes the likelihood of the expert action:

$$
\mathcal{L}_{\text{BC}}(\theta)
=-\mathbb{E}_{(o,\ell,a)\sim D_E}
\log \pi_\theta(a\mid o,\ell).
$$

The action $a$ may be one discrete bin, a sequence of FAST tokens, a continuous chunk, or a diffusion denoising target. This choice determines the loss, inference path, control rate, and level at which a later correction can assign credit.

The action head can change while retaining the pretrained visual-language model. OpenVLA originally predicts action tokens autoregressively. [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) replaces them during adaptation with parallel continuous chunks trained through an L1 loss. In its experiments, the new head improves both control speed and task success.

SFT should be the first serious baseline on a new robot. Compare full tuning with adapters and try several chunk lengths. Include both clean demonstrations and recoveries, using only action heads that meet the deployment deadline. Measure task success, robot-data efficiency, control rate, latency, and semantic forgetting together.

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

*FAST transforms continuous action chunks into frequency coefficients and compresses the resulting discrete sequence before autoregressive prediction. Source: [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html).*

FAST exposes a categorical token likelihood that fits directly into SFT and preference objectives. Its token order also affects credit assignment: early low-frequency tokens define the broad motion, while later tokens refine it. A sequence-level preference can therefore penalize an otherwise valid approach because one high-frequency correction is wrong.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) uses a different action representation at each stage. FAST tokens allow web and robot tasks to share a discrete pretraining objective. A continuous expert added during post-training provides finer control and faster inference. The representation used for heterogeneous pretraining is therefore separated from the representation used during execution.

I would evaluate the tokenizer in closed loop, reporting reconstruction error and sequence length alongside control latency, contact-heavy success, and perturbation recovery. A tokenizer can improve offline likelihood while attenuating a high-frequency correction required during execution.

### Alpamayo keeps reasoning tokenized and trajectories continuous

Driving provides a different division between discrete and continuous outputs. [Alpamayo-R1](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html) generates a tokenized Chain of Causation that names the relevant actors, causal factors, and decision. A diffusion decoder then uses that state to produce a continuous, dynamically feasible trajectory.

![Alpamayo-R1 separates tokenized causal reasoning from continuous diffusion trajectory prediction](/assets/images/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail-paper-figure.webp)

*Alpamayo-R1 generates a tokenized Chain of Causation and conditions a diffusion decoder that produces the continuous driving trajectory. Source: [Alpamayo-R1](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html).*

Post-training must align both outputs. SFT teaches the causal trace, a large reasoning critic scores its quality, and RL rewards consistency between the explanation and action. The trace must describe the scene correctly, and the diffusion decoder must produce a feasible plan. The reward must also detect disagreement between them.

FAST tokenizes the trajectory so one decoder can predict actions autoregressively. Alpamayo tokenizes the explanation while leaving the trajectory continuous. The relevant design choice is which output benefits from discrete language supervision and which must preserve metric continuity.

## Interactive imitation moved supervision onto policy states

The closed-loop problem predates VLAs. A supervised policy trains on expert states, then deploys under the state distribution created by its own actions. One mistake changes the next observation and can compound across the remaining horizon. [DAgger](/paper%20shorts/2011/04/11/dagger-reduction-of-imitation-learning-to-no-regret-online-learning.html) addresses this shift through iterative data collection.

DAgger runs the learner, queries the expert in the states the learner reaches, and adds those corrected actions to the dataset. Human takeovers, joystick corrections, recovery demonstrations, and successful reruns are modern forms of the same loop.

Behavior cloning learns an action in the states we collected. Interactive imitation changes which states we collect in the first place.

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

The deployment event must therefore support both the comparison label and the likelihood assumed by the optimizer. Implementing DPO does not establish either condition.

## Process supervision localized the failure

Suppose the gripper misses the handle at step 42 and a human takes over at step 47. The terminal bit says the episode failed. The intervention says behavior was unacceptable by step 47. Neither tells us that every earlier action was wrong.

[![Animation comparing episode outcomes, Action Preference Optimization, and process or interactive feedback on the same robot failure](/assets/images/blog-vla-feedback-attribution.gif)](/assets/images/blog-vla-feedback-attribution.gif)

*A terminal outcome labels the whole rollout. An intervention narrows the failure to a local window. A process critic can narrow it further, but only if the critic reads the state correctly. Custom synthesis based on [Action Preference Optimization](/paper%20shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html), [VLAC](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html), and [RIPT-VLA](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html).*

A process critic replaces an episode-level failure label with an estimate of progress at intermediate states. [VisualPRM](/paper%20shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html) provides the general recipe: label intermediate errors, train a critic, and validate it against held-out human judgment before optimization. [VLAC](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html) applies this idea to robotics by predicting signed progress and completion between two observations.

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

Clipping bounds the size of the policy update, but does not establish that the reward assigns the correct credit.

The policy gradient also has to match the action generator. For a diffusion actor, [DPPO](/paper%20shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html) treats the denoising steps themselves as the stochastic policy. A denoised trajectory does not have the same likelihood as one categorical token or Gaussian action. Using the wrong likelihood assigns credit to the wrong part of generation.

Binary success can still provide a useful reward when the rollout system creates comparable groups. [RIPT-VLA](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html) and [SimpleVLA-RL](/paper%20shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html) run multiple attempts and learn from their relative outcomes. A group in which every attempt succeeds or every attempt fails contains no ranking signal, making task sampling part of the learning algorithm.

The rollout system should sample tasks near the policy's competence boundary and keep resets comparable. It should also record the policy version and reject groups with no reward variation. These controls determine whether the optimizer receives an informative comparison.

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

## The rollout system became the training system

Once deployment data trains the next policy, every trajectory needs a history. Store the policy, critic, tokenizer, action head, controller, task, robot, sensor calibration, timestamps, interventions, reward components, termination reason, and evaluator version. Also record whether the rollout trained SFT, preferences, the critic, RL, or nothing at all.

This provenance is part of the experiment. A fleet may collect a rollout from a policy several updates behind the learner. A new tokenizer can make old likelihoods incomparable, while a critic update can relabel the same trajectory. Without versioned records, the evidence used for a gradient update cannot be reconstructed.

The launch gate should check lower-confidence-bound success, unsafe contacts, intervention rate, latency, regression slices, and automatic rollback. A higher average is not enough if a safety-critical slice gets worse.

The metric I ultimately care about is reliable policy improvement per robot-hour, human-hour, annotation-hour, and unit of compute.

## How to read a robot post-training paper

Start with what happened on the robot, not the optimizer name. Ask how local the label is, which policy created the state, and what likelihood or reward the update assumes.

| Question | What it reveals |
| --- | --- |
| What is one action event? | token, chunk, diffusion path, or continuous expert output |
| Who created the state? | expert, current policy, stale fleet policy, or simulator |
| What did the evaluator observe? | outcome, correction, preference, progress, or safety constraint |
| Where is credit assigned? | whole episode, intervention window, action token, or denoising step |
| What prevents reward exploitation? | held-out humans, critic ensembles, constraints, or real task success |
| What must not regress? | semantics, old tasks, control rate, safety, or calibration robustness |
| What is the next evidence layer? | offline, simulation, reproducible robot, or canary deployment |

The optimizer follows from these answers. PPO cannot correct a reward that assigns the wrong credit. DPO cannot create a matched counterfactual, and a process critic cannot infer contact from observations that do not contain it.

## So.... let's recap

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

The feedback becomes more precise as we move down the table. A terminal bit says the rollout ended badly. An intervention says behavior crossed a boundary near this state. A correction says what to do from the reached state. A process critic marks which transition made progress. A matched replay is the rare case that can show what another action would have caused.

The update should use the most specific feedback supported by the deployment event, rather than the most complex objective available in the training stack.

## A testable thesis

The central requirement for VLA post-training is accurate attribution. Each feedback event should identify the state, action, or transition it provides evidence about before that evidence changes the policy.

I would start with an action representation that matches the controller and use SFT as the baseline. Corrections should be collected in states reached by the policy. Preference optimization requires a defensible comparison, while RL requires a reward and rollout system that have been validated through cheaper evaluations.

My strongest bet is a process critic conditioned on persistent objects, geometry, contact, controller state, and task progress. It should identify the earliest defensible failure window and support a conservative update. That update must then be tested for physical success and retention of older skills before deployment.

A policy can learn reliably from deployment only when the update remains no broader than the evidence. Every consequence must also be traceable to the policy that created it.

## Selected references

- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](/paper%20shorts/2011/04/11/dagger-reduction-of-imitation-learning-to-no-regret-online-learning.html)
- [OpenVLA-OFT: Fine-Tuning VLAs for High-Throughput Robot Control](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html)
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html)
- [Pi0.5: A Vision-Language-Action Model with Open-World Generalization](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html)
- [Alpamayo-R1: Bridging Reasoning and Action Prediction](/paper%20shorts/2025/10/30/alpamayo-r1-bridging-reasoning-and-action-prediction-for-generalizable-autonomous-driving-in-the-long-tail.html)
- [Direct Preference Optimization](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html)
- [KTO: Model Alignment as Prospect Theoretic Optimization](/paper%20shorts/2024/02/02/kto-model-alignment-as-prospect-theoretic-optimization.html)
- [Action Preference Optimization for Robotic Policy Refinement](/paper%20shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html)
- [VisualPRM: Process Reward Models for Multimodal Reasoning](/paper%20shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html)
- [VLAC: Vision-Language-Action Critic for Real-World RL](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html)
- [DPPO: Diffusion Policy Policy Optimization](/paper%20shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html)
- [RIPT-VLA: Interactive Post-Training for Vision-Language-Action Models](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html)
- [SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](/paper%20shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html)
- [RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning](/paper%20shorts/2024/12/13/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning.html)
- [RobustVLA: Robustness-Aware Reinforcement Post-Training](/paper%20shorts/2025/11/03/robustvla-robustness-aware-reinforcement-post-training.html)
- [LIBERO-Para: Paraphrase Robustness in VLA Models](/paper%20shorts/2026/03/30/libero-para-paraphrase-robustness-in-vla-models.html)
- [SIMPLER: Evaluating Real-World Robot Policies in Simulation](/paper%20shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html)
- [VLA-REPLICA: Reproducible Real-World Evaluation](/paper%20shorts/2026/05/20/vla-replica-low-cost-reproducible-real-world-evaluation.html)
