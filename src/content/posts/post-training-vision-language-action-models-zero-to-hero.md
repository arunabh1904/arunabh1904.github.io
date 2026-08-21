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
summary: A systems guide to turning demonstrations, failures, interventions, critics, and fleet rollouts into better robot policies without erasing their pretrained capabilities.
---

# Post-Training for Robotics

A robot fails while closing a drawer.

The rollout gives us an outcome: failure. It does not tell us whether the camera missed the handle, the model chose the wrong subtask, the trajectory approached at a bad angle, the gripper slipped, the controller lagged, or the success detector fired too early. Post-training begins in the gap between outcome and explanation.

This is why “make the pretrained model behave better” is an inadequate description. A robot's action changes its next observation. One small error can create an unfamiliar state, an irreversible contact, or a recovery opportunity that no offline demonstration contains. The hard part is not taking another gradient step. It is deciding what the gradient is justified to say.

The right object is therefore not an optimizer. It is a closed-loop policy improvement system:

> Pretrained policy → task adaptation → deployment rollouts → failure mining → local supervision → conservative optimization → retention and safety evaluation → canary deployment → repeat.

The loop is the product. SFT, DPO, critics, actor-critic RL, adapters, and distillation are replaceable components inside it.

This is Part III of a three-part reading course. [Part I](/blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html) asks what a vision-language model must preserve before it can support action. [Part II](/blog/2026/07/15/omni-model-pretraining-decisions.html) asks how pretraining combines semantic priors, heterogeneous robot experience, and an action distribution. This part asks how evidence collected after pretraining should change the policy.

The modern vision-language-action model is the running case, but the principles apply more broadly to learned robot policies. I use “post-training” to include downstream supervised adaptation, interactive imitation, preference optimization, reinforcement fine-tuning, distillation, continual learning, and the deployment system that supplies their evidence. The literature cutoff is August 21, 2026; many of the newest results are preprints, so I treat their reported mechanisms and numbers as evidence, not settled recipes.

## Four gaps define the problem

Most post-training failures can be traced to four gaps.

| Gap | What went wrong | The question post-training must answer |
| --- | --- | --- |
| **Distribution gap** | The policy was trained on expert states but deploys in states created by its own actions | Which new states are worth collecting and learning from? |
| **Attribution gap** | A trajectory outcome does not identify the causal action or decision | What is the smallest segment that the evidence can honestly label? |
| **Interface gap** | The optimizer assumes a likelihood, horizon, or action distribution that the policy does not expose | Does the objective match discrete tokens, chunks, diffusion, flow, or continuous control? |
| **Retention gap** | Narrow adaptation improves one task while erasing broad semantic, spatial, or motor priors | Which parameters should move, how quickly, and under what regression tests? |

These gaps are coupled. Online rollouts close the distribution gap but create an attribution problem. Dense critics improve attribution but introduce reward-model error. Full fine-tuning gives the optimizer freedom but increases retention risk. Freezing the backbone protects semantics but may block necessary adaptation.

A useful post-training proposal should make three claims explicit:

1. **Where was the policy wrong?**
2. **What behavior should replace it?**
3. **How much of the pretrained system is allowed to move?**

A weak recipe answers all three with one terminal reward and one global optimizer.

> **Deep insight:** Post-training is not the art of extracting the largest update from a rollout. It is the art of extracting the smallest justified claim from that rollout, then making the smallest update that can test it.

## Post-training is a ladder, not one stage

It is tempting to divide robot learning into pretraining and RL. In practice, there is a ladder of increasingly expensive evidence.

| Stage | Evidence | Typical method | Main risk |
| --- | --- | --- | --- |
| Offline specialization | Expert demonstrations from the target task or embodiment | SFT, adapters, action-head replacement | Covariate shift and forgetting |
| Interactive correction | On-policy states plus human takeovers or recoveries | DAgger, correction SFT, intervention learning | Supervisor cost and ambiguous intervention boundaries |
| Preference or critic learning | Desirable/undesirable actions, segment comparisons, progress labels | DPO, KTO, APO, process critics | False counterfactuals and reward shortcuts |
| Online policy improvement | Repeated rollouts with task rewards, critics, or constraints | Actor-critic RL, group-relative RL, reinforcement fine-tuning | Exploration cost, instability, reward exploitation |
| Continual or fleet learning | Asynchronous rollouts across tasks, robots, and policy versions | Replay, distillation, online post-training systems | Staleness, interference, provenance failure |

The order matters. A system that cannot beat correction SFT under the same robot-hour and human-hour budget has not earned a more complicated optimizer. Conversely, a policy that already reaches near-success states, has cheap resets, and exposes a trustworthy success signal may be wasting time by collecting only more demonstrations.

The right question is not “Should we use RL?” It is “What is the cheapest evidence that directly attacks the diagnosed failure?”

## Build an SFT baseline that can actually be improved

Supervised fine-tuning hides most of its engineering inside the action target. Is the target one quantized joint token, a whole trajectory chunk, a diffusion denoising target, a flow field, or a parallel continuous vector? How much history does the model see? Does it act at 3 Hz or 50 Hz? Does it predict one step, replan a receding horizon, or temporally ensemble overlapping chunks?

Those are not implementation details. They determine the policy distribution, the effective horizon, the feedback granularity, and which post-training objectives are mathematically meaningful.

| Action interface | Representative work | What it buys | What it makes harder |
| --- | --- | --- | --- |
| Discrete action tokens | [RT-1](/paper%20shorts/2022/12/13/rt-1-robotics-transformer-for-real-world-control-at-scale.html), [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) | One autoregressive interface and tractable token likelihoods | Quantization error and sequential decoding latency |
| Transformer action chunks | [ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html) | Temporal coherence and a shorter effective horizon | Reduced reactivity inside the chunk |
| Diffusion trajectories | [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html) | Multimodal continuous behavior | Iterative sampling and nontrivial policy likelihoods |
| Frequency-domain tokens | [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) | Compact autoregressive trajectories | Compression can suppress sharp corrective motion |
| Flow action expert | [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) | Smooth continuous chunks alongside a semantic backbone | A separate expert, sampling process, and optimization path |
| Parallel continuous chunks | [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) | High-throughput control with a simple regression objective | Regression can average genuinely multimodal actions |

[OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html) is an important warning against treating the pretraining loss as sacred. Replacing OpenVLA's autoregressive action tokens with parallel continuous chunks and an $L_1$ objective improves both throughput and downstream performance. The pretrained representation remains useful even when the adaptation interface changes.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) adds a second axis: semantic subtask prediction. Long-horizon behavior becomes easier when one component represents what should happen next and another represents how to execute it. The decomposition also creates a new failure boundary. A perfect controller can efficiently execute the wrong subgoal.

The minimum credible SFT study is therefore an adaptation matrix:

- frozen backbone, partial tuning, adapters, and full tuning;
- one-step actions and several chunk horizons;
- discrete, regression, diffusion, and flow heads where feasible;
- expert successes alone versus successes plus recoveries and interventions;
- per-task adapters versus a shared multi-task adapter;
- task data alone versus task data mixed with broad replay.

Measure success, control frequency, latency, robot-data efficiency, old-task retention, paraphrase robustness, and semantic grounding together. A policy that succeeds 3 percentage points more often but halves the control rate or loses instruction robustness may already be worse.

### The data mixture is part of the policy

Fine-tuning datasets should not be described only by trajectory count. Their composition determines what the policy is permitted to forget.

A useful mixture contains four distinct roles:

- **Task demonstrations** teach the nominal behavior.
- **Recoveries and interventions** cover states produced by the learner.
- **Broad replay** anchors capabilities acquired before specialization.
- **Constraint and failure examples** define what must not be optimized away.

The weights should follow a diagnosed purpose. Increasing recovery data may improve robustness but also overrepresent awkward states that a mature policy rarely visits. Increasing broad replay can preserve semantics but dilute the downstream gradient. Increasing only successful trajectories can make the policy look clean offline while leaving its failure boundary untouched.

Treat mixture weights as a policy-design choice and report them as carefully as learning rate.

## Preserve what pretraining bought you

A broadly pretrained robot model is not just an initialization. It contains the reason to use a foundation model at all: visual concepts, language grounding, cross-task structure, and a prior over plausible motion. Full fine-tuning implicitly assumes that every shared parameter is expendable in service of the downstream benchmark.

That assumption is increasingly difficult to defend.

[PriorVLA](https://arxiv.org/abs/2605.10925) keeps a frozen Prior Expert and trains a separate Adaptation Expert that queries both semantic and motor priors. The paper reports stronger few-shot and out-of-distribution performance while updating 25% as many parameters as full fine-tuning. The architectural message is more important than any one benchmark result: adaptation can consume a prior without rewriting it.

[TEMPO](https://arxiv.org/abs/2608.07314) makes a related claim for online RL. It freezes the vision-language backbone, updates a semantic projection slowly, and updates the low-level action expert more frequently. This two-timescale design reflects the fact that semantic grounding and contact control do not receive equally dense evidence from a rollout.

A recent mechanistic study, [From Recovery to Drop-off](/paper%20shorts/2026/08/14/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability.html), probes one weight-matched VLM and VLA pair and finds weaker depth decodability throughout the VLA, with an additional late-layer collapse linked to MLP writes. One model pair and one spatial probe do not establish a universal law. They do establish that “the VLM probably keeps its geometry” is not an evaluation plan.

> **Deep insight:** Specialization gain and retained intelligence are separate axes. A narrow task benchmark can hide a broad regression.

Every adaptation run should include a retention suite:

| Retention axis | Example test |
| --- | --- |
| Language grounding | Instruction paraphrases, compositional instructions, distractor nouns |
| Object and spatial knowledge | Object identity, relative position, affordance, depth or geometry probes |
| Broad robot competence | Held-out tasks and old embodiments from before adaptation |
| Control prior | Smoothness, action range, latency sensitivity, recovery behavior |
| Representation drift | Layerwise probes, feature similarity, parameter/update norms |
| Behavioral drift | KL or action divergence from the reference policy on anchor states |

The intervention should match the diagnosed drift. Freeze or slowly update the semantic backbone when control feedback is dense but semantic evidence is weak. Use adapters when tasks are narrow. Keep a frozen prior expert when the downstream data are few-shot. Mix anchor examples when the same parameters must serve old and new tasks. Full fine-tuning remains a valid hypothesis, but it should no longer be the unexamined default.

## Deployment is a data-collection policy

A modern VLA begins with two useful priors. Vision-language pretraining supplies concepts, objects, instructions, and scene semantics. Robot pretraining supplies a distribution over physically plausible behavior. [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) demonstrates transfer through a language-token interface. [Open X-Embodiment](/paper%20shorts/2023/10/13/open-x-embodiment-robotic-learning-datasets-and-rt-x-models.html), [Octo](/paper%20shorts/2024/05/20/octo-an-open-source-generalist-robot-policy.html), and [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) make cross-embodiment robot pretraining concrete.

Neither prior guarantees that the deployed policy occupies familiar states. [DAgger](/paper%20shorts/2011/04/11/dagger-reduction-of-imitation-learning-to-no-regret-online-learning.html) explains why. In sequential prediction, an error changes the next observation. A small supervised error under the expert distribution can compound because the learner visits states the expert never visited.

This gives the first mental model to keep:

> SFT learns what to do in the states represented by its data. Interactive post-training changes which states become data.

That difference is why another million nominal demonstrations can be worth less than ten thousand carefully selected recoveries.

### Mine the boundary, not the average

Rollouts are not automatically useful. A thousand identical successes provide little new gradient. A thousand catastrophic failures may be unsafe and too far outside the recoverable region. The valuable middle consists of near-boundary states: recoverable mistakes, ambiguous objects, distribution shifts, and segments where a different local decision changes the outcome.

Human intervention has three roles at once:

1. It keeps deployment safe.
2. It reveals states where the current policy becomes unacceptable.
3. It supplies corrective behavior from those states.

[ThriftyDAgger](https://proceedings.mlr.press/v164/hoque22a.html) makes intervention timing a budgeted decision based on novelty and risk. [Fleet-DAgger](https://proceedings.mlr.press/v205/hoque23a.html) extends the problem to allocating limited human attention across multiple robots. [RLIF](https://arxiv.org/abs/2311.12996) treats intervention signals as rewards, relaxing the requirement that every human correction be a near-optimal action. [HIL-SERL](https://arxiv.org/abs/2410.21845) shows that demonstrations, corrections, and efficient off-policy RL can solve demanding real-world skills within practical training windows. [HELP](https://arxiv.org/abs/2607.09776) pushes the same resource question to a twelve-robot system, separating high-skill teleoperation from fleet monitoring and physical resets, then using a process critic to retain progress and recovery segments while filtering idle and failure-inducing portions.

The common lesson is not simply “add a human.” It is that supervisor attention is a scarce sensing and control resource. Query policies, intervention latency, takeover duration, reset cost, operator specialization, and disagreement should all be measured.

### Route failures before optimizing them

A useful taxonomy separates six causes:

| Failure class | Typical symptom | First place to look |
| --- | --- | --- |
| Semantic | Wrong object, goal, or subtask | Instruction grounding, task decomposition, retained VLM prior |
| Perceptual or metric | Correct identity, wrong pose, depth, geometry, or contact | Camera calibration, temporal perception, structured state, data augmentation |
| Planning | Locally plausible actions commit to a globally bad sequence | History, subgoal supervision, planner evaluation, process rewards |
| Control | Timing, latency, chunking, or actuator mismatch ruins the plan | Action interface, control rate, chunk horizon, low-level expert |
| Safety or constraint | Task progresses through forbidden contact or workspace violation | Constraint critic, shield, recovery policy, separate safety objective |
| Evaluation or systems | Correct behavior is mislabeled, delayed, or corrupted | Success detector, synchronization, dropped frames, controller and hardware logs |

This taxonomy is operational. Semantic failures should not be routed directly to a low-level RL loop. Controller latency should not be “fixed” by changing a VLM. A broken success detector should stop training, not generate more negative examples.

The next 1,000 robot hours should go where expected marginal information is highest: uncertain critic regions, recoverable failures, rare safety cases, new environments, and tasks that discriminate between candidate methods. Uniform collection is easy to schedule and often a poor research allocation.

## Label only what the rollout actually observed

The classic language post-training pipeline is documented by [InstructGPT](/paper%20shorts/2022/02/28/training-language-models-to-follow-instructions-with-human-feedback.html): supervised fine-tuning, a Bradley-Terry preference model, then [PPO](/paper%20shorts/2017/07/01/proximal-policy-optimization-ppo.html) against the learned reward with a KL penalty. [DPO](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html) removes the explicit reward model and optimizes the chosen response against a rejected response relative to a reference policy.

![The three-stage InstructGPT training pipeline from demonstrations to reward-model-guided PPO](/assets/images/training-language-models-to-follow-instructions-with-human-feedback-paper-figure.png)
_The feedback units are clean in language: demonstrations supervise SFT, matched rankings supervise the reward model, and prompts drive PPO rollouts. Physical feedback rarely arrives in such clean pairs. Source: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)._

For a matched prompt $x$ and preference $y^+ \succ y^-$, DPO optimizes a margin between policy and reference log-ratios:

$$
\mathcal{L}_{\text{DPO}}
=-\mathbb{E}\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}
\right]
\right).
$$

The elegance depends on a shared context. Robot trajectories rarely provide that counterfactual. If a human intervenes after a bad grasp, the corrected action occurs in the state produced by the bad grasp. If one rollout succeeds and another fails, camera pose, friction, initialization, object identity, or policy version may differ. Treating the episodes as a clean preference pair can teach the policy the wrong cause.

[KTO](/paper%20shorts/2024/02/02/kto-model-alignment-as-prospect-theoretic-optimization.html) is often a better conceptual starting point because it can learn from individual desirable and undesirable outputs. [Action Preference Optimization](/paper%20shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html) adapts binary intervention feedback to robot actions and reweights local token updates using decoded continuous-action error.

The feedback interface should match what deployment actually observed:

| Deployment signal | Honest interpretation | Common mistake |
| --- | --- | --- |
| Successful or failed episode | Outcome label for a trajectory | Assuming the final action caused the outcome |
| Human takeover | Evidence that behavior became unacceptable near an intervention boundary | Rejecting the entire prefix |
| Corrective action | Preferred local behavior in the reached state | Pairing it with an action from a different state |
| Repeated matched resets | Comparative evidence for segments under approximately shared conditions | Ignoring reset mismatch or stochastic contact |
| Progress judgment | Supervision over a transition or short segment | Assuming temporal order always means progress |
| Unsafe contact | Constraint violation with severity and context | Folding it into one scalar where task reward can compensate |
| Smooth versus oscillatory motion | Style or control preference | Letting smoothness hide task failure |

### Which action caused the failure?

Suppose the gripper misses the handle at step 42 and a human takes over at step 47. The terminal bit says the rollout failed. The intervention says the policy became unacceptable by step 47. Neither observation proves that every earlier action was wrong.

Penalizing the whole trajectory can erase a good approach because of one bad contact. Training only on the human suffix can also be misleading if the human begins from a state that the policy would never deliberately create.

[![Animation comparing episode outcomes, Action Preference Optimization, and process or interactive feedback on the same robot failure](/assets/images/blog-vla-feedback-attribution.gif)](/assets/images/blog-vla-feedback-attribution.gif)

*A failed episode labels an outcome without identifying a causal action. Action Preference Optimization narrows supervision toward desirable and undesirable actions around intervention. Process critics estimate progress between observations, while interactive RL constructs advantages from repeated rollouts. Each narrower label requires a stronger evidence assumption. Custom synthesis based on [Action Preference Optimization](https://arxiv.org/abs/2506.07127), [VLAC](https://arxiv.org/abs/2509.15937), and [RIPT-VLA](https://arxiv.org/abs/2505.17016).*

The safest useful label is usually local:

- preserve the prefix while it still makes progress;
- mark the first defensible failure window;
- record the exact reached state and policy version;
- attach the corrective continuation to that state;
- use a pairwise objective only when the alternative begins from a meaningfully matched context.

The method should follow the evidence unit, not the fashion cycle.

## Choose the optimizer after choosing the feedback unit

The useful taxonomy is not SFT versus RL. It is the relationship between evidence and policy update.

### 1. Correction SFT

Train on recoveries, human takeovers, successful reruns, or local expert actions at on-policy states. This is stable, easy to debug, and compatible with any differentiable action objective.

Its limitation is not that it is “only imitation.” It is that it treats the corrective action as the target without modeling why the original action was bad. It can also overfit to awkward states that appear only after takeover.

Correction SFT should be the default post-training baseline. Every preference or RL method should beat it under matched robot, human, reset, and compute budgets.

### 2. Binary and preference optimization

Use DPO when alternatives begin from a defensibly matched context and the policy exposes meaningful likelihoods. Use KTO or APO-style objectives when outcomes or interventions arrive independently. Keep task success, safety, efficiency, and style as separate labels until their conflicts are visible.

Continuous action policies require care. A diffusion or flow policy does not automatically expose the same sequence likelihood as an autoregressive model. [DPPO](/paper%20shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html) treats the denoising process as part of the stochastic policy. [CrossVLA](https://arxiv.org/abs/2605.21854), currently a workshop draft, explores a surrogate flow-matching log-probability for applying DPO to continuous-action backbones. These works are useful precisely because they do not pretend every action head has the language-model interface.

### 3. Critic-guided and actor-critic RL

Use a learned value or reward model when repeated interaction is available and imitation has reached a ceiling. This is the regime where RL can improve beyond the demonstrated behavior, but it is also where unstable values and poor exploration can rapidly destroy a useful pretrained policy.

[RLIF](https://arxiv.org/abs/2311.12996) uses intervention events as reward signals. [HIL-SERL](https://arxiv.org/abs/2410.21845) combines demonstrations, human corrections, and off-policy RL for real-world dexterous skills. [FORCE](https://arxiv.org/abs/2606.26006) targets the offline-to-online transition with a value-calibrated warm-up, then filters policy and expert proposals through the learned Q-function. [BORA](https://arxiv.org/abs/2605.30226) freezes the VLA base and adds lightweight online residual adaptation after an offline critic stage, an especially useful pattern for high-dimensional dexterous control. [EXPO-FT](https://arxiv.org/abs/2605.25477) reports stable VLA reinforcement fine-tuning across precision and dynamic real-robot tasks, reaching 30 out of 30 successes on its evaluated tasks after an average of 19.1 minutes of online robot data.

Those results move real-world RL fine-tuning from a speculative appendix into the main post-training toolbox. They do not make it universal. Each result depends on its task suite, resets, reward construction, hardware, and pretrained policy.

RL is warranted when most of the following are true:

- the SFT policy already reaches informative near-success states;
- success, progress, or constraint signals can be validated independently;
- resets and exploration are sufficiently cheap and safe;
- the action interface supports a correct policy objective;
- the system can detect initial unlearning and roll back;
- improvement is measured on held-out real trials, not only critic reward.

### 4. Sparse-reward group-relative RL

[RIPT-VLA](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html) and [SimpleVLA-RL](/paper%20shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html) show that binary task success can provide a useful learning signal when rollouts are cheap, parallel, and grouped into comparable conditions. [VLA-RL](https://arxiv.org/abs/2505.18719) formulates autoregressive VLA trajectories as multimodal multi-turn conversations and adds a process reward model trained from automatically segmented pseudo-labels. It reports a 4.5% improvement over its strongest fine-tuned baseline across 40 LIBERO tasks.

The hidden systems requirement is reward variation. If every rollout in a group succeeds or every rollout fails, relative advantages collapse. Rollout scheduling must therefore create informative contrasts by sampling near the competence boundary, controlling task and reset variation, and rejecting uninformative groups.

### 5. Process-rewarded or continual reinforcement fine-tuning

Terminal reward is often too sparse for long action chunks. [LifeLong-RFT](https://arxiv.org/abs/2602.10503) assigns multiple chunk-level process rewards for discrete consistency, continuous trajectory alignment, and output format. On continual LIBERO, it reports a 22% average-success gain over SFT while using 20% of the training data. [CRL-VLA](https://arxiv.org/abs/2602.03445) attacks the stability-plasticity problem from the value side, using a frozen goal-conditioned critic as an anchor and a trainable critic for new-task adaptation. The broader lesson is that continual post-training needs an explicit mechanism for preserving old-task value, not only a replay buffer and hope.

### 6. Specialist RL followed by distillation

[RLDG](/paper%20shorts/2024/12/13/rldg-robotic-generalist-policy-distillation-via-reinforcement-learning.html) offers a useful escape hatch when direct RL on a generalist is unstable. Train task-specific RL specialists, collect their improved trajectories, and distill those behaviors into the foundation policy. RL changes the data distribution without directly moving every shared parameter during exploration.

This trades one difficult problem for two more controlled ones: specialist optimization and generalist distillation. It is often attractive when task rewards are reliable but preserving broad capabilities is non-negotiable.

## Build critics as measurement instruments

A terminal success detector is sparse. A generic VLM reward may miss geometry, contact, occlusion, and temporal progress. A dense hand-engineered reward can teach the simulator rather than the task. The right critic is rarely “a bigger model asked whether the robot did well.”

A robot post-training system may need several critics with different contracts:

| Critic | Output | Proper use |
| --- | --- | --- |
| Terminal verifier | Task complete or incomplete | Episode filtering and final evaluation |
| Process critic | Progress, regression, stagnation, subgoal completion | Credit assignment and segment mining |
| Value critic | Expected future task return | Actor-critic updates and proposal selection |
| Safety critic | Constraint type, severity, uncertainty | Shielding, stopping, and separate optimization constraints |
| OOD or uncertainty critic | Novelty, confidence, disagreement | Human query allocation and canary gating |

[VisualPRM](/paper%20shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html) provides a methodology that transfers beyond its original reasoning setting: construct process labels, train a critic, and evaluate intermediate error localization before using the critic to optimize a generator. [VLAC](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html) makes the robotics version concrete by predicting signed progress and completion from a goal and pairs of observations. Its training data include regression, stagnation, irrelevant goals, and mismatches, not only positive temporal order.

For each critic, record:

- exact input modalities and temporal window;
- output semantics and calibration;
- label source and policy distribution used for training;
- held-out error-localization test;
- known shortcuts and blind spots;
- abstention or disagreement behavior;
- whether it is allowed to train the policy, gate data, stop execution, or only evaluate.

Do not collapse progress, task success, safety, style, and efficiency too early. A single scalar allows fast completion to compensate for unsafe contact or smooth motion to conceal a missed task.

[Scaling Laws for Reward Model Overoptimization](/paper%20shorts/2022/10/19/scaling-laws-for-reward-model-overoptimization.html) shows that proxy reward can continue rising after a stronger gold measure peaks. [Reward Model Ensembles](/paper%20shorts/2023/10/04/reward-model-ensembles-help-mitigate-overoptimization.html) shows that disagreement-aware conservative optimization can reduce the problem, though correlated critics can still share one blind spot.

> **Deep insight:** A reward model is a model of available evidence, not a source of truth. Once the policy is optimized against it, the critic becomes part of the environment.

Track independent task success, human judgment, intervention rate, unsafe contact, critic disagreement, entropy, action divergence from SFT, and the actual content of high-reward rollouts. Stop when gold performance turns down, even if proxy reward keeps rising.

## Match the learning clock to the control hierarchy

Robotic decisions live on different horizons:

- an instruction or goal may persist for an entire episode;
- a subgoal may last tens or hundreds of actions;
- a grasp approach may last several chunks;
- contact stabilization may require corrections every few milliseconds.

Updating every component with the same reward, horizon, and frequency mismatches the evidence available at each level.

[Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html) already separates semantic subtask prediction from continuous action generation. [HiRoC](https://arxiv.org/abs/2608.05999) extends the idea into hierarchical post-training: a planner proposes executable subgoals, an executor is first aligned to those planner-generated subgoals, and low-level behavior is then improved with RL. [TEMPO](https://arxiv.org/abs/2608.07314) takes a two-timescale route, updating semantic projection slowly and the action expert quickly while keeping the pretrained vision-language backbone frozen.

A useful design table is:

| Component | Evidence horizon | Typical update cadence | Primary regression test |
| --- | --- | --- | --- |
| Vision-language backbone | Broad, cross-task, semantic | Frozen or very slow | Grounding, paraphrases, spatial and object probes |
| Planner or semantic projection | Subtask and long-horizon progress | Slow, using aggregated evidence | Goal decomposition and plan consistency |
| Action expert | Local control and task reward | Faster, using online interaction | Success, recovery, latency, smoothness |
| Safety layer or critic | Contact and constraint events | Fast detection, conservative model updates | False negatives, calibration, abstention |

The central idea is not that every robot needs exactly four modules. It is that credit horizon and update horizon should agree. Dense low-level feedback should not rewrite broad semantics at the same rate. Sparse task completion should not be expected to teach every contact correction directly.

## World models must learn the failures too

A learned world model is attractive because real rollouts are expensive. If the model can predict action-conditioned futures, one real trajectory might support many synthetic policy rollouts. The trap is that a visually convincing future is not necessarily a useful control simulator.

[VLA-RFT](https://arxiv.org/abs/2510.00406) trains a data-driven world simulator from real interaction and uses action-conditioned future observations plus trajectory-level verified rewards for reinforcement fine-tuning. [VLAW](https://arxiv.org/abs/2602.12063) identifies a deeper problem: world models trained mainly on successful demonstrations often lack the contact-rich failures that matter for policy improvement. It therefore alternates between real policy rollouts that improve the world model and synthetic rollouts that improve the policy. [WoVR](https://arxiv.org/abs/2602.13977) treats hallucination as corruption of the RL objective itself, shortening imagined error depth through keyframe-initialized rollouts and periodically realigning the world model to the evolving policy.

This creates a co-improvement loop:

> Real policy rollout → failure-rich world-model update → uncertainty-filtered synthetic rollout → policy update → real validation.

Synthetic data should enter the training mixture only after four tests:

1. **Action sensitivity:** changing the action should change the predicted contact and outcome in the right direction.
2. **Failure fidelity:** the model should reproduce slips, collisions, occlusions, and recovery boundaries, not only nominal success videos.
3. **Policy ranking:** policies that are better in the model should usually be better on the real robot.
4. **Uncertainty control:** synthetic rollouts should be truncated, rejected, or downweighted when the model leaves its validated region.

Pixel quality is not the main metric. A world model becomes useful for post-training when its errors are smaller than the policy improvement it is being asked to choose.

## Evaluation is a prediction hierarchy

Post-training claims are only as strong as the next evaluation layer they predict. Each cheap metric should name the expensive decision it is expected to forecast.

### 1. Data and instrumentation checks

Before model evaluation, verify timestamps, dropped frames, action alignment, calibration, reset state, intervention latency, reward version, and policy version. A corrupted rollout can look like a learning failure or a new capability.

### 2. Offline diagnostics

Measure action error, chunk likelihood, critic calibration, preference accuracy, representation probes, instruction grounding, and retained capability. These are debugging tools. They do not measure recovery from the policy's own actions.

### 3. Closed-loop simulation

[LIBERO](/paper%20shorts/2023/06/05/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning.html) separates spatial, object, goal, and mixed transfer. [RoboTwin 2.0](/paper%20shorts/2025/06/20/robotwin-2-scalable-data-generator-and-benchmark.html) adds bimanual tasks, structured randomization, and synthetic data generation. Report success by shift and failure class, not only the mean.

### 4. Real-to-sim correlation

[SIMPLER](/paper%20shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html) asks whether simulation preserves real policy rankings and failure sensitivities. That correlation must be rechecked when the policy family changes. A simulator calibrated for one action interface may not rank a new diffusion or flow policy correctly.

### 5. Reproducible real trials

[VLA-REPLICA](/paper%20shorts/2026/05/20/vla-replica-low-cost-reproducible-real-world-evaluation.html) standardizes a lower-cost physical setup for independent reproduction. Report confidence intervals, intervention frequency, unsafe contacts, completion time, latency, recovery rate, and hardware-versus-policy faults.

### 6. Natural variation and retained intelligence

[LIBERO-Para](/paper%20shorts/2026/03/30/libero-para-paraphrase-robustness-in-vla-models.html) shows large drops under instruction paraphrases and attributes many failures to planning divergence. A robot that succeeds only when the user repeats the fine-tuning phrase has not retained semantic grounding.

[RobustVLA](/paper%20shorts/2025/11/03/robustvla-robustness-aware-reinforcement-post-training.html) moves robustness into the optimization objective through observation-sensitivity and action-smoothness regularization. Evaluation and optimization should meet on the same perturbations: camera shifts, occlusion, latency, calibration error, actuation noise, object substitutions, instruction variation, and recovery from mild disturbance.

### 7. Canary and fleet deployment

Aggregate success is not a launch criterion. A policy should pass lower-confidence-bound success, maximum unsafe-contact and intervention rates, latency limits, regression gates by task family, and an automatic rollback rule. Promote through a small canary population before broad deployment.

One metric sits above every layer:

> Reliable policy improvement per robot-hour, human-hour, annotation-hour, reset, and unit of compute.

## The post-training stack is a distributed system

An RL trainer is one component of a real robot program. The surrounding system needs versioned policies, rollout workers, synchronized logs, calibration records, feedback provenance, immutable dataset snapshots, critic versions, canary deployment, and rollback.

[SOP](https://arxiv.org/abs/2601.03044) makes this explicit through distributed, multi-task online post-training. A fleet streams on-policy experience and intervention signals to a centralized learner, then asynchronously receives updated policies. The paper instantiates the system with both interactive imitation and reinforcement learning, emphasizing that fleet architecture can be algorithm-agnostic. [RLinf-VLA](https://arxiv.org/abs/2510.06710) makes the corresponding training-system problem explicit in simulation by standardizing multiple VLA architectures, RL algorithms, simulators, and resource-allocation modes. [HELP](https://arxiv.org/abs/2607.09776) makes human labor equally explicit by separating teleoperation from fleet monitoring and reset work.

At minimum, every trajectory should identify:

- policy, backbone, action head, adapter, critic, and controller versions;
- task, environment, embodiment, sensors, and calibration;
- observation and action timestamps plus dropped-frame indicators;
- human interventions, operator identity, and intervention latency;
- reward components, uncertainty, termination reason, and evaluator version;
- reset procedure and initial-state metadata;
- whether the trajectory was used for SFT, preferences, critic training, RL, evaluation, or nowhere;
- dataset snapshot and training run that eventually consumed it.

Asynchronous fleets add policy staleness. A rollout may be generated by a policy several learner updates behind. Off-policy correction cannot repair missing provenance. Staleness should be recorded, bounded, and analyzed as part of the method.

The operational loop should be append-only and reproducible. Given a promoted checkpoint, the team should be able to recover the exact rollout set, labels, critic, objective, code version, and evaluation report that justified promotion.

> **Deep insight:** Once robots learn from deployment, data lineage becomes part of model correctness.

## A practical recipe

A robust post-training program can be organized into ten decisions.

### 1. Define the deployment slice

Specify tasks, objects, environments, embodiments, language variation, control frequency, safety constraints, and allowed recovery behavior. “Improve manipulation” is not a testable target.

### 2. Freeze a gold evaluation set

Create real and simulated trials that are never used for training or critic construction. Include old capabilities, natural variation, and safety slices before collecting new data.

### 3. Build the SFT adaptation matrix

Compare update scope, action interface, chunk horizon, task/replay mixture, and latency. Select the simplest policy that reaches informative states reliably.

### 4. Instrument deployment before scaling it

Log policy and critic versions, synchronized observations and actions, interventions, resets, reward components, controller state, calibration, and hardware faults.

### 5. Mine and route failures

Assign semantic, metric, planning, control, safety, or evaluation causes. Select near-boundary cases where a local change might alter the outcome.

### 6. Choose the smallest honest label

Use local correction targets, unpaired desirable/undesirable labels, matched preferences, progress labels, or terminal rewards according to what the rollout truly supports.

### 7. Choose the optimizer that consumes that label

Start with correction SFT. Escalate to preference optimization, critic-guided RL, group-relative RL, or specialist distillation only when the added evidence and system requirements exist.

### 8. Move the smallest useful module

Test adapters, a separate action expert, a slow semantic projection, or a prior-preserving branch before moving the entire backbone. Monitor retention continuously.

### 9. Climb the evaluation ladder

Require gains in closed loop, retained capability, real-world robustness, safety, and cost efficiency. Stop on gold-metric regression even when training reward rises.

### 10. Promote through canaries and preserve lineage

Deploy to a limited population, enforce automatic rollback, and record the complete evidence chain. Feed only validated new experience into the next iteration.

This recipe is intentionally conservative. Robotics rewards teams that iterate quickly, but fast iteration requires cheap falsification and reliable rollback, not larger blind updates.

## A decision guide

Use the cheapest method that attacks the diagnosed failure.

| Observed problem | First intervention | Escalate when |
| --- | --- | --- |
| New embodiment or action interface | OpenVLA-OFT-style SFT matrix | Demonstrations cover the states but control still plateaus |
| Covariate shift and recovery | DAgger or correction SFT | Corrections are abundant but failed and preferred behavior remain confused |
| Unpaired success/failure or takeover logs | KTO or APO-style binary training | Labels are too sparse or causal attribution remains weak |
| Defensible matched alternatives | DPO-style preference optimization | Continuous policy likelihood or reset matching is unreliable |
| Multimodal continuous behavior | Diffusion or flow action head | Imitation plateaus despite broad state coverage |
| Reliable reward and practical real interaction | HIL-SERL, EXPO-FT, or FORCE-style actor-critic RL | Value estimates remain calibrated and real gold metrics improve |
| Cheap parallel rollouts and binary success | RIPT-VLA or SimpleVLA-RL-style group-relative RL | Rollout groups contain reward variation and matched conditions |
| Long horizon with sparse terminal reward | Process critic, VLA-RL, or hierarchical post-training | Intermediate labels pass held-out localization tests |
| Direct generalist RL forgets or destabilizes | Specialist RL plus RLDG-style distillation | Distilled policy preserves both specialist gain and generality |
| Narrow adaptation erases priors | Prior expert, frozen backbone, adapters, slow semantic updates | Retention suite shows the protected path is insufficient |
| Synthetic rollout is attractive but contact fidelity is weak | Improve the world model with real failure rollouts | Policy ranking and uncertainty are validated against the real robot |
| Many robots and limited supervision | Thrifty/Fleet-DAgger or SOP-style allocation | Provenance, staleness, canaries, and rollback are production-ready |

## A reading course

Read the literature as six passes through one hypothetical robot failure. Produce an artifact after each pass so your own proposal becomes harder to hand-wave.

**Pass 1: understand closed-loop distribution shift.** Read [DAgger](/paper%20shorts/2011/04/11/dagger-reduction-of-imitation-learning-to-no-regret-online-learning.html), [ThriftyDAgger](https://proceedings.mlr.press/v164/hoque22a.html), [RLIF](https://arxiv.org/abs/2311.12996), and [HIL-SERL](https://arxiv.org/abs/2410.21845). **Output:** a map of which states the current policy visits, when a human intervenes, and which states are missing from offline demonstrations.

**Pass 2: understand the action interface.** Read [ACT](/paper%20shorts/2023/04/23/action-chunking-with-transformers-act.html), [Diffusion Policy](/paper%20shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html), [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html), [OpenVLA-OFT](/paper%20shorts/2025/02/27/openvla-oft-optimizing-speed-and-success.html), and [Pi0.5](/paper%20shorts/2025/04/22/pi0-5-vision-language-action-model-with-open-world-generalization.html). **Output:** an interface sheet with action units, chunk horizon, control rate, sampling cost, latency, and tractable likelihood for each candidate policy.

**Pass 3: match feedback to an objective.** Read [InstructGPT](/paper%20shorts/2022/02/28/training-language-models-to-follow-instructions-with-human-feedback.html), [DPO](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html), [KTO](/paper%20shorts/2024/02/02/kto-model-alignment-as-prospect-theoretic-optimization.html), [Action Preference Optimization](/paper%20shorts/2025/06/08/action-preference-optimization-for-robotic-policy-refinement.html), and [DPPO](/paper%20shorts/2024/09/01/dppo-diffusion-policy-policy-optimization.html). **Output:** an annotation protocol naming the state, temporal window, alternative, label provenance, and policy version behind every update.

**Pass 4: learn when RL earns its cost.** Read [RIPT-VLA](/paper%20shorts/2025/05/22/ript-vla-interactive-post-training-for-vision-language-action-models.html), [VLA-RL](https://arxiv.org/abs/2505.18719), [SimpleVLA-RL](/paper%20shorts/2025/09/11/simplevla-rl-scaling-vla-training-via-reinforcement-learning.html), [EXPO-FT](https://arxiv.org/abs/2605.25477), [FORCE](https://arxiv.org/abs/2606.26006), and [LifeLong-RFT](https://arxiv.org/abs/2602.10503). **Output:** a budgeted RL plan with reset cost, reward contract, exploration boundary, warm-up, stopping metric, and rollback condition.

**Pass 5: protect the prior and separate timescales.** Read [PriorVLA](https://arxiv.org/abs/2605.10925), [From Recovery to Drop-off](/paper%20shorts/2026/08/14/from-recovery-to-drop-off-how-action-post-training-reduces-a-vlms-late-layer-depth-decodability.html), [TEMPO](https://arxiv.org/abs/2608.07314), and [HiRoC](https://arxiv.org/abs/2608.05999). **Output:** a parameter-update map that states what is frozen, what moves slowly, what moves quickly, and which retention test guards each component.

**Pass 6: validate the complete loop.** Read [VisualPRM](/paper%20shorts/2025/03/13/visualprm-process-reward-model-for-multimodal-reasoning.html), [VLAC](/paper%20shorts/2025/09/19/vlac-vision-language-action-critic-for-real-world-rl.html), [VLA-RFT](https://arxiv.org/abs/2510.00406), [VLAW](https://arxiv.org/abs/2602.12063), [WoVR](https://arxiv.org/abs/2602.13977), [SOP](https://arxiv.org/abs/2601.03044), [HELP](https://arxiv.org/abs/2607.09776), [SIMPLER](/paper%20shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html), and [VLA-REPLICA](/paper%20shorts/2026/05/20/vla-replica-low-cost-reproducible-real-world-evaluation.html). **Output:** an evidence graph connecting every cheap metric to the real deployment decision it is expected to predict.

## A testable thesis

The next major gains in robot post-training will not come from one universally superior optimizer. They will come from closing evidence and timescale gaps.

My strongest bet is a policy with a broad semantic prior that is frozen or updated slowly, a faster adaptable action expert, an explicit planner-controller boundary for long-horizon work, and critics that separate progress, task success, safety, and uncertainty. Real deployment supplies near-boundary failures. Human intervention supplies local corrections and safety. Failure-rich world models amplify only the regions where their predictions are validated. Fleet infrastructure preserves provenance, limits staleness, and makes every promotion reversible.

The deeper principle is evidence compression. A rollout contains millions of pixels, thousands of actions, one physical outcome, and perhaps a few seconds of human feedback. A good post-training system does not turn all of that into one reward. It extracts the smallest defensible supervision unit, routes it to the right component, and asks a stronger evaluation layer whether the update survived reality.

That is the path from a model that can act to a robot that can keep learning without quietly becoming less capable.
