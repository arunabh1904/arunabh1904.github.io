---
title: 'PPO, DPO, GRPO, and On-Policy Distillation'
date: '2026-07-27T16:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: from-ppo-to-grpo-rl-for-reasoning-and-vlas
legacyPath: /blog/2026/07/27/from-ppo-to-grpo-rl-for-reasoning-and-vlas.html
tags:
  - Reinforcement Learning
  - Reasoning
  - Robotics
summary: 'How PPO, DPO, GRPO, and on-policy distillation construct their learning signals—and which assumptions survive contact with physical action.'
---

# PPO, DPO, GRPO, and On-Policy Distillation

Reinforcement learning for language models can look like a parade of acronyms. PPO adds a critic and clips updates. DPO removes the online loop. GRPO removes the critic and compares samples. On-policy distillation brings a teacher back. For vision-language-action models, the same names reappear beside diffusion policies, sparse success rewards, simulators, and robot rollouts.

The useful history is not the sequence of names. It is the sequence of compromises around one estimator:

> Given behavior sampled from the current policy, what evidence should increase or decrease its probability, and relative to what baseline?

This essay follows that question from PPO through verifiable-reward reasoning and into VLA post-training. It is intentionally narrower than [Post-Training VLAs: A Reading Guide to Closed-Loop Improvement](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html). That guide covers the whole deployment loop: action interfaces, failure mining, feedback collection, critics, evaluation, and reproducibility. Here, those are held at the boundary. The subject is the update machinery itself—sampling distribution, advantage construction, likelihood ratio, and credit assignment.

The evidence cutoff is July 27, 2026. PPO, DPO, DeepSeekMath/GRPO, GKD, DeepSeek-R1, and the early VLA-RL systems are reported evidence. The final hybrid design is synthesis: a falsifiable proposal, not a result already established across robots.

## The policy-gradient objective

For a trajectory $\tau=(s_0,a_0,\ldots,s_T)$ sampled from policy $\pi_\theta$, a policy-gradient update has the form

$$
\nabla_\theta J
\approx
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t
\right].
$$

The score-function term says which behavior becomes more likely. The advantage $\hat A_t$ says whether that behavior was better than an appropriate baseline. Almost every practical dispute is hidden inside four choices:

| Choice | Question | Failure when chosen poorly |
| --- | --- | --- |
| Sampling distribution | Which policy generates the states and actions? | Offline data miss the learner's own failures; stale online data bias the update |
| Feedback | Is supervision a terminal reward, process reward, pairwise preference, or teacher distribution? | The signal cannot identify which decision mattered |
| Baseline | Learned critic, group mean, reference policy, or no scalar baseline? | Variance explodes, or useful differences are normalized away |
| Update restraint | Clipped ratio, KL penalty, reference-relative margin, or supervised divergence? | The policy exploits noisy evidence faster than the evidence improves |

This frame also separates two ideas that are routinely conflated. **On-policy** describes where training states come from: the current learner. **Reinforcement learning** describes how future reward weights an action's log-probability. A student can generate its own prefixes and receive teacher-token supervision there; that is on-policy distillation, even though no reward or policy gradient is required.

The figure holds the prompt fixed and changes only the source of contrast. It is not one four-stage pipeline. Each panel answers a different question about what evidence exists before the update begins.

[![Animation comparing how PPO, DPO, GRPO, and on-policy knowledge distillation construct a learning signal](/assets/images/blog-rl-learning-signals.gif)](/assets/images/blog-rl-learning-signals.gif)

*PPO compares sampled actions with a learned value baseline. DPO starts from an already collected chosen–rejected pair. DeepSeekMath's GRPO compares several current-policy completions of the same prompt. GKD samples the student's own prefix and asks a teacher for a dense next-token distribution there. Custom explanatory synthesis based on [PPO](https://arxiv.org/abs/1707.06347), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath](https://arxiv.org/abs/2402.03300), and [On-Policy Distillation](https://arxiv.org/abs/2306.13649).*

This controlled view explains why the methods are not interchangeable. PPO and GRPO need fresh policy samples because their advantages describe current behavior. DPO avoids generation because the contrast is already stored in the dataset. GKD is on-policy in state coverage but supervised in its target: teacher logits repair the student's visited prefixes without estimating future return. The later sections differ mainly in how expensive, noisy, or physically defensible each contrast becomes.

## PPO: Value baselines and clipped updates

[Proximal Policy Optimization](/paper%20shorts/2017/07/01/proximal-policy-optimization-ppo.html) made actor-critic policy gradients operationally simple. The actor collects trajectories under $\pi_{\mathrm{old}}$. A value network estimates expected future return, and Generalized Advantage Estimation turns temporal-difference residuals into lower-variance $\hat A_t$. The actor then reuses the rollout for several minibatch epochs under a clipped surrogate:

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right],
$$

where $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$.

The clipping rule answers a narrow question: how much should the optimizer trust an advantage computed under the old policy? Once a probability ratio moves beyond the allowed band in the apparently beneficial direction, the objective stops rewarding additional movement. PPO does not guarantee a bounded global KL, validate the reward, or make old trajectories current again. It merely makes a few passes over fresh data less destructive.

That compromise fit early RLHF. A language model supplies tractable token log-probabilities; a learned reward model scores a completion; a value model predicts return; and a frozen reference policy supplies a KL anchor. But the complete training stack can require the actor, reference, reward model, and critic in memory, while generation dominates wall-clock time. The critic is especially awkward for sparse sequence rewards: it must predict the eventual quality of a long answer from every partial prefix.

PPO therefore established the durable skeleton—online sampling, relative probability updates, and reference control—while making the value model the obvious component to challenge.

## DPO: Offline preference optimization

[Direct Preference Optimization](/paper%20shorts/2023/05/01/direct-preference-optimization-dpo.html) is often placed between PPO and GRPO as if it were a newer policy-gradient algorithm. It changes the problem more radically. DPO assumes the evidence already arrives as fixed pairs: for prompt $x$, response $y_w$ is preferred to $y_l$. Under a Bradley–Terry preference model and KL-regularized reward optimum, the implicit reward can be expressed through policy-to-reference log-ratios. The training objective becomes

$$
\mathcal{L}_{\mathrm{DPO}} =
-\mathbb{E}\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

There is no current-policy rollout, scalar advantage, reward model, or exploration. That is why DPO is stable and inexpensive relative to a full PPO loop. It is also why it cannot discover a reasoning strategy or robot state absent from the preference dataset.

DPO belongs in the evolution because it clarified what online RL was buying. If good matched alternatives already cover the deployment distribution, policy gradients may be unnecessary. If the model must generate new reasoning traces, expose its own failures, or enter states never labeled offline, removing the online loop removes the mechanism that creates those states.

For VLAs, the “matched” condition is particularly fragile. Two text answers can share the same prompt. Two physical trajectories rarely share exactly the same camera pose, friction, object state, and intervention history. An apparent preference can encode an environment difference rather than an action difference. DPO remains useful for replayable simulations, local action alternatives from the same logged state, or carefully constructed intervention windows. It is not a generic conversion from “one robot rollout succeeded” and “another failed” into a clean preference pair.

## GRPO: Group-relative baselines

[DeepSeekMath](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html) makes a different trade. For each question, the current policy samples $G$ answers. An exact or learned verifier produces rewards $r_1,\ldots,r_G$, and GRPO normalizes them within that question:

$$
\hat A_i =
\frac{r_i-\bar r}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

The normalized completion reward replaces the learned value model. PPO-style likelihood ratios, clipping, and a reference-policy KL penalty remain. The baseline is no longer “what return did the critic expect from this prefix?” It is “how did this answer compare with the other answers sampled for the same question?”

![PPO uses a learned value model; GRPO removes it and uses a same-prompt group baseline](/assets/images/deepseekmath-grpo-paper-figure.png)
_GRPO keeps the actor, reference model, and reward signal while removing PPO's critic. The saved model is exchanged for multiple completions per prompt. Source: [DeepSeekMath](https://arxiv.org/abs/2402.03300)._

This is an excellent bargain when three conditions hold:

1. several independent attempts at the same task are cheap;
2. an answer can be verified more cheaply than it can be demonstrated;
3. the policy is competent enough that a group contains both successes and failures.

Math and code fit unusually well. A sampler can produce eight solutions without changing the problem, and an exact checker can often score the final answer. DeepSeekMath reports gains after GRPO-based training across GSM8K, MATH, and multilingual mathematics. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) pushes the idea further: R1-Zero shows that verifiable-reward RL without supervised reasoning traces can elicit longer reasoning, self-checking, and strategy changes. Its language mixing and poor readability also show what correctness-only reward leaves unconstrained. The final R1 recipe adds cold-start data and staged training rather than treating pure RL as sufficient.

GRPO removes a model, not the need for a baseline. The group is the baseline, which makes group composition part of optimization.

## GRPO's zero-gradient failure

If all sampled answers are correct, every normalized reward is zero. If all are wrong, the same thing happens. Easy and impossible questions consume generation without supplying a gradient. Near-zero variance also makes normalization noisy. A fixed prompt distribution therefore becomes progressively inefficient as the policy improves.

Systems such as [DAPO](https://arxiv.org/abs/2503.14476) turn this observation into engineering: dynamically sample prompts that still produce informative outcomes, prevent overly long trajectories from dominating the token budget, and make clipping asymmetric enough to avoid suppressing useful low-probability actions. DAPO reports 50 on AIME 2024 with Qwen2.5-32B, but its larger lesson is that “GRPO” names only the core estimator. Sampling, filtering, token accounting, and reward design determine whether the estimator receives usable batches.

Later work names the pathology directly. [Advantage Collapse in GRPO](https://arxiv.org/abs/2605.21125) reports that homogeneous groups erase the learning signal and proposes sampling around advantage variance. Its experiments report substantially fewer collapsed groups and higher reasoning accuracy. The precise numbers are system-dependent; the structural result is not. A group-relative method cannot learn from a group with no relative information.

A second failure is temporal credit. Under outcome supervision, the same $\hat A_i$ weights every token in an answer. A correct final result reinforces the dead ends and lucky guesses along its path; an incorrect result penalizes any sound steps it contained. Larger groups reduce baseline variance but do not identify the decisive token.

Process rewards can address that, but they move difficulty into the verifier. A process critic must distinguish a recoverable detour from an invalid step, remain calibrated on current-policy traces, and resist optimization. The value model disappeared; a trustworthy process model can quietly reintroduce equivalent complexity.

## On-policy distillation

[On-Policy Distillation of Language Models](/paper%20shorts/2023/06/23/on-policy-distillation-language-models-gkd.html) attacks the same credit problem from another direction. Generalized Knowledge Distillation samples prefixes from the student and asks a teacher for the full next-token distribution on those prefixes. The student minimizes a divergence such as forward KL, reverse KL, or Jensen–Shannon:

$$
\mathcal{L}_{\mathrm{GKD}}
=
\mathbb{E}_{y\sim \pi_S}
\sum_t
D\left(
\pi_T(\cdot\mid x,y_{<t}),
\pi_S(\cdot\mid x,y_{<t})
\right).
$$

The state distribution is on-policy; the learning signal is dense supervised imitation. This distinction changes the error being corrected. Offline distillation teaches the student on teacher-written prefixes. At inference, one student mistake creates a prefix the teacher never wrote, and subsequent errors compound. GKD deliberately visits those student-created states and supplies a teacher distribution there.

![On-policy GKD consistently improves students more than fixed-data distillation across three generation tasks](/assets/images/on-policy-distillation-language-models-paper-figure.png)
_The source figure compares distillation methods on summarization, translation, and arithmetic. On-policy targets are useful across tasks, although the best divergence and mixture remain task-dependent. Source: [GKD](https://arxiv.org/abs/2306.13649)._

Distillation and RL optimize different evidence:

- GRPO can surpass the behavior distribution of a teacher when an external verifier rewards a better answer.
- GKD can tell the learner what to do at every visited prefix, but it inherits the teacher's distribution and errors.
- A mixed objective can use sparse task reward to choose outcomes and dense teacher divergence to shape the path.

That hybrid has become more relevant as reasoning policies improve. [Self-Supervised On-Policy Distillation for Reasoning LMs](https://arxiv.org/abs/2605.17497) uses the correct answers inside mixed GRPO groups as teachers for incorrect answers, extracting dense supervision from rollouts that would otherwise provide only a binary contrast. The 2026 paper reports a 65.6 macro Avg@12 for Qwen3-8B, 1.6 points above its GRPO baseline. This is frontier evidence, not yet a settled recipe, but it reveals the direction: reuse on-policy computation to manufacture denser, state-matched targets.

## Comparing feedback and estimation methods

| Method | Training states | Feedback unit | Baseline or anchor | Main cost | Native blind spot |
| --- | --- | --- | --- | --- | --- |
| PPO | Current-policy rollouts | Return or learned reward per trajectory/step | Learned value model; old and reference policies | Rollouts plus critic training | Critic error and stale on-policy data |
| DPO | Fixed preference dataset | Matched chosen/rejected pair | Frozen reference log-ratio | Preference collection | No exploration beyond dataset coverage |
| GRPO | Current-policy groups | Usually one verifiable reward per completion | Same-prompt group mean; reference KL | Multiple generations and verification | Homogeneous groups and coarse token credit |
| On-policy GKD | Student-generated prefixes | Teacher next-token distribution | Teacher distribution | Student generation plus teacher inference | Cannot exceed a flawed teacher without another objective |
| RL + distillation | Current-policy groups/prefixes | External outcome plus teacher/process targets | Group/critic plus teacher/reference | Most expensive stack | Conflicting reward and imitation signals |

The table prevents a common category error. DPO is not “GRPO but offline,” because its pairwise reference-relative margin encodes a different statistical object. On-policy distillation is not “RL with soft rewards,” because teacher logits supervise local action distributions without estimating future return. PPO and GRPO are closest: both weight policy ratios with advantages, but they construct the baseline differently.

## Why VLA policies need different assumptions

Transferring these estimators to a VLA requires more than replacing “token” with “action.” Three assumptions change at once.

### Environment resets

A math question can be sampled eight times from exactly the same prompt. A robot rollout changes the world. Resetting a deformable object, spill, drawer, or cluttered scene can require human labor and may not recreate the same state. Group-relative advantages are defensible only when rollouts are exchangeable enough that reward differences mostly reflect the policy.

Simulation restores cheap resets, but creates a new estimator boundary: an advantage can be accurate for the simulator and wrong for reality. The important budget is not episodes; it is independent, correctly reset, deployment-relevant state transitions per robot-hour.

### Continuous-action likelihoods

Autoregressive action tokens expose $\log\pi(a_t\mid s_t)$ directly. Modern VLAs often produce continuous chunks through diffusion or flow matching. A naïve PPO ratio over the final action ignores the stochastic denoising trajectory that generated it.

[DPPO](https://arxiv.org/abs/2409.00588) treats denoising as an augmented Markov decision process and applies policy optimization to the diffusion policy's actual stochastic transitions. This is more than an implementation detail. If the likelihood ratio does not correspond to the deployed sampler, clipping controls the wrong distribution.

### Long-horizon credit assignment

Copying one answer reward across 1,000 reasoning tokens is noisy. Copying one success bit across a long robot trajectory also confounds perception, planning, contact, controller latency, and environment dynamics. An early grasp may be correct even if the final placement fails; a bad approach may succeed because the gripper catches by chance.

VLA post-training therefore needs a feedback unit between token and episode: action chunk, skill segment, state transition, or critic-estimated progress. That unit must remain replayable or observable enough to compare alternatives. Otherwise denser reward only creates denser confidence, not better credit.

## VLA reinforcement-learning methods

The early VLA-RL literature can be organized by what replaces the critic and where high-quality trajectories come from.

### Direct policy optimization in simulation

DPPO is the clean direct-RL branch for diffusion actors: retain online interaction, model the correct action likelihood, and optimize reward. [RIPT-VLA](https://arxiv.org/abs/2505.17016) shows a critic-free alternative for autoregressive VLAs. It uses sparse binary success, dynamic rollout sampling, and leave-one-out relative advantages. The paper reports a 21.2-point gain for QueST and 97.5% for an OpenVLA-OFT setup; in a one-demonstration case, it reports improvement from 4% to 97% within 15 iterations.

Those numbers are striking because the environment supplies cheap, repeated comparisons. They should not be interpreted as a general conversion rate from demonstrations to physical mastery. The estimator needs resets, reliable success detection, and tasks whose exploration boundary is reachable from the starting policy.

[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) pushes the same systems direction on LIBERO and RoboTwin: scale parallel rollouts, filter uninformative groups, and make the policy update compatible with the VLA's action interface. As in reasoning, the optimizer's name hides the rollout scheduler.

### RL-generated training data

[RLDG](https://arxiv.org/abs/2412.09858) trains task-specific RL specialists and distills their trajectories into a general robot policy. The source paper reports up to 40% higher success than human-demonstration training in its evaluated settings. The architectural lesson is broader than the number: RL can improve the state distribution without directly exposing a shared foundation model to every unstable reward update.

This resembles on-policy distillation at a different boundary. A specialist visits difficult states and discovers successful behavior; the generalist learns from the resulting trajectories with supervised objectives. The teacher may be a policy, planner, privileged simulator agent, or previous checkpoint. Direct RL spends less storage and can adapt rapidly; RL-generated-data pipelines make updates reviewable, replayable, and easier to mix across embodiments.

### Process rewards

The frontier is moving toward intermediate feedback. [LifeLong-RFT](https://arxiv.org/abs/2602.10503) reports chunk-level on-policy RL and a multidimensional process reward, with a 22% average success gain over SFT while using 20% of the data in its evaluated suite. [RobustVLA](https://arxiv.org/abs/2511.01331) adds observation-sensitivity and action-smoothness regularization so reward improvement does not purchase brittle control. [VLA-RFT](https://arxiv.org/abs/2510.00406) uses a world simulator and verified rewards to reduce dependence on physical rollouts.

These systems are recent and heterogeneous. Their reward definitions, simulators, embodiments, action heads, and evaluation protocols differ, so their headline gains should not be ranked as if they were optimizer ablations. What they collectively support is narrower: VLA RL is becoming a joint design of rollout distribution, feedback granularity, and action likelihood.

## Choosing an estimator

The method decision can be made without starting from an acronym:

| Available evidence | Best first estimator | Why | Required control |
| --- | --- | --- | --- |
| Exact verifier, cheap identical resets, mixed outcomes | GRPO or leave-one-out group RL | Same-context samples form a cheap baseline | Measure zero-variance group rate and reset fidelity |
| Dense trusted teacher on student states | On-policy distillation | Corrects visited mistakes without terminal credit assignment | Compare teacher tokens and wall-clock to refreshed offline replay |
| Continuous diffusion actions with simulator reward | DPPO-style policy gradient | Matches update ratio to actual sampler | Verify augmented-MDP likelihood and sim-to-real ranking |
| Fixed matched preferences, no safe online exploration | DPO | Uses the evidence that actually exists | Audit state matching and reference coverage |
| Privileged specialist or simulator, risky shared backbone | RL-generated data plus distillation | Separates exploration from generalist update | Hold environment steps and final data quality fixed |
| Sparse physical success with costly resets | Correction SFT or local process critic first | Episode-relative RL wastes too much interaction | Earn a matched-state or segment-level feedback unit |

The last row is deliberately conservative. “On-policy” does not make evidence causal. If two robot episodes cannot be matched well enough for their reward difference to identify a policy decision, a relative advantage is numerically valid and scientifically weak.

## A hybrid training hypothesis

The evolution suggests a shared endpoint: sparse external reward should select outcomes, while dense on-policy supervision should repair the trajectory states that produced them.

For reasoning, sample a same-prompt group, use the verifier to identify successful completions, then distill their token distributions or step structure into failed student prefixes. Retain a small KL anchor so locally dense imitation does not erase useful diversity. This converts group generation from one scalar per answer into state-matched supervision while preserving the ability to prefer solutions that beat an older teacher.

For VLAs, the corresponding unit should be an action chunk or state transition, not an entire episode. Sample comparable rollouts in simulation or controlled resets. Use terminal task success for the group-level objective. Use a privileged teacher, process critic, or successful neighboring rollout to supervise only the segments where state matching is defensible. Distill successful behavior into the generalist, then validate the resulting policy under real closed-loop trials.

This hypothesis is falsifiable. Compare four methods under identical model initialization, environment transitions, reset labor, and inference compute:

1. correction SFT on successful and intervention segments;
2. sparse group-relative RL;
3. on-policy distillation from a privileged teacher;
4. the hybrid of sparse RL and segment-level distillation.

Report success and safety, but also zero-variance group rate, critic/teacher calls, policy KL, action entropy, recovery success, real robot-hours, and performance after the teacher or simulator distribution shifts. The hybrid earns its complexity only if it improves reliable success per unit of interaction—not merely final reward after consuming more rollouts and teacher inference.

## Where each method gets its learning signal

PPO learns a critic so each action can be compared with expected return. DPO assumes the contrast has already been collected as a chosen/rejected pair. GRPO obtains the contrast from other attempts at the same prompt. On-policy distillation replaces a scalar contrast with a teacher distribution at the learner's own states.

Reasoning models made GRPO powerful because prompts reset perfectly and verifiers are cheap. VLAs expose the limits because physical state does not reset cleanly, action likelihood may pass through diffusion, and a terminal bit is far from the causal motion. The transferable object is therefore not GRPO itself. It is the estimator-design discipline:

> Sample where the policy will act, compare only what the environment makes comparable, and make feedback no coarser than the decision it is supposed to credit.

That principle explains the past decade of policy optimization better than the acronym sequence—and gives VLA post-training a testable path beyond copying language-model RL.

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588)
- [RLDG](https://arxiv.org/abs/2412.09858)
- [RIPT-VLA](https://arxiv.org/abs/2505.17016)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [RobustVLA](https://arxiv.org/abs/2511.01331)
- [LifeLong-RFT](https://arxiv.org/abs/2602.10503)
- [Self-Supervised On-Policy Distillation for Reasoning LMs](https://arxiv.org/abs/2605.17497)
- [Advantage Collapse in GRPO](https://arxiv.org/abs/2605.21125)
