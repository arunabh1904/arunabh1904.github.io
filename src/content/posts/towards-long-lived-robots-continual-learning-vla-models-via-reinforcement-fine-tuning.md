---
title: 'Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning'
date: '2026-02-11T00:00:00.000Z'
section: paper-shorts
postSlug: towards-long-lived-robots-continual-learning-vla-models-via-reinforcement-fine-tuning
legacyPath: /paper shorts/2026/02/11/towards-long-lived-robots-continual-learning-vla-models-via-reinforcement-fine-tuning.html
tags:
  - Vision-Language-Action
  - Reinforcement Learning
  - Continual Learning
field: 'Robot Post-Training & Evaluation'
summary: '2026 – Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning'
---

## 2026 – Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning

**Paper:** [Liu et al., arXiv:2602.10503v2](https://arxiv.org/abs/2602.10503v2) · **Project:** [LifeLong-RFT](https://yuan-liu-lifelong-rft.github.io/)

## Summary

> LifeLong-RFT applies group-relative reinforcement fine-tuning to discrete-action VLAs using expert-demo observations and three chunk-level rewards: token consistency, continuous trajectory alignment, and output-format validity. On NORA-Long, it improves LIBERO's sequential-learning AUC from 54.4 to 90.3 on LIBERO-Goal and reduces real-world negative backward transfer from 18.3 to 6.1. The evidence is a strong post-training result for this Fast+-tokenized, discrete-action model and its demonstration protocol; it is not yet online continual learning from newly encountered robot states.

## Core Insights

The paper starts from a specific mismatch between a pretrained VLA and a long-lived robot. Supervised fine-tuning can learn a new task from demonstrations, but the update is tied to the current dataset and can erase earlier skills. The authors therefore keep the observations and language instructions from demonstration data, sample several candidate action chunks from the current policy, and compare those candidates with a reward that can be computed without a simulator, a physical robot, or a pretrained reward model.

This is a narrower claim than “RL solves continual learning.” The policy is on-policy with respect to the generated action chunks, while the observation distribution remains the expert-demonstration buffer. The approach avoids expensive environment resets and makes the reward reproducible, but its closed-loop success tests do not isolate recovery from deliberately off-demonstration states.

### Three rewards connect tokens to control

The source method diagram makes the interface explicit: a VLA maps an observation and instruction to discrete action tokens, decodes them into a chunk of continuous robot actions, and receives three complementary signals. The method inherits NORA-Long's visual-language backbone and action interface. This report and its project page do not specify the image resolution, visual connector dimensions, numerical chunk horizon, or control cadence; those remain prerequisites for reproducing the complete system.

![LifeLong-RFT samples action chunks from a VLA and scores them with quantized consistency, continuous trajectory alignment, and format compliance rewards.](/assets/images/lifelong-rft-source-figure-2.png)
*Fig 1: The method's source Figure 2 connects the discrete action-token sequence to a decoded continuous chunk and its three process rewards. Rewards compare sampled actions with demonstration references; they do not observe the environmental consequences of those actions during training. | source: [Towards Long-Lived Robots, Figure 2](https://arxiv.org/html/2602.10503v2#S3.F2)*

[Open the figure at full size](/assets/images/lifelong-rft-source-figure-2.png)

Quantized Action Consistency Reward (QACR) first checks that the generation has the shape required by the Fast+ action tokenizer. If it is valid, the reward counts position-wise matches against the demonstration's token sequence and divides by the longer of the two lengths:

$$
\operatorname{QACR}=
\frac{\sum_{\ell=1}^{\min(U,V)}\mathbf{1}[a_\ell=\tilde a_\ell]}{\max(U,V)}.
$$

Continuous Trajectory Alignment Reward (CTAR) decodes the candidate tokens and scores each timestep's pose and gripper state. The pose term is an exponential penalty on normalized $L_1$ distance, $r_t^{\mathrm{pose}}=\exp(-\alpha d_t)$; the gripper term is exact agreement. The chunk reward averages $\beta r_t^{\mathrm{pose}}+(1-\beta)r_t^{\mathrm{grip}}$ across the horizon. Format Compliance Reward (FCR) is binary: malformed action dimensions or horizons receive zero, while a decodable chunk receives one.

The final reward is

$$
r=\omega\operatorname{QACR}+(1-\omega)\operatorname{CTAR}+\lambda\operatorname{FCR}.
$$

This decomposition matters because token agreement and physical agreement are not interchangeable. A token mismatch does not reveal how far the decoded robot motion deviates; malformed token strings may have no executable trajectory at all. The paper's ablation isolates the difference within its setup: removing QACR lowers average LIBERO success from 95.6% to 92.8%, removing FCR gives 93.0%, and removing CTAR gives 4.7%. The CTAR result is a large task-specific dependency, not evidence that every VLA needs this exact weighting.

The optimizer is GRPO, the same group-relative estimator discussed in [DeepSeekMath](/paper%20shorts/2024/02/05/deepseekmath-group-relative-policy-optimization-grpo.html) and in the site's [PPO-to-GRPO guide](/blog/2026/07/27/from-ppo-to-grpo-rl-for-reasoning-and-vlas.html). For each demonstration observation and instruction, the old policy samples a group of eight chunks. Their rewards are normalized by the group's mean and standard deviation, and the clipped policy objective includes a KL penalty to a reference policy. The implementation uses full-parameter updates, AdamW at $10^{-6}$, $\alpha=5$, $\beta=0.8$, $\omega=0.7$, $\lambda=0.1$, and eight H20 GPUs. Appendix A specifies sampling temperature 0.8 and reference-KL weight 0.001. LIBERO multi-task training uses batch 128 for 15 epochs, except Long, which uses batch 256 for 35; incremental updates use batch 32 for ten epochs.

### The evaluation measures both plasticity and stability

For ordinary multi-task adaptation, the backbone is NORA-Long. SimplerEnv uses BridgeData V2 for WidowX and Fractal for the Google Robot, with 24 trials per task. LifeLong-RFT raises the NORA-Long average from 65.5% to 69.0% on WidowX and from 74.7% to 79.1% on the Google Robot. On LIBERO, the model sees all ten tasks in each suite with third-person and wrist cameras; 500 trials per suite produce a 95.6% average versus 91.8% for SFT. The largest suite-specific gain is on LIBERO-Long, 82.4% to 89.0%.

The real-world test uses a Franka robot and four tasks: Pick Banana, Pick Bread, Pull Drawer, and Hang Chinese Knot. The authors jointly train on 40 demonstrations for each of the first three tasks and 50 for the last, then run 20 trials per task. The average success rate rises from 78.8% for SFT to 87.5% for LifeLong-RFT; the hardest reported task, Hang Chinese Knot, improves from 60% to 75%. These are task-level success results, not a measurement of failure recovery under distribution shift.

The continual-learning protocol follows [LIBERO's benchmark distinction between forward transfer, forgetting, and sequential performance](/paper%20shorts/2023/06/05/libero-benchmarking-knowledge-transfer-for-lifelong-robot-learning.html). Each suite first trains on six tasks with 50 demonstrations each. Four new tasks then arrive one at a time with ten demonstrations each, while five demonstrations per previous task are retained for experience replay. Forward Transfer (FWT) measures performance on the new task, Negative Backward Transfer (NBT) measures the drop on earlier tasks, and AUC summarizes success across the whole sequence.

On LIBERO-Goal, LifeLong-RFT reaches FWT 92.4, NBT 3.1, and AUC 90.3, compared with the NORA-Long SFT baseline's 72.8, 25.2, and 54.4. On the full LIBERO table, the gains are consistent: AUC rises from 79.7 to 94.8 on Object, 71.7 to 91.2 on Spatial, and 47.3 to 64.5 on Long. The real-world sequence uses 20 demonstrations per new task and five retained per earlier task. It is smaller but points in the same direction: FWT goes from 56.3 to 80.0, NBT from 18.3 to 6.1, and AUC from 44.2 to 75.9.

The data-efficiency result is also bounded. In a representative adaptation test, LifeLong-RFT reaches 100% on Pick Orange Juice with five demonstrations where SFT reaches 98% with 50; on Pick Bowl, five demonstrations match the SFT result at 50 and ten reach 100%. Those comparisons use selected tasks and the paper's checkpoint and evaluation procedure. They show that the reward can use a small new-task set effectively; they do not establish that ten demonstrations suffice for arbitrary embodiments or long-horizon skills.

### What the paper leaves open

The method's cleanest advantage is its reward construction: it replaces privileged environment success and a learned reward model with comparisons against expert action chunks. That makes training cheap and repeatable, but it also makes the demonstration trajectory the central reference. QACR rewards token identity, CTAR rewards decoded proximity, and FCR rewards syntax; none directly observes whether an unseen object was grasped, whether contact forces were safe, or whether the task succeeded after a long sequence of closed-loop corrections.

Appendix E also reports improved retention with RoboBrain-X0, extending the result beyond NORA-Long within discrete-action VLAs. Continuous-action models remain future work. A convincing next test would preserve the same task sequence while charging for environment interaction, varying the amount of replay, and evaluating states reached after the policy departs from the demonstrations. If the method retains its low NBT under those controls, the claim would move from efficient demonstration-conditioned adaptation toward continual robot learning in the stronger sense.

## High-Level Takeaways

- Chunk-level rewards make a discrete VLA's update legible: token agreement, decoded motion, and output validity are measured separately before they are combined.
- The reported continual-learning gains are large on LIBERO and positive on a four-task Franka sequence, but the observation states still come from demonstrations and replay.
- CTAR is the dominant ablation in this paper's setting; removing it collapses average LIBERO success, which makes the continuous decoder and trajectory reference part of the method rather than an implementation detail.
- GRPO removes a value model, while the reward design supplies the missing structure; the training system still depends on group sampling, a reference policy, a tokenizer contract, and expert trajectories.
- A decisive follow-up would deliberately perturb closed-loop states away from demonstrations, vary replay, and test long-horizon recovery and continuous-action VLAs.
