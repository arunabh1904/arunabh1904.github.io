---
title: 'WPT: World-to-Policy Transfer via Online World Model Distillation'
date: '2025-11-25T00:00:00.000Z'
section: paper-shorts
postSlug: wpt-world-to-policy-transfer-via-online-world-model-distillation
legacyPath: /paper shorts/2025/11/25/wpt-world-to-policy-transfer-via-online-world-model-distillation.html
tags: [Other]
field: 'Autonomous Driving: VLA & Planning'
summary: '2025 – WPT: training-time world-model reasoning for a lightweight driving policy'
---
## 2025 – WPT: World-to-Policy Transfer via Online World Model Distillation

**arXiv:** [2511.20095](https://arxiv.org/abs/2511.20095)

**Paper:** [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_WPT_World-to-Policy_Transfer_via_Online_World_Model_Distillation_CVPR_2026_paper.html)

## Summary

> WPT uses a predictive world model as a training-time teacher for autonomous driving. A reward model scores candidate trajectories using imitation and simulated interaction signals, and two distillation losses transfer the teacher’s planning representation and preferred trajectory into a lightweight student. The student does not call the world model at deployment, so the paper measures the value of predictive reasoning against the latency cost of keeping it online.

## Core Insights

### Spend predictive compute while learning

The paper separates the policy that can afford to reason about futures from the policy that must run in real time. During training, the teacher generates multiple candidate trajectories. A frozen world model rolls those candidates forward, and a reward model evaluates what each candidate would do to the future scene. The student generates one trajectory and learns from the teacher’s internal plan and its selected outcome.

The world-model choice follows the benchmark. On nuScenes, WPT uses Drive-OccWorld, an occupancy model that predicts future 4D occupancy. On Bench2Drive, it uses an instance-based model that forecasts future agent states and lane topology. This is a practical design rather than one universal simulator: the teacher sees future structure in the representation available for each dataset.

![WPT framework showing teacher and student policies interacting with a world model during training](/assets/images/wpt-world-to-policy-transfer-via-online-world-model-distillation-paper-figure.webp)
*Fig 1: During training, a world model rolls out candidate futures and a reward model selects the teacher trajectory; query and reward distillation then train a world-model-free student. | source: [WPT: World-to-Policy Transfer via Online World Model Distillation, Figure 2](https://arxiv.org/abs/2511.20095)*

Follow Fig. 1 from the teacher branch to the student branch. The teacher’s planning queries are aligned with the student’s queries using `L_policy = ||Q_S − Q_T||₂`. The teacher’s best trajectory `τ_T*` is selected by its final reward, and the student is trained to match that score with `L_reward = ||r_final(τ_S) − r_final(τ_T*)||₂`. The student is therefore asked to reproduce both a useful internal plan and the world-model-informed preference over trajectories. After training, the world model, teacher, and reward model leave the inference path.

### Turn future interaction into inspectable rewards

WPT combines an imitation reward with five simulation signals: no collision (NC), drivable-area compliance (DAC), ego progress (EP), time to collision (TTC), and comfort. The imitation reward is derived from the trajectory’s distance to the expert path. The simulation rewards inspect the candidate in the predicted future: is it collision-free, inside the drivable area, progressing forward, safe from imminent contact, and within acceleration and jerk limits? The final reward fuses the imitation and simulation terms.

![Figure 1 from WPT comparing training paradigms for autonomous-driving policies](/assets/images/wpt-world-to-policy-transfer-via-online-world-model-distillation-source-figure-1.webp)
*Fig 2: The paper contrasts imitation learning, runtime world-model policies, simulator-based reinforcement learning, and WPT’s training-only world-model interaction. | source: [WPT: World-to-Policy Transfer via Online World Model Distillation, Figure 1](https://arxiv.org/abs/2511.20095)*

Fig. 2 explains the deployment claim. WPT is not saying that a student can forecast the world by itself. It says the teacher can use future-aware evaluation during training, then compress the resulting planning preference into a policy whose inputs and runtime graph look like the baseline policy. This is why the student must be reported separately from the teacher.

### The open-loop numbers show what transfers

The nuScenes experiment uses 1,000 twenty-second scenes with approximately 1.4 million images and 23-class 3D boxes at 2 Hz. The standard split is 700/150/150 scenes for train, validation, and test. On the validation set, the same student architecture without WPT reaches 0.88 m average L2 displacement and 1.06% average collision. The WPT student reaches 0.66 m and 0.24%; the teacher reaches 0.61 m and 0.11%. At the 3-second horizon, the teacher’s collision rate is 0.10%, showing where future-aware scoring has its clearest safety effect.

Those values are open-loop planning metrics. They are useful for controlled trajectory comparison, but they do not replace closed-loop driving. Bench2Drive provides that second test: WPT trains on 950 of 1,000 clips, reserves 50 for open-loop validation, and evaluates closed loop on 220 predefined routes. The baseline reports 65.23 driving score and 34.10% success. The student reaches 72.61 and 45.45%; the teacher reaches 79.23 and 54.54%. The student’s 64 ms planning latency matches the baseline, while the teacher takes 312 ms with online reward scoring (286 ms without it), giving the student a 4.9× latency advantage over the full teacher path.

The student keeps much of the teacher’s efficiency profile as well: its reported efficiency is 188.52 versus 188.63 for the teacher. It does not retain every teacher gain, however. That gap is the practical point of WPT: predictive reasoning can be distilled, but compression still changes behavior in the cases where the teacher’s rollout search matters most.

### Ablations identify the expensive ingredients

The reward ablation starts from a teacher without the proposed reward model at 0.72 m L2 and 0.71% collision. Adding imitation reward during training gives 0.78 m and 0.23%; adding all simulation rewards during training gives 0.62 m and 0.14%; enabling reward scoring at inference reaches 0.61 m and 0.11%. Removing TTC increases collision to 0.25%, the largest single simulation-reward degradation in the table. The result makes safety supervision more specific than the generic statement that “more reward” helps: TTC carries the largest reported safety contribution, while the other signals add smaller improvements.

The occupancy-source ablation separates the world model from the reward design. With imitation reward only, ground-truth occupancy gives 0.64 m and 0.16% collision, while world-model occupancy gives 0.69 m and 0.22%. With all rewards, ground-truth occupancy gives 0.65 m and 0.11%, while world-model occupancy gives 0.61 m and 0.11%. Predictive occupancy is not always the cleaner signal, but it produces the best combined planning result when the student is trained on the same kind of future representation it will inherit from the teacher.

The distillation ablation starts at the baseline’s 0.88 m and 1.06%. Query distillation alone gives 0.69 m and 0.86%; adding imitation-reward distillation gives 0.68 m and 0.25%; adding simulation-reward distillation gives the final 0.66 m and 0.24%. This is evidence for the two-loss design rather than for a single feature-matching shortcut.

WPT’s boundary is the world model’s own forecast. A rare collision or interaction can dominate driving risk, and a reward model can transfer a forecast error as confidently as a useful preference. The paper also uses different world-model structures across nuScenes and Bench2Drive, so the results establish a training paradigm across two settings rather than a universal world model. The teacher’s better closed-loop score and the student’s lower latency are both part of the claim.

## High-Level Takeaways

- WPT makes predictive world modeling a teacher-time resource and removes it from the deployed student.
- Query alignment transfers planning structure; reward alignment transfers the teacher’s preference among candidate futures.
- TTC is the clearest safety-critical ablation, while the student retains most—but not all—of the teacher’s closed-loop advantage.
- Open-loop L2, closed-loop driving, and latency measure different parts of the system and should stay separate when reading the result.
