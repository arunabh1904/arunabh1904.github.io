---
title: 'SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation'
date: '2024-05-09T00:00:00.000Z'
section: paper-shorts
postSlug: simpler-evaluating-real-world-robot-policies-in-simulation
legacyPath: /paper shorts/2024/05/09/simpler-evaluating-real-world-robot-policies-in-simulation.html
tags:
  - Robotics
  - Evaluation
field: 'Robot Post-Training & Evaluation'
summary: "2024 – SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation"
---

## 2024 – SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation

**arXiv:** [2405.05941](https://arxiv.org/abs/2405.05941)

**Project:** [simpler-env.github.io](https://simpler-env.github.io/)

## Summary

> SIMPLER asks a more useful question than whether a simulator is photorealistic: does an evaluation inside it predict how real policies rank and fail? The benchmark recreates common Google Robot and WidowX setups, reduces control and visual mismatches, and runs real-data-trained policies without retraining them in simulation.

## Core Insights

![SIMPLER comparison of expensive real-robot evaluation with reproducible simulated evaluation matched to the same task](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-paper-figure.png)
*Figure 1 states the benchmark proposition directly: replace repeated physical evaluations with purpose-built simulated replicas, then validate the replica by whether policy rankings correlate with real-world performance. source: [SIMPLER](https://arxiv.org/abs/2405.05941)*

![Figure 2 from SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-source-figure-2.webp)
*Figure 2 Fig. 2 : We perform system identification (SysID) for closing the control gap between real and simulated environments. We visualize the open-loop execution of demonstration actions (using 6D end-effector pose control) for picking up a coke can before and after SysID ( Section IV-A ). Afterwards, the simulated robot arm tracks the real motion much more accurately and successfully reproduces the pick-up behavior. source: [SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941)*

![Figure 0 from SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](/assets/images/simpler-evaluating-real-world-robot-policies-in-simulation-source-figure-0.webp)
*Figure 0 Fig. 0 : We introduce SIMPLER, a suite of open-source simulated evaluation environments for common real robot manipulation setups, namely the Google Robot evaluations from the RT-series of works [ 6 , 5 , 11 ] , and environments from the BridgeData V2 dataset [ 66 ] . All environments can be imported with a single line of code and can be interacted with through a standard Gym interface. source: [SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941)*


Across RT-1, RT-1-X, RT-2-X, and Octo and roughly 1,500 evaluation episodes, simulated success correlates strongly with paired real-world performance. The simulator also reflects behavioral sensitivities under several distribution shifts. The result makes simulation a screening layer, not a replacement for real trials.

The main design choice is calibration at the policy interface. Robot control mode, camera pose, observation preprocessing, and object appearance must be close enough that differences between policies survive the domain gap. A visually impressive environment that changes the controller distribution can give worse rankings.

| Evaluation layer | Strength | Boundary |
| --- | --- | --- |
| SIMPLER | Cheap, reproducible, high-volume comparison | Correlation is specific to tasks, policies, and matched interfaces |
| Real robot | Captures actual physics and operations | Expensive, noisy, difficult to reproduce |

## High-Level Takeaways

- SIMPLER informs which regressions can be screened in simulation before spending robot hours. Its atomic unit is a closed-loop simulated episode executed by a policy trained on real data. The scaling claim concerns rank correlation and failure-mode similarity, not simulation realism by itself.
- A missing test repeatedly recalibrates the correlation as new policy families, controllers, and tasks arrive. At ten times the capability breadth, one simulator may preserve rankings for some skills and invert them for others. The central claim fails if improvements selected by SIMPLER do not predict real gains prospectively rather than retrospectively.
- SIMPLER supplies the real-to-sim middle layer in an evaluation pyramid.
- Correlation on existing policies can break after architecture or action-interface changes.
- Use simulation when it predicts a decision you care about; measure that prediction continuously.
