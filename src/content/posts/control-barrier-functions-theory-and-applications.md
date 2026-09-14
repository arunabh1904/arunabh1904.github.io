---
title: "Control Barrier Functions: Theory and Applications"
date: "2019-03-27T00:00:00.000Z"
section: paper-shorts
postSlug: control-barrier-functions-theory-and-applications
legacyPath: /paper shorts/2019/03/27/control-barrier-functions-theory-and-applications.html
tags: ["Control Theory", "Safety"]
field: "Motion Forecasting & Planning"
summary: "2019 – Control Barrier Functions: Theory and Applications"
---

## 2019 – Control Barrier Functions: Theory and Applications

**Paper:** [arXiv:1903.11199](https://arxiv.org/abs/1903.11199) · [PDF](https://arxiv.org/pdf/1903.11199)

## Summary

> Control Barrier Functions: Theory and Applications explains how a controller can preserve an explicitly defined safe set under stated dynamics and feasibility assumptions. A quadratic program minimally modifies a desired action to satisfy a barrier constraint. The guarantee comes from the system model and admissible-control condition, not from a learned representation being organized in BEV coordinates.

## Core Insights

### Preserve a set rather than predict a safe-looking action

Consider a control-affine system $\dot x=f(x)+g(x)u$ and a safe set $C=\{x:h(x)\geq0\}$. A control barrier function permits controls satisfying

$$
L_fh(x)+L_gh(x)u\geq-\alpha(h(x)).
$$

Here the Lie derivatives describe how the safety margin changes under the uncontrolled dynamics and the chosen input; $\alpha$ is an extended class-K function. At the boundary, the condition prevents the state from flowing outward. Inside the set, it can allow the margin to shrink, avoiding the excessive conservatism of requiring it to increase everywhere.

The theorem requires regularity, including a nonzero gradient on the boundary, and a suitable locally Lipschitz controller choosing admissible inputs. Safety means forward invariance from the safe set for the modeled system. It does not certify arbitrary initial conditions, unmodeled dynamics, or an inaccurate estimate of the state.

### Turn the desired policy into a constrained control problem

Given a nominal action $u_{\mathrm{nom}}$, a common implementation solves

$$
\min_{u\in U}\frac{1}{2}\lVert u-u_{\mathrm{nom}}\rVert^2
\quad\text{subject to}\quad
L_fh(x)+L_gh(x)u\geq-\alpha(h(x)).
$$

The objective preserves the desired behavior when possible. The inequality limits how much safety margin may be spent. The paper first presents the unconstrained-input case and then addresses input bounds: a barrier is useful only if an admissible action can enforce it. Choosing a safe set without considering braking or steering authority can create an infeasible constraint precisely when intervention is needed.

Read the source diagram as a feedback controller with an inserted filter. The nominal controller still proposes actions, while the filter evaluates them using the system state before execution.

![Desired action filtered by an active set invariance filter before reaching the system; source Figure 4](/assets/images/control-barrier-functions-source-figure-4.webp)
*Fig 1: The safety filter sits between a nominal controller and the physical system. This crop preserves the original feedback diagram while omitting surrounding page text. | source: [Paper, Figure 4](https://arxiv.org/abs/1903.11199)*

[View full-size figure](/assets/images/control-barrier-functions-source-figure-4.webp)

The survey develops extensions for higher-relative-degree constraints, where the input affects a safety quantity only after multiple time derivatives. It also describes applications to stepping-stone walking, automotive lane keeping and speed regulation, Segway balance, and long-duration autonomy. These are demonstrations of particular modeled constraints rather than a shared benchmark with one overall success rate. In the lane-keeping example, the barrier includes lateral velocity and available lateral acceleration, so current position alone is insufficient to assess recoverability.

A learned driving policy could supply the nominal action, but connecting it to this framework requires an explicit physical state, constraints, dynamics, and a feasible control interface. An opaque BEV feature grid does not supply those objects automatically. Sensor error, discretization, computation delay, and disturbance assumptions must be handled in the actual controller design; the continuous-time theorem cannot simply be relabeled as a deployment guarantee.

## High-Level Takeaways

- A safety claim needs a defined safe set and a proof that available controls can preserve it for the modeled dynamics.
- Keep nominal performance and constraint enforcement separate. A learned planner may propose the action while a model-based filter enforces a specific constraint.
- Test feasibility, state-estimation error, delays, and actuator limits before transferring a theoretical guarantee to a physical driving system.
