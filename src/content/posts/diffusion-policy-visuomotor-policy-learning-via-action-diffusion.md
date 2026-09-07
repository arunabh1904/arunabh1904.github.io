---
title: 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion'
date: '2023-03-07T00:00:00.000Z'
section: paper-shorts
postSlug: diffusion-policy-visuomotor-policy-learning-via-action-diffusion
legacyPath: /paper shorts/2023/03/07/diffusion-policy-visuomotor-policy-learning-via-action-diffusion.html
tags:
  - Robotics
  - Diffusion
field: 'Vision-Language-Action & Robotics'
summary: "2023 – Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
---

## 2023 – Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**arXiv:** [2303.04137](https://arxiv.org/abs/2303.04137)

**Project:** [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)

## Summary

> Diffusion Policy predicts a distribution over short action trajectories by denoising noise into a sequence of robot commands. It pairs that expressive trajectory model with receding-horizon execution, so the controller can coordinate several actions and still replan when the next image disagrees with the prediction.

## Core Insights

### Why a trajectory distribution helps

A direct behavior-cloning regressor maps an observation to one action, often averaging demonstrations that contain different valid choices. The paper instead models $p(A_t\mid O_t)$, where $A_t$ is a finite action sequence and $O_t$ is a history of observations. It samples a clean trajectory $A_t^0$ from demonstrations, adds noise at a random diffusion step $k$, and trains a network to predict that noise:

$$\mathcal{L}=\operatorname{MSE}\left(\epsilon_k,\epsilon_\theta(O_t,A_t^0+\epsilon_k,k)\right).$$

At inference, the model begins with a noisy action sequence and repeatedly removes predicted noise. The observation conditions the denoising network but is not itself diffused, which lets the visual representation be computed once per control cycle. A CNN version applies FiLM conditioning at every temporal convolution. The paper's time-series transformer sends each action token through causal attention to itself and earlier action tokens, while cross-attending to the observation tokens. The two designs share the same conditional trajectory objective.

The controller predicts $T_p$ actions, executes only $T_a$, then observes again. This is the central compromise: a whole chunk gives the model room to express a coherent contact sequence, while the short executed prefix limits the damage from a stale prediction. The next inference can be warm-started from the previous sequence. In the real-world experiments, DDIM reduces a 100-step training schedule to 10 inference steps and reports about 0.1 seconds per policy call on an Nvidia 3080; the real-world hyperparameter table uses 16 inference iterations for the reported tasks.

### The design at a glance

![Diffusion Policy conditions action-sequence denoising on an observation history](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-paper-figure.png)
*Fig 1: The policy starts from noisy actions, repeatedly predicts the denoising direction, and uses either FiLM-conditioned temporal convolutions or a causal action transformer with observation cross-attention. | source: [Diffusion Policy, Figure 2](https://arxiv.org/abs/2303.04137)*

Figure 1 explains why this is more than “diffusion for actions.” The output is a sequence with internal temporal structure; the observation enters as context at every denoising stage. Causal attention prevents a later action token from leaking information backward through the action sequence, while cross-attention preserves visual conditioning. The policy therefore represents multiple smooth ways to reach the same local goal without an explicit mixture-of-Gaussians head.

### Evidence from simulation and real robots

The paper evaluates behavior cloning on 15 tasks across four benchmarks, covering 2-DoF to 6-DoF actions, single-arm and bimanual systems, rigid and fluid objects, and state or image observations. Across its comparison set it reports a 46.9% average improvement. The controlled state benchmarks help separate distribution modeling from camera perception: the transformer variant reaches 0.99/0.94 on the hardest reported multi-stage BlockPush metrics and 0.99/0.96 on Kitchen's difficult metrics, while the baselines are lower (Table 4). The authors also report that an action horizon of eight steps works best for most tasks; longer horizons smooth motion but react too slowly, while a one-step policy loses temporal consistency.

The Push-T experiment shows the cost of averaging modes. In the real setup, the task first pushes a T-shaped block into a target and then moves the end effector to an end zone. The end-state IoU must exceed the minimum IoU achieved by human demonstrations. The end-to-end transformer version succeeds in 19 of 20 trials (0.95), with average IoU 0.80, while the best IBC and LSTM-GMM variants succeed in 0 and 0.20 of trials. A camera occlusion causes only a brief jitter; when the block is shifted during pushing or while moving to the end zone, the policy replans and approaches from the needed direction.

![Diffusion Policy executes the multimodal sequence needed for sauce pouring and spreading](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-source-figure-10.webp)
*Fig 2: The six-degree-of-freedom sauce tasks combine scooping, pouring or periodic spreading, and self-termination; the table below the source figure compares human, LSTM-GMM, and Diffusion Policy outcomes. | source: [Diffusion Policy, Figure 10](https://arxiv.org/abs/2303.04137)*

The sauce experiment is a useful stress test because idle actions and periodic contact are part of the task. For pouring, Diffusion Policy obtains IoU 0.74 and success 0.79 versus human IoU 0.79 and success 1.00; for spreading, it obtains coverage 0.77 and success 1.00 versus human 0.79 and 1.00. LSTM-GMM reaches 0.06/0 for pouring and 0.27/0 for spreading. These numbers are not just a win on a clean trajectory: the controller handles viscosity, varied initial positions, and perturbations to the dough, but still has failures when grasping or contact geometry is wrong.

![Position control gives Diffusion Policy a different latency and smoothness trade-off from velocity control](/assets/images/diffusion-policy-visuomotor-policy-learning-via-action-diffusion-source-figure-4.webp)
*Fig 3: The source ablation compares velocity and position control across the simulated tasks; position control improves the diffusion variants, while latency and horizon still impose a responsiveness trade-off. | source: [Diffusion Policy, Figure 4](https://arxiv.org/abs/2303.04137)*

Figure 3 is a control-interface result, not an architectural footnote. Position commands let the receding-horizon policy absorb small image and network delays without integrating velocity errors at every step. The paper warns that comparisons are asymmetric: Diffusion Policy uses position control, while several baselines are strongest with velocity control. A matched action-space comparison is therefore needed before attributing every gain to the sampler.

### What the ablations actually establish

On the robomimic Square proficient-human task, the vision encoder choice matters. A CLIP ViT-B/16 trained end to end reaches 0.98 success after 50 epochs, compared with 0.70 when frozen and 0.22 when trained from scratch. ResNet-18 reaches 0.92 with fine-tuning, 0.58 frozen, and 0.94 from scratch. The result argues for adapting the visual representation to the action loss; “pretrained” alone is not a sufficient recipe.

The diffusion representation also has a deployment cost. It requires multiple denoising evaluations, and its training objective does not expose the same convenient normalized action likelihood as an autoregressive policy. The benchmarks show strong imitation under the paper's horizons, data, and hardware. They do not establish superiority under a strict high-frequency control budget, online reinforcement learning, or a policy-preference objective.

## High-Level Takeaways

- Diffusion Policy learns multimodal, temporally coherent action chunks instead of averaging incompatible demonstrations.
- Receding-horizon execution turns those chunks into a closed-loop controller; the action horizon must balance smoothness against reaction time.
- The strongest evidence comes from contact-rich and multimodal tasks, including Push-T and sauce manipulation, with explicit success and geometry metrics.
- End-to-end visual fine-tuning and the choice of position versus velocity control materially affect the comparison.
- A deployment decision should compare diffusion, flow, autoregressive, and parallel regression heads at the same control rate and hardware budget.
