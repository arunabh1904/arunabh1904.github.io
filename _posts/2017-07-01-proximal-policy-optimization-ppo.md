---
layout: content
title: "Proximal Policy Optimization Algorithms (PPO)"
date: 2017-07-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Reinforcement Learning
---

## 2017 – Proximal Policy Optimization Algorithms (PPO)

**arXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)

**GitHub:** [openai/baselines](https://github.com/openai/baselines), [openai/spinningup](https://github.com/openai/spinningup)

**PDF:** [1707.06347.pdf](https://arxiv.org/pdf/1707.06347.pdf)

**Conference:** Not formally peer-reviewed; influential tech report later presented at ICML workshops and widely adopted.

**Summary of the abstract**
PPO is a family of first-order policy-gradient methods that balances the stability of Trust-Region Policy Optimization (TRPO) with the simplicity of REINFORCE. Instead of a hard KL trust region, PPO uses a clipped surrogate objective (or an adaptive KL penalty) to keep updates within a reliable neighbourhood while allowing multiple epochs of SGD per on-policy batch. On Atari and MuJoCo benchmarks, PPO matches or beats TRPO with better wall-clock time and sample efficiency.

**Novel insights**
- **Clipped likelihood ratio:** Caps the policy ratio $r = \pi_\theta/\pi_{\text{old}}$ at $1\pm\epsilon$ to avoid destructive updates.
- **Multiple epochs per batch:** Safely reuses on-policy rollouts for several SGD passes.
- **Simpler than TRPO:** No Hessian-vector products or conjugate gradients—just back-prop and Adam.
- **Versatility:** Works in both discrete (Atari) and continuous (MuJoCo) control; became the default algorithm for RLHF pipelines.

**Evals / Benchmarks**

| Domain | Metric | PPO (clip) | TRPO | A2C | Notes |
| ------ | ------ | ---------- | ---- | --- | ----- |
| MuJoCo HalfCheetah-v1 | Average return | ~3300 | 3000 | 2500 | 1 M timesteps |
| Atari Pong | Mean score | +20.7 | 19.8 | 18.3 | 50 M frames |
| Wall-clock | Time to 2000 reward (Ant-v1) | 2 h | 5 h | 3 h | NVIDIA K40 GPU |

PPO consistently strikes a balance between performance, simplicity and compute budget among on-policy algorithms of its era.

**Critiques**
| What works well | Limitations / open questions |
| --------------- | --------------------------- |
| Easy to implement; stable with minimal tuning ($\epsilon\approx0.1$, epochs $\approx4$). | On-policy method → lower sample efficiency than off-policy algorithms for sparse rewards. |
| Robust across tasks; backbone of OpenAI Five and RLHF. | Sensitive to reward scaling and advantage normalisation. |
| First-order ⇒ fast on GPUs/TPUs. | Requires fresh rollouts each update, expensive for real-world robotics. |
| Simple objective aids theoretical analysis. | Clipping is heuristic; adaptive-KL variant needs $\beta$ tuning. |

**Tiny PyTorch snippet – PPO clipped surrogate loss**
```python
def ppo_clip_loss(policy, old_logp, obs, act, adv, clip_eps=0.2):
    """Compute the PPO clipped surrogate objective."""
    logp = policy.log_prob(obs, act)
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss = -torch.mean(torch.min(unclipped, clipped))
    return loss
```
Run several epochs of Adam on batches of rollout data, add a small value-function loss and entropy bonus, and you have a working PPO agent in about 50 lines. This simplicity is why PPO became the "hello world" of modern deep RL.


