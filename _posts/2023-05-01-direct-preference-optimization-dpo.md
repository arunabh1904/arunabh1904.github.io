---
layout: content
title: "Direct Preference Optimization"
date: 2023-05-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Reinforcement Learning
---

## 2023 – Direct Preference Optimization: Your Language Model Is Secretly a Reward Model

**arXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

**GitHub:** [direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)

**Community forks:** TRLX `dpo.py`, Microsoft & Hugging Face ports.

**Conference:** NeurIPS 2023 (spotlight)

**Plain-language summary**
Traditional RLHF first fits a reward model from ranked human preferences and then runs PPO under a KL penalty.
DPO shows that the same KL-regularised objective admits a closed-form solution, so you can optimise the policy directly.
This turns alignment into a supervised task: one pass over each preference pair with a binary-cross-entropy loss.

**Novel insights**
- Closed-form link between reward and policy under the Bradley–Terry model.
- Eliminates explicit reward networks or on-policy sampling.
- Training is stable and scales like standard SFT.

**Evals / Benchmarks**

| Task & Metric | PPO-RLHF | DPO | Notes |
| ------------- | -------- | --- | ----- |
| IMDB sentiment control – win-rate ↑ | 0.26 | 0.36 | 2.8 B LM, GPT-4 judge |
| Reddit TL;DR summarisation – GPT-4 win-rate ↑ | 0.42 | 0.48 | 6 B LM, temp 0.25 |
| CNN/DailyMail OOD – win-rate ↑ | 0.23 | 0.31 | zero extra fine-tuning |
| Anthropic-HH dialogue – win-rate vs “chosen” ↑ | 0.54 | 0.60 | 6 B LM, temp 0.25 |

Training takes about 4 GPU-days for a 6 B model—roughly ten times less than PPO.

**Tiny DPO objective (PyTorch-style)**
```python
def dpo_loss(policy, ref_policy, batch, beta=0.1):
    """batch = dict(prompt, chosen_txt, reject_txt)"""
    chosen_logp = policy.log_prob(batch["prompt"], batch["chosen_txt"])
    reject_logp = policy.log_prob(batch["prompt"], batch["reject_txt"])
    with torch.no_grad():
        chosen_ref = ref_policy.log_prob(batch["prompt"], batch["chosen_txt"])
        reject_ref = ref_policy.log_prob(batch["prompt"], batch["reject_txt"])
    logits = beta * ((chosen_logp - reject_logp) - (chosen_ref - reject_ref))
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
```

**Take-home message**
DPO reduces alignment to a contrastive classification loss.
With no reward model or RL loop, it matches or beats PPO while being much cheaper to train.
