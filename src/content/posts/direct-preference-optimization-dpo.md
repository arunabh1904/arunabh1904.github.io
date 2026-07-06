---
title: Direct Preference Optimization
date: '2023-05-01T04:00:00.000Z'
section: paper-shorts
postSlug: direct-preference-optimization-dpo
legacyPath: /paper shorts/2023/05/01/direct-preference-optimization-dpo.html
tags:
  - Other
field: Reinforcement Learning
summary: DPO replaced explicit reward modeling and PPO with a direct preference loss derived from the RLHF objective.
---
## 2023 – Direct Preference Optimization: Your Language Model Is Secretly a Reward Model

**arXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

**GitHub:** [direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)

**Community forks:** TRLX `dpo.py`, Microsoft & Hugging Face ports.

**Conference:** NeurIPS 2023 (spotlight)

## Paper map

DPO turns preference optimization into a supervised classification-style objective and removes the explicit reward model plus RL loop used in RLHF. Starting from a reference policy and preference pairs, the derivation shows that the optimal policy under a KL-constrained reward objective can be written directly in terms of preference likelihoods. Training then increases the log-probability gap between chosen and rejected responses while keeping the policy near the reference model through the implicit KL term. The empirical case compares against PPO-style RLHF on summarization, dialogue, and instruction-following style tasks. The caveat is that DPO inherits the preference dataset's coverage and quality; it is simpler than RLHF, but not a replacement for good preference data or careful safety evaluation.

![Figure 1: DPO optimizes for human preferences while avoiding reinforcement learning from Direct Preference Optimization](/assets/images/direct-preference-optimization-dpo-paper-figure.png)
_Figure 1: DPO optimizes for human preferences while avoiding reinforcement learning. From the [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290), via arXiv HTML._

**Plain-language summary:** DPO removes the reinforcement learning loop from preference tuning. Traditional RLHF fits a reward model from ranked human preferences, then uses PPO under a KL penalty to update the policy. DPO shows that the same KL-regularized objective has a closed-form relationship between reward and policy under the Bradley-Terry preference model, so the policy can be optimized directly.

That turns alignment into a supervised contrastive classification task over preference pairs. There is no explicit reward network, no on-policy sampling loop, and no PPO stability tax. Training looks much closer to standard supervised fine-tuning.

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

**Take-home message:** DPO reduces preference optimization to a contrastive classification loss. By removing the reward model and RL loop, it matches or beats PPO while being much cheaper and easier to train.
