---
layout: content
title: "Training Language Models to Follow Instructions with Human Feedback"
date: 2022-03-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Alignment
---

## 2022 – Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**arXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

**GitHub:** [CarperAI/trlx](https://github.com/CarperAI/trlx) (open-source RLHF pipeline)

**Project / blog:** [OpenAI announcement](https://openai.com/index/instruction-following/)

**Conference:** NeurIPS 2022 (spotlight)

**Plain-language abstract**
Large GPT-3 models were impressive, but they often ignored or misunderstood user instructions. The authors show that reinforcement learning from human feedback (RLHF) can align a language model with user intent. The process involves supervised fine-tuning on human-written demonstrations, training a reward model from ranked outputs, and then policy optimisation with PPO while penalising divergence from the supervised model. Even a 1.3&nbsp;B parameter InstructGPT matches or beats the 175&nbsp;B GPT-3 on real prompts and reduces toxicity and hallucination.

**Novel insights**
- RLHF pipeline crystalised: SFT ➔ RM ➔ PPO became the de-facto recipe for alignment.
- Human preference beats scale: a 1.3&nbsp;B RLHF model is preferred to the 175&nbsp;B GPT-3 on customer prompts.
- 3&nbsp;H framing (helpfulness, honesty, harmlessness) guides evaluation.
- KL penalty prevents reward hacking while allowing large behavioural shifts.
- Provided rigorous alignment evaluation, inspiring red-teaming and scalable oversight.

**Evals / Benchmarks**

| Metric / task | Vanilla GPT-3 175&nbsp;B | InstructGPT 1.3&nbsp;B | InstructGPT 175&nbsp;B |
| ------------- | ----------------------- | --------------------- | ----------------------- |
| Human preference (API prompts) | — | 58&nbsp;% preferred | 85&nbsp;% preferred |
| TruthfulQA (higher = better) | 22&nbsp;% | 37&nbsp;% | 40&nbsp;% |
| RealToxicityPrompts (toxicity ↓) | 6.6&nbsp;% | 4.9&nbsp;% | 4.6&nbsp;% |
| Academic NLP (avg.) | 70.0 | 69.2&nbsp;(-0.8) | 70.3&nbsp;(+0.3) |

Key takeaway: alignment gains come without sacrificing standard benchmark scores.

**Tiny RLHF-style PPO loop (PyTorch-like pseudocode)**
```python
for step in range(num_updates):
    # 1. sample model outputs
    prompts = dataset.sample(batch_size)
    with torch.no_grad():
        responses, logp = policy.generate(prompts, return_logprobs=True)

    # 2. compute rewards
    with torch.no_grad():
        reward = reward_model(prompts, responses)  # ≈ human preference
        kl_penalty = kl_coef * (logp - ref_logp)
        advantage = (reward - kl_penalty).detach()

    # 3. PPO policy update
    ratio = (policy.logp(prompts, responses) - logp).exp()
    loss = -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage,
    ).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
A reference policy (the frozen supervised model) keeps the RL step from drifting too far.

**Critiques**
- **What we liked:** Clear pipeline, quality beats size, reduced toxicity, open-source ecosystem.
- **Limitations / open questions:** Costly human effort and compute,
  reward models can be gamed, alignment is narrow.

**Why it matters**
RLHF transformed large language models from clever autocomplete into instruction-following assistants,
 laying the groundwork for ChatGPT, GPT-4 and beyond.

