---
title: Training Language Models to Follow Instructions with Human Feedback
date: '2022-03-01T04:00:00.000Z'
section: paper-shorts
postSlug: training-language-models-to-follow-instructions-with-human-feedback
legacyPath: >-
  /paper
  shorts/2022/02/28/training-language-models-to-follow-instructions-with-human-feedback.html
tags:
  - Other
field: Alignment
summary: InstructGPT showed that human preference data can make smaller language models follow instructions better than larger base models.
---
## 2022 – Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**arXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

**GitHub:** [CarperAI/trlx](https://github.com/CarperAI/trlx) (open-source RLHF pipeline)

**Project / blog:** [OpenAI announcement](https://openai.com/index/instruction-following/)

**Conference:** NeurIPS 2022 (spotlight)

![RLHF Pipeline](/assets/images/DQN Atari RLHF.png)

**Plain-language abstract:** GPT-3 could generate impressive text, but it often missed what users actually asked for. InstructGPT showed that reinforcement learning from human feedback could move a language model toward user intent. The pipeline has three stages: supervised fine-tuning on human demonstrations, reward-model training from ranked outputs, and PPO policy optimization with a KL penalty that keeps the model close to the supervised baseline.

The striking result is that preference data can beat scale for instruction following. A 1.3B-parameter InstructGPT model was preferred to the 175B GPT-3 on real customer prompts, while also reducing toxicity and hallucination. The paper also helped crystallize the helpfulness, honesty, and harmlessness framing that shaped later alignment evaluations.

**Evals / Benchmarks**

| Metric / task | Vanilla GPT-3 175&nbsp;B | InstructGPT 1.3&nbsp;B | InstructGPT 175&nbsp;B |
| ------------- | ----------------------- | --------------------- | ----------------------- |
| Human preference (API prompts) | — | 58&nbsp;% preferred | 85&nbsp;% preferred |
| TruthfulQA (higher = better) | 22&nbsp;% | 37&nbsp;% | 40&nbsp;% |
| RealToxicityPrompts (toxicity ↓) | 6.6&nbsp;% | 4.9&nbsp;% | 4.6&nbsp;% |
| Academic NLP (avg.) | 70.0 | 69.2&nbsp;(-0.8) | 70.3&nbsp;(+0.3) |

The key result is that alignment gains did not require sacrificing standard benchmark scores.

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

**Critiques:** The pipeline is clear, and the quality-over-size result is still important. But RLHF is expensive in both human labor and compute, reward models can be gamed, and the alignment target is narrow: a model can become better at satisfying raters without becoming robustly truthful or safe in every setting.

**Why it matters**
RLHF transformed large language models from clever autocomplete systems into instruction-following assistants, laying the groundwork for ChatGPT, GPT-4, and beyond.
