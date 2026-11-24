---
title: 'Learning to summarize from human feedback'
date: '2020-09-02T09:00:00.000Z'
section: paper-shorts
postSlug: learning-to-summarize-from-human-feedback
legacyPath: /paper shorts/2020/09/02/learning-to-summarize-from-human-feedback.html
tags:
  - Reinforcement Learning
  - Human Feedback
  - Summarization
field: "Alignment & Post-Training"
summary: "2020 – Learning to summarize from human feedback"
---

## 2020 – Learning to summarize from human feedback

**Paper:** [Stiennon et al., arXiv:2009.01325v3](https://arxiv.org/abs/2009.01325v3) · **Data and code:** [OpenAI summarize-from-feedback](https://github.com/openai/summarize-from-feedback)

## Summary

> Stiennon et al. train a reward model on human choices between two summaries, then use its score to fine-tune a summarization policy. Human evaluators prefer the resulting Reddit summaries to those from larger supervised policies and to the original Reddit references. The study also shows why human evaluation stays necessary after training: better scores on a learned reward can eventually coincide with worse summaries.

## Core Insights

### Demonstrations initialize the policy; comparisons define the reward

The authors start with a language model fine-tuned to imitate the reference TL;DR for each Reddit post. They collect candidate summaries from multiple sources, including supervised policies, feedback-trained policies, and original references. Labelers read the original post and choose the better of two summaries. The released training data contains 64,832 TL;DR comparisons.

![Two candidate summaries for one post are ranked by a human; the ranking trains the scorer, which then rates new policy samples for PPO.](/assets/images/summarize-from-feedback-training-loop.svg)
*Fig 1: An original schematic separating feedback collection from policy optimization. Human judgments train the scorer; during PPO, that scorer evaluates new policy samples. | source: [Stiennon et al., Figure 2](https://arxiv.org/html/2009.01325v3#S2.F2)*

[Open the figure at full size](/assets/images/summarize-from-feedback-training-loop.svg)

The reward model represents each post–summary pair with a scalar score. On a labeled pair, the sigmoid of the preferred summary's score minus the rejected summary's score predicts the probability of the observed preference. The model learns which summaries rank higher for the same post without asking a labeler to define what an absolute score means.

The policy starts from the supervised checkpoint and uses PPO to improve the reward on its own samples. A penalty proportional to the log probability ratio against the supervised policy limits drift; in expectation under the sampled policy this is a KL penalty. The reward model scores the completed summary, while PPO updates a token-generating policy with a separate value network. See the [PPO guide](/blog/2026/07/27/from-ppo-to-grpo-rl-for-reasoning-and-vlas.html) for how rollout advantages become policy updates.

### Check whether the feedback signal survives optimization

In the TL;DR evaluation, the 1.3B feedback-trained model was preferred to reference summaries 61% of the time, versus 43% for the roughly ten-times-larger supervised baseline, each compared against the references. Because the new policies tended to write longer summaries, the paper also adjusts for length. The 6.7B feedback-trained policy still beat the Reddit references about 65% of the time after adjustment, roughly five percentage points below its unadjusted result. On CNN/DailyMail, it transferred without news-specific fine-tuning and nearly matched the news-tuned T5 at comparable lengths, although that comparison changes with length.

Human preference is a measure tied to this dataset, rubric, and pool of labelers. The 6.7B reward model preferred edits that improved but shortened summaries 62.6% of the time, compared with 76.4% for human evaluators. In the optimization analysis, stronger selection for high reward eventually lowered human preference for some reward models. These results distinguish a useful training signal from a reliable final judge. The paper also lacked a supervised baseline trained on an equivalent amount of newly collected, high-quality demonstrations, so it does not isolate the effect of choosing comparisons over demonstrations at equal annotation cost.

## High-Level Takeaways

- Pairwise comparisons turn an editorial rubric into a trainable reward without requiring consistent absolute scores.
- A reference-policy penalty can restrain drift during optimization; it cannot make a learned scorer capture all the intended criteria.
- Use independent human comparisons and length-controlled evaluations to see where reward gains stop translating into better summaries.
