---
title: Training Language Models to Follow Instructions with Human Feedback
date: '2022-03-04T00:00:00.000Z'
section: paper-shorts
postSlug: training-language-models-to-follow-instructions-with-human-feedback
legacyPath: >-
  /paper
  shorts/2022/02/28/training-language-models-to-follow-instructions-with-human-feedback.html
tags:
  - Other
field: 'Alignment & Post-Training'
summary: "2022 – Training Language Models to Follow Instructions with Human Feedback"
---
## 2022 – Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**arXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

**GitHub:** [CarperAI/trlx](https://github.com/CarperAI/trlx) (open-source RLHF pipeline)

**Project / blog:** [OpenAI announcement](https://openai.com/index/instruction-following/)

**Conference:** NeurIPS 2022 (spotlight)

## Summary

> InstructGPT tests whether a language model can be trained for user intent instead of only next-token continuation. Starting from GPT-3, the authors collect about 13k prompts with labeler demonstrations, 33k prompts with ranked model outputs for reward modeling, and 31k API prompts for PPO; the data is over 96% English and was produced with about 40 contractors. The three-stage recipe is supervised fine-tuning (SFT), reward-model (RM) training, and PPO with a KL penalty to the SFT model. On held-out customer prompts, 175B InstructGPT outputs are preferred to 175B GPT-3 outputs 85 ± 3% of the time and to few-shot GPT-3 outputs 71 ± 4% of the time; the paper also reports a 1.3B InstructGPT model beating 175B GPT-3 in the same evaluation. PPO-ptx adds pretraining updates to reduce the alignment tax, but the gains remain tied to the prompt and labeler distribution: bias does not improve, the toxicity advantage depends on a respectful instruction, and capability regressions remain on some public NLP tasks.

## Core Insights

### The target is a behavior distribution, not an abstract notion of alignment

GPT-3 already contains broad knowledge and can be prompted into many tasks, but its pretraining objective does not say that an answer should follow a user's intent, avoid fabrication, or refuse harmful requests. InstructGPT starts with the same GPT-3 architecture at 1.3B, 6B, and 175B parameters and changes the post-training target. Prompts come from early OpenAI API Playground use and labeler-written tasks spanning generation, question answering, dialogue, summarization, and extraction. Train, validation, and test splits are separated by user ID, and personally identifiable information is filtered from the training split.

The prompt distribution is broad in task type but narrow in coverage: it is over 96% English, and its notion of a good answer is supplied mainly by the contractors and researchers who wrote the guidance. The held-out test prompts are from customers not represented in training, which makes the preference comparison more meaningful than evaluating on the same prompts used to collect demonstrations. It still measures agreement with this particular deployment population rather than a universal human value function.

![Local replot of Figure 1: Human evaluations on the API prompt distribution](/assets/images/training-language-models-to-follow-instructions-with-human-feedback-paper-figure.png)
*Fig 1: Local replot of the paper’s human evaluation on the API prompt distribution, shown as win rate against the 175B SFT model; the dashed midpoint is the comparison baseline. | source: [Training Language Models to Follow Instructions with Human Feedback, Figure 1](https://arxiv.org/abs/2203.02155)*

The chart makes the scale result easy to read. At each size, the SFT and PPO variants move above the GPT-3 baselines, and the 1.3B PPO-ptx point is already above the 175B GPT-3 point. At 175B, the PPO-ptx curve is the highest of the plotted variants. These points are comparisons against the 175B SFT baseline, so the paper's separate head-to-head numbers—85 ± 3% against 175B GPT-3 and 71 ± 4% against few-shot 175B GPT-3—should not be read as the y-values of this chart.

### Three stages put different supervision at different interfaces

![Figure 2: A diagram illustrating the three steps of our method](/assets/images/training-language-models-to-follow-instructions-with-human-feedback-source-figure-2.webp)
*Fig 2: The three-step pipeline: collect demonstrations for SFT, rank sampled outputs to train an RM, then optimize a policy with PPO on new prompts. | source: [Training Language Models to Follow Instructions with Human Feedback, Figure 2](https://arxiv.org/abs/2203.02155)*

The diagram is more than a workflow summary: each step changes what counts as a training signal.

1. **SFT turns demonstrations into a starting policy.** A labeler writes the response they consider appropriate for a prompt, and GPT-3 is fine-tuned on these prompt-response pairs. The SFT dataset contains about 13k training prompts from the API and labeler-written prompts.

2. **The RM turns relative judgments into a scalar.** The SFT or PPO policies produce between four and nine completions for a prompt. A labeler ranks them, and a 6B reward model is trained to assign the preferred completion a higher score. For a pair $(y_w, y_l)$, the paper uses

   $$
   \mathcal{L}_{\mathrm{RM}}=-\mathbb{E}\left[\log\sigma\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)\right].
   $$

   The authors keep all pairwise comparisons from one ranking as a single batch element because treating correlated pairs as independent examples made the RM overfit. The RM's held-out-labeler accuracy is 69.6 ± 0.9%, versus 72.4 ± 0.4% on the labelers whose comparisons were used for training.

3. **PPO changes the policy while constraining drift.** The RL environment is a one-step bandit: sample a customer prompt, generate a response, receive the RM score, and end the episode. A per-token KL penalty keeps the policy near the SFT model, and the value function is initialized from the RM. The PPO-ptx variant adds a pretraining likelihood term:

   $$
   J(\pi_\phi)=\mathbb{E}_{x\sim D,\,y\sim\pi_\phi}\left[r_\theta(x,y)-\beta\log\frac{\pi_\phi(y\mid x)}{\pi_{\mathrm{SFT}}(y\mid x)}\right]
   +\gamma\,\mathbb{E}_{x\sim D_{\mathrm{pretrain}}}\left[\log\pi_\phi(x)\right].
   $$

   Here $\beta$ controls the KL penalty and $\gamma$ controls the strength of the pretraining updates; ordinary PPO sets $\gamma=0$. The PPO update uses a value-based advantage estimate internally. The scalar RM reward and the KL term are ingredients of that RL objective, not an advantage by themselves.

Steps 2 and 3 can be iterated: collect comparisons from the current policy, train a new RM, and optimize again. That loop explains both the strength and the cost of the method. The policy can improve against a current preference target, but every new round can also amplify errors in the target.

### Preference gains and side effects need separate denominators

The headline preference result is not a single general-purpose score. The source reports several measurements, each with its own comparator and population:

| Source measurement | Reported result | Comparator and scope |
| --- | --- | --- |
| Direct human preference | 175B InstructGPT preferred 85 ± 3% of the time | 175B GPT-3, held-out API-prompt evaluation |
| Few-shot human preference | 175B InstructGPT preferred 71 ± 4% of the time | Few-shot 175B GPT-3 |
| Closed-domain hallucination | About 21% versus 41% | InstructGPT versus GPT-3 on API tasks where answers should stay within the input |
| Reward-model transfer | 69.6 ± 0.9% versus 72.4 ± 0.4% | Held-out labelers versus training labelers |

The closed-domain result is a behavior-level measure: it asks whether the output invents information that is absent from the prompt. It does not establish factuality on open-domain questions. Likewise, the preference result says that labelers favored one completion under the study instructions; it does not say that the chosen completion is always true or safe.

### Truthfulness can improve by becoming more willing to abstain

![Figure 3: Results on the TruthfulQA dataset](/assets/images/training-language-models-to-follow-instructions-with-human-feedback-source-figure-6.webp)
*Fig 3: TruthfulQA results; gray bars rate truthfulness, while colored bars rate answers that are both truthful and informative, for ordinary and instruction-plus-question prompts. | source: [Training Language Models to Follow Instructions with Human Feedback, Figure 6](https://arxiv.org/abs/2203.02155)*

The two panels separate a model's willingness to answer from its willingness to answer confidently. The instruction-plus-question prompt tells the model to say “I have no comment” when it is uncertain. PPO and PPO-ptx therefore improve truthfulness partly by choosing the safer abstention; the colored truthfulness-and-informativeness bars show the cost of that choice. The paper reports small but significant truthfulness improvements over GPT-3, with the 1.3B PPO-ptx model as an exception that is slightly worse than the same-size GPT-3 baseline.

The other safety measurements are similarly conditional. On RealToxicityPrompts, a respectful instruction reduces toxicity relative to GPT-3, but the advantage disappears without that instruction; when explicitly prompted for toxic text, InstructGPT can be more toxic. On Winogender and CrowS-Pairs, the paper finds no significant bias improvement, and a respectful prompt can make the PPO-ptx model more certain in ways that increase measured bias.

### PPO-ptx reduces, but does not erase, the alignment tax

PPO training can reduce performance on public NLP tasks such as SQuAD, DROP, HellaSwag, and WMT 2015 French-to-English translation. The paper calls this an alignment tax because a model that is easier to steer on customer prompts may lose capabilities that matter elsewhere. PPO-ptx mixes updates from the original pretraining distribution and reverses many of these regressions without sacrificing labeler preference. It surpasses GPT-3 on HellaSwag, but still lags on DROP, SQuADv2, and translation. Increasing the KL coefficient alone lowers validation reward and does not fully recover the lost task performance, which is why the pretraining term is a distinct intervention rather than a cosmetic regularizer.

### Decision test and boundary

Use this pipeline when the target prompt distribution is concrete, comparison labels can express the desired behavior more reliably than demonstrations alone, and the team can afford online rollouts plus an RM/PPO loop. Evaluate the result with a specified comparator and keep preference, truthfulness, toxicity, bias, and capability metrics separate. PPO-ptx is the relevant choice when public-task regressions matter, but neither variant removes the core boundary: the policy is optimized for the coverage and judgments represented in the data. The paper's held-out-labeler result is encouraging evidence of transfer, not evidence that the learned reward captures broad human values.

## High-Level Takeaways

- Human demonstrations give GPT-3 a usable instruction-following starting point; ranked comparisons and PPO then optimize behavior against a learned preference signal.
- Preference supervision can outweigh parameter count for this API prompt distribution: 175B InstructGPT beats GPT-3 in direct comparisons, and the 1.3B model beats the 175B GPT-3 baseline in the paper's plotted evaluation.
- The reward model is a proxy with measurable transfer loss: held-out-labeler accuracy is 69.6 ± 0.9%, below its 72.4 ± 0.4% training-labeler accuracy.
- Truthfulness and toxicity gains depend on how the model is prompted, and measured bias does not improve; a preference win rate cannot stand in for those tests.
- PPO-ptx preserves more pretrained capability than PPO alone, while leaving an alignment tax on some public NLP tasks.
