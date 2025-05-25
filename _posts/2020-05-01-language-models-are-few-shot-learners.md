---
layout: content
title: "Language Models are Few-Shot Learners"
date: 2020-05-01 00:00:00 -0400
categories: ["Paper Shorts"]
field: Natural Language Processing
---

## 2020 – Language Models are Few-Shot Learners (GPT-3)

**arXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)

**GitHub:** n/a (no official code release)

**Project page / blog:** [OpenAI blog announcement](https://openai.com/blog/openai-api/)

**Conference:** NeurIPS 2020 (oral)

**Plain-language abstract**
GPT-3 is an autoregressive Transformer 175 billion parameters strong—around ten times larger than any dense model before it. Trained on roughly 500 billion tokens of Internet text, it showed that scaling a vanilla decoder unlocks in-context learning: at test time users embed a task description plus a few demonstrations in the prompt and the frozen model performs the task with no gradient updates. Across more than twenty-five benchmarks spanning translation, QA, cloze, reasoning and arithmetic, GPT-3's zero-, one- and few-shot scores often match or surpass fine-tuned systems.

**Novel insights**
- Compound scaling hypothesis: performance in the no-fine-tune regime rises smoothly with parameter count and data, with "emergent" jumps beyond about ten billion parameters.
- Prompt-programming paradigm: the model acts as a general purpose text interpreter; tasks are "programmed" via natural-language examples instead of SGD.
- General knowledge in one network: without supervised tuning GPT-3 writes news blurbs rated fifty percent human-like and solves SAT analogies above the average college applicant.

**Evals / Latency benchmarks**

| Task (metric, setting) | GPT-3 175B | Prior SOTA (method) |
| --------------------- | ---------- | ------------------- |
| LAMBADA narrative completion (acc., zero-shot) | 76 % | 68 % (fine-tuned GPT-2 XL) |
| TriviaQA open-book QA (exact-match, zero-shot) | 64 % | 50 % (T5-11B, fine-tuned) |
| SuperGLUE (avg., few-shot = 32) | 71.8 | 89.3 (T5-11B, fine-tuned) |
| SAT analogies (few-shot = 20) | 65 % | 57 % (avg. human) |

Inference speed at a 2048-token context is about 0.4 seconds on an A100-80&nbsp;GB GPU. Memory footprint is roughly 350&nbsp;GB, necessitating model-parallel serving.

**Critiques & discussion**
- **What I liked:** Minimalist recipe; scaling plus a standard decoder stack yields new capabilities. Introduced few-shot prompting that changed how we interact with LMs. Empirical proof of the scaling hypothesis that inspired later LLMs. Sparked an ecosystem via the OpenAI API.
- **What I didn't like:** Resource heavy—training requires around three hundred sextillion FLOPs so only megascale labs can replicate. Bias and toxicity persist and can amplify stereotypes. Reasoning remains brittle on multi-step logic. Weights are closed, hampering full scientific scrutiny.

**Take-home message**
GPT-3's debut was a paradigm shift: sheer size and data delivered a single model that could tackle an unprecedented breadth of NLP tasks with clever prompting alone.

### Few-shot prompt example
```python
import openai

prompt = """Translate English to French:\n\nEnglish: good morning\nFrench: bonjour\n\nEnglish: how are you?\nFrench:"""

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=5,
    temperature=0.0,
)
print(response["choices"][0]["text"].strip())
```
