---
title: Language Models are Few-Shot Learners
date: '2020-05-01T04:00:00.000Z'
section: paper-shorts
postSlug: language-models-are-few-shot-learners
legacyPath: /paper shorts/2020/05/01/language-models-are-few-shot-learners.html
tags:
  - Other
field: Natural Language Processing
summary: GPT-3 showed that scale can turn language models into few-shot learners, but also exposed the limits of prompting alone.
---
## 2020 – Language Models are Few-Shot Learners (GPT-3)

**arXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)

**GitHub:** n/a (no official code release)

**Project page / blog:** [OpenAI blog announcement](https://openai.com/blog/openai-api/)

**Conference:** NeurIPS 2020 (oral)

![Figure 2.1 from GPT-3: zero-shot, one-shot, few-shot, and fine-tuning evaluation strategies](/assets/images/gpt3-paper-figure-2-1-eval-strategies.png)
_Figure 2.1 from the [GPT-3 paper](https://arxiv.org/abs/2005.14165), via ar5iv._

**Plain-language abstract:** GPT-3 made scale itself feel like a new interface. It is a 175B-parameter autoregressive Transformer, roughly ten times larger than prior dense language models, trained on about 500B tokens of internet text. At test time, users can put a task description and a few examples in the prompt, and the frozen model often performs the task without gradient updates.

The paper's central claim is that no-fine-tune performance improves smoothly with model size, data, and compute, with some capabilities appearing more sharply beyond roughly ten billion parameters. It also made prompt programming feel real: instead of creating a dataset and fine-tuning, users could steer one general-purpose model with natural-language examples.

**Evals / Latency benchmarks**

| Task (metric, setting) | GPT-3 175B | Prior SOTA (method) |
| --------------------- | ---------- | ------------------- |
| LAMBADA narrative completion (acc., zero-shot) | 76 % | 68 % (fine-tuned GPT-2 XL) |
| TriviaQA open-book QA (exact-match, zero-shot) | 64 % | 50 % (T5-11B, fine-tuned) |
| SuperGLUE (avg., few-shot = 32) | 71.8 | 89.3 (T5-11B, fine-tuned) |
| SAT analogies (few-shot = 20) | 65 % | 57 % (avg. human) |

Inference speed at a 2048-token context is about 0.4 seconds on an A100-80&nbsp;GB GPU. Memory footprint is roughly 350&nbsp;GB, necessitating model-parallel serving.

**Critiques & discussion:** The minimalist recipe is the point: scale a standard decoder stack and new behaviors appear. Few-shot prompting changed how people interact with language models and helped launch the API-first LLM ecosystem. The costs are just as central. Training required enormous compute, replication was limited to megascale labs, bias and toxicity remained visible, multi-step reasoning was brittle, and closed weights made full scientific scrutiny difficult.

**Take-home message:** GPT-3 shifted the field from task-specific models toward general models steered by prompts. Its breadth came from scale, and its limitations showed what scale alone could not yet solve.

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
