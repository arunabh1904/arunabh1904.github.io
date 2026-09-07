---
title: 'Visual-RFT: Visual Reinforcement Fine-Tuning'
date: '2025-03-03T00:00:00.000Z'
section: paper-shorts
postSlug: visual-rft-visual-reinforcement-fine-tuning
legacyPath: /paper shorts/2025/03/03/visual-rft-visual-reinforcement-fine-tuning.html
tags: [Vision-Language Models, Reinforcement Learning]
field: 'Alignment & Post-Training'
summary: '2025 – Visual-RFT: Visual Reinforcement Fine-Tuning'
---

## 2025 – Visual-RFT: Visual Reinforcement Fine-Tuning

**arXiv:** [2503.01785](https://arxiv.org/abs/2503.01785)

## Summary

> Visual-RFT applies reinforcement learning with verifiable rewards to visual perception tasks. The policy generates a reasoning trace and a structured answer, then receives a task-specific rule-based score. With roughly 100 examples, the paper reports a 24.3-point gain over the baseline in one-shot fine-grained classification, plus gains of 21.9 on two-shot COCO detection and 15.4 on LVIS.

## Core Insights

### The visual output contract determines the reward

Visual-RFT starts from a practical mismatch: a fluent answer is not necessarily a correct box, class, or grounding mask. For each image and prompt, the LVLM samples a group of responses containing a <think> trace and an <answer> with the required structure. GRPO compares their rewards within the group, using

$$
A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)}
$$

and updates the policy with a verifiable reward plus a KL term to a reference policy. The reward is defined by the task contract rather than by a separately trained preference model.

The source’s headline comparison shows whether that contract improves several visual tasks at once:

![Visual-RFT results across open-vocabulary detection, few-shot detection, reasoning grounding, and fine-grained classification](/assets/images/visual-rft-visual-reinforcement-fine-tuning-source-figure-1.webp)
*Fig 1: Headline results across open-vocabulary detection, few-shot detection, LISA grounding, and fine-grained classification. | source: [Visual-RFT: Visual Reinforcement Fine-Tuning, Figure 1](https://arxiv.org/abs/2503.01785)*

For detection, predicted boxes are sorted by confidence and matched to ground-truth boxes using an IoU threshold. The reward is

$$
R_d=R_{\mathrm{IoU}}+R_{\mathrm{conf}}+R_{\mathrm{format}}.
$$

$R_{\mathrm{IoU}}$ averages the matched box overlaps. For a matched box, the confidence term rewards high confidence; for an unmatched box with zero IoU, it rewards low confidence through $1-c_i$. The format term checks the required <think> and <answer> tags. For classification, the corresponding contract is

$$
R_{\mathrm{cls}}=R_{\mathrm{acc}}+R_{\mathrm{format}},
$$

where accuracy is exact class agreement. This decomposition is the important mechanism: the policy is trained against geometry, confidence calibration, answer identity, and output syntax at the same time.

### Reasoning tokens become useful when the verifier can score the answer

The qualitative classification examples explain why the paper asks the model to produce a reasoning trace. SFT tends to emit a short label; Visual-RFT can spend tokens on visual evidence before committing to the class:

![Qualitative fine-grained classification examples comparing SFT with Visual-RFT](/assets/images/visual-rft-visual-reinforcement-fine-tuning-source-figure-4.webp)
*Fig 2: SFT and Visual-RFT qualitative classification examples with explicit visual reasoning. | source: [Visual-RFT: Visual Reinforcement Fine-Tuning, Figure 4](https://arxiv.org/abs/2503.01785)*

The training settings are deliberately data-limited. The fine-grained classification and few-shot detection experiments use small per-category sets, with the classification study described as roughly 100 examples in the one-shot setting and the main short runs using 200 training steps. LISA reasoning grounding uses 239 images and 500 fine-tuning steps. These details matter because the result is about adapting a capable LVLM with a small, verifiable signal, not about replacing large-scale visual pretraining.

### The gains follow the reward contract

On four fine-grained datasets—Flower102, Pets37, FGVC Aircraft, and Cars196—the one-shot Visual-RFT average is 24.3 points above the Qwen2-VL-2B baseline; the SFT comparison is 4.3 points lower than the baseline average in the reported table. On eight COCO categories, the two-shot setting improves by 21.9 points over the baseline. On six rare LVIS categories, the reported gain is 15.4 points.

The transfer tests are also specific. In LISA reasoning grounding, the Qwen2-VL-2B Visual-RFT model improves test mIoU by 10.7 points and test gIoU by 9.1 points over the SFT comparison shown in the paper. In open-vocabulary detection, the model is trained on 6,000 COCO annotations from 65 base classes and tested on 15 new COCO classes plus 13 rare LVIS classes. The reported Qwen2-VL-2B mAP on new COCO classes rises from 9.8 to 31.3, and the selected rare-LVIS result rises from 2.7 to 20.7. These are different tasks with different verifiers; the shared recipe is group-relative policy optimization, not a shared reward definition.

## High-Level Takeaways

- A visual RL fine-tuning run needs an output contract before it needs more data. Boxes require overlap and confidence checks; classes require exact identity; all tasks still need a format check.
- The method is data-efficient because the verifier supplies dense selection pressure among sampled answers. That efficiency depends on the verifier being cheap, correctly matched to the evaluation metric, and hard to exploit.
- The reported gains are strongest in few-shot settings and across task-specific metrics. They do not show that visual reasoning traces are faithful or that the same reward can be reused across detection, grounding, and classification.
- For a new visual task, first write the parser and reward decomposition, then test whether reward rankings agree with held-out task quality before scaling the policy update.
