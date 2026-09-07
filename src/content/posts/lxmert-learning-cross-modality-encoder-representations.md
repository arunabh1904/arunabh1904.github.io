---
title: 'LXMERT: Learning Cross-Modality Encoder Representations from Transformers'
date: '2019-08-20T00:00:00.000Z'
section: paper-shorts
postSlug: lxmert-learning-cross-modality-encoder-representations
legacyPath: /paper shorts/2019/08/20/lxmert-learning-cross-modality-encoder-representations.html
tags:
  - Vision-Language Models
  - Multimodal Pretraining
field: 'Vision-Language Models'
summary: '2019 – LXMERT: Learning Cross-Modality Encoder Representations from Transformers'
---

## 2019 – LXMERT: Learning Cross-Modality Encoder Representations from Transformers

**arXiv:** [1908.07490](https://arxiv.org/abs/1908.07490)

**Code:** [airsplay/lxmert](https://github.com/airsplay/lxmert)

## Summary

> LXMERT separates object relationships, language context, and cross-modal reasoning into three Transformer encoders. Five pretraining tasks supervise the modalities both alone and together, using 9.18M image-sentence pairs from COCO and Visual Genome sources. After fine-tuning, the model reports 72.5 VQA accuracy, 60.3 GQA accuracy, and 76.2 NLVR2 accuracy on the paper’s test splits. Its strongest transfer evidence is also its cleanest boundary: the model still depends on fixed detector regions, and VQA/GQA images and questions are part of pretraining.

## Core Insights

### Object structure should be learned before cross-modal reasoning

![LXMERT’s object, language, and cross-modality encoders](/assets/images/lxmert-learning-cross-modality-encoder-representations-source-figure-1.webp)
*Fig 1: LXMERT first models detected objects and words in separate encoders, then exchanges information in a stack of bidirectional cross-attention layers. The model exposes language, vision, and joint outputs for downstream heads. | source: [LXMERT, Figure 1](https://arxiv.org/abs/1908.07490)*

LXMERT receives 36 object regions from a Faster R-CNN detector pretrained on Visual Genome. Each region combines a 2,048-dimensional RoI feature with a learned embedding of its bounding-box coordinates; the two projected vectors are layer-normalized before they are averaged. Words use BERT’s WordPiece and position embeddings. This representation makes location part of every visual token, while the object order itself remains arbitrary.

The architecture then gives each modality its own context. An object-relationship encoder and a language encoder apply self-attention independently, with five visual layers and nine language layers in the released configuration. A five-layer cross-modality encoder performs bidirectional cross-attention: language queries read object features and object queries read language features, followed by self-attention and feed-forward sublayers. The three outputs are useful for different heads, while the `[CLS]` vector from the cross-modality stream serves as the joint representation.

That staging is more than an implementation detail. A language-only BERT initialization can look attractive because it already solves masked text prediction, but it has no reason to learn the image-text connection. LXMERT instead trains the visual and cross-modal pathways from scratch, so the objective mixture—not the language checkpoint alone—has to make the two streams agree.

### Five tasks divide the failure modes of a pair

![LXMERT pretraining tasks for masked objects and words, matching, and image question answering](/assets/images/lxmert-learning-cross-modality-encoder-representations-source-figure-2.png)
*Fig 2: The pretraining branches hide words or object features and ask the encoders to recover them, then use the same joint representation for pair matching and image question answering. | source: [LXMERT, Figure 2](https://arxiv.org/abs/1908.07490)*

The masked cross-modality language task hides 15% of words and predicts them from both visible language and image regions. The visual side is trained with masked object prediction: 15% of RoI features are zeroed, then the model either regresses the detector feature with an L2 loss or predicts the detector’s object label with cross-entropy. The label target is itself a detector output, so this is distillation from a noisy visual vocabulary rather than human object annotation. The paper also evaluates a soft-label KL variant to avoid treating the detector’s top class as certain.

The two explicitly joint tasks complete the picture. Cross-modality matching replaces a sentence with a sentence from another image half of the time and predicts whether the pair matches. Image question answering uses the matched image-question pairs already present in VQA, GQA, and Visual Genome QA, with a joint answer table of 9,500 candidates covering roughly 90% of the questions. The model therefore sees both generic caption alignment and answer-shaped supervision before downstream fine-tuning.

The pretraining mixture contains 9.18M image-sentence pairs over 180K distinct images: COCO captions, Visual Genome captions, VQA v2, GQA, and VG-QA. It contains approximately 100M words and 6.5M detected objects. The detector is frozen and every image is represented by exactly 36 regions, which avoids padding variability but fixes the visual interface. LXMERT trains for 20 epochs—ten without the image-QA loss and ten with it—on four Titan Xp GPUs, taking about ten days. Fine-tuning uses only the task-specific changes for four epochs.

### The ablations identify the useful supervision

| Evaluation | LXMERT result | What the protocol measures |
| --- | ---: | --- |
| VQA v2.0 | 72.5 accuracy | test-standard; binary/number/other subcategories are also reported |
| GQA | 60.3 accuracy | test-standard; raw questions and images at fine-tuning |
| NLVR2 | 76.2 accuracy / 42.1 consistency on Test-U; 74.5 / 39.7 on public Test-P | two image-statement pairs per example; the paper reports both splits |
| NLVR2 controls | 50.9 Test-U for BERT+2/3/4/5 CrossAtt, Train+BERT, and Train+scratch; BERT+1 CrossAtt reaches 52.4 | no LXMERT pretraining; Table 3 controls |

The headline NLVR2 comparison is a 22-point absolute increase over the prior 54% result on the unreleased Test-U split, alongside a consistency score of 42.1. The paper’s footnote reports 74.5 accuracy and 39.7 consistency on the public Test-P split, so those numbers should not be mixed. NLVR2 is a stronger transfer test than VQA or GQA because its images and statements are not used in LXMERT pretraining. For VQA and GQA, the pretraining mixture already contains related image-question examples, so their scores demonstrate a strong initialization but do not isolate zero-shot dataset transfer.

The ablations give the mechanism more resolution. Removing the image-QA loss lowers the development scores from 69.9/60.0/74.9 to 68.9/58.2/72.4 on VQA/GQA/NLVR2. Replacing the two visual objectives with no vision tasks gives 66.3/57.1/50.9; RoI regression and detected-label classification together recover 69.9/60.0/74.9. Adding QA pretraining beats simply adding other QA examples at fine-tuning, and loading BERT weights helps early but eventually underperforms the from-scratch LXMERT pretraining. These controls support complementary visual and cross-modal supervision rather than a generic “more data” explanation.

### The strongest transfer result leaves the pretraining images behind

NLVR2 matters because the model must carry its learned image-language interaction to new images and statements, while VQA and GQA already contribute questions during pretraining. That distinction makes the visual-loss and QA-loss ablations more useful than the headline score alone. The detector remains frozen throughout: replacing it with raw patches would change both the representation and the targets of the masked-object tasks, so the five-task recipe has not been established for that different setting.

## High-Level Takeaways

- LXMERT separates object relations, language context, and bidirectional cross-modal attention.
- Masked language, masked object, matching, and QA losses supervise distinct parts of the representation.
- Image-question pretraining and the two visual losses account for meaningful ablation gains, especially on NLVR2.
- Fixed detector regions and related VQA/GQA pretraining data constrain how broadly its headline scores transfer.
