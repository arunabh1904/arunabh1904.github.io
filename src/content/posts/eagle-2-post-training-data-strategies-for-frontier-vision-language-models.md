---
title: 'Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models'
date: '2025-01-20T00:00:00.000Z'
section: paper-shorts
postSlug: eagle-2-post-training-data-strategies-for-frontier-vision-language-models
legacyPath: /paper shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html
tags:
  - Other
field: 'Vision-Language Models'
summary: "2025 – Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models"
---

## 2025 – Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models

**arXiv:** [2501.14818](https://arxiv.org/abs/2501.14818)

**Code:** [NVlabs/EAGLE](https://github.com/NVlabs/EAGLE)

## Summary

> Eagle 2 treats post-training as an inspectable data system. Starting from a Qwen2.5-7B language model, a SigLIP-400M encoder, and an MLP connector, the authors iteratively collect, filter, format, select, augment, and rebalance multimodal examples. A three-stage recipe uses 1.2M alignment examples, 21.6M diverse Stage-1.5 examples, and a final 4.6M high-quality Stage-2 set. In the paper's 14-benchmark average, the Eagle2-9B ablation climbs from 58.8 for the baseline to 73.5 after the full data, recipe, and encoder changes; the final table reports 68.2 on its OpenCompass aggregation. The result is a strong recipe, not proof that any one filter or mixture transfers unchanged.

## Core Insights

Eagle 2 is unusually useful because it exposes the work between “we fine-tuned a VLM” and “here is the checkpoint.” The starting point is intentionally ordinary: ALLaVA's 1.2M samples train the connector, then a 5.2M filtered Cambrian subset trains the full model. The authors do not claim that the baseline architecture is novel. Their claim is that the data distribution and its update loop determine which visual failures the model can repair.

![Eagle 2 step-by-step ablation](/assets/images/eagle-2-post-training-data-strategies-for-frontier-vision-language-models-source-figure-2.webp)
*Source Figure 2. The average over 13 benchmarks rises as data categories and later curation stages are added; the plotted star values are 58.8 for the baseline, 69.7 after Stage-1.5, and 73.5 after the mixture of visual encoders. [Eagle 2](https://arxiv.org/abs/2501.14818)*

The loop has two halves. Passive collection watches new datasets; proactive searching starts from error analysis and asks which missing category might address a weak benchmark. The final pool spans general VQA, OCR, charts and tables, science, mathematics, grounding and counting, captioning, knowledge, and text-only data. The authors use a similarity score based on the product of image and text similarity to find overlap, then use manual inspection and rules to remove mismatched question-answer pairs, irrelevant images, repeated text, and numerical answers whose precision is not supported by the image. They also cluster image embeddings with k-means before subset selection, so chart data does not collapse into whichever format happened to be most common.

The distribution matters more than a single impressive data count. Stage-1.5 contains 21.6M diverse examples; Stage-2 contains 4.6M selected examples. Captioning and knowledge occupy more of the early mixture, while Stage-2 shifts weight toward general VQA, OCR, science, mathematics, and text-only examples and sharply reduces captioning. Grounding/counting is present in both distributions but is a smaller Stage-2 share in the plotted mixture. This is a curriculum with a specific division of labor: use a broad, high-capacity stage to build a strong foundation, then use a smaller, cleaner stage for rapid iteration.

![Eagle 2 data mixture by training stage](/assets/images/eagle-2-post-training-data-strategies-for-frontier-vision-language-models-source-figure-4.webp)
*Source Figure 4. Stage-1.5 is broad and captioning-heavy; Stage-2 shifts the mixture toward general VQA, OCR, science, math, and text-only examples while reducing captioning and grounding/counting. The chart reports source proportions, not capability scores. [Eagle 2](https://arxiv.org/abs/2501.14818)*

That staged view changes how to read the ablation table. Under the initial two-stage recipe, adding chart, table, and OCR QA gives the largest immediate jump in the average, from 61.3 to 65.0. Under the three-stage recipe, introducing Stage-1.5 moves the average from the Cambrian-7B reference to 69.7, and adding Stage-2 reaches 70.9. Naive subset selection falls to 70.6; formatting and filtering recover 71.2; advanced selection, augmentation, re-updating Stage-1.5, and the mixture of vision encoders bring the sequence to 71.8, 72.1, 72.4, and 73.5. The paper's text reports a 45-point OCRBench gain after formatting and filtering, so “cleaning” changes the learned behavior rather than merely shrinking a file.

Two engineering decisions make the loop affordable. First, Stage-1.5 lets the team test Stage-2 ideas on a stronger checkpoint, then feed the successful curation decisions back into the large stage. Second, the balance-aware greedy knapsack packs long and short sequences together rather than letting a greedy algorithm produce separate length islands. The paper reports a 2–3× training acceleration from packing. This is a systems detail with a modeling consequence: more uniform packs keep long examples from contributing a different effective loss weight simply because the batch contains more padding.

The final Eagle2-9B is built on Qwen2.5-7B with a tiled mixture of SigLIP and ConvNeXt encoders. Pixel shuffle makes the two encoders produce matching 16×16 feature maps before channel concatenation. In Table 7, Eagle2-9B scores 92.6 on DocVQA, 86.4 on ChartQA, 77.2 on InfoVQA, 868 on OCRBench, and 63.8 on MathVista, with a reported OpenCompass aggregation of 68.2. It leads Qwen2-VL-7B on 9 of 14 listed benchmarks, but this is a broad comparison across different data and model recipes. The durable result is the visible iteration path: quality filters, format control, targeted augmentation, and stage feedback compound, while random data reduction can erase the gain.

## High-Level Takeaways

- Eagle 2 turns VLM post-training into a control loop: observe failure, add a targeted source, filter its pathologies, and measure the whole mixture again.
- The key split is 21.6M diverse Stage-1.5 examples versus 4.6M selected Stage-2 examples. The latter is small enough to iterate, while the former gives later conclusions a strong base.
- Data formatting is a learned behavior intervention. Removing a fixed LaTeX wrapper from OCR examples is reported to improve OCRBench by 45 points, a reminder that the answer format can become a shortcut.
- The 2–3× packing speedup and the image/text similarity score are part of the recipe's reproducibility story; without them, “curation” is too vague to rerun.
- Eagle2-9B's 73.5 ablation endpoint and 68.2 final OpenCompass aggregation come from different reporting views. They should not be collapsed into one headline number or treated as a cost-matched comparison to larger models.
