---
title: 'DINOv2 Meets Text: dino.txt'
date: '2024-12-20T00:00:00.000Z'
section: paper-shorts
postSlug: dinov2-meets-text-dino-txt
legacyPath: /paper shorts/2024/12/20/dinov2-meets-text-dino-txt.html
tags:
  - Vision-Language Models
  - Dense Vision
field: 'Vision-Language Models'
topics:
  - multimodal
summary: '2024 – DINOv2 Meets Text: dino.txt'
---

## 2024 – DINOv2 Meets Text: dino.txt

**arXiv:** [2412.16334](https://arxiv.org/abs/2412.16334)

**Conference:** CVPR 2025

## Summary

> dino.txt adds a language interface to DINOv2 without replacing its self-supervised visual geometry. The DINOv2 ViT-L/14 stays frozen, two trainable vision blocks adapt its tokens to web image-text data, and a text encoder is learned from scratch. Concatenating the updated class token with averaged patch tokens makes one contrastive objective serve image classification, retrieval, and dense open-vocabulary segmentation. The result is efficient alignment with a clear tradeoff: the frozen backbone preserves dense features, while the new text encoder remains weaker than CLIP’s for general language tasks.

## Core Insights

### One representation carries global and local alignment

![Figure 1 from DINOv2 Meets Text: dino.txt](/assets/images/dinov2-meets-text-dino-txt-source-figure-2.webp)
*Fig 1: Overview of dino.txt: frozen DINOv2 features, two trainable vision blocks, and a text encoder align global and patch tokens for classification and open-vocabulary segmentation; the right panel shows test-time class-name queries. | source: [DINOv2 Meets Text: dino.txt, Figure 2](https://arxiv.org/abs/2412.16334)*

DINOv2 already provides a strong image representation, but it does not expose a text query interface. A naive LiT recipe—freeze DINOv2 and train only a text tower against its class token—reaches 78.8% ImageNet zero-shot accuracy and 30.2 COCO retrieval R@1, yet only 8.3 ADE20K mIoU. The global class token is useful for image recognition, but the patch tokens have not been trained to occupy the same text-aligned space.

dino.txt keeps the ViT-L/14 backbone frozen and appends two trainable Transformer blocks, written as $\psi$. If the backbone emits a class token $c$ and patch tokens $f_1,\ldots,f_N$, the added blocks produce $c'$ and $f'_p$. The global descriptor is

$$
g = [c';\;\operatorname{avg}(f'_1,\ldots,f'_N)].
$$

The text encoder is trained from scratch and maps its end-of-sentence representation into the same $2D$ dimension. The average patch path is the important design choice: the class token continues to carry global context, while gradients from the pooled patches give every local token a reason to become text-discriminative. At inference, the global descriptor supports classification and retrieval; each patch can be compared with the text embedding for a class name to produce a segmentation map.

Figure 1 is a useful architecture walkthrough. On the left, the frozen DINOv2 feature maps already show object structure before language alignment. In the middle, only the two small vision blocks and text encoder receive gradients; the loss sees one concatenated image descriptor rather than separate global and dense objectives. On the right, the same learned text space is applied either to one global vector or to the patch grid. The design avoids a segmentation-specific decoder, but it asks the text-aligned patches to preserve enough locality for pixel prediction while the class token learns the language interface.

### Data curation is part of the alignment method

The training pool starts from 2.3 billion CommonCrawl image-text pairs. Text curation follows a balanced sampling procedure based on caption frequencies; image curation uses DINOv2 embeddings and hierarchical k-means to suppress overrepresented visual clusters. The final LVTD-2.3B data selection is the intersection of those text- and image-curated choices. The point is not simply to remove bad captions. A caption can mention a balanced word while its image distribution remains dominated by near-duplicates or head concepts, so the authors rebalance both modalities.

The ablation makes that decision visible. Starting from the reference LiT recipe at 78.8/30.2/8.3 on ImageNet/COCO/ADE20K, increasing the batch to 65k reaches 79.8/35.1/18.2. One and two trainable vision blocks raise retrieval to 40.8 and 42.1 while leaving classification near 79.8. Increasing the text embedding from 768 to 1,280 reaches 80.8/43.9/20.5, and adding image-based curation reaches 81.4/45.4/20.6. The largest retrieval and dense gains do not come from unfreezing DINOv2; they come from a small adaptation path and better paired data.

Training lasts 50k iterations, equivalent to 1.6 billion pairs at batch 32k or 3.2 billion at batch 65k. The paper reports 128 A100 GPUs for 19 hours to reach 81.4% ImageNet accuracy. A CLIP model trained on the same LVTD-2.3B data needs 110 hours to reach 79.0%, while a CLIP run constrained to dino.txt’s compute reaches 73%. This is a comparison of the proposed initialization and frozen-backbone recipe with a from-scratch baseline; the data curation and implementation are still part of the cost.

### The global–dense tradeoff is measured, not assumed

| Representation or inference | ImageNet | COCO retrieval | ADE20K mIoU |
| --- | ---: | ---: | ---: |
| DINOv2 LiT, class token | 78.8 | 30.2 | 8.3 |
| dino.txt, 65k batch + two vision blocks | 79.7 | 42.1 | 20.4 |
| dino.txt, large text encoder + image curation | 81.4 | 45.4 | 20.6 |
| dino.txt, high-resolution inference | — | — | 25.1 |

The pooling ablation shows why concatenation matters. Using only the class token preserves 78.8% ImageNet accuracy but has weak dense features. Average or max pooling alone improves ADE20K to 13.3 or 18.0 while harming classification. Concatenating class and average patch tokens reaches 79.2/34.7/18.2, so the model does not have to choose between a global and local objective. The two added vision blocks then restore retrieval and classification as the alignment becomes more flexible.

At 224 pixels, dino.txt reaches 81.4 ImageNet, 45.4 COCO retrieval R@1, and 20.6 ADE20K mIoU. At 336 pixels, the reported numbers are 81.6, 44.9, and the same global-evaluation family; high-resolution dense inference reaches 25.1 ADE20K, 41.0 Cityscapes, 67.6 Pascal VOC, 24.1 Pascal Context, and 36.7 COCO-Stuff mIoU. The 800-crop protocol visits each pixel about 40 times and takes around ten seconds on an A100, so the dense ceiling is not a free property of the representation.

![Figure 4 from DINOv2 Meets Text: dino.txt showing high-resolution inference](/assets/images/dinov2-meets-text-dino-txt-source-figure-4.webp)
*Fig 2: High-resolution inference. Left: input image. Middle: result of k-means clustering (k=32) on the features. Right: open-vocabulary predictions with the ADE20K class names. | source: [DINOv2 Meets Text: dino.txt, Figure 4](https://arxiv.org/abs/2412.16334)*

Figure 2 shows the dense protocol rather than a new model head. The image is processed through overlapping crops, the patch features are clustered with $k=32$, and class-name embeddings label the clusters. The visual intuition is that high-resolution views reveal small or separated regions that a single 224-pixel grid would merge. The cost is repeated encoding and a clustering step, and the labels are only as good as the text query and benchmark ontology.

### The remaining error is semantic as well as geometric

Giving the segmentation system ground-truth masks for the k-means step raises ADE20K from 25.1 to a 38.9 mIoU boundary topline. That gap is not all boundary localization: the model can predict “shower” where the annotation says “wall,” and overlapping objects can be omitted because the dataset assigns one label where the image contains several concepts. Class names themselves are unstable; replacing ADE20K names with nearest words found from the ground-truth mask embeddings adds 2.1 mIoU. These tests make the benchmark limitation concrete rather than attributing every error to patch quality.

The text encoder is another boundary. On MTEB it trails CLIP’s text encoder by 4.2 points on average, and removing the two trainable vision blocks makes it 3.2 points worse. Freezing the strong visual tower saves compute and protects DINOv2’s dense geometry, but it also limits how well the text space can adapt to a new image-text distribution. dino.txt is therefore a strong image-alignment interface, not a general-purpose language encoder.

### When dense geometry is worth a frozen tower

Use dino.txt when a self-supervised visual backbone already serves dense tasks and the new requirement is open-vocabulary querying. Compare it with frozen DINOv2 plus class-token LiT, partial unfreezing, and end-to-end CLIP at matched image-text pairs, text capacity, and dense inference cost. Evaluate class-name sensitivity and overlapping concepts separately from boundary quality. The paper supports an efficient adapter recipe; it does not establish that strict freezing is always optimal, nor that its private curation pipeline transfers unchanged to another domain.

## High-Level Takeaways

- dino.txt aligns a frozen DINOv2 backbone with text using two vision blocks and a concatenated class-plus-patch representation.
- The same contrastive training supports global classification, retrieval, and patch-level open-vocabulary segmentation.
- Image and text curation together raise the reported ImageNet/COCO/ADE20K result to 81.4/45.4/20.6.
- High-resolution inference reaches 25.1 ADE20K mIoU but costs about 800 crops and ten seconds on an A100; benchmark labels and text quality remain limiting factors.
