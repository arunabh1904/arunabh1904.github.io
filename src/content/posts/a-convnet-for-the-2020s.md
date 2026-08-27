---
title: 'A ConvNet for the 2020s'
date: '2022-01-10T18:59:10.000Z'
section: paper-shorts
postSlug: a-convnet-for-the-2020s
legacyPath: /paper shorts/2022/01/10/a-convnet-for-the-2020s.html
tags:
  - Other
field: 'Vision Foundations'
topics:
  - learning
summary: '2022 – A ConvNet for the 2020s: modernizing ResNet into ConvNeXt'
---

## 2022 – A ConvNet for the 2020s

**Paper:** [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

**Code:** [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

**Venue:** [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)

## Summary

> ConvNeXt shows that attention is not required to build a competitive post-ViT vision backbone. The study first separates training from architecture: a modern recipe raises ResNet-50 ImageNet-1K top-1 accuracy from 76.1% to 78.8%. The network is then redesigned while that recipe stays fixed. ConvNeXt-T reaches 82.1% at 4.5 GFLOPs versus Swin-T's 81.3% at the same nominal compute, and larger variants remain competitive on COCO and ADE20K. The evidence supports a pure CNN as a serious generic-backbone option. It does not establish that convolution is universally better than attention, especially for multimodal or structured-output tasks.

## Core Insights

### The comparison separates the training recipe from the operator

The paper starts from [ResNet-50](/paper%20shorts/2015/12/01/deep-residual-learning-for-image-recognition.html). It does not compare the original 90-epoch recipe directly with a modern Transformer. AdamW and 300 training epochs improve the optimization setup; Mixup, CutMix, RandAugment, random erasing, stochastic depth, and label smoothing strengthen augmentation and regularization. Together, they move the baseline from 76.1% to 78.8% ImageNet-1K top-1 accuracy. That 2.7-point gain is a warning about architecture comparisons: an old training system can make an old operator look weaker than it is.

The architecture study then keeps that recipe fixed and changes the network in stages. At the macro level, ConvNeXt adopts a 3-3-9-3 stage ratio and a non-overlapping $4\times4$ stride-4 stem. Its blocks use depthwise convolution with wider channels, an inverted bottleneck, a $7\times7$ spatial kernel, GELU, fewer activations and normalization layers, and LayerNorm. Separate layers handle downsampling between stages. The ResNet-50-scale model moves from 78.8% to 82.0% across the accepted changes while staying near the Swin-T compute regime. Intermediate FLOPs vary, and the sequence is not a full factorial ablation. The chart therefore establishes a practical recipe, not an independent effect size for every component.

![ConvNeXt modernization path from ResNet-50 and ResNet-200 through macro design, depthwise convolution, inverted bottlenecks, large kernels, and micro-design changes](/assets/images/a-convnet-for-the-2020s-source-figure-2.webp)
*Figure 2 separates the training baseline from the accepted architecture changes and records ImageNet-1K accuracy and GFLOPs after each step. source: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)*

### The ConvNeXt block separates spatial and channel mixing

ConvNeXt keeps a residual hierarchy, but its block is closer to a Transformer block in how work is divided. A $7\times7$ depthwise convolution mixes spatial evidence independently within each channel. Two $1\times1$ layers expand the channel width by four and contract it again, so channel mixing happens separately from spatial mixing. One LayerNorm and one GELU replace the repeated BatchNorm-ReLU pattern used by a ResNet bottleneck.

This separation matters for dense camera features. The model keeps convolution's translation-equivariant, sliding-window computation while gaining a larger local receptive field and a Transformer-like inverted bottleneck. Unlike the [Vision Transformer](/paper%20shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html), ConvNeXt does not build pairwise token interactions. Its spatial mixing therefore avoids the quadratic attention cost that grows with image resolution.

![Swin Transformer, ResNet, and ConvNeXt blocks compared side by side](/assets/images/a-convnet-for-the-2020s-source-figure-4.webp)
*Figure 4 shows the shared inverted-bottleneck shape and the operator change: Swin uses windowed self-attention, while ConvNeXt uses one large depthwise convolution for spatial mixing. source: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)*

### The downstream evidence makes ConvNeXt a backbone result

The paper evaluates classification on ImageNet-1K, object detection and instance segmentation on COCO, and semantic segmentation on ADE20K. The comparable model pairs use similar parameter counts and nominal FLOPs. Larger models use ImageNet-22K pretraining where marked.

| Setting | ConvNeXt | Swin | Reported difference |
| --- | ---: | ---: | ---: |
| ImageNet-1K, Tiny, 224 px, top-1 | 82.1% | 81.3% | +0.8 points |
| ImageNet-22K pretraining, Base, 384 px, ImageNet-1K top-1 | 86.8% | 86.4% | +0.4 points |
| COCO, Cascade Mask R-CNN, Base, box / mask AP | 54.0 / 46.9 | 53.0 / 45.8 | +1.0 / +1.1 points |
| ADE20K, UperNet, Base, multi-scale mIoU | 53.1% | 51.7% | +1.4 points |

These results are why ConvNeXt belongs in a camera-backbone reading path, not only an ImageNet architecture history. The same hierarchical feature extractor transfers into detection and segmentation heads. The paper also reports comparable or higher throughput than Swin in its V100 and A100 tests. That deployment claim is hardware- and implementation-dependent. The A100 advantage uses PyTorch 1.10, TensorFloat32, and channels-last memory layout, while several comparison numbers come from official baseline repositories rather than one jointly retrained experiment.

## High-Level Takeaways

- ConvNeXt changes the architecture question from “convolution or attention?” to “which spatial mixer, hierarchy, normalization, and training system fit the task?” Its reported gains show that several advantages credited to vision Transformers also survive when their design choices are translated back into a pure CNN.
- For a dense camera backbone, ConvNeXt is the practical middle path between a classic ResNet and a hierarchical Transformer: it preserves multiscale convolutional feature maps and optimized sliding-window execution while using a wider inverted bottleneck and a larger depthwise kernel.
- The modernization study is sequential, not factorial. It does not isolate every interaction among the patchify stem, stage ratio, kernel size, normalization, activation count, width, and training recipe, and its throughput results should not be assumed to transfer unchanged across compilers or accelerators.
- The decision should be retested on the actual camera stack with the same detector or segmenter, data, resolution, augmentation, parameter budget, and measured P99 latency, memory, and energy. Prefer the Transformer if that matched test gives better accuracy or robustness within the deployment budget, especially when cross-modal interaction or sparse structured outputs are central.
