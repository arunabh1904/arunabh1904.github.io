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

The architecture study then keeps that recipe fixed and changes the network in stages. At the macro level, ConvNeXt adopts a 3-3-9-3 stage ratio and a non-overlapping $4\times4$ stride-4 stem. Its blocks use depthwise convolution with wider channels, an inverted bottleneck, a $7\times7$ spatial kernel, GELU, fewer activations and normalization layers, and LayerNorm. Separate layers handle downsampling between stages. The ResNet-50-scale model moves from 78.8% to 82.0% across the accepted changes while staying near the Swin-T compute regime. Intermediate FLOPs vary, and the sequence is not a full factorial ablation. The chart therefore establishes a practical recipe, not an independent effect size for every component. Table 10 reports means and standard deviations over three seeds for this small-model sequence; the final 81.97% is separate from the 82.1% headline result, whose training uses exponential moving averages. Appendix A explicitly disables EMA during modernization because it hurt the BatchNorm models.

![ConvNeXt modernization path from ResNet-50 and ResNet-200 through macro design, depthwise convolution, inverted bottlenecks, large kernels, and micro-design changes](/assets/images/a-convnet-for-the-2020s-source-figure-2.webp)
*Fig 1: Separates the training baseline from the accepted architecture changes and records ImageNet-1K accuracy and GFLOPs after each step. | source: [A ConvNet for the 2020s, Figure 2](https://arxiv.org/abs/2201.03545)*

Read the foreground bars as the small-model trajectory and the gray bars as the larger-model experiment. The hatched kernel-size rows are trials that were not adopted, while the star-marked line tracks GFLOPs. Accuracy and computation therefore move on different scales: the chart is useful precisely because a step can improve one and worsen the other.

### Some useful changes initially make the model worse

The sequence is more revealing when its regressions are kept visible. Replacing the spatial convolution with depthwise convolution cuts compute from 4.42 to 2.35 GFLOPs but reduces accuracy from 79.51% to 78.28%. Widening the network then reaches 80.50% at 5.27 GFLOPs. Depthwise convolution creates room in the budget; it does not provide the whole accuracy gain on its own.

Moving that depthwise operation ahead of the channel expansion also initially hurts: 80.64% becomes 79.92%, while compute falls from 4.64 to 4.07 GFLOPs. But the spatial operator now acts on the narrow representation, making a larger kernel affordable. Increasing it from $3\times3$ to $7\times7$ recovers 80.57% at 4.15 GFLOPs. A $9\times9$ kernel gives the same mean and $11\times11$ is slightly worse. In the larger ResNet-200 regime, the appendix finds saturation already around $5\times5$. The transferable insight is the ordering of expensive operations, rather than a universal optimum of seven pixels.

The smaller changes also resist an easy slogan. ReLU-to-GELU alone barely changes the score; retaining only the activation between the two channel-mixing layers raises 80.62% to 81.27%. Separate downsampling initially causes training divergence. LayerNorm before each downsampling layer, after the stem, and after final global average pooling stabilizes the network. Several choices work because the surrounding block has changed.

### The ConvNeXt block separates spatial and channel mixing

ConvNeXt keeps a residual hierarchy, but its block is closer to a Transformer block in how work is divided. A $7\times7$ depthwise convolution mixes spatial evidence independently within each channel. Two $1\times1$ layers expand the channel width by four and contract it again, so channel mixing happens separately from spatial mixing. One LayerNorm and one GELU replace the repeated BatchNorm-ReLU pattern used by a ResNet bottleneck.

This separation matters for dense camera features. The model keeps convolution's translation-equivariant, sliding-window computation while gaining a larger local receptive field and a Transformer-like inverted bottleneck. Unlike the [Vision Transformer](/paper%20shorts/2020/10/01/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.html), ConvNeXt does not build pairwise token interactions. Its spatial mixing scales linearly with the number of spatial positions for fixed kernel size and channels. Swin also restricts attention to fixed local windows, so avoiding global quadratic attention is not an advantage unique to ConvNeXt in the comparison.

For a concrete first-stage block, a tensor of shape $B\times96\times56\times56$ stays at 96 channels through the depthwise convolution. LayerNorm normalizes channels at each spatial location, and the pointwise layers expand to 384 channels, apply GELU, and return to 96 before residual addition. Ignoring biases and normalization, spatial mixing uses $49C$ weights and the two channel projections use $8C^2$. Moving the depthwise convolution before expansion makes its own cost four times smaller. Layer Scale, initialized at $10^{-6}$, and stochastic depth regulate the residual branch in the training recipe.

At the network level, a $224\times224$ image becomes a $56\times56$ feature map after the stride-4 stem. Separate stride-2 layers produce $28\times28$, $14\times14$, and $7\times7$ maps with 192, 384, and 768 channels. This hierarchy is what detection and segmentation heads consume. Resolution changes need no positional-embedding interpolation, although they still increase compute and alter the training and testing setup.

![Swin Transformer, ResNet, and ConvNeXt blocks compared side by side](/assets/images/a-convnet-for-the-2020s-source-figure-4.webp)
*Fig 2: Shows the shared inverted-bottleneck shape and the operator change: Swin uses windowed self-attention, while ConvNeXt uses one large depthwise convolution for spatial mixing. | source: [A ConvNet for the 2020s, Figure 4](https://arxiv.org/abs/2201.03545)*

### Scaling works with both model capacity and pretraining data

ConvNeXt-T and -S share widths of 96, 192, 384, and 768, but -S increases the third stage from 9 to 27 blocks. Base, Large, and XL retain the deeper block counts and increase widths. The headline 87.8% ImageNet result belongs to the 350-million-parameter XL, pretrained on approximately 14 million ImageNet-22K images and fine-tuned at $384\times384$. It is not a result from training the tiny model on ImageNet-1K.

The main comparisons use 300 epochs for ImageNet-1K training, or 90 epochs of ImageNet-22K pretraining followed by 30 epochs of ImageNet-1K fine-tuning. At Base size and 384-pixel resolution, ConvNeXt reaches 85.1% with ImageNet-1K training and 86.8% with ImageNet-22K pretraining; Swin-B reaches 84.5% and 86.4%. The larger dataset helps the convolutional architecture too.

The isotropic ablation goes further by removing the resolution hierarchy and keeping ViT-like fixed-size feature maps throughout the network. ConvNeXt-B then reports 82.0% against ViT-B’s 81.8% at similar parameter counts, and the large pair ties at 82.6%. These are supervised ImageNet-1K comparisons using improved ViT recipes. They show that the block remains useful outside a multiscale hierarchy, while leaving broader pretraining objectives and multimodal interaction outside the experiment.

### The downstream evidence makes ConvNeXt a backbone result

The paper evaluates classification on ImageNet-1K, object detection and instance segmentation on COCO, and semantic segmentation on ADE20K. The comparable model pairs use similar parameter counts and nominal FLOPs. Larger models use ImageNet-22K pretraining where marked.

| Setting | ConvNeXt | Swin | Reported difference |
| --- | ---: | ---: | ---: |
| ImageNet-1K, Tiny, 224 px, top-1 | 82.1% | 81.3% | +0.8 points |
| ImageNet-22K pretraining, Base, 384 px, ImageNet-1K top-1 | 86.8% | 86.4% | +0.4 points |
| COCO, Cascade Mask R-CNN, Base, IN-22K pretraining, box / mask AP | 54.0 / 46.9 | 53.0 / 45.8 | +1.0 / +1.1 points |
| ADE20K, UperNet, Base, IN-22K pretraining, multi-scale mIoU | 53.1% | 51.7% | +1.4 points |

These results are why ConvNeXt belongs in a camera-backbone reading path, not only an ImageNet architecture history. The same hierarchical feature extractor transfers into detection and segmentation heads. The paper also reports comparable or higher throughput than Swin in its V100 and A100 tests. That deployment claim is hardware- and implementation-dependent. The A100 advantage uses PyTorch 1.10, TensorFloat32, and channels-last memory layout, while several comparison numbers come from official baseline repositories rather than one jointly retrained experiment.

The efficiency numbers also have concrete boundaries. On a V100, ConvNeXt-T processes 774.7 images/s against Swin-T’s 757.9: a small advantage. With A100 TF32 and channels-last execution, the same pair reaches 1943.5 and 1325.6 images/s, about 47% apart. Neither ratio is a latency guarantee for a single image on another device. On COCO, the Base Cascade Mask R-CNN setup uses 17.4 GB of peak training memory against Swin-B’s 18.5 GB at two images per GPU.

Robustness is similarly mixed at small scale. ConvNeXt-T improves ImageNet-C mean corruption error over Swin-T (53.2 versus 62.0, lower is better), but the explicitly robustness-oriented RVT-S reaches 49.4. The XL model performs strongly after larger-data pretraining, yet these comparisons combine architecture, capacity, and data. The paper’s evidence supports revisiting a convolutional backbone; it does not make the operator alone a guarantee of robustness.

## High-Level Takeaways

- The improved training recipe contributes 2.7 ImageNet points before the architecture changes. Comparisons against an old ResNet recipe obscure that contribution.
- Depthwise spatial mixing becomes effective alongside width and operation ordering. Some accepted steps first reduce accuracy or compute before enabling a later improvement.
- One large depthwise convolution and a channel-expanding MLP are enough for a competitive residual block; the best kernel and normalization choices depend on the surrounding architecture.
- ImageNet-22K pretraining benefits ConvNeXt, and its hierarchy transfers to COCO and ADE20K. The 87.8% headline belongs to the large, pretrained XL configuration.
- Throughput depends strongly on hardware and memory layout. The paper also leaves multimodal interaction and sparse or structured outputs as settings where attention may offer greater flexibility.
