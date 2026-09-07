---
title: 'GuideFormer: Transformers for Image-Guided Depth Completion'
date: '2022-06-19T04:00:00.000Z'
section: paper-shorts
postSlug: guideformer-transformers-for-image-guided-depth-completion
legacyPath: /paper shorts/2022/06/19/guideformer-transformers-for-image-guided-depth-completion.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2022 – GuideFormer: transfer RGB structure into sparse-depth features with guided attention'
---
## Summary

> GuideFormer asks when image context should enter a sparse-depth network. Its best configuration first processes the color image through an encoder–decoder, then uses those decoded features to guide the depth encoder through cross-attention. Each branch predicts depth, and learned confidence weights combine the two outputs. On KITTI test data it reaches 721.48 mm RMSE, while controlled ablations show that guidance order matters beyond replacing convolutions with attention. The improvement has a substantial compute cost, and the model still uses convolutional components for local detail despite its “fully transformer-based” description.

## Core Insights

### Depth tokens ask for context from color tokens

Sparse depth supplies metric anchors, but most image locations have no measurement. Color features supply dense edges, appearance, and contextual structure. GuideFormer processes these inputs in separate branches so they can build complementary representations, then connects them with guided attention.

The depth feature supplies the query, while the color feature supplies the keys and values. A depth location therefore selects color information according to a cross-modal similarity rather than simply receiving a concatenated feature vector. In compact form, the transfer is $\operatorname{softmax}(Q_dK_c^T/\sqrt{d}+B)V_c$, where $B$ is relative positional bias. Self-attention inside either branch uses queries, keys, and values from that same branch; guided attention changes the source of the information being retrieved.

![GuideFormer Figure 1 shows two branches, guided attention, and confidence-weighted depth fusion](/assets/images/guideformer-source-figure-1.png)
*Fig 1: Each branch emits one depth map and one confidence map. The four labeled outputs are therefore two estimates plus their weights, not four candidate depths. Guided attention transfers features before the final depth estimates are combined. | source: [GuideFormer, Figure 1](https://openaccess.thecvf.com/content/CVPR2022/papers/Rho_GuideFormer_Transformers_for_Image_Guided_Depth_Completion_CVPR_2022_paper.pdf)*

The attention operates through window-based blocks in a hierarchical encoder–decoder. It should not be pictured as every full-resolution depth pixel attending to every camera pixel globally in one step. Patch merging, bottleneck processing, upsampling, and skip connections carry information across scales while preserving dense output detail.

### Guide the sparse branch before asking it to organize itself

GuideFormer compares pre-guide and post-guide modules. Pre-guide introduces color information before depth self-attention; post-guide first lets depth tokens interact, then adds color guidance. A bidirectional variant also sends information back from depth into color.

The intuition is tied to sparsity: early depth features may lack enough observations to form useful internal relationships. Providing image context first gives their subsequent self-attention a richer starting point. Under parallel guidance, pre-guide reaches 756.15 mm validation RMSE versus post-guide's 759.98, while bidirectional guidance improves to 755.09 at a larger parameter count.

![GuideFormer Figure 3 compares pre-guide, post-guide, and bidirectional guidance](/assets/images/guideformer-source-figure-3.png)
*Fig 2: Read upward from each pair of inputs. Pre-guide sends color information into depth before depth self-attention; post-guide reverses that order. The bidirectional variant adds a second cross-modal transfer, increasing both interaction and model size. | source: [GuideFormer, Figure 3](https://openaccess.thecvf.com/content/CVPR2022/papers/Rho_GuideFormer_Transformers_for_Image_Guided_Depth_Completion_CVPR_2022_paper.pdf)*

A separate architectural choice determines which color features provide that guidance. Parallel guidance connects the two encoders at corresponding stages. Sequential guidance runs the color encoder–decoder first, then uses its decoded features to guide the depth encoder at matching resolutions. The best sequential pre-guide result is 754.17 mm, slightly better than the 755.09-mm parallel bidirectional result, while using 130 million rather than 163 million parameters. Richer, decoded guidance can therefore be more useful than adding another direction of communication.

The matched sequential comparison also separates guided attention from simple fusion: concatenation gives 765.38 mm, pre-guide 754.17, and post-guide 757.52. The guided modules use 130 million parameters versus concatenation's 99 million, so this is not a parameter-matched attribution to the attention operation alone.

### Local convolutions remain important inside the transformer design

The network uses a shallow residual convolutional stem rather than a single large-stride patch embedding. It also inserts a depthwise convolution into the feed-forward blocks and uses transposed convolutions for reconstruction. Its central encoder–decoder blocks are attention-based, but the complete network is not convolution-free.

The supplementary ablation makes this concrete. In the concatenation baseline, replacing the residual embedding with strided convolution worsens validation RMSE from 765.38 to 798.46 mm. Removing the depthwise convolution gives 768.94, and removing encoder–decoder skip connections gives 779.85. Dense completion benefits from local processing and spatial detail even when attention supplies content-dependent interaction.

The CNN–transformer comparison also exposes the cost. With sequential concatenation in every configuration, a CNN encoder–decoder reports 772.78 mm RMSE, 132 million parameters, 748 GFLOPs, and 0.053-second inference. The transformer encoder–decoder gives 765.38 mm, 99 million parameters, 1,802 GFLOPs, and 0.101 seconds. Fewer parameters here coexist with more computation and slower inference. These are baseline ablation timings, not the latency of the final 130-million-parameter guided model.

### Final fusion and the loss determine which errors are rewarded

Both branches produce dense depth and confidence scores. A softmax over the two scores yields per-pixel fusion weights. Sequential pre-guide improves from 758.26 mm when using only the depth-branch output to 754.17 with the fused output. The color branch still contains useful complementary information after its features have guided the depth branch.

Training uses MSE on valid ground-truth pixels, with auxiliary losses for each branch. The model trains on KITTI alone for 30 epochs on eight V100 GPUs, without additional pretraining data. The authors report that their Adam recipe with small weight decay trains more reliably than the larger-weight-decay AdamW settings they tried; that is a task-specific observation, not a general optimizer ranking.

On the held-out KITTI test set, GuideFormer reaches 721.48 mm RMSE versus PENet's 730.08 and NLSPN's 741.68. Its MAE is 207.76 mm, worse than NLSPN's 199.59. GuideFormer uses L2 supervision while NLSPN combines L1 and L2, which helps explain why the methods rank differently across metrics. KITTI's aggregated ground truth labels only part of each image, so the reported metric evaluates annotated locations rather than every predicted pixel. The paper does not test deployment under calibration drift, unfamiliar scan patterns, or sensor loss.

## High-Level Takeaways

- Guided attention gives depth queries a content-dependent route to color evidence. The direction and timing of that transfer are architectural decisions.
- Sequential decoded-color guidance slightly outperforms a larger bidirectional parallel model, showing that feature maturity can matter more than another interaction path.
- Local stems, depthwise convolutions, and skip connections remain important to the attention-based network's dense predictions.
- Parameter count is an incomplete efficiency measure: the transformer ablation is smaller in parameters but larger in FLOPs and latency.
- Improved KITTI RMSE does not imply the best MAE or known robustness beyond the evaluated data. Training losses and supervision coverage shape the comparison.
