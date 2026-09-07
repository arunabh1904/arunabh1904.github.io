---
title: 'NLSPN: Non-Local Spatial Propagation Network for Depth Completion'
date: '2020-07-20T04:00:00.000Z'
section: paper-shorts
postSlug: nlspn-non-local-spatial-propagation-network-for-depth-completion
legacyPath: /paper shorts/2020/07/20/nlspn-non-local-spatial-propagation-network-for-depth-completion.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2020 – NLSPN: Non-Local Spatial Propagation Network for Depth Completion'
---
## Summary

> NLSPN refines a dense depth estimate by learning both where to borrow neighboring depth and how strongly to borrow it. Fixed local propagation can mix foreground and background at an object edge; learned offsets let a pixel instead sample relevant locations beyond that fixed window. A second mechanism suppresses messages from low-confidence source pixels, and a learned normalization scale controls the range of neighbor affinities. The model reaches 741.68 mm KITTI test RMSE and 0.092 m on NYUv2 with 500 sampled depth points. Its controlled ablations show incremental gains from these components, rather than attributing the full cross-paper improvement to non-local sampling alone.

## Core Insights

### Learn the neighborhood before learning how much it should contribute

A convolutional spatial propagation network repeatedly updates each depth pixel from a fixed local neighborhood. Learned affinity weights can suppress unrelated neighbors, but they cannot introduce a useful neighbor outside that window. At the edge of a thin pole, many nearby pixels belong to the background, while useful pixels from the same pole may lie farther away along its length.

NLSPN predicts eight two-dimensional offsets per pixel from RGB and sparse depth. The offsets can be fractional, so differentiable sampling retrieves depths between grid locations. Learned affinities then determine how those sampled values contribute to refinement. “Non-local” means that the learned neighbors are not restricted to a fixed local stencil; it does not mean comparing every pixel against every other pixel with dense global attention.

![NLSPN source Figure 3f shows learned neighbors aligned with surfaces near depth boundaries](/assets/images/nlspn-source-figure-3f.png)
*Fig 1: The learned connections remain mostly within the same colored surface even near an edge. Choosing where to sample reduces the burden on affinity weights to suppress irrelevant foreground–background mixtures afterward. | source: [NLSPN, Figure 3f](https://arxiv.org/abs/2007.10042)*

A ResNet-34 encoder–decoder predicts the initial dense depth, confidence, offsets, and raw affinities. The propagation layer then performs 18 refinement steps in the default setting, using deformable-convolution sampling for the learned neighborhood. The system still needs sparse depth at inference and pays for repeated dense image-space updates.

### Affinity normalization decides how much mixing is available

Previous absolute-sum normalization divides each neighbor affinity by the sum of their absolute values. This puts every normalized neighbor-weight vector on the boundary where its absolute sum equals one. The network cannot express a smaller total neighbor-weight magnitude through that normalization alone.

NLSPN first applies a hyperbolic tangent to raw affinities and divides by a learned scale, gamma. It then performs absolute-sum normalization only when the magnitude sum exceeds one. This lets the model use weight combinations inside the bounded region as well as on its boundary. In the paper's two-neighbor illustration, the distinction is between being confined to a diamond's outline and being allowed to use its interior. That interior includes configurations with weaker overall propagation.

The learned scale adapts to the task: starting from eight, gamma finishes at 5.2 on NYUv2 and 6.3 on KITTI. The affinities are signed values rather than attention probabilities, so the mechanism should not be interpreted as an ordinary nonnegative average over neighbors. The reported normalization is a control on propagation weights; the paper's benchmark evidence, rather than the normalization alone, establishes its practical usefulness.

### Confidence belongs to the pixel sending the message

Affinity and reliability answer different questions. Two pixels may have similar appearance, yet one depth estimate may be wrong because sparse returns mixed across an occlusion boundary. A high similarity should not give that unreliable estimate unrestricted influence over its neighbors.

NLSPN multiplies each candidate neighbor affinity by the confidence of the sampled source pixel before normalization. It predicts confidence for the initial dense depth map, and learns it indirectly through the final reconstruction loss without confidence labels. This targets the flow of unreliable values during propagation. It differs from blending sparse input and a predicted depth map before refinement: an erroneous value can otherwise still spread once propagation begins.

![NLSPN Figure 2 shows joint prediction of initial depth, confidence, offsets, and affinities](/assets/images/nlspn-non-local-spatial-propagation-network-for-depth-completion-source-figure-2.webp)
*Fig 2: The shared network predicts both the starting depth and the rules for refining it. Confidence modifies source contributions inside propagation, while learned offsets determine which source pixels are considered. | source: [NLSPN, Figure 2](https://arxiv.org/abs/2007.10042)*

### Small matched gains clarify the larger benchmark result

The component ablations use only 10,000 KITTI training images, smaller 912-by-228 crops, and 20 training epochs. Their errors should not be compared directly with the full-data test score.

| Reduced-data KITTI validation configuration | RMSE, mm |
| --- | ---: |
| Fixed local neighbors, learned normalization, confidence | 890.4 |
| Non-local neighbors, absolute-sum normalization, confidence | 889.5 |
| Non-local neighbors, learned normalization, no confidence | 891.3 |
| Non-local neighbors, learned normalization, binary confidence | 892.9 |
| Non-local neighbors, learned normalization, continuous confidence | 884.1 |

Keeping learned normalization and confidence fixed, replacing local neighbors with non-local ones improves 890.4 to 884.1 mm. Keeping non-local neighbors and confidence fixed, changing absolute-sum to learned normalization improves 889.5 to 884.1. Continuous confidence improves 891.3 to 884.1 relative to no confidence, while thresholding it into a binary mask is worse. These comparisons support complementary, measured benefits rather than a claim that any one component explains the entire method.

On the full KITTI test benchmark, NLSPN reports 741.68 mm RMSE and 199.59 mm MAE, versus CSPN++ at 743.69 and 209.28. The RMSE gain is small, while the MAE improvement is larger. The model trains with both L1 and L2 losses on KITTI, so its metric balance also reflects its objective.

The indoor NYUv2 result is 0.092 m RMSE with 500 randomly sampled depth pixels, compared with DepthNormal's 0.112 m. Those cross-paper results have their own protocol differences: the listed Sparse-to-Dense baseline uses only 200 input points, whereas the other listed methods use 500. NYUv2 is trained separately with L1 loss. It is not an outdoor model evaluated indoors without adaptation.

## High-Level Takeaways

- Learned offsets expand the set of useful neighbors; learned affinities decide how strongly those neighbors should influence depth refinement.
- Confidence should regulate unreliable values while they propagate, not merely mark which input pixels contain measurements.
- Conditional normalization allows weaker neighbor mixing as well as strongly normalized updates. The learned scale differs across the two training tasks.
- The matched ablations show modest complementary gains. Full-data leaderboard improvements also depend on backbone, supervision, losses, and input-point protocols.
- Sparse metric input and iterative dense refinement remain part of deployment. The paper does not establish calibration robustness or universal behavior under unfamiliar sensor noise.
