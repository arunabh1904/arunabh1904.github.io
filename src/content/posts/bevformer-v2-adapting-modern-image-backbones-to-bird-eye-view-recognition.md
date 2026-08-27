---
title: 'BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision'
date: '2022-11-18T00:00:00.000Z'
section: paper-shorts
postSlug: bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition
legacyPath: /paper shorts/2022/11/18/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2022 – BEVFormer v2: strengthening BEV learning with perspective supervision'
---
## 2022 – BEVFormer v2

**arXiv:** [2211.10439](https://arxiv.org/abs/2211.10439)

**Paper:** [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.html)

### Method and reported result

BEVFormer v2 asks why stronger 2D image backbones do not automatically produce stronger BEV detectors. Its answer is optimization: when supervision arrives only after view transformation, the image backbone receives an indirect signal for depth, orientation, and velocity. The model adds a perspective-view 3D detection head, then encodes its proposals as object queries for the BEV head.

## Summary

> Perspective supervision gives the image backbone a direct 3D learning signal before BEV projection. The first-stage proposals also give the BEV decoder scene-conditioned queries instead of relying only on a fixed learned query bank.

## Core Insights

Across ResNet-50, DLA-34, ResNet-101, VoVNet-99, and InternImage-B backbones, adding perspective supervision improves nuScenes validation NDS by roughly three points and mAP by roughly two points. With InternImage-XL, the paper reports 63.4 NDS and 55.6 mAP on the nuScenes test set.

![BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision source figure: Overall architecture of BEVFormer v2.](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-paper-figure.webp)
*Fig 1: BEVFormer v2 adds a perspective 3D head and hybrid object queries to temporal BEV encoding, jointly supervising perspective and BEV predictions from modern image backbones. | source: [BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](https://arxiv.org/abs/2211.10439)*

![Figure 2 from BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-source-figure-2.webp)
*Fig 2: Comparison of perspective supervision (a) and BEV supervision (B). The supervision signals of the perspective detector are dense and direct to the image feature, while those of the BEV detector are sparse and indirect. | source: [BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](https://arxiv.org/abs/2211.10439)*

![Figure 3 from BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](/assets/images/bevformer-v2-adapting-modern-image-backbones-to-bird-eye-view-recognition-source-figure-3.webp)
*Fig 3: The decoder of the BEV head in BEVFromer v2. The projected centers of the first-stage proposals are used as per-image reference points (purple ones), and they are combined with per-dataset learnded content queries and positional embeddings (blue ones) as hybrid object queries. | source: [BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision](https://arxiv.org/abs/2211.10439)*


| Comparison | NDS | mAP |
| --- | ---: | ---: |
| BEV-only, ResNet-101 | 42.6 | 35.5 |
| Perspective + BEV, ResNet-101 | 45.1 | 37.4 |
| BEVFormer v2, InternImage-XL test | 63.4 | 55.6 |

The comparison supports the optimization claim more strongly than a claim about universal architecture quality. The paper evaluates a spectrum of backbones, but explicitly notes that compute limited its exploration of still larger models.

## High-Level Takeaways

- A shared BEV loss may be too remote to train an image backbone to preserve the 3D cues that projection needs.
- Auxiliary supervision is most convincing here because the gain repeats across backbone families and model sizes.
- Perspective proposals serve two roles: they improve the image features and initialize the BEV decoder with image-conditioned hypotheses.
- The result does not remove projection error; it improves the features and queries presented to the BEV stage.
