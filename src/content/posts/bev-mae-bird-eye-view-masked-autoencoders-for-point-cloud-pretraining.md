---
title: "BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining"
date: '2022-12-12T05:00:00.000Z'
section: paper-shorts
postSlug: bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining
legacyPath: /paper shorts/2022/12/12/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2022 – BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining"
---
## 2022 – BEV-MAE

**arXiv:** [2212.05758](https://arxiv.org/abs/2212.05758)

**Code:** [VDIGPKU/BEV-MAE](https://github.com/VDIGPKU/BEV-MAE)

### Method and reported result

BEV-MAE pretrains a sparse LiDAR encoder by masking vertical BEV columns and predicting normalized point coordinates plus density inside the missing regions. A shared learnable token occupies each masked column so the encoder sees a receptive-field pattern closer to downstream fine-tuning.

## Summary

> The objective is designed around outdoor LiDAR physics. Density falls with range and scan pattern, so reconstructing it forces the encoder to learn where a feature lives instead of only copying local appearance.

## Core Insights

BEV-guided masking removes an entire vertical column rather than independent occupied voxels. That choice lets the model keep a sparse encoder and use a one-layer convolutional decoder. The coordinate target describes local geometry; density supplies a coarse location cue. On Waymo with 20% labeled fine-tuning data, the paper reports 1.42 mAP and 1.34 mAPH over training from scratch.

![BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining source figure: Overall pipeline of BEV-MAE.](/assets/images/bev-mae-bird-eye-view-masked-autoencoders-for-point-cloud-pretraining-paper-figure.webp)
_Overall pipeline of BEV-MAE. Source: [BEV-MAE: Bird's-Eye-View Masked Autoencoders for Point-Cloud Pretraining](https://arxiv.org/abs/2212.05758), Figure 3, via arXiv HTML._


| Pretraining design | L2 mAP | Memory | Training cost |
| --- | ---: | ---: | ---: |
| BEV-guided masking | 66.70 | 4.1 GB | $1\times$ |
| Random occupied-voxel masking | 66.63 | 12.6 GB | $1.4\times$ |

Using only 5% of Waymo labels, BEV-MAE improves L2 mAP from 44.41 to 51.63. With the full labeled set, the gain shrinks from 68.50 to 69.35, suggesting that downstream model capacity or supervised data eventually dominates initialization. Combining nuScenes and Waymo for pretraining improves transfer more than either source alone, but sensor-density differences still create a domain gap.

The strongest modified nuScenes model reports 69.6 mAP and 73.6 NDS. That number combines pretraining with a changed encoder, so the cleaner evidence for the objective comes from the controlled data-efficiency and masking ablations.

## High-Level Takeaways

- BEV-MAE informs what a LiDAR foundation objective should reconstruct. Its atomic unit is a masked BEV column containing a set of points, not a generic voxel token. The encoder is the reusable asset; the small decoder and reconstruction targets are discarded after pretraining.
- The rejection test holds unlabeled frames, encoder, and fine-tuning budget fixed while comparing coordinate-density reconstruction with occupancy, contrastive, and future-geometry objectives. BEV-MAE loses if gains vanish with more diverse labeled data or fail across sensors with different beam patterns. At 10× pretraining scale, scenario redundancy and data I/O are likely to dominate before the small decoder does.
- BEV-MAE is a LiDAR-only pretraining reference. UniM²AE adds camera-LiDAR masked reconstruction in a shared 3D volume; UniWorld, ViDAR, and DriveWorld add temporal prediction so the latent must retain persistence and motion.
- A useful driving pretext task reconstructs measurement geometry and range-dependent density, not an arbitrary token merely because it was masked.
