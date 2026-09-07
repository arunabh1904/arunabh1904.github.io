---
title: 'Sparse4D v2: Recurrent Temporal Fusion with a Sparse Model'
date: '2023-05-23T04:00:00.000Z'
section: paper-shorts
postSlug: sparse4dv2-recurrent-temporal-fusion-with-sparse-model
legacyPath: /paper shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – Sparse4D v2: Recurrent Temporal Fusion with a Sparse Model'
---
## 2023 – Sparse4D v2

**arXiv:** [2305.14018](https://arxiv.org/abs/2305.14018)

**Code:** [linxuewu/Sparse4D](https://github.com/linxuewu/Sparse4D)

## Summary

> Sparse4D v2 changes what a temporal detector carries between frames. The original model repeatedly samples a window of historical image features; v2 propagates a fixed set of instance features and geometric anchors, then retrieves fresh evidence from the current images. That makes temporal state and fusion cost independent of elapsed history length at a fixed anchor budget. The paper reports 55.7 mAP and 63.8 NDS on nuScenes test, but its more instructive contribution is the combination of recurrence, explicit geometric state, and a fused sampling operator that makes the sparse design practical to train.

## Core Insights

### Move the geometry and retain the instance feature

An instance contains three different objects: a structured 3D box, an appearance/semantic feature, and an embedding of the box parameters. Temporal propagation advances the box using its estimated velocity, transforms it with ego motion, rotates its orientation and velocity, and recomputes the geometric embedding. The instance feature itself is carried unchanged through this propagation step; the current decoder then updates it with temporal attention and new image evidence.

![Sparse4D v2 propagates instance features through time](/assets/images/sparse4dv2-recurrent-temporal-fusion-with-sparse-model-source-figure-1.webp)
*Fig 1: In the recurrent panel, each frame samples its own image evidence and combines it with the carried instance feature. Earlier observations survive through that feature rather than through an expanding image-feature queue. | source: [Sparse4D v2, Figure 1b](https://arxiv.org/abs/2305.14018)*

This separation is useful because geometry has an explicit transformation rule while an arbitrary semantic vector does not. The model need not learn ego-motion compensation by rotating hidden features. It updates the physical state, re-encodes that state, and lets attention decide which historical content remains useful. Recurrence can carry information from many previous frames, but a fixed state cannot preserve every observation losslessly, and confidence-based selection can drop an object.

The decoder reserves capacity for new arrivals. A single-frame layer first evaluates 900 current-frame hypotheses, then contributes its best 300 to five temporal layers alongside 600 propagated instances. The total stays at 900 rather than growing whenever memory is added. In the paired ablation without camera-parameter encoding, keeping this proposal layer raises mAP/NDS from 38.4/50.4 to 41.9/52.4. The benefit comes from letting new objects compete before historical hypotheses occupy most of the capacity.

### A sparse algorithm still needs an efficient memory-access pattern

Sparse sampling sounds inexpensive, but a straightforward implementation materializes the sampled features, reshapes them, multiplies them by weights, and writes intermediate arrays before reducing over cameras and scales. Sparse4D v2 fuses bilinear sampling and weighted aggregation into one CUDA operator. It avoids those large intermediate arrays and the repeated trips to GPU memory; this changes execution efficiency without introducing another perception task.

![Sparse4D v2 combines fresh proposals with propagated instances](/assets/images/sparse4dv2-recurrent-temporal-fusion-with-sparse-model-paper-figure.webp)
*Fig 2: One current-frame layer supplies fresh hypotheses, while five subsequent layers combine them with projected historical instances through temporal cross-attention, self-attention, and image sampling. Anchor count stays fixed across the transition. | source: [Sparse4D v2, Figure 2](https://arxiv.org/abs/2305.14018)*

On RTX 3090 with ResNet-50 and 256×704 images, the operator ablation gives:

| Measurement | Basic aggregation | Fused aggregation |
| --- | ---: | ---: |
| Training memory, batch 1 | 6328 MB | 3100 MB |
| Maximum training batch | 3 | 8 |
| Inference memory, batch 1 | 925 MB | 432 MB |
| Inference throughput | 13.7 FPS | 20.3 FPS |
| 100-epoch training time on eight GPUs | 23.5 h | 14.5 h |

The training-time comparison uses each implementation's maximum batch size, so it combines the operator improvement with the larger batch that lower memory enables. The separate history-window comparison reports v1 falling from 21.5 FPS at one frame to 6.1 at nine, while recurrent v2 runs at 19.4 FPS with 432 MB of inference memory. That is the direct evidence for avoiding growth with history length; the 19.4 and 20.3 FPS figures come from separate reported comparisons.

### Camera conditioning and training-only depth change the recipe

In v1, aggregation weights are predicted without explicitly encoding camera parameters. Camera identity and geometry must be absorbed into learned weights. V2 embeds each camera's projection parameters, adds that embedding to the instance feature, and predicts weights for that view. This gives calibration and augmentation a direct route into the sampling weights. In the ablation with the proposal layer and dense-depth supervision, adding camera conditioning raises mAP from 41.9 to 43.9 and reduces orientation error from 0.523 to 0.475.

V2 also replaces v1's instance-level depth-reweight module with multi-scale dense depth supervision from LiDAR points during training. A small per-scale depth head predicts depth at an equivalent focal length and rescales it for the actual camera; those heads are inactive at inference. The resulting model remains camera-only at deployment, but its training requirements differ from v1's 3D-box-center depth targets.

Removing this supervision drops the displayed mAP/NDS from 43.9/53.9 to 35.4/43.5. The authors mark the latter run as unstable, with gradient collapse under the revised ImageNet initialization and augmentation recipe. That is evidence that depth targets stabilize this training setup, not an equal-convergence estimate of a universal 8.5-point benefit. The full paper also trains for 100 epochs, so comparing its strongest model with v1 does not isolate recurrence alone.

### The speed comparison changes with resolution and initialization

At 256×704, the paper's ResNet-50 v2 reaches 43.9 mAP and 53.9 NDS at 20.3 FPS, while its listed StreamPETR row reaches 43.2/53.7 at 26.7 FPS. V2 has the slightly higher detection score and lower throughput in that comparison. At 512×1408 with nuImages-pretrained ResNet-101, v2 reaches 50.5/59.4 at 8.4 FPS versus StreamPETR's 50.4/59.2 at 6.4 FPS. The resolution-independent sampling workload of the sparse head becomes more useful as image resolution grows, though the backbone still pays for every pixel.

The 60.8 NDS validation row uses an additional future frame and runs at 7.1 FPS; it is an offline variant, not the causal 59.4-NDS model. The 55.7/63.8 test result uses VoVNet-99 with DD3D pretraining. Keeping these rows separate shows the actual choice: recurrent state controls history cost, while resolution, pretraining, and future-frame access set different accuracy and runtime regimes.

## High-Level Takeaways

- Propagate explicit box geometry and recompute its embedding; carry the semantic feature until the decoder incorporates current evidence.
- Reserve fresh-proposal capacity before temporal fusion. A fixed memory budget otherwise risks favoring yesterday's objects over new arrivals.
- Fused sampling removes large intermediate memory traffic; its training-time gain also benefits from the larger batch it enables.
- Dense LiDAR-derived depth is training-only in v2, but it is a real change in supervision from v1 and stabilizes the reported recipe.
- Read speed and accuracy within the same resolution, initialization, and temporal protocol. The best future-frame row is not available to a causal online detector.
