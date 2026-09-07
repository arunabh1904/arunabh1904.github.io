---
title: 'StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection'
date: '2023-03-21T00:00:00.000Z'
section: paper-shorts
postSlug: streampetr-object-centric-temporal-modeling-for-multiview-3d-detection
legacyPath: /paper shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection'
---
## 2023 – StreamPETR

**arXiv:** [2303.11926](https://arxiv.org/abs/2303.11926)

**Code:** [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR)

## Summary

> StreamPETR makes object queries the memory of a camera detector. Historical features are compressed into a bounded queue of object hypotheses, updated with current images and propagated forward. This keeps recurrent temporal modeling inexpensive while allowing global interaction between objects. Its key distinction is between explicitly aligning coordinates for ego motion and implicitly conditioning features on motion. The strongest results also depend on training the model long enough in sequence to behave well during continuous video inference.

## Core Insights

### Long history can live inside a short query queue

A dense temporal BEV method carries a scene grid forward. A perspective-history method repeatedly retrieves features from multiple image frames. StreamPETR instead carries object state: context features, 3D centres, velocity, relative time, and ego pose. The default queue stores 256 top-scoring queries from each of four timestamps, giving 1,024 memory entries. New entries replace the oldest ones in first-in, first-out order.

That storage limit is not a four-frame reasoning horizon. Each saved query has already interacted with previous memory, so it can carry information older than its own timestamp. The compression is recursive. It also has a cost: selection preserves the hypotheses the classifier currently finds most useful, rather than every piece of visual evidence that might matter later.

![StreamPETR comparison of dense BEV, perspective-history, and object-query temporal fusion.](/assets/images/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection-source-figure-1.webp)
*Fig 1: Compare what crosses time in each column: a dense grid, repeated access to image features, or updated object queries. The right column propagates a compact state, allowing old evidence to survive without retaining every historical image feature. | source: [StreamPETR, Figure 1](https://arxiv.org/abs/2303.11926)*

Current queries include fresh learned hypotheses and propagated queries from the previous frame. The default uses 644 fresh queries plus 256 propagated ones. Hybrid attention lets them interact with all stored queries, then spatial cross-attention retrieves current image evidence. Fresh queries remain necessary for objects absent from memory; recurrence alone cannot discover a new arrival.

The memory ablation supports the compression argument. Moving from no memory to one stored frame raises mAP from 31.7 to 39.4 at the same reported 27.7 FPS. Two frames reach 40.1 mAP at 27.4 FPS, and four reach 40.2 at 27.1 FPS. Most of the gain arrives with a small queue, consistent with the fact that recurrent entries already summarize earlier observations.

### Coordinate alignment and feature conditioning solve different motion problems

Historical centres are first transformed into the current ego coordinate system using the known poses. This explicit operation assumes a static world: it corrects where the vehicle moved, without claiming that surrounding objects stayed still.

Motion-aware layer normalization then encodes relative ego pose, estimated object velocity, and elapsed time into learned scale and shift vectors. These modulate normalized context features and positional embeddings. The network receives motion information without being forced to trust a single velocity-extrapolated object position. This distinction matters early in training, when estimated velocities can be poor.

![StreamPETR recurrent memory queue and propagation transformer.](/assets/images/streampetr-paper-figure-3.png)
*Fig 2: Ego transformation updates the memory coordinates before the propagation transformer combines memory, fresh queries, and current images. Top-scoring output queries replenish the queue. The future segment depicts subsequent online updates, not future-frame input to the current prediction. | source: [StreamPETR, Figure 3](https://arxiv.org/abs/2303.11926)*

Table 6 reports 37.8 mAP and 48.3 NDS without the proposed motion conditioning. Explicit object-motion compensation gives 38.0 and 48.1, slightly worsening NDS. Ego-pose-conditioned normalization reaches 39.8 and 50.1; including time and velocity reaches 40.2 and 50.5. The authors attribute the weak explicit-compensation result to error propagation. That is their explanation for the ablation, not a claim that explicit motion models always fail.

### Streaming behavior must be learned, not assumed from the memory design

Training uses local sequences, while inference runs continuously. Short training windows produce a mismatch: with two training frames, mAP is 32.8 when evaluated in a matching window but 31.5 in online video. With eight training frames, online evaluation instead reaches 40.2 versus 39.6 in a window. Twelve frames give little additional mAP benefit, so the authors choose eight for training efficiency.

The appendix specifies truncated gradient propagation: the first six frames build state without gradients, while the final two contribute gradients and losses. This lets the model encounter a more mature memory state without backpropagating through every historical frame. FlashAttention also reduces training memory from 61 GB to 27 GB in the reported V2-99 setup; it is disabled for the paper’s inference-speed measurements.

The temporal representation ablation separates memory from initialization. Object memory without directly propagating last-frame queries gives 39.5 mAP; adding those queries reaches 40.2. A perspective-history configuration reaches 36.1 at 18.9 FPS, while object memory plus propagation reaches 40.2 at 27.1 FPS. Combining both histories remains at 40.2 mAP but drops to 18.6 FPS. Under this setup, replaying image history adds cost without improving detection over the compressed recurrent state.

### Strong aggregate scores still leave localization and tracking gaps

The lightweight model uses ResNet-50, 256 × 704 images, perspective pretraining, and a reduced 300-fresh-plus-128-propagated query set. It reaches 45.0 mAP and 55.0 NDS at 31.7 FPS on an RTX 3090 in FP32. That is separate from the 900-query ablations and the much larger test model. Table 1 lists SOLOFusion at 11.4 FPS, giving about 2.78 times the throughput; the paper’s phrase “1.8× faster” describes the approximate increase over that baseline, not a 1.8 throughput ratio.

The strongest ViT-L model reaches 62.0 mAP and 67.6 NDS on nuScenes test without test-time augmentation or future frames. Against the listed LiDAR CenterPoint result, its NDS is slightly higher, but translation error is 0.470 versus 0.262 metres. Tracking reaches 65.3 AMOTA versus CenterPoint’s 63.8, yet has more identity switches, 1,037 versus 760, and worse AMOTP, 0.876 versus 0.555. “Comparable to LiDAR” describes some aggregate scores, not parity in every useful attribute.

![StreamPETR detections with remote false positives highlighted in source examples.](/assets/images/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection-source-figure-6.webp)
*Fig 3: The BEV view compares blue predictions with green ground truth; red circles mark failures. Nearby crowded objects are handled well in this scene, while remote objects generate false positives. Recurrent object memory does not eliminate uncertain distant evidence. | source: [StreamPETR, Figure 6](https://arxiv.org/abs/2303.11926)*

The approach also transfers to DETR3D: the appendix reports 34.7 to 39.6 mAP with throughput changing only from 6.3 to 6.2 FPS. This supports object-query recurrence as an interface that can wrap different spatial retrieval mechanisms, while the smaller gain leaves room for the underlying image-sampling design to matter.

## High-Level Takeaways

- A bounded query queue can carry longer history through recursive updates. Its size measures retained explicit state, not the full age of the information inside that state.
- StreamPETR explicitly aligns centres for ego motion, then conditions features on pose, velocity, and time. Those are different operations with different assumptions.
- Eight-frame training closes the measured window-to-streaming mAP gap; the first six frames build memory without gradient propagation. A recurrent architecture still needs training that exposes it to realistic state.
- In the temporal ablation, adding perspective history to object memory preserves mAP while reducing throughput. Compact state is effective here because it retains sufficient detection evidence under the tested conditions.
- The lightweight throughput result and ViT-L test result are separate configurations. Strong NDS and AMOTA coexist with substantial translation error, identity switches, and remote false positives.
