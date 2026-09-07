---
title: 'VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation'
date: '2020-05-08T00:00:00.000Z'
section: paper-shorts
postSlug: vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation
legacyPath: /paper shorts/2020/05/08/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2020 – VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation"
---
## 2020 – VectorNet

**arXiv:** [2005.04259](https://arxiv.org/abs/2005.04259)

## Summary

> VectorNet encodes HD-map elements and observed trajectories as vectors grouped into polylines, then lets the polyline features interact through attention. Its Argoverse validation result improves three-second displacement error from 4.49 m for the strongest tested raster baseline to 3.67 m. The efficiency advantage is substantial but depends on how many agents are predicted: the reported vector encoder is recomputed in each target's coordinate frame, while the ego-centered raster backbone can be shared. The paper primarily demonstrates a better scene encoder, not a solution to multimodal future prediction.

## Core Insights

### Keep the structure that rendering would hide

Lane boundaries, crosswalks, and tracked motion already come with coordinates and identities. Rendering them into colored pixels makes a convolutional network recover relationships that were explicit before rendering. VectorNet instead represents each directed segment by its start and end coordinates, attributes such as timestamps or semantic labels, and the polyline it belongs to.

A lane becomes a sequence of spatial segments; an agent history becomes a sequence of temporally sampled segments. The network's set operations do not require a fixed storage order, but that does not mean direction and time disappear. Endpoints and attributes retain the relevant ordering information. Coordinates are centered on the target agent's last observation and rotated using its heading, so the same local geometry does not have to be relearned at every global position and orientation.

The hierarchy below makes two different relationships explicit. Vectors within one polyline describe one entity's shape or motion. Attention between polyline features describes how different entities interact. Pooling at the first level reduces the number of objects passed into the second, more global operation.

![VectorNet source Figure 2: vector inputs, local polyline graphs, global interaction, and prediction](/assets/images/vectornet-source-figure-2-hierarchy.png)
*Fig 1: Local graphs compress the segments of each map element or agent history into one polyline feature. Global attention exchanges information between those features before trajectory prediction and the training-only completion task. | source: [VectorNet, Figure 2](https://arxiv.org/abs/2005.04259)*

### Local pooling and global attention do different jobs

Inside a polyline, a shared MLP embeds each segment. Max pooling summarizes the embedded segments, and that summary is concatenated back onto each segment's own feature. Repeating the operation lets a segment representation incorporate information about the entire polyline. A final max pool yields one polyline embedding. The implementation uses three local graph layers, with layer normalization and ReLU in the MLPs.

The global graph is fully connected and implemented with self-attention. An agent feature can therefore retrieve a relevant lane or another agent without waiting for information to diffuse through a raster receptive field. The reported implementation uses one global attention layer. At inference, it only needs the updated feature for the prediction target, followed by an MLP decoder producing future coordinate offsets.

This division is supported by the architecture ablation. Increasing local depth from one to three layers changes Argoverse three-second error from 3.89 to 3.67 m. Adding a second global layer gives 3.69 m instead. Widening the local MLP from 64 to 128 hidden units worsens that result to 3.93 m. More computation is not uniformly useful: in this experiment, constructing the local entity representation matters more than deepening the global exchange.

### Completion predicts hidden features, not missing roads directly

The auxiliary task randomly masks polyline features before global attention and trains a decoder to recover their learned representations. It is not supervised reconstruction of an entire missing map or a raw future trajectory. A coordinate-based identifier remains available so the unordered graph can distinguish which hidden polyline it is reconstructing.

The objective adds a Huber feature-reconstruction loss to the trajectory likelihood loss with weight one. Polyline features are L2-normalized before the global graph; otherwise, shrinking the learned features could make reconstruction artificially easy. The auxiliary decoder is discarded at inference. Its purpose is to make the shared interaction model extract enough context to predict a missing representation.

The Argoverse validation ablation separates the contributions:

| Context and training | Three-second error, m | ADE, m |
| --- | --- | --- |
| Target history only | 5.24 | 2.36 |
| Add map context | 3.94 | 1.75 |
| Add map and other-agent context | 3.84 | 1.72 |
| Map and agents, plus completion | 3.67 | 1.66 |

The map supplies the largest improvement in these rows; other agents and completion add smaller gains. That does not mean interactions are unimportant in every scene. It identifies what drives the aggregate result in this dataset and prevents attributing the full gain to the auxiliary objective.

### Attention can find the relevant lane without predicting the right motion

The figure crop below pairs a prediction with attention over the same scene. Gray lines show lanes, blue shows the predicted future, and brighter red on the right indicates stronger attention. The visual question is whether the model selects a plausible route through the map—not whether attention itself certifies the forecast.

![VectorNet source Figure 4, one example pair: trajectory prediction and corresponding attention](/assets/images/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation-source-figure-4.webp)
*Fig 2: This example pair is cropped from the paper's qualitative figure. The prediction is shown beside attention over map and agent features; stronger attention on relevant lanes illustrates context selection, not a guarantee of correct motion. | source: [VectorNet, Figure 4, example crop](https://arxiv.org/abs/2005.04259)*

The full paper includes a case where attention selects a reasonable lane but the trajectory is still inaccurate. That separates representation from decoding: finding the road geometry does not determine how fast an agent will travel or which plausible future it will choose. The simple decoder is deliberately compatible with a later multimodal predictor. The reported Argoverse test comparison uses one predicted trajectory, with 4.01 m final displacement error and 1.81 m ADE.

### Count targets before interpreting the compute advantage

The raster baseline stacks rendered history frames and uses a ResNet-18-based encoder. The authors vary resolution, kernel size, and feature cropping rather than comparing against only one weak raster configuration. Cropping along the observed trajectory improves its Argoverse validation error to 4.49 m, showing that access to spatial context matters for the baseline too.

Table 4 reports 10.56 GFLOPs and 246K parameters for that raster encoder, versus $0.041n$ GFLOPs and 72K parameters for VectorNet, excluding the prediction decoder. Here $n$ is the number of prediction targets. The raster features are ego-centered and reusable; VectorNet recenters the inputs and recomputes features for each target.

For the paper's illustrative average of thirty targets, the vector figure is about 1.23 GFLOPs, giving roughly an 8.6-fold encoder advantage rather than the more than 200-fold single-target ratio. This arithmetic also depends on the paper's average scene size: seventeen road polylines and fifty-nine agent polylines in the in-house data. It is not a scene-independent latency guarantee, and batching or a different sharing strategy would change the deployment comparison.

Finally, the datasets emphasize different behavior. Argoverse supplies two seconds of history and predicts three; the in-house dataset uses one second of history and predicts three. VectorNet and the best raster model both reach 1.00 m three-second error on the in-house set, which contains many stationary vehicles. The larger gain on curated Argoverse cases supports the value of scene context where motion is less trivial, while limiting a claim of uniform superiority across driving distributions.

## High-Level Takeaways

- VectorNet preserves directed segments and entity groupings before learning global interactions. It avoids asking a raster encoder to rediscover structure already present in maps and tracks.
- Three local graph layers help more than extra global depth in the reported ablation, making the quality of polyline features a central design choice.
- The auxiliary objective reconstructs normalized learned polyline features; it is not direct supervision for reconstructing missing roads or complete trajectories.
- Correct attention over a lane can coexist with an inaccurate forecast. The encoder's context advantage and the decoder's ability to represent multiple futures are separate questions.
- The FLOP advantage depends on scene size and target count because the original implementation recomputes target-centered features. The single-agent ratio should not be presented as a full-scene speedup.
