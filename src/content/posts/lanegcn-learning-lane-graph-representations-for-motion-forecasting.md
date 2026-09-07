---
title: 'LaneGCN: Learning Lane Graph Representations for Motion Forecasting'
date: '2020-07-27T00:00:00.000Z'
section: paper-shorts
postSlug: lanegcn-learning-lane-graph-representations-for-motion-forecasting
legacyPath: /paper shorts/2020/07/27/lanegcn-learning-lane-graph-representations-for-motion-forecasting.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2020 – LaneGCN: Learning Lane Graph Representations for Motion Forecasting"
---
## 2020 – LaneGCN

**arXiv:** [2007.13732](https://arxiv.org/abs/2007.13732)<br>
**Code:** [uber-research/lanegcn](https://github.com/uber-research/lanegcn)

## Summary

> LaneGCN makes road connectivity an explicit part of motion forecasting: lane-segment nodes exchange different messages with predecessors, successors, and lateral neighbors. Actor features then update the map before the map updates the actors. On the paper's Argoverse test comparison, the six-trajectory predictor reaches 1.36 m minFDE and a 0.16 miss rate. The module ablation shows why the interaction cycle matters: routing actor information through the lane graph supplies much of the benefit otherwise obtained from direct actor-to-actor messages. The map informs prediction without guaranteeing the chosen trajectory or covering missing traffic signals.

## Core Insights

### Nearby lanes can mean very different things

Two lane segments can be close in Euclidean distance yet represent opposite travel directions, neighboring lanes, or a continuation through a junction. A raster encoder must infer those relationships from rendered geometry. LaneGCN receives centerlines and connectivity directly, preserving four edge types: predecessor, successor, left neighbor, and right neighbor.

The node is a short centerline segment rather than an entire lane polyline. Its feature combines an embedding of endpoint displacement, which captures orientation and length, with an embedding of the segment center. This gives an actor access to fine local geometry while allowing messages to travel over the larger road network.

That choice differs from the concurrent [VectorNet](/paper%20shorts/2020/05/08/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation.html) design. VectorNet first pools each polyline and then uses a fully connected interaction graph. LaneGCN keeps segment-level map nodes and uses sparse, typed connections following the lane topology. The comparison is about the level and structure of map interaction, not simply whether both methods use vectors.

The overview separates temporal evidence from road structure. ActorNet encodes observed displacements with a 1D convolutional network, including a validity mask for padded history. MapNet encodes the lane graph. FusionNet combines their features before predicting multiple possible futures.

![LaneGCN source Figure 1: separate actor and lane encoders followed by fusion and prediction](/assets/images/lanegcn-learning-lane-graph-representations-for-motion-forecasting-paper-figure.png)
*Fig 1: Actor history and lane topology enter through different encoders. Their features meet in actor-map fusion, so the trajectory predictor receives both observed motion and structured road context. | source: [LaneGCN, Figure 1](https://arxiv.org/abs/2007.13732)*

### Dilation follows the road instead of crossing empty space

An ordinary graph convolution can use the same transformation for every neighbor. LaneConv instead uses a distinct weight matrix for each connection type:

$$
Y=XW_0+\sum_{r\in\{\mathrm{pre,suc,left,right}\}}A_rXW_r.
$$

A message from a predecessor can therefore mean something different from a message from a left neighbor. The graph below shows the conversion from centerline points to segment nodes. At a branch, multiple successor connections can remain explicit; a nearby but unconnected road does not automatically become a successor.

![LaneGCN source Figure 3: centerlines converted into lane-segment nodes with typed neighbors](/assets/images/lanegcn-source-figure-3-typed-graph.png)
*Fig 2: The red node has predecessor, successor, left, and right relationships shown in different colors. These edge types retain road meaning that spatial proximity alone cannot supply. | source: [LaneGCN, Figure 3](https://arxiv.org/abs/2007.13732)*

Long-range propagation uses powers of the predecessor and successor adjacency matrices, at dilation sizes 1, 2, 4, 8, 16, and 32. These are graph-hop distances, not fixed meter ranges. A distant successor can send information along a curved lane without requiring a large square receptive field that also covers irrelevant surroundings. Left and right connections remain separate from these along-lane dilations.

The operator ablation supports both ingredients. With the same actor-map fusion setup, a vanilla graph-convolution model reports six-mode minFDE of 1.41 m; the full residual, multi-type, dilated operator reaches 1.10 m. The experiment varies several components, so the full improvement should not be attributed to dilation alone. Separate rows show useful gains from both typed connections and dilation.

### Actors write into the map before reading it back

FusionNet applies actor-to-lane, lane-to-lane, lane-to-actor, and actor-to-actor updates in that order. Actor-to-lane messages put current traffic information into lane features. Lane-to-lane propagation carries that information through the road graph. Lane-to-actor messages then give an actor map context that already reflects nearby actors.

For example, an occupied lane can influence another actor through the shared lane representation. This is an interpretation of the information path, not a separately supervised occupancy prediction. It explains why the map is more than a static feature lookup: the same geometry can provide different context when the surrounding traffic changes.

Follow the arrows inside FusionNet below. A2L and L2A have opposite directions, and the intervening LaneGCN determines where the actor-conditioned map information travels. The other interaction blocks use spatial attention with relative position features; the model does not apply one undifferentiated attention operation to everything.

![LaneGCN source Figure 2: ActorNet, MapNet, ordered interaction blocks, and multimodal header](/assets/images/lanegcn-learning-lane-graph-representations-for-motion-forecasting-source-figure-2.webp)
*Fig 3: FusionNet sends actor features into lane nodes, propagates them through the lane graph, returns map context to actors, and then applies direct actor interaction. The order makes map-conditioned social information available before decoding. | source: [LaneGCN, Figure 2](https://arxiv.org/abs/2007.13732)*

The module ablation makes this interpretation testable:

| Included information paths | Six-mode minFDE, m |
| --- | --- |
| ActorNet only | 1.66 |
| ActorNet plus direct actor interaction | 1.29 |
| ActorNet, MapNet, and lane-to-actor | 1.23 |
| Add actor-to-lane and lane-to-lane | 1.10 |
| Add direct actor interaction to the complete map cycle | 1.08 |

Direct actor interaction helps much more without the map cycle: 1.66 to 1.29 m, compared with 1.10 to 1.08 m after the cycle. The paper interprets the smaller final gain as evidence that actor information already propagates through lanes. It does not show that actor-to-actor reasoning is universally redundant.

### Six plausible modes are different from one correct choice

The prediction head produces six future trajectories and confidence scores. During training, the trajectory with the closest final point is selected as the positive mode. A smooth-L1 loss supervises all timesteps of that selected trajectory, while a margin loss encourages its score to exceed the other modes' scores. Selecting by endpoint is therefore not the same as selecting by average trajectory error.

The Argoverse task supplies two seconds of history at 10 Hz and predicts three seconds ahead. Inputs include actors and lanes within 100 meters of the target. In the paper's test comparison, the model reports 1.71 m ADE and 3.78 m FDE with one prediction, versus 0.87 m minADE and 1.36 m minFDE with six. The corresponding miss rate falls from 0.59 to 0.16, using a two-meter endpoint threshold.

The six-mode metrics select the best candidate using the ground truth. They measure whether the candidate set contains a good forecast, not whether a downstream system will select that candidate correctly. The gap from the one-mode result is therefore part of the evidence about remaining ambiguity, rather than a reason to quote only the lowest error.

### Road topology does not explain every acceleration

The qualitative examples show plausible turning modes when observed motion is limited, but all compared models struggle with an extreme-acceleration case. The test labels are unavailable in the paper's own plotted predictions, so those visual examples establish plausible geometry more directly than exact trajectory accuracy.

The model also uses centerlines and connectivity without traffic-light or traffic-sign state. Those signals could help distinguish waiting, slowing, and proceeding at an intersection. An explicit lane graph narrows the geometric possibilities; it cannot supply a missing cause of motion. The decoder remains a learned predictor rather than a hard guarantee of traffic-rule compliance.

## High-Level Takeaways

- LaneGCN preserves segment-level geometry and typed road connectivity, allowing nearby lanes with different meanings to exchange different messages.
- Along-lane dilation reaches distant graph neighbors without expanding a square raster receptive field; graph hops should not be confused with a fixed metric radius.
- The actor-map-actor cycle carries traffic context through lane features. Its ablation explains why direct actor interaction adds less once that path is present.
- Best-of-six accuracy measures coverage of possible futures, while one-mode accuracy also exposes the difficulty of ranking them correctly.
- A lane graph supplies geometry and connectivity, not unobserved traffic signals or a complete explanation of future acceleration. Plausible lane-following remains different from accurate motion prediction.
