---
title: 'Adding Spatial Grids and Object Tokens to Vision-Language Models'
date: '2026-10-06T20:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: adding-spatial-grids-and-object-tokens-to-vision-language-models
legacyPath: /blog/2026/10/06/adding-spatial-grids-and-object-tokens-to-vision-language-models.html
tags: [Research, Multimodal Learning, Autonomous Driving, Adapters]
topics: [autonomy, multimodal]
summary: A practical guide to separate grid and object adapters, small-token projection, spatial and temporal alignment, and integration with Qwen3-VL.
---

# Adding Spatial Grids and Object Tokens to Vision-Language Models

A driving system can know more than its camera image shows. A perception model may supply a spatial feature grid. A tracker may supply boxes, velocities, and uncertain object attributes. A map may supply lane geometry. A vision-language model needs an interface that preserves what these inputs mean, where they apply, and when they were observed.

The first implementation question often sounds simple: how do I turn a small model's 256-dimensional tokens into the 4,096-dimensional tokens of a larger reader? A learned projection solves that shape mismatch. It does not establish that two tokens describe the same place, that a velocity uses the right coordinate frame, or that the reader will use the new evidence.

This guide develops two separate interfaces: a **spatial-grid adapter**, followed by an **object adapter for boxes and attributes**. It then connects them to a concrete Qwen3-VL checkpoint. The architectures and experiments proposed here are design choices, not results from a trained driving model. The literature and implementation review is current to October 6, 2026.

## Define the interface before choosing a projector

Each source token needs a contract. Specify its feature width, spatial support, coordinate frame, observation time, validity, and source model version. Spatial support means the area or volume from which the feature was computed. Two features at array index seven may cover different regions after different crops, strides, or camera projections.

There are four distinct alignment problems. **Width alignment** makes the tensors fit. **Semantic alignment** makes the reader interpret the features. **Spatial alignment** associates features with locations or objects. **Temporal alignment** makes observations comparable at the time of the question. A successful matrix multiplication proves only the first.

Use a small diagnostic task to expose each problem. Ask which side of the ego vehicle contains an actor. Move that actor while keeping its appearance fixed. Change its class while keeping its position fixed. Delay its observation without changing its feature. These controlled changes give the interface a testable meaning before a large instruction dataset hides mistakes behind fluent answers.

### What the connector literature provides

[LLaVA](https://arxiv.org/abs/2304.08485) establishes a direct projection and visual instruction tuning route. [BLIP-2](https://arxiv.org/abs/2301.12597) instead uses a query transformer to extract a compact visual representation before connecting it to a frozen language model. [Flamingo](https://arxiv.org/abs/2204.14198) combines a visual resampler with gated cross-attention inside the language model. These are different choices about where to compress evidence and where to let language interact with it.

[Honeybee](https://arxiv.org/abs/2312.06742) makes locality explicit through convolutional or deformable-attention projectors. [Cambrian-1](https://arxiv.org/abs/2406.16860) uses a Spatial Vision Aggregator to combine several visual encoders while retaining spatial organization. Their relevance here is structural: a connector can do more than change channel count, but its spatial assumptions must match the source features.

Start with a per-token linear map when the source already has useful context and its token count is affordable. Add a two-layer multilayer perceptron, or MLP, to test whether nonlinear channel mixing helps. Add local merging when sequence length is the limiting cost. Use learned-query resampling when a fixed output budget matters more than preserving one token per source cell. Each added operation should solve an observed failure.

The recent [frontier adapter study](https://arxiv.org/abs/2610.05897) is a useful warning: a strong frozen reader with a trained visual connector can still struggle on visual interactions outside the connector's training distribution. The [paper note](/paper%20shorts/2026/10/05/fitting-vision-adapters-at-frontier-scales.html) discusses its blind controls and multi-image limitations. A bigger reader does not remove the need for aligned supervision.

## Spatial-grid adapters

Assume a small encoder produces a tensor with batch, height, width, and channel axes. For a concrete design example, use a sixteen-by-sixteen grid with 256 channels per cell. There are 256 tokens per sample. The grid could be an image feature map or a bird's-eye-view feature map, but those two cases require different position information.

An image cell refers to a region of a particular camera image. A bird's-eye-view, or BEV, cell refers to an area in a metric coordinate frame. A row in one does not correspond to a row in the other. Converting image evidence into BEV requires geometry or learned correspondence; reshaping the array cannot perform that conversion. The [mapping guide](/blog/2026/10/01/from-bev-features-to-lane-graphs-and-changing-maps.html) develops the depth, projection, and temporal-fusion choices in more detail.

### Project each cell without changing the grid

The simplest adapter normalizes each 256-channel feature, then applies a learned map to 4,096 channels. It produces the same 256 tokens. A linear layer needs 256 times 4,096 weights, plus an optional bias: approximately one million parameters. This is a useful baseline because every output token retains a clear source cell.

For each input cell, a linear projection multiplies its feature vector by a learned matrix and adds a bias. The matrix has 4,096 rows and 256 columns.

$$
z_i=W x_i+b.
$$

The projection expands the representation width, but a linear map can have rank at most 256. A nonlinear MLP can change the representation more flexibly; it still cannot recover evidence that the encoder discarded.

Choose normalization deliberately. Layer normalization removes per-token scale information. If feature magnitude carries calibrated confidence, supply that confidence as a separate field before normalization destroys it. Measure the projected feature norms against the reader's existing embeddings, and test an explicit learned scale. Similar norms help numerical conditioning; they do not prove semantic compatibility.

### Concatenate neighboring cells inside the projector

To reduce the number of tokens, group each two-by-two neighborhood. Concatenate its four feature vectors along the **channel axis**. Four 256-channel vectors become one 1,024-channel vector. An MLP then maps that vector to 4,096 channels. The sixteen-by-sixteen input grid becomes an eight-by-eight output grid, with 64 output tokens.

This is different from sequence concatenation. Appending four vectors along the token axis retains four 256-channel tokens. Channel concatenation makes one wider token. State the axis in code reviews, because both operations are commonly called concatenation and only one reduces sequence length.

![Four adjacent grid cells become one wider vector before projection; the output has fewer spatial tokens.](/assets/images/spatial-adapter-grid.svg)
*Author's proposed implementation. Merge adjacent cells in a fixed order, project their combined channels, and update the output grid metadata. The operation preserves the grouped values before the learned projection, but reduces their independent access in the reader.*

The PyTorch component implements this local merge and projection. It expects an ordinary row-major grid, not the internal packed order of a particular vision encoder. It intentionally excludes padding and position construction so those contracts remain visible.

```python
import torch
from torch import nn

class GridProjector(nn.Module):
    def __init__(self, source_dim=256, reader_dim=4096):
        super().__init__()
        self.norm = nn.LayerNorm(source_dim)
        merged_dim = 4 * source_dim
        self.project = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, reader_dim),
        )

    def forward(self, grid):
        # grid: [batch, height, width, source_dim]
        batch, height, width, channels = grid.shape
        if height % 2 or width % 2:
            raise ValueError("The grid must have even height and width")
        x = self.norm(grid)
        x = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(batch, (height // 2) * (width // 2), 4 * channels)
        return self.project(x)  # [batch, merged_cells, reader_dim]
```

The permutation is necessary. A direct reshape into groups of four would collect consecutive entries in memory, which need not form two-by-two neighborhoods. The explicit axes group top-left, top-right, bottom-left, and bottom-right cells. A coordinate-valued toy grid is a better test of this behavior than an output-shape assertion.

If the grid has an odd size, either reject it as above or define a padding policy. For padding, carry a four-element validity vector for each merged cell. An all-invalid group can be masked from the reader. A partly valid group must retain which members were missing; replacing missing observations with zeros alone makes absence ambiguous. Include the validity features in the projector's declared input width.

Merging does not mean that the four source cells occupy the same location. The output token covers their combined support. Keep its center, extent, and scale in the metadata. If fine localization matters, preserve local offsets within the feature or retain a second, finer feature path. A distant pedestrian may occupy less than one coarse cell even when the total token budget looks generous.

### Combine features from different small encoders

Suppose a semantic encoder and a depth encoder each produce a grid. Channel concatenation is reasonable only after the grids refer to the same image crop or metric cells. Their heights and widths being equal is insufficient. Different resize rules, receptive fields, or camera timestamps can still break correspondence.

For aligned cells, normalize each encoder's features separately, concatenate them, and train a joint projector. Separate normalization prevents one source's scale from dominating by accident. Include a source-validity flag and train with missing sources. A source-specific projection followed by learned fusion is another useful baseline when the feature distributions differ substantially.

For unaligned grids, first establish correspondence. Image grids may need resampling in a shared crop coordinate system. Camera and LiDAR features may need calibrated projection and visibility checks. BEV grids may need an ego-motion transform. Learned cross-attention can search for correspondence, but it still needs position information and examples that teach the correct association.

[PETR](https://arxiv.org/abs/2203.05625) provides a relevant geometric precedent: it encodes 3D coordinates into multi-view image features for detection. That is evidence for a geometry-aware representation, not evidence that those features can be inserted unchanged into a language model. The language interface remains a separate training problem.

### Decide whether the compressed output is still a grid

A local merge preserves an explicit coarser lattice. A learned-query resampler generally produces slots whose spatial support depends on attention. Those slots should not receive invented row and column coordinates merely because their count is a square.

[MapLightning](https://arxiv.org/abs/2610.01905) is a useful adjacent example: its learned one-dimensional map tokens summarize image evidence before map decoding. The slots are not BEV cells. If a new adapter follows that design, carry explicit geometry through its features or train a spatially anchored query scheme; do not infer a metric map from slot index alone.

The choice affects the evaluation. For a grid, test cell correspondence after downsampling and coordinate transforms. For learned slots, inspect whether small or occluded objects disappear as the slot budget falls. In both cases, compare at a fixed final reader-token budget. Otherwise, a larger projector may win simply because it passes more evidence downstream.

## Object adapters: boxes and attributes

An object interface starts with a set of detected or tracked instances. Each instance carries appearance, geometry, attributes, and observation quality. This representation makes object-level questions direct: which vehicle is crossing a lane boundary, which pedestrian is occluded, or which traffic light controls this approach?

It also inherits the detector's omissions. An undetected object produces no object token. The grid can retain weak evidence that has not become a discrete detection, while the object representation gives explicit structure to accepted hypotheses. This is the main reason to evaluate both interfaces separately before combining them.

### Define a typed object record

For a 3D driving object, begin with center coordinates, dimensions, orientation, velocity, class evidence, confidence, observation age, and validity. State the coordinate frame and units. A practical convention is an ego frame at the query time, with forward, left, and up axes; document it explicitly and transform every source into it. Other conventions are valid if they are consistent.

Represent heading with sine and cosine when the object rotates about the vertical axis. These values avoid a jump at the angle wrap. They do not solve full three-dimensional orientation, for which an explicit rotation representation is needed. Normalize coordinates and velocities using fixed documented scales, and preserve flags for out-of-range or unknown values.

For a 2D box, record both the image coordinate convention and the camera identity. Normalized box coordinates remove dependence on image size, but not on crop or camera geometry. A box in the left camera and a box at the same coordinates in the front camera are different observations. Do not turn them into one 3D object without association evidence.

| Field group | Example contents | Reason to keep it explicit |
| --- | --- | --- |
| Appearance | Region-pooled image or point features | The class label alone may omit useful evidence |
| Geometry | Center, size, orientation, box frame | A position has meaning only in a coordinate system |
| Motion | Velocity, age, track history | Old observations need different interpretation |
| Attributes | Class probabilities, signal state, occlusion | Unknown and uncertain must differ from false |
| Provenance | Sensor, validity, association confidence | The reader needs to distinguish absence from disagreement |

The record separates measured state from inferred attributes. For example, an indicator light may be visible, while an intention to turn is a hypothesis. Encode that distinction in training labels. A low-confidence class distribution can be more informative than an incorrect hard class label, provided the reader is trained to use uncertainty rather than treating every field as equally reliable.

### Build one object token, then test whether one is enough

Use separate small encoders for appearance, geometry, and attributes. Concatenate their outputs and project the result to the reader width. Geometry can pass through an MLP or a Fourier feature encoding followed by an MLP. Fourier features express coordinates at several frequencies; their scales must match the physical distances the task needs to distinguish.

For each object, the proposed projector receives three feature groups: normalized appearance, encoded geometry, and encoded attributes. The equation uses lowercase f for appearance, g for geometry, and a for attributes. Uppercase G and A denote the geometry and attribute encoders.

$$
z_j=P\bigl([\operatorname{LN}(f_j);G(g_j);A(a_j)]\bigr).
$$

The three feature groups are concatenated along the channel axis within one object. The projector mixes appearance, location, and state into a reader-width vector. It does not merge different objects. If one token loses important detail, allocate a small fixed group per object, such as appearance and state tokens, and include an object-group identifier so the association remains explicit.

Region features must follow the same transforms as the boxes. If an image is cropped or resized, transform the box before pooling its features. If point features are pooled inside a 3D box, specify whether coordinates are local to the object or global to the scene. Local coordinates describe shape; scene coordinates describe position. Both may be useful, but neither should silently replace the other.

[Kosmos-2](https://arxiv.org/abs/2306.14824) grounds language through location tokens and bounding boxes. That offers a useful alternative baseline: serialize object records as text or coordinate tokens and retain the model's ordinary image path. Continuous object embeddings may be shorter, but they require new learned semantics. Compare them against a readable structured-text baseline before assuming that an opaque vector interface is better.

[3D-MoE](https://arxiv.org/abs/2501.16698) provides an object-centered 3D example with encoded object point clouds and geometry-aware routing. Its indoor reasoning and manipulation setting does not establish a driving result. The transferable idea is to give object appearance and spatial relations an explicit path, then train and evaluate that path in the intended environment.

### Handle sets, identities, and missing objects

An object set has no natural first element. A causal language model does see an order. Sorting by distance is deterministic, but small motion can swap neighboring objects and change their sequence positions. Sorting by tracker identifier stabilizes a clip but should not make the numeric identifier a semantic property.

Use a set encoder before the reader if permutation robustness matters. Alternatively, randomize object order during training and measure answer consistency under permutations. This encourages robustness but does not mathematically guarantee it. For referring questions, include explicit local object identifiers and keep the mapping stable throughout the prompt and answer.

Track identity is useful within a sequence. It should not leak scene identity across train and test splits. A reassociated track must carry uncertainty rather than silently asserting continuity. Cross-camera observations of one actor need association before deduplication; otherwise, counting questions can double-count that actor.

Reserve a clear representation for no detected objects. Padding is not a detected object at the origin. Mask padded slots and keep a separate frame-validity signal. Also distinguish an empty scene from an unavailable detector. These cases may have identical tensor shapes while requiring different answers.

Finally, a box is an extent, not just a center. Questions about lane overlap, collision corridors, or partial occlusion require size and orientation. Relations can be encoded as pairwise features in a set encoder or as selected relation tokens. Avoid materializing every pair by default: the number of pairs grows quadratically, and many relations will be irrelevant to the query.

## Position information has two jobs

Position can enter the **content** of a token through coordinate features. It can also affect **attention** through a positional encoding. These mechanisms are complementary. A geometry MLP can tell the reader that a box is twelve meters ahead. A positional encoding can influence which tokens interact as neighbors. One does not automatically substitute for the other.

[RoFormer](https://arxiv.org/abs/2104.09864) introduces rotary position embeddings, or RoPE, which rotate query and key components according to position. The resulting attention scores depend on relative position. RoPE is not simply an extra coordinate vector added to the input embedding.

For a new BEV adapter, a conservative starting point is to encode metric geometry in the token content and use an ordinary sequence-position policy for the external prefix. This avoids pretending that metric x and y are native image-patch indices. A custom rotary policy may be useful later, but it changes attention behavior and needs its own training and ablation.

Keep coordinate conventions stable across distance scales. If one dataset uses centimeters and another uses meters, an unchanged Fourier or rotary scale will represent different physical relations. If the BEV resolution changes, update the cell centers and extents even when the number of output tokens remains fixed.

### Time, cameras, and ego motion

Before combining observations from different times, decide the reference time. A static map feature can be transformed with ego motion. A moving actor also needs a motion estimate or a retained observation time. Warping a pedestrian with ego motion alone treats the pedestrian as static and can place it incorrectly.

Preserve both the observation age and the uncertainty of any propagation. Do not present an extrapolated state as a fresh measurement. During training, prohibit future frames, future tracks, or labels computed with unavailable future context from entering the input. A model can otherwise appear temporally capable by using information that deployment will never provide.

Camera identity and sensor identity are categorical metadata. They are not additional spatial axes by default. Encode them in content or through a learned source embedding, and train examples in which camera order changes. A model that relies on array order instead of camera identity can fail when a stream is missing.

## Integrating the adapters with Qwen3-VL

Use a named checkpoint instead of the phrase “Qwen-sized.” For **Qwen3-VL-8B-Instruct**, the reviewed [configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json) has a 4,096-wide language state and a 1,152-wide vision state. Its spatial merge size is two. Four vision features therefore form a 4,608-wide merger input, which is projected to 4,096 channels. Other Qwen3-VL sizes have their own configurations.

That native merger does not accept arbitrary 256-channel features. The example grid adapter has its own learned input map. Matching the native merger's input width with another linear layer would make the shapes compatible, but would not make the features follow the distribution on which the native merger was trained.

The [Qwen3-VL report](https://arxiv.org/abs/2511.21631) describes interleaved multidimensional RoPE, or MRoPE, plus DeepStack connections from intermediate vision features into early language layers. It also uses textual video timestamps. These mechanisms are part of the pretrained image and video interface. A new modality should enter with a defined contract rather than borrowing the image interface by coincidence.

### Preserve the native image path first

My proposed first implementation keeps Qwen's camera-image path and adds an external context prefix for the grid or object tokens. This isolates the new adapter and preserves the pretrained visual representation. The reader receives native image evidence, external context, and the question before generating its answer.

Implement the prefix in a wrapper around a pinned model version. The wrapper must own embedding insertion, valid-token masks, position construction, label masking, and generation-cache handling. Its contract should say which positions are native images and which are external context. The adapter tokens need answer-loss supervision through the reader, but should not be treated as vocabulary targets themselves.

The inspected [Transformers v4.57.1 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) scatters native visual embeddings into image or video placeholders. It also uses visual-position masks for DeepStack additions. Its position builder derives native grid coordinates from image metadata and retains offsets for cached generation. A wrapper must preserve those associations after inserting context.

Do not reuse image placeholder IDs for arbitrary object tokens. That can trigger native image-count checks or associate DeepStack features with the wrong positions. Likewise, passing an `inputs_embeds` tensor is not a complete integration: a valid prefill can still be followed by incorrect cached positions during generation.

Test the wrapper with padding, mixed image counts, empty external context, variable object counts, and more than one generated token. Compare cached and uncached next-token logits within the expected numerical tolerance. Also test that the unmodified native path produces the same output when the external-context length is zero.

### Treat deeper fusion as a separate experiment

Adding external features to intermediate language layers could make them easier to access. It also adds new normalization, scale, and position-association choices. Begin with explicit learned gates and measure whether the extra paths improve grounding at the same input-token budget. A zero-initialized gate can preserve initial behavior, but check that gradients reach the gate and then the adapter; multiplying a whole branch by zero initially blocks gradients to that branch's internal weights.

An alternative is to fuse new evidence with the native image features before the reader. This requires correspondence at the image-token locations. It may suit depth features aligned to the same image, but is less direct for a variable object set or a metric BEV grid. Cross-attention can bridge those layouts at the cost of another learned association stage.

Keep the native image path, prefix-only path, and deeper-fusion path as separate experiments. Changing all of them at once makes it difficult to identify whether an improvement came from the representation, the training budget, or the reader modifications.

## Train the reader to use the evidence

Start with the source encoders and reader frozen, and train only the new connector on paired inputs and answers. Freezing reader weights reduces trainable parameters; it does not eliminate the backward computation through the reader. The adapter still needs gradients through those operations. Wrapping the reader forward pass in `torch.no_grad()` would prevent that learning.

Use tasks that require the new modality. A caption that is already obvious from the image gives little pressure to use a box or velocity token. Better examples include a hidden object retained by tracking, a metric distance unavailable from appearance alone, or a stale observation that must be discounted. Keep such examples labeled by the evidence that was available at the query time.

After a connector-only baseline, compare limited reader adaptation, such as LoRA, with the frozen-reader run. LoRA modifies selected reader weights; it is distinct from the modality projector. Evaluate text and ordinary image tasks as well as the new driving tasks so that improved context use does not hide a regression elsewhere.

Feature distillation can initialize a student connector when teacher and student tokens have known correspondence. Match cells only after accounting for resize, crop, and merge order. For object features, match object identities or use an explicit assignment. Comparing equal sequence indices without this step can teach the student to reproduce the wrong target.

Normalize a feature loss only when its invariances are intended. Cosine similarity ignores magnitude; mean squared error penalizes it. Neither proves that the reader uses the feature correctly. Retain an answer-level objective and counterfactual evaluation beside the alignment loss.

[PaLM-E](https://arxiv.org/abs/2303.03378) provides a broader precedent for incorporating continuous embodied observations into a language-model interface. It supports the general direction, but an implementation still needs a task-specific input contract, paired data, and deployment-relevant tests. This guide does not imply that a projector alone produces a trained driving policy.

## A proposed hybrid and the tests that can reject it

My starting hybrid keeps native camera tokens, adds a locally merged BEV grid for scene context, and adds one projected token per tracked object. The grid supplies distributed evidence. The objects supply explicit extents and state. Ego motion and route context enter through separately typed features, with the same reference time and frame.

![Separate grid and object adapters feed external context while the native image path remains intact.](/assets/images/spatial-adapter-hybrid.svg)
*Author's proposed architecture. The two adapters have separate input contracts and share a reader-width output. Geometry, time, and validity remain explicit; no performance gain is claimed without the ablations below.*

An illustrative budget is 64 grid tokens and up to 32 object tokens, in addition to the native image tokens and text. These numbers are experimental choices, not literature optima. Increasing them changes prefill work and cache memory. Measure end-to-end latency from sensor input through adapter construction and answer generation, rather than timing only the final projection.

| Comparison | Hold fixed | What a useful improvement should show |
| --- | --- | --- |
| Structured text versus continuous object tokens | Same object records and questions | Better accuracy or lower token cost without lost attributes |
| Per-cell projection versus local merge | Source encoder and training data | Acceptable localization at the reduced token budget |
| Grid only versus objects only versus both | Reader adaptation and evaluation set | Complementary gains on weak evidence and explicit relations |
| Correct context versus shuffled context | Image and question | A clear dependence on the relevant external evidence |
| Fresh versus stale observations | Physical scene and answer target | Appropriate use of time and uncertainty |
| Prefix versus deeper fusion | Final token count and training budget | A gain that justifies the extra model changes |

Report answers by failure type: object omission, wrong association, wrong frame, wrong time, wrong attribute, and unsupported inference. Split data by scene or route where possible, and inspect overlap. A model can memorize place-specific answers without learning the new sensor interface.

Include small-object localization, box extent, track identity, cross-camera counting, and missing-sensor cases. Ask the same question after an object-order permutation. Translate or rotate the whole scene and update the labels consistently. Break only one input field at a time to determine whether the response follows the changed evidence.

Reject an adapter that gives nearly the same answer with unrelated context. Reject a spatial adapter that passes ordinary questions but fails coordinate-transform checks. Reject a temporal adapter whose advantage disappears when future information is removed. These tests turn “the tokens fit” into a claim about observable behavior.

The first deliverable should be a small, reproducible interface experiment: explicit grid and object schemas, checked tensor layouts, a pinned reader wrapper, and matched baselines. Once those pieces work, larger encoders and more training data have a clear role. Until then, increasing token width mostly makes an ambiguous interface more expensive.
