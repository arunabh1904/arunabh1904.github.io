---
title: 'Tracing the VLM Progression'
date: '2026-07-05T20:00:00.000Z'
section: blog
blogGroup: research-guides
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
legacyPath: /blog/2026/07/05/from-seeing-to-doing-the-evolution-of-vision-language-models.html
tags:
  - Research
  - Vision-Language Models
summary: A technical history of how vision-language models moved from task-specific fusion and image-text alignment to visual assistants, grounded reasoning, video, and robot action.
---
# Tracing the VLM Progression

A vision-language model is not one capability. It is an interface that decides which parts of an image survive into a language-conditioned output. The VLM progression is therefore a progression in output contracts: recognition, alignment, generation, grounding, temporal reasoning, and finally action.

That progression matters because a model can name a mug, explain that mugs hold liquid, and still drive a gripper into the table beside it. Recognizing the mug, binding the word *mug* to pixels, estimating its pose, predicting contact, and controlling a wrist are different jobs. Each tolerates a different amount of lost information.

This is the history I care about. A new model can look like a clean capability jump while quietly inheriting an older compression boundary. Once location, time, or geometry disappears, a fluent language model cannot talk it back into existence.

The usual timeline, CLIP to LLaVA to video to robotics, is useful but incomplete. It hides two important transitions. First, CLIP did not begin vision-language learning. Earlier systems already learned joint representations through region-level fusion. CLIP changed the scaling economics and the interface: natural language became an open vocabulary for visual recognition. Second, visual chat did not emerge directly from contrastive alignment. A generation of models learned how to connect pretrained visual encoders and language models before instruction tuning turned that machinery into an assistant.

A more durable history follows what the representation became responsible for:

1. **Task-specific fusion** learned interactions between detected regions and words.
2. **Image-text alignment** made visual concepts addressable through natural language.
3. **Conditional generation** made visual evidence usable by a language decoder.
4. **Instruction tuning** taught the system how to behave as a visual assistant.
5. **Grounding and video** forced the representation to preserve location, detail, and time.
6. **Decision and action models** made errors consequential in the next state of the world.

Each stage inherits machinery from the previous one, but none is a free upgrade. Global semantic alignment can discard location. Fluent generation can hide weak eyesight. More frames can consume context without teaching dynamics. A chain of thought can organize evidence that survived the encoder, but it cannot reconstruct pixels that were never preserved. Language can specify a goal while saying nothing about metric geometry or control frequency.

This is Part I of a three-part reading course. [Part II: Pre-Training for Robotics](/blog/2026/07/15/omni-model-pretraining-decisions.html) asks how multimodal and robot data should shape a pretrained policy. [Part III: Post-Training for Robotics](/blog/2026/07/16/post-training-vision-language-action-models-zero-to-hero.html) asks how deployment failures should justify a policy update. The sequence matters: post-training cannot recover visual evidence that pretraining threw away, and pretraining cannot anticipate every state the deployed policy will create.

The scope is deliberately selective. I care about architectural and data decisions that survive model generations, not a complete leaderboard. Reported results belong to the cited papers. The interface taxonomy, comparisons, and reading order are my synthesis.

## How to use this guide

Do not read every VLM paper with the same question.

For an alignment paper, identify the unit of comparison and the structure that the loss can ignore. For a visual-assistant paper, trace where spatial detail can disappear before it reaches the language model. For a grounding paper, ask whether the target is a word, box, point, mask, track, or metric relation. For video, separate temporal compression from actual state modeling. For driving and robotics, write down the action representation, control deadline, and source of closed-loop evidence.

The recurring exercise is simple:

> Draw the path from raw observation to evaluated output, then circle every irreversible compression step.

That picture usually explains more than the model name.

## The progression is a sequence of output contracts

“Vision-language model” now covers systems with very different outputs:

| System contract | Typical output | What vision must preserve | Failure that matters |
| --- | --- | --- | --- |
| Image-text alignment | similarity score | global semantic identity | wrong neighborhood in embedding space |
| Visual assistant | text tokens | evidence needed across a response or conversation | fluent answer unsupported by the image |
| Grounded perceiver | boxes, points, masks, regions, text | identity, location, and fine detail | right noun, wrong object or position |
| Temporal observer | event, timestamp, track, state change | persistence, ordering, and brief events | right scene, wrong moment or entity |
| Decision model | rationale, tool call, plan, trajectory proposal | state, rules, geometry, and uncertainty | plausible explanation for a bad decision |
| Embodied policy | action token, chunk, or control distribution | state transitions, embodiment, and timing | compounding closed-loop error |

This table is the first defense against vague capability claims. A retrieval model can be excellent without being able to generate text. A visual assistant can describe a scene while failing to localize a small object. A model can localize an object in two dimensions yet lack the depth, pose, or control rate required to manipulate it.

The common implementation pattern is deceptively simple. An observation is converted into visual features or tokens. A connector, cross-attention path, or shared transformer allows those representations to interact with text. A loss then decides what “interaction” means: contrastive agreement, next-token prediction, masked modeling, region grounding, video prediction, denoising, or action imitation.

> **The loss is the contract. The architecture determines which evidence survives long enough to satisfy it, and which evidence no later stage can recover.**

The animation below keeps one mug-and-tray scene fixed. Only the evaluated output changes. Watch which evidence becomes mandatory rather than assuming every later model is simply a larger version of the first.

[![Animation showing how CLIP, LLaVA, Molmo, and Pi0 require progressively different visual evidence from the same scene](/assets/images/blog-vlm-evidence-contract.gif)](/assets/images/blog-vlm-evidence-contract.gif)

*CLIP's contrastive objective can succeed with global semantic identity; it does not require an inspectable location. LLaVA projects visual features into a language model so they can condition generation. Molmo's point supervision makes a spatial binding externally testable. Pi0 adds temporally conditioned continuous action through a flow-based expert. These panels compare output contracts. They do not claim that one model literally evolves into the next. Synthesis based on [CLIP](https://arxiv.org/abs/2103.00020), [LLaVA](https://arxiv.org/abs/2304.08485), [Molmo](https://arxiv.org/abs/2409.17146), and [Pi0](https://arxiv.org/abs/2410.24164).*

The progression is therefore not “more modalities.” It is a sequence of stricter information obligations. A global vector may identify the mug while discarding the handle location. A generative assistant needs enough visual tokens to support an answer but may still lack an explicitly supervised binding. Point grounding exposes that binding. Action raises the standard again: the representation must remain useful across changing observations and an embodiment-specific deadline.

## Before CLIP: task-specific cross-modal fusion

It is tempting to begin VLM history with CLIP because CLIP became the visual backbone for so many later systems. Historically, however, the field had already established a rich language of multimodal pretraining.

[ViLBERT](https://arxiv.org/abs/1908.02265) used separate visual and language streams connected through co-attention. [LXMERT](https://arxiv.org/abs/1908.07490) separated object, language, and cross-modality encoders. [UNITER](https://arxiv.org/abs/1909.11740) used a joint transformer with masked language, masked region, image-text matching, and word-region alignment objectives.

These models solved a real problem: they let words and image regions interact deeply enough to support visual question answering, referring expressions, retrieval, and reasoning. They also exposed three limitations that shaped the next generation.

First, the visual interface was often a set of object proposals produced by a detector. The detector had already decided which regions deserved representation. Missed proposals, detector vocabulary, and pretraining biases became an upstream bottleneck.

Second, cross-modal fusion is expensive for retrieval. If every candidate image and sentence must interact through a transformer, large-scale indexing becomes difficult. A dual encoder can embed each side once and compare them cheaply.

Third, the pipelines were heavily coupled to curated tasks and annotations. Scaling them required more than collecting noisy image-caption pairs from the web.

CLIP's contribution makes more sense against this backdrop. It traded dense pairwise interaction for a scalable alignment interface.

### A recurring architectural fork

The field repeatedly revisits the same tradeoff in different forms:

| Interface | How modalities interact | Main advantage | Main bottleneck |
| --- | --- | --- | --- |
| Dual encoder | compare independently encoded image and text embeddings | scalable retrieval and open-vocabulary transfer | weak explicit token-region interaction |
| Cross-modal encoder | fuse visual and text tokens through attention | rich pairwise reasoning and grounding | expensive per pair, often task-coupled |
| Visual prefix or resampler | map visual features into a language decoder | reuses a strong pretrained language model | connector or token bottleneck can discard detail |
| Early-fusion token model | process all modalities in one autoregressive stream | one training and decoding interface | tokenization and representation conflicts |
| Semantic trunk plus specialist expert | share high-level semantics, specialize geometry or actions | preserves task-specific bandwidth and timing | coordination and training complexity |

Model names change. This tradeoff does not.

## Image-text alignment at web scale

Before large-scale image-text pretraining, a vision classifier usually learned a closed vocabulary. Its final layer represented the categories chosen by the dataset designer. Adding a new concept meant collecting labels and retraining the head or model.

[CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html) changed the interface. It trains an image encoder $f_I$ and text encoder $f_T$ so matched image-caption pairs receive high similarity and mismatched pairs receive lower similarity. In simplified form,

$$
s_{ij}=\frac{f_I(I_i)^\top f_T(T_j)}{\tau},
$$

where $\tau$ controls the sharpness of the similarities. At inference, class names become text prompts. Classification becomes retrieval against language rather than prediction through a fixed learned head.

![Figure 1 from the CLIP paper, showing contrastive pretraining and zero-shot transfer through text prompts](/assets/images/clip-paper-figure-1-contrastive-pretraining.png)

*The CLIP interface is the important change: images and text are aligned during pretraining, then written class prompts replace a fixed classifier head. Source: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), Figure 1.*

The important result was not merely zero-shot ImageNet accuracy. CLIP showed that noisy natural language at web scale can define a broad visual supervision space. A prompt can name a category, attribute, style, activity, or relation that never appeared as a dedicated class label.

That interface changed the economics of transfer. One pretrained image encoder could support retrieval, open-vocabulary classification, filtering, and later multimodal systems. It also made the language model's vocabulary a practical control surface for visual recognition.

The tradeoff sits inside the objective. Batch-softmax contrastive learning treats other examples in a batch as negatives. Large and diverse batches improve the comparison set, but they demand synchronization and can introduce false negatives when multiple captions describe compatible content.

[SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) changes the unit of competition. Each image-text pair receives an independent positive or negative sigmoid loss rather than participating in one batch-wide softmax. The broader lesson matters beyond SigLIP:

> **An alignment loss determines the unit of competition. If the unit is the batch, batch composition and distributed systems become part of the learning algorithm.**

[SigLIP 2](https://arxiv.org/abs/2502.14786) shows how far this line can be extended without abandoning the dual-encoder interface. It combines sigmoid alignment with captioning-style supervision, self-supervised objectives, data curation, multilingual training, and multi-resolution support. The goal is no longer only strong global retrieval. It is to make the visual encoder more useful for localization, dense prediction, and later multimodal models.

This is an important correction to a common narrative. Contrastive pretraining does not have to remain purely global. Local and self-supervised objectives can pressure the encoder to preserve more structure. Still, the output contract matters. If success is measured only by image-sentence similarity, fine spatial evidence remains optional.

A shared embedding can tell us that an image and sentence belong together without preserving which patch supports which phrase, whether the left or right mug is referenced, or how to generate a multi-sentence answer. Global semantic alignment is a strong visual prior, not a complete multimodal interface.

## Connecting vision to a language generator

The historical jump from CLIP to LLaVA can look immediate in hindsight. It was not. A central question between them was:

> How little must we train to make a strong pretrained language model condition on images?

[Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884) showed that a trainable visual prefix could feed images into a frozen language model. [Flamingo](https://arxiv.org/abs/2204.14198) made the idea substantially more capable by combining a vision encoder, a Perceiver Resampler, and gated cross-attention layers inserted into a pretrained language model. It could consume interleaved images, video, and text, then perform new tasks from a few in-context examples.

[BLIP-2](https://arxiv.org/abs/2301.12597) sharpened the bottleneck idea. Its Querying Transformer, or Q-Former, learns a compact set of visual queries that bridge a frozen image encoder and a frozen language model. The design is efficient because most parameters remain fixed. It is also conceptually revealing: a small learned interface decides which visual evidence is allowed to enter the language model.

A parallel line trained more of the multimodal system end to end. [PaLI](https://arxiv.org/abs/2209.06794) used a visual encoder and encoder-decoder language model across many vision-language tasks and languages. The distinction is useful:

| Strategy | Typical recipe | What it buys | What it risks |
| --- | --- | --- | --- |
| Bridge frozen components | frozen vision encoder + small adapter/resampler + frozen or mostly frozen LM | low training cost, reuse of strong unimodal models | a narrow interface may cap visual fidelity |
| Joint multimodal training | co-train visual and language pathways on a task mixture | deeper adaptation and shared representations | more compute, data balancing, and optimization complexity |

The early generative era was therefore not mainly about chat. It was about deciding where modality adaptation should live. Should visual evidence be compressed into a short prefix? Should the language model receive visual tokens through inserted cross-attention? Should both sides move together during training?

Those questions remain alive in current systems. The adapter may be small, but it defines an information boundary.

## Instruction-tuned visual assistants

Conditional generation gives a model the mechanical ability to continue text from an image. Instruction tuning teaches it how to behave when a user asks a visual question.

[InstructBLIP](https://arxiv.org/abs/2305.06500) instruction-tuned a broad mixture of vision-language tasks and made its Q-Former instruction-aware. [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html) made the transition unusually legible: start with a pretrained visual encoder and an instruction-tuned language model, learn a projector into the language model's token space, then fine-tune on image-instruction-response examples.

The minimal architecture looks almost underwhelming:

$$
H_v=W f_I(I), \qquad
p(y\mid I,x)=\prod_t p(y_t\mid H_v,x,y_{<t}).
$$

The projector $W$ is small relative to the components around it. Yet the system feels qualitatively different because the output contract changed. The model must use visual evidence while maintaining the conversational behavior already learned by the language model.

![The LLaVA paper's visual instruction-tuning pipeline](/assets/images/visual-instruction-tuning-llava-paper-figure.png)

*LLaVA separates data generation from model adaptation: GPT-4 first converts image metadata into instruction data, then those examples tune the projected vision-language model. Source: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485).*

LLaVA's durable contribution was not the invention of visual generation. Frozen, Flamingo, PaLI, and BLIP-2 had already established that path. LLaVA made a simple, reproducible recipe for a visual assistant and showed how synthetic instruction data could shape the interaction policy.

This distinction matters:

- **Multimodal pretraining** determines which visual and linguistic regularities the model can represent.
- **Instruction tuning** determines how those capabilities are elicited and formatted.
- **Preference and rejection data** shape which answers the model favors, calibrates, or refuses.
- **Tool and action data** determine whether the output becomes an operation rather than a sentence.

Instruction tuning solves a behavioral problem. It does not automatically repair a perceptual bottleneck.

If a document image is reduced to too few tokens, eloquent instructions cannot recover the missing characters. If pretraining rarely requires spatial reference, the model can learn the statistical form of grounded answers without consistently binding phrases to regions. If captions name only salient foreground objects, background detail remains safe to ignore.

### The connector is often not the main bottleneck

It is tempting to treat the vision-language connector as the central research problem because it is the visible seam between two pretrained models. Controlled studies in [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html) point elsewhere. In its studied recipe, image encoder quality, resolution, number of visual tokens, and the mixture of caption, interleaved, and text-only data had larger effects than increasingly elaborate connector designs.

That result should change experiment allocation:

> Sweep the variables that determine what evidence enters the language model before polishing the bridge that carries it.

[Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html) makes the complementary point from post-training. Its central object is the data strategy: quality, task balance, filtering, and training stages. A smaller model with a deliberate curriculum can compete with a much larger model trained on a poorly organized mixture.

The operational question is no longer “How many multimodal examples do we have?” It is:

> Which behavior does each dataset teach, and which capability regresses when we increase its weight?

### From attached vision to native multimodal pretraining

Many early visual assistants were assembled from separately pretrained components. That approach remains effective, but the field is moving vision earlier into the model lifecycle.

[InternVL3](https://arxiv.org/abs/2504.10479) describes this as native multimodal pretraining: multimodal and text capabilities are developed in one pretraining stage rather than treating vision only as a post hoc attachment. The model still initializes strong pretrained components, so “native” should be read as a training strategy, not as learning every subsystem from scratch.

[Qwen3-VL](https://arxiv.org/abs/2511.21631) pushes the same progression through long interleaved context, multi-level visual features, dynamic visual tokenization, and explicit timestamp alignment for video. The architectural direction is clear: visual information is no longer a small prefix consumed once at the beginning of a chat. It can be interleaved, revisited, temporally indexed, and injected at multiple depths.

This creates a spectrum rather than a binary distinction:

1. frozen visual encoder plus frozen language model;
2. trainable connector between mostly frozen components;
3. jointly tuned multimodal stack;
4. multimodal data present throughout pretraining;
5. shared generative model over multiple modalities.

The later positions allow deeper adaptation. They also make it harder to attribute gains. Better data, more visual tokens, a stronger language model, longer context, and joint training often move together.

## The visual-token budget became a first-class design variable

Text tokenization compresses language into a sequence of discrete symbols. Images do not arrive with an obvious equivalent. A $1024\times1024$ image can be encoded as one global vector, a fixed patch grid, multiple high-resolution tiles, a variable-length native-resolution sequence, or a small learned set of queries.

Every choice trades detail for sequence length. More visual tokens can preserve small objects and text, but consume attention, memory, and latency. Fewer tokens improve throughput, but may create an irreversible perceptual bottleneck.

[PaliGemma](/paper%20shorts/2024/07/10/paligemma-a-versatile-3b-vlm-for-transfer.html) offers a clean transfer recipe: a SigLIP vision encoder feeds a compact Gemma language model, and later resolution upcycling pays more visual compute where downstream tasks need it. [Qwen2-VL](/paper%20shorts/2024/09/01/qwen2-vl-enhancing-vision-language-model-perception-of-the-world-at-any-resolution.html) makes token count depend on input resolution and adapts positional treatment across images and video.

[Qwen2.5-VL](https://arxiv.org/abs/2502.13923) extends this direction with dynamic resolution, absolute time encoding for video, and stronger support for structured visual outputs such as boxes, points, documents, and charts. [InternVL3.5](https://arxiv.org/abs/2508.18265) introduces a visual resolution router and deployment strategies intended to allocate visual compute more selectively. Qwen3-VL adds multi-level feature injection so early and late visual representations do not have to be squeezed through one final-layer bottleneck.

These systems answer one shared question:

> Where should variable visual complexity be paid for?

| Visual interface | Benefit | Failure mode | Serving implication |
| --- | --- | --- | --- |
| Fixed low-resolution grid | predictable cost and batching | small text and objects disappear | stable but blunt |
| Multi-tile high resolution | preserves local detail | duplicated boundaries and weaker global layout | input-dependent token count |
| Native or dynamic resolution | allocates tokens with image size or shape | worst-case sequences can become expensive | harder batching and latency tails |
| Learned resampler or query tokens | fixed compact interface | compressor decides what is permanently lost | efficient downstream context |
| Multi-level feature injection | exposes both local and semantic features | more complex routing and optimization | extra memory and attention paths |

The token budget is not only an architecture choice. It is a compute allocation policy. It determines which images receive enough representational bandwidth and which do not.

A convincing comparison must therefore match more than parameter count. It should report:

- input pixels and resizing policy;
- number and distribution of visual tokens;
- training and inference FLOPs;
- latency, memory, and batchability;
- performance on small text, counting, localization, and global layout;
- behavior under controlled reductions in visual evidence.

Otherwise, a “better architecture” may simply be buying more pixels.

## Grounding reconnects words to visible evidence

A model can answer “the traffic light is red” for at least three reasons. It may localize the light and read its state. It may use a language prior about the scene. Or it may guess from dataset regularities. Standard answer accuracy often fails to distinguish them.

Grounding asks the model to bind language to a box, point, mask, region, track, or spatial relation. The historical path matters.

[MDETR](https://arxiv.org/abs/2104.12763) trained an end-to-end text-conditioned detector with explicit phrase-object alignment. [GLIP](https://arxiv.org/abs/2112.03857) unified object detection and phrase grounding so detection categories could be expressed through language. [Kosmos-2](https://arxiv.org/abs/2306.14824) made location tokens part of generated text, allowing one autoregressive output to mix words and coordinates.

Later visual assistants brought the same obligation into general-purpose models. [LocCa](/paper%20shorts/2024/03/28/locca-visual-pretraining-with-location-aware-captioners.html) pushes location information into caption-style pretraining. [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html) treats visual representation, connector design, and vision-centric evaluation as controlled variables rather than assuming a stronger language model will absorb every visual weakness.

[Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html) makes point-based grounding and openly described high-quality data central to its recipe. [Molmo 2](https://arxiv.org/abs/2601.10611) extends this interface across video and multiple images through pointing, tracking, timestamps, and dense descriptions.

The deeper lesson is about supervision granularity:

- caption loss can be satisfied by global semantics;
- region captions require spatial binding;
- boxes expose extent;
- points expose correspondence but not shape;
- masks expose object support;
- tracks require identity persistence;
- metric targets require physical scale.

This gives a useful hierarchy for evaluating visual evidence:

1. **Recognition:** is the concept present?
2. **Localization:** which pixels or region support the answer?
3. **Relation:** how do the relevant entities interact spatially?
4. **Persistence:** is the same entity tracked consistently over time?
5. **Counterfactual dependence:** does the answer change when the relevant evidence changes?
6. **Metric consistency:** are distance, orientation, scale, and motion correct in physical coordinates?

The fifth level is often the most revealing. If an answer remains unchanged after the traffic light is masked or its state is edited, the model's explanation was not causally grounded in that evidence.

### Grounding is not geometry

A box or point can establish that a word refers to a location in a two-dimensional image. It does not by itself provide depth, camera pose, object orientation, free space, contact geometry, or uncertainty in metric units.

This distinction is easy to miss because grounded outputs look precise. A point such as $(0.63, 0.41)$ is numerically precise but may still be physically ambiguous. The same pixel can correspond to different world positions under different depth and calibration assumptions.

[SpatialVLM](https://arxiv.org/abs/2401.12168) attacks part of this gap with explicit quantitative spatial supervision. For robotics and autonomous driving, the broader implication is stronger: image-language grounding should complement, not silently replace, geometric representations such as depth, multi-view structure, camera calibration, object pose, maps, and proprioception.

Language can describe geometry. The training target must still make geometric error measurable.

## A parallel branch: unified multimodal generation

Most visual assistants use one representation for understanding images and another model family for generating them. A parallel line asks whether images and text can share one autoregressive token stream.

[Chameleon](https://arxiv.org/abs/2405.09818) trains an early-fusion model over interleaved image and text tokens. [Emu3](https://arxiv.org/abs/2409.18869) similarly frames image, text, and video tasks as next-token prediction over discrete sequences. [Unified-IO 2](https://arxiv.org/abs/2312.17172) pushes the interface further by representing a wide collection of vision, language, audio, and action tasks in a shared token space.

This direction is attractive because it offers one training objective, one context, and one decoder interface. It also exposes a representation conflict. Visual understanding benefits from invariance and semantic abstraction. Image generation needs enough local detail to reconstruct appearance. Action prediction needs precise temporal and embodiment-specific structure.

[Janus](https://arxiv.org/abs/2410.13848) makes that conflict explicit by decoupling visual encoders for understanding and generation while retaining a shared transformer. The design is a useful warning against a simplistic version of unification:

> A shared token stream is an interface choice. It is not proof that every output contract wants the same visual representation.

A model can be architecturally unified while remaining representationally specialized. In practice, this hybrid pattern appears repeatedly: shared semantic reasoning, separate high-bandwidth routes for pixels, geometry, audio, or actions.

## Time: video is not just more images

Video looks like a longer image sequence, but it changes the representation problem. Adjacent frames are highly redundant. Important events can be brief. The correct sampling rate depends on the question. Uniformly encoding every frame wastes context; aggressive sampling can delete the event.

[LLaVA-OneVision](https://arxiv.org/abs/2408.03326) showed that a common model can handle single images, multiple images, and video, with useful transfer from image tasks into video understanding. [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html) combines variable-resolution visual encoding with similarity-based reduction of redundant video tokens. Qwen3-VL adds explicit textual timestamp alignment. Molmo 2 makes temporal grounding inspectable through video pointing and tracking.

These are meaningful advances, but temporal compression and temporal modeling are not the same thing.

A video VLM can answer four increasingly difficult classes of questions:

| Temporal contract | Required capability | Typical shortcut |
| --- | --- | --- |
| Scene description | aggregate visible content | rely on a few representative frames |
| Event localization | identify when a change occurred | infer event from before or after state |
| State tracking | preserve entity identity and state over time | re-detect independently in each frame |
| Action-conditioned prediction | predict how a chosen action changes the future | generate a plausible but action-insensitive continuation |

The last row is where “video understanding” begins to approach a world model.

Temporal coherence alone is not enough. A useful world model must preserve state and respond consistently to interventions. If two different actions produce the same plausible continuation, the model may understand video statistics without representing controllable dynamics.

For video systems, I would separate four tests:

1. **Event localization:** when did the relevant change occur?
2. **State persistence:** what remained true across frames?
3. **Causal ordering:** which event enabled or prevented the next?
4. **Intervention consistency:** does changing an action or state change the predicted consequence?

The first three measure temporal understanding. The fourth begins to test an action-conditioned model of the world.

[V-JEPA 2](https://arxiv.org/abs/2506.09985) is useful precisely because it comes from a different objective family. It learns predictive visual representations from large-scale video, then adds action-conditioned robot data for planning. Its existence is a reminder that a language decoder is not the only route from video to intelligence. Predictive latent objectives may preserve different structure than next-token text generation.

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) returns to a tension that has been present since CLIP: global semantic quality does not guarantee dense spatial quality. It adds dense predictive losses and deep self-supervision across intermediate layers to make video features more spatially grounded and temporally structured. The progression is revealing. Even a strong predictive model eventually needs explicit pressure on the local evidence required by depth, anticipation, and robot interaction.

### Reasoning cannot reconstruct missing evidence

Recent VLMs increasingly produce long reasoning traces. This can improve multi-step computation after perception, but it creates a dangerous evaluation ambiguity: did the model reason better, or did it simply describe a plausible chain around a weak visual guess?

A major post-training shift in 2025 was reinforcement fine-tuning with verifiable multimodal rewards. [Visual-RFT](https://arxiv.org/abs/2503.01785) uses task-specific visual rewards, such as intersection-over-union for detection, rather than rewarding only the final language answer. InternVL3.5 combines offline and online reinforcement learning in a staged recipe. [GRIT](https://arxiv.org/abs/2505.15879) takes a complementary route by interleaving language reasoning with explicit bounding-box references, making parts of the reasoning trace visually inspectable.

These methods move in the right direction because reward design can make perception and grounding part of the optimized behavior. They also sharpen the warning. A reward for answer correctness, formatting, or reasoning length can improve the appearance of reasoning without proving that the relevant pixels caused the answer. Reward functions inherit the same output-contract problem as pretraining losses: whatever is not measured remains optional.

A useful diagnostic separates three error sources:

1. **Perception:** was the relevant object, text, relation, or event represented?
2. **Binding:** was the internal evidence connected to the correct phrase or entity?
3. **Inference:** given the correct facts, did the model derive the correct conclusion?

Oracle experiments can isolate them. Give the model a crop or structured scene description that preserves the relevant evidence. If performance recovers, perception or binding was the bottleneck. If it still fails, the remaining problem is inference. Conversely, increase reasoning tokens while holding visual evidence fixed. If confidence rises without improved evidence sensitivity, the model may be rationalizing rather than looking.

Reasoning is valuable. It should be evaluated as computation over evidence, not as a substitute for evidence.

## From answers to decisions: autonomous driving as a stress test

Autonomous driving compresses nearly every VLM weakness into one domain: small distant objects, metric geometry, traffic rules, rare hazards, temporal prediction, uncertainty, and hard latency limits.

The literature explores several placements for language. [GPT-Driver](/paper%20shorts/2023/10/01/gpt-driver-learning-to-drive-with-gpt.html) and [Driving with LLMs](/paper%20shorts/2023/10/01/driving-with-llms-fusing-object-level-vector-modality.html) represent driving state in forms a language model can reason over. [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html) combines scene reasoning with conventional planning. [AsyncDriver](/paper%20shorts/2024/06/01/asyncdriver-asynchronous-llm-enhanced-planner-for-autonomous-driving.html) addresses the latency mismatch by decoupling the slower language path from faster planning. [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html) uses VLM supervision to improve a deployable policy rather than placing the full VLM in the control loop.

These designs should not be collapsed into “VLMs for driving.” They encode different system bets:

| Placement of language | Main benefit | Primary risk |
| --- | --- | --- |
| Online planner | semantic flexibility and explicit reasoning | latency, instability, and geometric imprecision |
| High-level route or behavior selector | separates semantics from low-level control | interface errors between levels |
| Auxiliary explanation head | inspectable supervision and debugging | rationale may not cause the action |
| Offline teacher or data labeler | rich supervision without online cost | teacher errors become training targets |
| Unified end-to-end policy | fewer hand-built interfaces | difficult attribution, calibration, and safety validation |

My current read is that the strongest near-term pattern is hybrid. VLMs contribute semantic knowledge, intent understanding, rare-scenario interpretation, data annotation, and auxiliary supervision. Metric perception, motion forecasting, constraints, and high-rate control retain explicit structure.

This is not an argument that end-to-end systems cannot work. It is a statement about evidence. Fluent rationales and open-loop trajectory metrics do not establish reliable closed-loop control.

## Benchmarks should force the model to look

[DriveBench](/paper%20shorts/2025/01/01/are-vlms-ready-for-autonomous-driving-drivebench.html), [IDKB](/paper%20shorts/2024/09/01/can-lvlms-obtain-a-drivers-license-idkb.html), [TOD3Cap](/paper%20shorts/2024/03/01/tod3cap-towards-3d-dense-captioning-in-outdoor-scenes.html), and [AutoTrust](/paper%20shorts/2024/12/01/autotrust-benchmarking-trustworthiness-in-large-vision-language-models-for-autonomous-driving.html) attack different versions of the same problem: plausible language can hide weak evidence use.

The same issue appears outside driving. OCR benchmarks can sometimes be solved through document templates. Visual question answering can reward answer priors. Video benchmarks can leak the event through a single frame. Robotics benchmarks can reward memorized scene layouts.

The benchmark question should be phrased as a causal audit:

- Does performance fall when the relevant region or frame is corrupted?
- Does the answer follow a changed sign, object state, position, timestamp, or instruction?
- Can the model distinguish “not visible” from “not present”?
- Do irrelevant edits leave the answer unchanged?
- Does confidence track evidence quality and distribution shift?
- Does the rationale identify evidence that actually changes the decision?
- Does an open-loop gain survive closed-loop execution?

A benchmark that tests only final-answer agreement can reward memorized priors. Corruption tests, counterfactual edits, evidence localization, and calibrated abstention make it harder to pass without looking.

The practical implication is uncomfortable: a higher aggregate score may be less valuable than a lower score with better evidence dependence, calibration, and recovery behavior. Deployment cares about the shape of failure, not only its average frequency.

## From decisions to actions

Robotics completes the transition from description to intervention. Once a model emits actions, its outputs change the next observation. Errors compound under the state distribution created by the policy itself.

[PaLM-E](https://arxiv.org/abs/2303.03378) was an important bridge. It interleaved visual observations, continuous state estimates, and text inside an embodied language model, showing how language-model capacity could support planning and embodied reasoning across tasks.

[RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html) expresses robot actions as tokens, allowing semantic and control outputs to share an autoregressive interface. [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html) adapts an open VLM backbone into a robot policy and makes a modern VLA recipe inspectable.

Tokenizing actions is not the only path. [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html) combines a semantic VLM trunk with a continuous flow-based action expert. [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html) compresses continuous action chunks in the frequency domain so autoregressive models need fewer action tokens. [DexVLA](/paper%20shorts/2025/02/01/dexvla-vision-language-model-with-plug-in-diffusion-expert.html) gives dexterous control a specialized diffusion route. [GR00T N1](https://arxiv.org/abs/2503.14734) uses a related dual-system pattern: a vision-language model handles semantic reasoning while a diffusion transformer produces continuous actions.

### From imitation to generalization, experience, and scale

The next VLA transition is less about inventing another decoder and more about changing the data contract.

[π0.5](https://arxiv.org/abs/2504.16054) combines data from multiple robots, web sources, object detections, semantic subtask prediction, and low-level actions to target open-world generalization. [π*0.6](https://arxiv.org/abs/2511.14759) then makes deployment experience part of post-training through demonstrations, autonomous rollouts, and expert corrections. [π0.7](https://arxiv.org/abs/2604.15483) expands conditioning beyond a task command to include strategy, metadata, and subgoal images, making one generalist policy more steerable across behavior modes.

[Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330) reports another scale step: more than 100,000 hours of real-world UMI trajectories are automatically labeled with language descriptions of state transitions, followed by cross-embodiment post-training on robot data and imperative instructions. The exact reported numbers should be read as paper claims, but the training pattern is more broadly important. Scalable pretraining data may not look like conventional robot demonstrations. It can be captured with cheaper interfaces, labeled in the language of state change, then aligned later to an embodiment and the instructions humans actually issue.

This progression makes the boundary between pretraining and post-training concrete. Pretraining broadens the action prior. Post-training aligns that prior to an embodiment, command distribution, safety envelope, and deployment state distribution. Experience closes the final loop by adding failures and corrections that were absent from demonstrations.

The action representation is not an output-format detail. It defines what the policy can express, how likelihood is computed, and how quickly the system can react.

| Action interface | Advantage | Cost |
| --- | --- | --- |
| Per-step regression | simple and fast | averages multimodal action distributions |
| Discrete action tokens | exact autoregressive likelihood and shared vocabulary | quantization and sequential decoding latency |
| Compressed action tokens | shorter sequences over long horizons | compression may remove abrupt corrections |
| Diffusion or flow chunks | expressive continuous distributions | iterative generation and harder likelihood-based RL |
| Separate action expert | specialization without discarding VLM semantics | extra parameters and coordination path |

A caption has no control frequency. An action does. A policy must fit sensing, inference, communication, and actuation inside a deadline. It must also decide how long an action chunk remains valid before new evidence should interrupt it. Longer chunks reduce inference calls and improve temporal coherence. Shorter chunks respond faster to disturbances.

The transfer from web-scale VLMs to robotics is asymmetric:

- semantic concepts, object knowledge, and instruction following can transfer;
- embodiment, contact, calibration, and control timing must be learned or represented explicitly;
- action data are expensive because they must cover not only desired behavior but recovery states created by the policy.

This is where “language as a universal interface” reaches its limit. Language can carry task semantics. It does not make units, embodiment, contact dynamics, or control latency disappear.

## How to read a VLM paper

I use eight questions to avoid being carried away by a capability collage.

1. **What is the output contract?** Retrieval score, text, point, mask, timestamp, trajectory, or action?
2. **What is one training unit?** Pair, interleaved document, region, frame, clip, episode, or action chunk?
3. **Where can evidence be lost?** Resolution, crop, detector, tokenizer, resampler, connector, context compression, or temporal sampling?
4. **What supervision forces the claimed capability?** Caption, contrastive pair, location, track, metric target, preference, or closed-loop return?
5. **Which component actually changed?** Encoder, connector, language model, data, objective, post-training, or decoder?
6. **What matched control supports the claim?** Same data, pixels, tokens, parameters, compute, and evaluation protocol?
7. **Does the evaluation require the model to use the claimed evidence?** Or can language priors and dataset regularities solve it?
8. **What is the deployment clock?** Offline retrieval, interactive chat, asynchronous planning, or high-rate control?

These questions turn a model paper into a decision record. If the answer to question five is “several things,” the paper may demonstrate a strong recipe without identifying why it works. That is still useful, but it supports adoption more than causal understanding.

## A compact reading course

The reading course is ordered by conceptual dependency. Each layer has a deliverable. Without one, “reading the paper” too easily becomes collecting model names.

### Layer 0: cross-modal fusion

Read [ViLBERT](https://arxiv.org/abs/1908.02265), [LXMERT](https://arxiv.org/abs/1908.07490), and [UNITER](https://arxiv.org/abs/1909.11740).

**Deliverable:** draw the visual and text streams, mark where they interact, and list every assumption introduced by detector-region features.

### Layer 1: alignment

Read [CLIP](/paper%20shorts/2021/02/28/learning-transferable-visual-models-from-natural-language-supervision.html), [SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html), and [SigLIP 2](https://arxiv.org/abs/2502.14786).

**Deliverable:** derive the softmax and sigmoid losses, then explain how batch composition becomes part of the learning algorithm. Add one paragraph on which spatial information each objective can ignore.

### Layer 2: generative bridges

Read [Flamingo](https://arxiv.org/abs/2204.14198), [BLIP-2](https://arxiv.org/abs/2301.12597), and [PaLI](https://arxiv.org/abs/2209.06794).

**Deliverable:** compare the number of trainable components, the visual bottleneck, and the route by which visual tokens enter the decoder.

### Layer 3: assistants and data mixtures

Read [LLaVA](/paper%20shorts/2023/04/01/visual-instruction-tuning-llava.html), [InstructBLIP](https://arxiv.org/abs/2305.06500), [MM1](/paper%20shorts/2024/03/14/mm1-methods-analysis-and-insights-from-multimodal-llm-pre-training.html), and [Eagle 2](/paper%20shorts/2025/01/01/eagle-2-post-training-data-strategies-for-frontier-vision-language-models.html).

**Deliverable:** build an ablation table with five columns: visual encoder, connector, pretraining mixture, instruction mixture, and preference data. Separate a good recipe from a causal result.

### Layer 4: visual evidence and geometry

Read [MDETR](https://arxiv.org/abs/2104.12763), [Kosmos-2](https://arxiv.org/abs/2306.14824), [Cambrian-1](/paper%20shorts/2024/06/01/cambrian-1-vision-centric-exploration-of-multimodal-llms.html), [Molmo](/paper%20shorts/2024/09/01/molmo-and-pixmo-open-weights-and-open-data-for-state-of-the-art-vision-language-models.html), and [SpatialVLM](https://arxiv.org/abs/2401.12168).

**Deliverable:** for one image, estimate the visual-token budget and label the strongest supervision available: caption, box, point, mask, relation, or metric target.

### Layer 5: time and predictive models

Read [LLaVA-OneVision](https://arxiv.org/abs/2408.03326), [VideoLLaMA 3](/paper%20shorts/2025/01/01/videollama-3-frontier-multimodal-foundation-models.html), [Molmo 2](https://arxiv.org/abs/2601.10611), [V-JEPA 2](https://arxiv.org/abs/2506.09985), and [V-JEPA 2.1](https://arxiv.org/abs/2603.14482).

**Deliverable:** design one temporal counterfactual that changes the correct answer and one irrelevant edit that should not. State whether the model predicts text, pixels, latent state, or action-conditioned state.

### Layer 6: decisions and actions

Read [DriveVLM](/paper%20shorts/2024/02/01/drivevlm-convergence-of-autonomous-driving-and-large-vision-language-models.html), [VLM-AD](/paper%20shorts/2024/12/19/vlm-ad-end-to-end-autonomous-driving-through-vision-language-model-supervision.html), [RT-2](/paper%20shorts/2023/07/28/rt-2-vision-language-action-models-transfer-web-knowledge-to-robotic-control.html), [OpenVLA](/paper%20shorts/2024/06/01/openvla-open-source-vision-language-action-model.html), [Pi0](/paper%20shorts/2024/10/01/pi0-vision-language-action-flow-model-for-general-robot-control.html), and [FAST](/paper%20shorts/2025/01/01/fast-efficient-action-tokenization-for-vision-language-action-models.html). Then compare [π0.5](https://arxiv.org/abs/2504.16054), [π*0.6](https://arxiv.org/abs/2511.14759), and [π0.7](https://arxiv.org/abs/2604.15483) as successive changes to the data, post-training, and conditioning contracts.

**Deliverable:** reconstruct the action distribution, horizon, control rate, inference path, and recovery mechanism. The output should be a latency and failure budget, not another model summary.

After these layers, move to [Part II: Pre-Training for Robotics](/blog/2026/07/15/omni-model-pretraining-decisions.html). The VLM literature tells us which visual and semantic priors are available. The robotics literature asks which of those priors survive contact with embodiment.

## A testable thesis

The VLM progression is a sequence of stricter evidence contracts.

Task-specific fusion established deep word-region interaction. Contrastive learning made images addressable through language at web scale. Generative bridges let language models condition on visual features. Instruction tuning made that interface conversational. Grounding tried to reconnect fluent words to visible evidence. Video introduced persistence, compression, and intervention. Driving and robotics exposed every shortcut because a plausible answer can become a bad physical decision.

My strongest architectural bet is a shared semantic layer with explicit high-bandwidth routes for geometry, time, generation, and control. A fully unified token stream should replace that hybrid only when, under matched data, pixels, tokens, parameters, compute, and latency, it wins on fine grounding, metric spatial reasoning, temporal counterfactuals, calibration, and closed-loop recovery.

Until then, “one model for everything” is a research program, not an architectural result.

## Selected references

- [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)
- [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
- [UNITER: Learning Universal Image-Text Representations](https://arxiv.org/abs/1909.11740)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343)
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)
- [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794)
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500)
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611)
- [Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models](https://arxiv.org/abs/2501.14818)
- [PaliGemma: A Versatile 3B VLM for Transfer](https://arxiv.org/abs/2407.07726)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://arxiv.org/abs/2504.10479)
- [InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency](https://arxiv.org/abs/2508.18265)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [MDETR: Modulated Detection for End-to-End Multi-Modal Understanding](https://arxiv.org/abs/2104.12763)
- [Grounded Language-Image Pre-training (GLIP)](https://arxiv.org/abs/2112.03857)
- [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/abs/2306.14824)
- [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860)
- [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146)
- [Molmo 2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611)
- [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168)
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)
- [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)
- [Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action](https://arxiv.org/abs/2312.17172)
- [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326)
- [VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding](https://arxiv.org/abs/2501.13106)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)
- [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)
- [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)
- [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
- [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [π*0.6: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)
- [π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483)
- [Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories](https://arxiv.org/abs/2607.15330)
