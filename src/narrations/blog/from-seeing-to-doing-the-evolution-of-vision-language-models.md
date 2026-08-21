---
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
sourceSha256: 36e8dc971d0ecaa9bd80b93c06a039907f8a619fc9f78b1422ac6c47bd9c1263
---

# Tracing the VLM Progression

A vision-language model is not one capability. It is an interface that decides which parts of an image survive into a language-conditioned output. The progression is therefore a progression in output contracts: recognition, alignment, generation, grounding, temporal reasoning, and finally action.

That distinction matters because a model can name a mug, explain what mugs are for, and still drive a gripper into the table beside it. Recognizing the mug, binding the word mug to pixels, estimating pose, predicting contact, and controlling a wrist are different jobs. Each tolerates a different amount of lost information.

The history I care about follows what the representation became responsible for. CLIP did not begin vision-language learning, and visual chat did not follow from contrastive alignment in one jump. Earlier models learned region-word interaction. CLIP changed the scaling economics. Generative bridges connected visual encoders to language models. Instruction tuning taught those systems to act like assistants. Grounding, video, driving, and robotics then made location, time, geometry, and control impossible to ignore.

Each stage inherits machinery from the previous one, but none is a free upgrade. Global alignment can discard location. Fluent generation can hide weak eyesight. More frames can consume context without teaching dynamics. Reasoning can organize evidence that survived the encoder, but it cannot reconstruct pixels that were never preserved.

## How to use this guide

Do not read every VLM paper with the same question. For alignment, identify the unit of comparison and what the loss can ignore. For a visual assistant, trace where spatial detail disappears before reaching the language model. For grounding, write down whether the target is a word, box, point, mask, track, or metric relation. For video, separate temporal compression from state modeling. For driving and robotics, record the action representation, control deadline, and source of closed-loop evidence.

The recurring exercise is simple: draw the path from raw observation to evaluated output, then circle every irreversible compression step. That picture usually explains more than the model name.

## The progression is a sequence of output contracts

The label VLM covers systems with radically different obligations. Image-text alignment produces a similarity score. A visual assistant produces text supported by an image. A grounded model must bind that text to a location. A temporal model must preserve identity, order, and brief events. A decision model adds rules, geometry, and uncertainty. An embodied policy must produce an action before its deadline and live with the next state that action creates.

The loss is the contract. The architecture determines which evidence survives long enough to satisfy it, and which evidence no later stage can recover. A global vector may identify a mug while discarding its handle location. Point supervision makes spatial binding testable. Action raises the standard again because the representation must stay useful across changing observations.

## Before CLIP: task-specific cross-modal fusion
<!-- covers: A recurring architectural fork -->

ViLBERT, LXMERT, and UNITER already let words and detected image regions interact through co-attention or a joint transformer. They supported visual question answering, referring expressions, retrieval, and reasoning. But their visual interface often began with detector proposals, so the detector had already decided what deserved representation. Missed regions and detector vocabulary became an upstream ceiling.

The fused encoder was also expensive for retrieval because every candidate image and sentence had to interact. And the pipeline remained coupled to curated tasks and annotations. CLIP makes more sense against that backdrop: it traded dense pairwise interaction for a scalable alignment interface. This fork keeps returning. Dual encoders scale cheaply; cross-modal encoders reason richly; compact prefixes reuse language models but can discard detail; shared trunks with specialist experts preserve task-specific bandwidth at the cost of coordination.

## Image-text alignment at web scale

CLIP trains separate image and text encoders so matched pairs have high similarity and mismatched pairs have lower similarity. At inference, written class prompts replace a fixed classifier head. Natural language becomes an open vocabulary for visual recognition.

The important result was not merely zero-shot ImageNet accuracy. Web-scale language created broad visual supervision and changed the economics of transfer. One encoder could support retrieval, open-vocabulary classification, filtering, and later multimodal systems.

The tradeoff lives inside the objective. CLIP's batch softmax makes every other example part of the competition, so batch composition and distributed systems become part of the learning algorithm. SigLIP instead gives each pair an independent sigmoid loss. Later variants add captioning, self-supervision, multilingual data, and multi-resolution training. Yet the contract remains decisive: if success is measured only by image-sentence similarity, fine spatial evidence is optional. A shared embedding is a strong visual prior, not a complete multimodal interface.

## Connecting vision to a language generator

The next question was how little we must train to make a strong language model condition on images. Frozen language-model work used a trainable visual prefix. Flamingo combined a vision encoder, a Perceiver Resampler, and gated cross-attention inside a pretrained language model. BLIP-2 used a compact Q-Former between a frozen image encoder and frozen language model. PaLI trained more of the multimodal stack together.

These systems established the bridge before visual chat. Freezing most parameters lowered training cost, but it made a small learned interface responsible for choosing which visual evidence could enter the language model. Joint training allowed deeper adaptation but demanded more compute and careful data balancing. The adapter may be small. Its information boundary is not.

## Instruction-tuned visual assistants
<!-- covers: The connector is often not the main bottleneck | From attached vision to native multimodal pretraining -->

Conditional generation provides the machinery to continue text from an image. Instruction tuning teaches the model how to behave when a user asks a visual question. InstructBLIP made its Q-Former instruction-aware. LLaVA paired a pretrained visual encoder with an instruction-tuned language model, learned a projector, and tuned the system on image-instruction-response examples.

The system felt qualitatively new because the output contract changed, not because the connector was large. Multimodal pretraining determines which regularities the model can represent. Instruction tuning determines how those capabilities are elicited. Preference data shapes which answers it favors. Tool and action data decides whether an output becomes an operation.

Instruction tuning cannot repair a perceptual bottleneck. MM1's controlled studies point toward image-encoder quality, resolution, visual-token count, and data mixture as larger levers than an elaborate connector. Eagle 2 makes the complementary post-training argument: quality, balance, filtering, and curriculum matter more than raw example count. Sweep the variables that determine what evidence enters the language model before polishing the bridge that carries it.

## The visual-token budget became a first-class design variable

Images do not arrive with an obvious tokenization. The same image can become one global vector, a fixed patch grid, high-resolution tiles, a variable native-resolution sequence, or a learned set of queries. Every choice trades detail for sequence length.

More tokens preserve small objects and text but consume attention, memory, and latency. Fewer tokens improve throughput but can create an irreversible bottleneck. PaliGemma uses resolution upcycling where downstream tasks need it. Qwen and InternVL variants make token count or routing depend on the input and expose features at several depths.

The token budget is a compute-allocation policy. A fair comparison must match pixels, resizing, visual tokens, training and inference compute, latency, memory, and performance on both small details and global layout. Otherwise, a supposedly better architecture may simply be buying more pixels.

## Grounding reconnects words to visible evidence
<!-- covers: Grounding is not geometry -->

A model may answer that a traffic light is red because it localized the light, because it used a scene prior, or because it guessed from dataset regularities. Final-answer accuracy often cannot distinguish them. Grounding binds language to a box, point, mask, region, track, or spatial relation.

MDETR and GLIP made phrase-object alignment part of detection. Kosmos-2 generated words and coordinates together. LocCa pushed location into caption-style pretraining. Molmo made point supervision central, and Molmo 2 extended inspectable grounding into video through pointing, tracking, and timestamps.

Supervision granularity defines what becomes testable. Captions can be satisfied by global semantics. Boxes expose extent. Points expose correspondence. Masks expose support. Tracks require persistent identity. Metric targets require physical scale. The strongest test is counterfactual: if the relevant traffic light is masked or edited, the answer should change. A two-dimensional point is still not geometry; depth, calibration, pose, free space, and uncertainty require their own measurable targets.

## A parallel branch: unified multimodal generation

Chameleon, Emu3, and Unified-IO 2 ask whether images, text, video, audio, and actions can share one autoregressive token stream. One objective and one decoder are attractive, but the modalities want different representations. Visual understanding benefits from semantic invariance. Image generation needs local appearance. Action prediction needs precise timing and embodiment-specific structure.

Janus makes that conflict explicit by separating visual encoders for understanding and generation while keeping a shared transformer. This hybrid is a useful warning: a shared token stream is an interface choice, not proof that every output contract wants the same representation. A model can be architecturally unified while retaining high-bandwidth specialist routes for pixels, geometry, audio, or actions.

## Time: video is not just more images
<!-- covers: Reasoning cannot reconstruct missing evidence -->

Adjacent frames are redundant, important events can be brief, and the correct sampling rate depends on the question. Uniformly encoding every frame wastes context; aggressive sampling deletes the event. LLaVA-OneVision, VideoLLaMA 3, Qwen3-VL, and Molmo 2 improve variable-resolution encoding, token reduction, timestamps, pointing, and tracking. These are real advances, but temporal compression is not temporal modeling.

A scene description can rely on representative frames. Event localization must identify when a change happened. State tracking must preserve entity identity. Action-conditioned prediction must change the future when the action changes. That last contract begins to look like a world model.

V-JEPA 2 and V-JEPA 2.1 remind us that language generation is not the only route. Predictive latent objectives can learn state and planning structure, while dense losses restore spatial detail. For reasoning models, separate perception, binding, and inference. More reasoning tokens cannot reconstruct missing visual evidence, and rewards for answer form can improve the appearance of thought without proving that pixels caused the answer.

## From answers to decisions: autonomous driving as a stress test

Driving puts small distant objects, metric geometry, traffic rules, rare hazards, temporal prediction, uncertainty, and hard latency in one domain. Language can sit in several places: an online planner, a high-level behavior selector, an explanation head, an offline teacher, or a unified policy. Those are different system bets, not one category called VLMs for driving.

My near-term bet is hybrid. VLMs contribute semantic knowledge, intent understanding, rare-scenario interpretation, annotation, and auxiliary supervision. Metric perception, forecasting, constraints, and high-rate control retain explicit structure. Fluent rationales and open-loop trajectory metrics do not establish reliable closed-loop control.

## Benchmarks should force the model to look

Plausible language can hide weak evidence use. The same shortcut appears in driving, OCR, visual question answering, video, and robotics. A benchmark should therefore act like a causal audit. Corrupt the relevant region or frame. Change the sign, object state, position, timestamp, or instruction. Confirm that irrelevant edits do not matter. Ask whether confidence follows evidence quality and whether an open-loop gain survives execution.

A higher aggregate score can be less valuable than a lower score with better evidence dependence, calibration, and recovery. Deployment cares about the shape of failure, not only its average frequency.

## From decisions to actions
<!-- covers: From imitation to generalization, experience, and scale -->

Robotics completes the move from description to intervention. Once a model acts, it creates its next observation and its errors compound. PaLM-E interleaved visual observations, continuous state, and text. RT-2 and OpenVLA made action tokens part of the language-style interface. Pi-zero and related systems split semantic reasoning from a continuous flow or diffusion action expert. FAST compresses action chunks so an autoregressive model emits fewer tokens.

The data contract then becomes central. Pi-zero-point-five broadens pretraining across robots and web data. Later versions add deployment experience, expert corrections, richer conditioning, and steerable strategies. The boundary is concrete: pretraining broadens the action prior; post-training aligns it to an embodiment, command distribution, safety envelope, and the states produced during deployment.

An action representation defines what the policy can express and how quickly it reacts. Regression is fast but can average multiple valid behaviors. Discrete tokens give exact autoregressive likelihood but add quantization and decoding latency. Diffusion or flow chunks model continuous multimodality but cost iterative generation. Language can carry task semantics. It does not make units, contact, calibration, or control frequency disappear.

## How to read a VLM paper

I use eight questions. What is the output contract? What is one training unit? Where can evidence be lost? Which supervision forces the claimed capability? Which component actually changed? What matched control supports the claim? Does evaluation require the model to use the claimed evidence? And what is the deployment clock?

These questions turn a paper into a decision record. When several components change at once, the work may demonstrate a strong recipe without identifying why it works. That supports adoption more than causal understanding, and the distinction is worth stating plainly.

## A compact reading course
<!-- covers: Layer 0: cross-modal fusion | Layer 1: alignment | Layer 2: generative bridges | Layer 3: assistants and data mixtures | Layer 4: visual evidence and geometry | Layer 5: time and predictive models | Layer 6: decisions and actions -->

Read the field by dependency. Begin with ViLBERT, LXMERT, and UNITER to understand region-word fusion. Move to CLIP and SigLIP for alignment. Use Flamingo, BLIP-2, and PaLI to compare generative bridges. Then read LLaVA, InstructBLIP, MM1, and Eagle 2 for assistants and data mixtures. Follow with MDETR, Kosmos-2, Cambrian-1, Molmo, and SpatialVLM for evidence and geometry. Study LLaVA-OneVision, VideoLLaMA 3, Molmo 2, and V-JEPA for time. Finish with DriveVLM, VLM-AD, RT-2, OpenVLA, Pi-zero, and FAST for decisions and actions.

At every layer, produce something concrete: a stream diagram, loss derivation, ablation table, token budget, temporal counterfactual, or latency and failure budget. Without a deliverable, reading too easily becomes collecting model names.

## A testable thesis

The VLM progression is a sequence of stricter evidence contracts. Task-specific fusion established word-region interaction. Contrastive learning made images addressable through language at web scale. Generative bridges let language models condition on vision. Instruction tuning made that interface conversational. Grounding reconnected words to evidence. Video introduced persistence and intervention. Driving and robotics exposed every shortcut because a plausible answer can become a bad physical decision.

My strongest architectural bet is a shared semantic layer with explicit high-bandwidth routes for geometry, time, generation, and control. A fully unified token stream should replace that hybrid only when, under matched data, pixels, tokens, parameters, compute, and latency, it wins on grounding, metric reasoning, temporal counterfactuals, calibration, and closed-loop recovery. Until then, one model for everything is a research program, not an architectural result.

## Selected references

The spoken version skips the reference list. The complete linked reading list remains in the written post.
