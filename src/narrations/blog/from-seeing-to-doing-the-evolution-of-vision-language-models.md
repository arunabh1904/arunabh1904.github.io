---
postSlug: from-seeing-to-doing-the-evolution-of-vision-language-models
sourceSha256: eb61d3b044197fa426b5b3c69fd0ca1a1f5e4650caa9bd52057db0bac6b3a121
---

# Tracing the VLM Progression

Vision-language models have become incredibly popular over the last few years. Rightfully so. Grounding language in images or video is a huge generalization unlock. It has pushed one model family across classification, detection, alignment, generation, grounding, temporal reasoning, and finally action.

VLMs really started to click with CLIP's image-text alignment, moved through LLaVA's visual instruction tuning, and continued into video understanding and robotics. That arc is useful but incomplete. CLIP did not begin vision-language learning. Earlier models already fused detected regions with words. Its real shift was scale: natural language became an open vocabulary for classification. Visual chat required a second bridge, connecting pretrained vision encoders to language models before instruction tuning turned them into assistants.

A better way to understand this progression is to ask what the representation needs to preserve. We start by connecting detected regions to words, then move through image-text alignment, conditional generation, and instruction tuning. Grounding and video add location, detail, and time. Decision and action add geometry, control timing, and consequences. Each step unlocks a new output, but it can also throw away information the next step needs.

## Image-text alignment

CLIP is the natural place to start because it made image-text alignment work at internet scale. But it was not the first model to connect images and words.

ViLBERT, LXMERT, and UNITER started with regions from an object detector and made those regions interact with words. They could answer visual questions, retrieve images, resolve referring expressions, and reason over detected objects. The recipe was powerful but heavy. The detector decided which regions reached the model, so a missed proposal was gone before language ever saw the image. Every image-text pair then had to run through cross-modal attention.

CLIP removed that expensive fusion step by encoding images and text independently, then comparing their vectors. Retrieval became a nearest-neighbor lookup instead of a fresh transformer pass for every pair. Contrast images against the text found alongside them on the internet, then build a similarity index of sorts in a shared vector space. This sidesteps a fixed class ontology. Adding a concept no longer means collecting labels and retraining a classifier head. Do it across a large enough corpus and batch, and classification becomes retrieval against language.

CLIP trained this recipe on 400 million image-text pairs. Its contrastive loss pulls matched pairs together and pushes mismatched pairs apart. At inference, class names are written as prompts and embedded by the text encoder. The predicted class is the prompt closest to the image.

The loss still couples every example in the batch. Each image competes against every caption, so larger batches demand synchronization across devices and can create false negatives. SigLIP replaces the batch-wide softmax with an independent sigmoid loss for every image-text pair. The change reduces the objective's dependence on batch size and makes distributed training easier.

The sigmoid objective still rewards global alignment. An image and caption can match even when the encoder ignores a small object or its exact location. SigLIP 2 keeps the dual-encoder architecture and adds captioning, self-supervision, multilingual data, curation, and multi-resolution training. Detic and OWL-ViT carry the open vocabulary into detection. They can place a box around a named object, but they still do not produce a multi-sentence answer. That required a language decoder conditioned on visual features.

## Connecting vision to a language generator

A pretrained language model already contained most of the machinery needed for text generation. The remaining problem was to map visual features into its input space without retraining the full stack.

Frozen language-model work learned a visual prefix that the decoder could treat like extra context. Flamingo used a Perceiver Resampler to compress an image or video into 64 visual tokens, then inserted gated cross-attention inside a frozen language model. BLIP trained more of the path. It used contrastive, matching, and caption-generation losses while its CapFilt pipeline generated and filtered better captions for noisy web images. PaLI went further by training the visual encoder and language encoder-decoder together across tasks and languages.

Underneath the model names are three choices: turn the image into a short prefix, expose it through cross-attention, or train vision and language together.

### From Q-Formers to MLP projectors

The connector has to map visual features into the language model's embedding space, and it may also reduce the number of visual tokens. BLIP-2's Q-Former learned both operations together. Thirty-two learned queries cross-attend to the image patches and produce 32 visual tokens regardless of input resolution.

BLIP-2 first teaches the Q-Former to extract visual features that are useful for language. Image-text contrast aligns the query outputs with text. Image-text matching predicts whether a pair is genuine. Image-grounded generation trains text to attend to the visual queries. A second stage projects those outputs into the frozen language model as soft prompt tokens. InstructBLIP also conditions the queries on the user's instruction, so a counting question and a description request can select different evidence.

LLaVA removed the learned query bank. It projects every CLIP patch directly into the language model. The first stage freezes both endpoints and trains the projector on image-caption pairs. The second keeps the vision encoder frozen while tuning the projector and language model on image-instruction-response examples. The connector could remain simple because the language model was allowed to adapt.

LLaVA-1.5 replaced the linear map with a two-layer MLP. Its ablation showed a modest improvement from that change inside the same recipe. The full gain also used higher visual resolution, more academic visual-question-answering data, response-format prompts, and a larger language model. PaliGemma likewise projects SigLIP patches directly into a Gemma decoder and expresses captioning, question answering, detection, and segmentation as text generation.

MM1 puts the connector result in context. Its controlled experiments found larger effects from image resolution and visual-token count than from the choice among several connector designs. MLP projectors became a strong default because they are cheap, preserve patch tokens, and train well with the language model. Learned compression still matters when the visual-token budget is the actual constraint.

### How an image becomes visual tokens

An image can become one fixed patch grid, several high-resolution tiles, a learned set of queries, or a variable sequence whose length grows with the input. More tokens preserve small text, crowded objects, and local layout, but increase attention cost, memory, and latency. Fewer tokens are cheaper, but any detail removed here is gone for the rest of the stack.

PaliGemma uses separate checkpoints at several resolutions. Qwen2-VL keeps the native aspect ratio, packs variable patch sequences, and merges neighboring patches before they reach the language model. Later Qwen and InternVL models extend dynamic resolution to documents and video or learn when the extra compute is worthwhile. A fair model comparison therefore has to match pixels, resizing, visual tokens, compute, and latency.

### Vision moved earlier into training

Once the connector worked, the next gains came from what the model saw and when it saw it. Eagle 2 studies data quality, balance, filtering, and curriculum rather than treating every instruction example as interchangeable. Other systems move multimodal data earlier into language-model pretraining, giving vision and language more chances to adapt to each other. This makes the resulting capability broader, but attribution harder. Better data, more visual tokens, a stronger decoder, and joint training often arrive together.

## Detour: models that also generate images

The systems above take images as input and primarily produce text. Unified multimodal models ask whether the same model can also generate images.

Unified-IO 2, Chameleon, and Emu3 convert several modalities into token sequences and train one autoregressive transformer. This lets a context mix text and images in any order. It also forces the representation to serve conflicting objectives. Recognition benefits from ignoring texture and lighting. Image generation must preserve texture, color, and local appearance. Long discrete image sequences also make generation expensive.

Transfusion shares the transformer but gives each output a suitable loss: next-token prediction for text and diffusion over continuous image patches. Janus separates the visual encoders used for understanding and generation while sharing the autoregressive transformer. These hybrids capture the broader result. A model can share context without forcing every modality to share a tokenizer, representation, or output objective.

## Grounding

Grounding attaches a word or phrase to the part of the image that supports it, usually a box, point, mask, region, or track. It makes an answer inspectable. A model that says the traffic light is red can also show which light it read.

Location-aware captioning is one bridge from generation to grounding. An ordinary caption may say two dogs are playing without specifying which dog is on the left. LocCa also asks the decoder to describe a region and emit its box, or to generate text for a specified region. Each word now carries more information about location.

MDETR aligns phrases with detected objects. GLIP expresses detection categories through language. Kosmos-2 places coordinates inside generated text. Molmo uses pointing data, and Molmo 2 carries those points into video as tracks and timestamps.

The supervision decides what the representation must preserve. A caption can be satisfied by global semantics. A region caption adds correspondence. A box adds extent, a mask adds shape, and a track adds identity through time. Grounding is still not geometry. A point identifies a pixel location, but not depth, camera pose, object orientation, free space, or uncertainty in physical units. SpatialVLM adds spatial relations and measurements, while driving and robotics introduce depth, calibration, maps, pose, and proprioception.

The cleanest test is counterfactual. If changing a light from red to green does not change the answer, the model was not using that evidence. If masking the light causes abstention while changing an irrelevant car leaves the answer alone, the prediction is easier to trust.

## Video-language models: sampling, packing, and time

A video-language model does not receive time for free. It samples frames, converts frames or short tubes into patches, merges or compresses them, packs the remaining tokens into the language-model context, and adds order or time. Every step can remove the brief event needed to answer a question.

LLaVA-OneVision reuses one visual interface across images, image groups, and sampled frames. Qwen2-VL encodes short temporal tubes and packs variable visual sequences. Qwen2.5-VL adds absolute time. VideoLLaMA 3 merges redundant tokens across frames. Molmo 2 exposes temporal evidence through timestamps and tracks.

Packing more frames helps only if the model preserves what changes between them. Uniform sampling spends tokens on repeated content. Aggressive sampling can skip a handoff, collision, or state change. Scene description may rely on a few representative frames. Event localization must preserve when something changed. State tracking must preserve identity. Action-conditioned prediction must produce different futures for different actions. That last contract begins to look like a world model.

## Detour: JEPA predicts in representation space

Video-language models still turn visual evidence into words. JEPA changes the target. A joint-embedding predictive architecture hides part of a video and predicts its representation from visible context. The target can discard unpredictable texture while preserving the state and motion needed to understand what happens next.

V-JEPA 2 first learns this objective from video without action labels. It then trains a smaller action-conditioned predictor on robot trajectories and uses it for image-goal planning. Video teaches how scenes tend to change. Robot data teaches which changes an action can cause.

This predictive representation is a learned bottleneck. Its value comes from discarding variation that does not help prediction. Its risk is discarding local geometry or motion needed by a later controller. V-JEPA 2.1 adds dense prediction and intermediate self-supervision so depth, anticipation, and interaction do not have to survive only through a global target. JEPA gives us a predictive latent state rather than a language answer. It is a parallel pretraining path, not the next step in the VLM lineage.

## From answers to decisions and actions

### Vision-language-action models for driving

Autonomous driving puts nearly every VLM weakness in the same frame: small distant objects, metric geometry, traffic rules, rare hazards, temporal prediction, uncertainty, and a hard latency limit.

Language can be the online planner, a high-level behavior selector, an explanation head, an offline teacher, or part of a unified policy. GPT-Driver and Driving with LLMs turn driving state into a form a language model can reason over. DriveVLM combines scene reasoning with a conventional planner. AsyncDriver separates the slower language path from faster planning. VLM-AD uses VLM supervision without putting the full model in the control loop.

My current read is that the near-term system will remain hybrid. Let the VLM handle semantics, intent, rare-scenario interpretation, annotation, and auxiliary supervision. Keep metric perception, forecasting, constraints, and high-rate control explicit. A fluent rationale or lower open-loop trajectory error does not establish reliable driving. The result has to survive closed-loop evaluation.

### Testing whether the model uses visual evidence

Final-answer accuracy can reward scene priors, document templates, or events visible in one frame. A useful benchmark should corrupt the relevant region, change the sign or object state, remove the decisive frame, and confirm that irrelevant edits do not matter. It should test whether confidence tracks evidence quality and whether an open-loop gain survives execution.

Post-training can reward these behaviors. Visual-RFT uses task-specific signals such as intersection over union, while GRIT interleaves reasoning with region references. These methods can teach a decoder to use surviving visual features. They cannot recover a sign, object, or motion that the encoder discarded. Supplying a crop tests reasoning once evidence is explicit. Editing the pixels tests whether the original answer depended on them.

### Robot actions close the loop

Robotics adds a closed-loop consequence: the output changes the next input. PaLM-E expands the language model's input with visual observations and continuous state. RT-2 puts control on the output side by writing robot actions as tokens. OpenVLA makes that recipe open and inspectable.

Tokenizing the action then becomes a model decision. Pi zero keeps a semantic VLM and gives continuous actions to a flow-based expert. FAST compresses action chunks in the frequency domain so the language decoder emits fewer tokens. DexVLA uses a diffusion expert, while GR00T N1 separates semantic reasoning from continuous action generation. A shared backbone does not require words and motor commands to use the same distribution.

### From imitation to generalization, experience, and scale

Later VLA systems broaden the training loop. Pi zero point five mixes web data, robots, detections, semantic subtasks, and low-level actions. Pi star zero point six adds autonomous rollouts and expert corrections. Pi zero point seven makes strategy and subgoals steerable. Xiaomi-Robotics-1 collects large-scale human manipulation through a cheaper interface and later aligns it with robot commands.

This is where pretraining and post-training separate. Pretraining gives the policy a broad starting point. Post-training adapts it to one robot, command distribution, safety envelope, and the states created during deployment. The action interface decides what motions the policy can express and how quickly it reacts. Regression is fast but may average valid strategies. Discrete tokens provide an autoregressive likelihood but add quantization and decoding latency. Diffusion and flow model continuous multimodality but need a specialist serving path.

A caption has no control frequency. An action does. Language is a good interface for the task, but it does not make units, embodiment, contact, or latency disappear.

## Recap: what the representation must preserve

The history becomes easier to read when each paper is reduced to four questions. What does the model produce? Where can visual evidence be lost? Which supervision forces the new capability? Does the evaluation require that evidence?

Region-based models connected words to detected objects. CLIP made images searchable through language at web scale. Generative bridges gave those features to a language model, and instruction tuning turned the result into an assistant. Grounding tied words back to pixels. Video added time. Driving and robotics made the remaining shortcuts expensive because a plausible sentence could now produce a bad physical decision.

My strongest bet is a shared semantic model with separate high-bandwidth paths for geometry, time, image generation, and control. I would replace that hybrid with one token stream only when it wins under matched data, pixels, tokens, parameters, compute, and latency. The win also has to hold on fine grounding, metric spatial reasoning, temporal counterfactuals, calibration, and closed-loop recovery.

For further reading, the robotics side continues in two posts. Pre-Training for Robotics looks at how multimodal and robot data shape a base policy. Post-Training for Robotics looks at how deployment feedback and failures refine that policy.
