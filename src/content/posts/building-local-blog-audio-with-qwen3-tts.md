---
title: Building Local Blog Audio with Qwen3-TTS
date: '2026-08-13T23:30:00.000Z'
section: blog
blogGroup: projects
postSlug: building-local-blog-audio-with-qwen3-tts
legacyPath: /blog/2026/08/13/building-local-blog-audio-with-qwen3-tts.html
tags:
  - AI
  - Software Engineering
  - Apple Silicon
  - Audio
summary: >-
  A local Qwen3-TTS pipeline that turns Blog Markdown into consistent, static
  narration.
---

# Building Local Blog Audio with Qwen3-TTS

A read-aloud control looks like a browser feature. The engineering problem is really an export pipeline: choose a narrator worth listening to, decide what part of a technical post should be spoken, generate durable assets, and attach them without turning a static site into an inference service.

I wanted the result to work on a commute. That ruled out a one-off desktop script and ruled out delegating the experience to whatever speech voice happens to be installed in a visitor's browser. The site now generates MP3s locally, commits them as ordinary static assets, and gives every Blog post a native player with play, pause, seeking, and speed controls. The browser only downloads audio. It never loads a model, sends post text to an API, or waits for synthesis.

> **Deep insight.** The difficult part is not adding a player. It is preserving the boundary between an authored document and a coherent narration, so synthesis is reproducible and serving stays static.

## The architecture: export once, serve statically

The source of truth remains the Markdown or MDX file under `src/content/posts`. A local exporter finds entries whose frontmatter says `section: blog`, extracts narration-safe text, synthesizes audio, and writes one MP3 per `postSlug` under `public/assets/audio/blog/`. Astro then maps a Blog route to the matching asset path.

```text
Blog Markdown or MDX
        |
        v
narration-safe text extraction
        |
        v
fixed synthetic voice reference + Qwen3-TTS ICL chunks
        |
        v
24 kHz mono MP3 plus manifest digest
        |
        v
Astro ReadAloud component and GitHub Pages
```

This separation matters. A static-site build should be deterministic and cheap; model loading is neither. Synthesis runs explicitly on the Apple Silicon machine where the model is available. The production build only copies committed MP3 files, so GitHub Pages serves them like images, CSS, or JavaScript.

The page component is deliberately small. It wraps a native `<audio>` element, adds a visible play or pause button, exposes a seek range and playback-rate control, and leaves browser-native controls available as a fallback. The route only passes an audio URL when `section === 'blog'`, so paper notes and other site sections do not acquire a misleading empty player.

## Narration is not a Markdown render

The first version treated Markdown as a string-cleaning problem. That is how a listener ends up hearing image alt text, a figure-source paragraph, a raw equation, or a table with its pipes removed. None of those failures is a model problem. They are extraction failures.

The exporter now treats a post as several kinds of material with different narration contracts:

| Source material | Spoken? | Reason |
| --- | --- | --- |
| Headings and paragraphs | Yes | They carry the argument and orient the listener. |
| Bulleted and numbered lists | Yes | They often contain the decision criteria or setup. |
| Markdown tables | Yes, as labelled rows | The values are evidence, but punctuation must make the columns intelligible aloud. |
| Fenced code and MDX components | No | Source code and live widgets are useful on screen, not in a long-form narration. |
| Image alt text and figure captions | No | The surrounding prose already explains the figure; captions become awkward duplication in audio. |
| Display and inline LaTex | No | Raw symbolic notation is not reliable speech. The prose around an equation must carry the listener's model. |
| References bibliography | No | Links and citation lists are retrieval tools, not a useful closing monologue. |

This is an editorial policy, not a claim that code, figures, or equations are unimportant. It says that the web page and the audio file have different jobs. If a technical idea only exists inside a figure caption or an equation, the post needs a prose explanation for readers as well as listeners.

The extractor preserves headings, strips Markdown links down to their labels, turns table rows into phrases such as `Runtime: MLX; TTFT: 684 ms`, and splits only on sentence boundaries. It also explicitly skips the caption immediately following an image rather than relying on a fragile global regular expression. These choices keep the voice from changing topic just because the source contains presentation markup.

## A single narrator needs a persistent reference

The first export used the 8-bit MLX conversion of [`Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://github.com/QwenLM/Qwen3-TTS) for every chunk. VoiceDesign accepts a natural-language description of a narrator, which was useful for finding the target sound: a mature low-mid register, warm and resonant tone, clean close-miked delivery, measured pacing, restrained emphasis, and no breathiness or theatrical performance. It was the wrong production interface for a long post. Each chunk was a new design request, so the description could hold the style roughly steady without giving the model a speaker identity to preserve. The result was audible identity drift from chunk to chunk.

The exporter now uses the 8-bit MLX conversion of `Qwen3-TTS-12Hz-1.7B-Base` in its in-context-learning mode. It conditions every chunk on the same short synthetic reference WAV and its exact transcript. The reference was designed locally; it is not a recording of, or an attempt to imitate, a real person. That one audio-and-text pair gives the base model a concrete timbre, cadence, and speaker identity to carry through the entire post. It is an identity anchor, not a persistent continuation cache: each chunk still starts a fresh acoustic generation. The exporter hashes both the reference file and transcript into every manifest digest, so changing either intentionally invalidates the corpus.

The `1.7B` model is heavier than a browser speech engine or a compact local TTS model, but its prosody is the point. A technical post needs sentence-level emphasis, pauses that land at argument boundaries, and a voice that does not turn every paragraph into the same synthetic cadence. The model runs once for the export, not once per listener, so export cost is a better trade than a permanently weak reading experience. It emits 12 Hz acoustic codes and the local [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) runtime decodes them to waveform audio. I encode the final waveform as mono `24 kHz` MP3 at `64 kbit/s`: enough bandwidth for spoken voice while keeping the static bundle reasonable. The original WAV chunks are temporary; only the MP3s are committed.

## Long-form synthesis needs a scheduler

One request per post is not reliable. Long technical posts exceed a comfortable generation window, and a failure near the end should not discard forty minutes of work. The exporter therefore breaks narration into sentence-bounded chunks of at most 360 characters. I tested 720- and 960-character chunks to reduce joins, but the local runtime made those work units slow enough to make a corpus rerun impractical. Pure greedy decoding was deterministic, but it could run long on an awkward passage instead of stopping naturally. The production compromise is low-temperature sampling (`0.05`, `top_p=0.9`) with a stable seed derived from the post slug and chunk index: it is repeatable across reruns without losing normal stopping behavior or prosody. Before concatenation, the exporter removes model-generated leading and trailing dead air while retaining only a 50 ms boundary release. A final waveform pass collapses any residual silence longer than 0.8 seconds to a 90 ms transition. The join is effectively gapless; ordinary rhetorical pauses remain natural.

The exporter loads the model once and keeps it resident for the entire corpus. Each direct synthesis call receives the same reference and transcript, then receives its deterministic seed. That choice is deliberate: Qwen's continuous batch API is useful for short parallel requests, but the default path keeps one chunk at a time so progress remains truthful and every join is easy to inspect.

```python
for result in model.generate(
    text=chunk,
    ref_audio="clean-warm-synthetic-reference.wav",
    ref_text=REFERENCE_TEXT,
    temperature=0.05,
    top_p=0.9,
    lang_code="english",
):
    audio_parts.append(result.audio)
```

The seed is part of the export profile and manifest. This matters operationally: changing the reference, sampling settings, punctuation shaping, chunk boundary, or source prose makes the old MP3 stale by definition. The final audio run must happen after the last editorial change, not alongside it.

The script also gives each direct call a text-length-derived acoustic-token ceiling: at least 128 tokens and 1.05 tokens per source character above that. This detail was a reliability fix, not a cosmetic optimization. An earlier version imposed a 768-token floor; at roughly 12.5 acoustic tokens per second, a short heading or paragraph could keep the decoder running after the text had finished. The audible result was occasional non-speech noise a few items into a post. Shorter source chunks and a ceiling that follows their text prevent that runaway tail. The cap is included in the manifest digest, so changing it deliberately invalidates the affected audio.

The script still exposes `--batch-size` for short segments and future runtime improvements, but its default is one chunk. The real export optimizations are one resident model, sentence-safe work units, bounded generation, resumable manifest checks, atomic post writes, and no wasted inference on captions or code. The fixed reference removes a different class of failure: it prevents each chunk from deciding who is speaking again.

## Narration is a second editorial pass

Stable timbre is necessary, but it is not the same as expressive delivery. The model has no reliable SSML-style emphasis control in this export path, so emphasis has to be carried by the narration text rather than by a different voice prompt for each clip. The exporter uses low-temperature sampling with a stable seed per post and chunk, rather than a pure greedy path that can make long passages run to an unhelpful acoustic limit. That makes every export reproducible while preserving natural stopping and prosody. It also keeps the work separate from the visible article: the renderer preserves actual claims and vocabulary, then adds only a few restrained punctuation cues at real argument turns.

The useful places to shape delivery are contrasts, consequences, and transitions. A sentence beginning with `But`, `However`, or `Instead` receives a brief beat after the pivot. Statements such as `The point is`, `The result is`, and `The trade-off is` get a colon before the payoff. `That means`, `In practice`, and `Put differently` get the same treatment. These are not theatrical directions; they mark the places where the written argument itself changes pressure.

The rule is deliberately narrow. Do not add capitals, filler, fake intensity, or a new claim merely to make a sentence sound dramatic. Do not add a cue to every paragraph. The narration should emphasize the decision that the prose has already earned, then return to a conversational baseline. When a post needs a stronger treatment than these mechanical cues can provide, write an audio-only narration script and keep its factual content aligned with the visible post.

After a post completes, the exporter writes temporary WAV chunks, concatenates them with `ffmpeg`, applies the residual-silence limit, and atomically replaces the target MP3. That final pass solves a separate failure: an otherwise valid chunk can decode a long quiet tail. Those were the conspicuous multi-second gaps in the middle of early files. The pass keeps normal pauses but reduces any quiet region longer than 0.8 seconds to 90 ms. The manifest is updated only after the post succeeds, so an interrupted run can resume only when its narration, model ID, reference identity, chunking profile, sampling seed version, generation limit, and encoder settings still match.

## Validation belongs before the voice model

Synthesis is expensive enough that validation should reject bad text before inference begins. The export checks every Blog post for a current digest and verifies that no audio file is missing. A separate extraction pass asserts that the final text contains none of the patterns the listener should never hear: image Markdown, code fences, the References heading, figure-caption boilerplate, or raw LaTex commands. Before release, the same freshness check runs after all prose edits; a changed post without a newly matching digest is a release blocker, not a minor follow-up.

That is only the mechanical part. I also read the Blog corpus as an audio corpus before generating it. A page can be visually coherent while sounding disjointed because a chart carries the transition or a short caption supplies the missing premise. The practical test is simple: headings, paragraphs, and lists must still form a route through the argument after images, equations, code, and citations disappear. Posts that fail that test need prose edits before they need a better model.

The final checks happen at three levels:

1. The exporter verifies manifest coverage for every Blog post.
2. The site runs its normal static build and CI checks.
3. Listening checks sample the beginning, middle, and end of each newly exported priority post. The test is specifically for non-speech tails, long joins, and speaker drift.
4. A browser test confirms that the control appears on Blog routes, remains absent from non-Blog routes, and can play, pause, seek, and change speed without console errors.

## Static audio has real trade-offs

Committing generated audio makes the site portable and private at read time, but it also creates a new asset lifecycle. A prose edit invalidates the corresponding digest and needs an audio rerun. A voice change invalidates the whole corpus. Large posts create large MP3s, and a static repository must absorb that cost.

Those costs are acceptable here because they are explicit. There is no hidden per-listen API charge, no runtime model service to keep alive, and no browser-dependent voice quality. The price is a deliberate local export step whenever the written corpus changes.

The deeper lesson is that spoken technical writing is a different rendering target. A good narration system does not merely make text audible. It preserves the post's reasoning while refusing to read the parts of a document that only work when seen.

## References

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- [Astro content collections](https://docs.astro.build/en/guides/content-collections/)
