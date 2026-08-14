---
title: Building Local Blog Audio with Qwen3-TTS
date: '2026-08-13T23:30:00.000Z'
section: blog
postSlug: building-local-blog-audio-with-qwen3-tts
legacyPath: /blog/2026/08/13/building-local-blog-audio-with-qwen3-tts.html
tags:
  - AI
  - Software Engineering
  - Apple Silicon
  - Audio
summary: >-
  How this site turns Blog Markdown into static MP3 narration with Qwen3-TTS,
  extracts prose rather than visual or code artifacts, and serves playback
  without a runtime model or API dependency.
---

# Building Local Blog Audio with Qwen3-TTS

A read-aloud control looks like a browser feature. The engineering problem is really an export pipeline: choose a narrator worth listening to, decide what part of a technical post should be spoken, generate durable assets, and attach them without turning a static site into an inference service.

I wanted the result to work on a commute. That ruled out a one-off desktop script and ruled out delegating the experience to whatever speech voice happens to be installed in a visitor's browser. The site now generates MP3s locally, commits them as ordinary static assets, and gives every Blog post a native player with play, pause, seeking, and speed controls. The browser only downloads audio. It never loads a model, sends post text to an API, or waits for synthesis.

The interesting work is not the button. It is the boundary between an authored document and a coherent narration.

## The architecture: export once, serve statically

The source of truth remains the Markdown or MDX file under `src/content/posts`. A local exporter finds entries whose frontmatter says `section: blog`, extracts narration-safe text, synthesizes audio, and writes one MP3 per `postSlug` under `public/assets/audio/blog/`. Astro then maps a Blog route to the matching asset path.

```text
Blog Markdown or MDX
        |
        v
narration-safe text extraction
        |
        v
sentence-bounded Qwen3-TTS chunks
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

## Why Qwen3-TTS

The voice is generated with the 8-bit MLX conversion of [`Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://github.com/QwenLM/Qwen3-TTS). Qwen's VoiceDesign model takes a natural-language description of the narrator rather than a fixed speaker ID. That matters here because the target is not a named person's voice; it is a set of audible constraints: mature low-mid register, warm and resonant tone, clean close-miked delivery, measured pacing, restrained emphasis, and no breathiness or theatrical performance.

The `1.7B` model is heavier than a browser speech engine or a compact local TTS model, but its prosody is the point. A technical post needs sentence-level emphasis, pauses that land at argument boundaries, and a voice that does not turn every paragraph into the same synthetic cadence. The model runs once for the export, not once per listener, so export cost is a better trade than a permanently weak reading experience.

The model emits 12 Hz acoustic codes and the local [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) runtime decodes them to waveform audio. I encode the final waveform as mono `24 kHz` MP3 at `64 kbit/s`: enough bandwidth for spoken voice while keeping the static bundle reasonable. The original WAV chunks are temporary; only the MP3s are committed.

## Long-form synthesis needs a scheduler

One request per post is not reliable. Long technical posts exceed a comfortable generation window, and a failure near the end should not discard forty minutes of work. The exporter therefore breaks narration into sentence-bounded chunks of at most 900 characters. Before concatenation, it removes model-generated leading and trailing dead air while retaining only a 50 ms boundary release. A final waveform pass collapses any residual silence longer than 0.8 seconds to a 90 ms transition. The join is effectively gapless; ordinary rhetorical pauses remain natural.

The exporter loads the model once and keeps it resident for the entire corpus. Each chunk then uses the same direct VoiceDesign path as the approved listening sample. That choice is deliberate: Qwen's continuous batch API is useful for short parallel requests, but for long designed-voice segments it makes progress opaque and can hold a group behind its slowest member. The direct path gives predictable chunk boundaries and a truthful per-post progress signal.

```python
for result in model.generate_voice_design(
    text=chunk,
    instruct=VOICE_INSTRUCTION,
    temperature=0.38,
    top_p=0.86,
    language="English",
):
    audio_parts.append(result.audio)
```

The script also gives each direct call a text-length-derived acoustic-token ceiling: at least 768 tokens, and 1.5 tokens per source character above that. Qwen's generous default is useful for open-ended generation, but a narration segment should not be able to occupy the exporter for several minutes after it has already said what the text contains. The cap is included in the manifest digest, so changing it deliberately invalidates the affected audio.

The script still exposes `--batch-size` for short segments and future runtime improvements, but its default is one chunk. The real export optimizations are one resident model, sentence-safe work units, bounded generation, resumable manifest checks, atomic post writes, and no wasted inference on captions or code. Those improvements reduce avoidable work without changing the narrator that the listener selected.

After a post completes, the exporter writes temporary WAV chunks, concatenates them with `ffmpeg`, and atomically replaces the target MP3. The manifest is updated only after the post succeeds. An interrupted run can therefore resume: posts whose extracted narration, model ID, voice instruction, chunking profile, generation limit, and encoder settings still match their manifest digest are skipped.

## Validation belongs before the voice model

Synthesis is expensive enough that validation should reject bad text before inference begins. The export checks every Blog post for a current digest and verifies that no audio file is missing. A separate extraction pass asserts that the final text contains none of the patterns the listener should never hear: image Markdown, code fences, the References heading, figure-caption boilerplate, or raw LaTex commands.

That is only the mechanical part. I also read the Blog corpus as an audio corpus before generating it. A page can be visually coherent while sounding disjointed because a chart carries the transition or a short caption supplies the missing premise. The practical test is simple: headings, paragraphs, and lists must still form a route through the argument after images, equations, code, and citations disappear. Posts that fail that test need prose edits before they need a better model.

The final checks happen at three levels:

1. The exporter verifies manifest coverage for every Blog post.
2. The site runs its normal static build and CI checks.
3. A browser test confirms that the control appears on Blog routes, remains absent from non-Blog routes, and can play, pause, seek, and change speed without console errors.

## Static audio has real trade-offs

Committing generated audio makes the site portable and private at read time, but it also creates a new asset lifecycle. A prose edit invalidates the corresponding digest and needs an audio rerun. A voice change invalidates the whole corpus. Large posts create large MP3s, and a static repository must absorb that cost.

Those costs are acceptable here because they are explicit. There is no hidden per-listen API charge, no runtime model service to keep alive, and no browser-dependent voice quality. The price is a deliberate local export step whenever the written corpus changes.

The deeper lesson is that spoken technical writing is a different rendering target. A good narration system does not merely make text audible. It preserves the post's reasoning while refusing to read the parts of a document that only work when seen.

## References

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- [Astro content collections](https://docs.astro.build/en/guides/content-collections/)
