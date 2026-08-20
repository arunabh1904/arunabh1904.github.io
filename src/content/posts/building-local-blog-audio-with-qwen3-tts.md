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
  How I built, repaired, and deployed one consistent local narrator for every
  Blog post.
---

# Building Local Blog Audio with Qwen3-TTS

I wanted the Blog to work on a commute. The obvious feature was a read-aloud button, but the real project became a small publishing system: derive a spoken version of each post, give every chunk the same narrator, export durable audio, and prove that the file on GitHub Pages matches the current article.

The word *prove* matters because I shipped this more than once before I had solved it. One version changed speaker identity between chunks. The next used a fixed voice reference and repeated part of that reference before nearly every chunk. A decoder change looked correct in code and passed manifest checks, yet the deployed MP3 still failed the only test a listener cares about: hearing the article without synthetic setup text interrupting it.

The final design removes spoken references from generation entirely. It uses Qwen3-TTS CustomVoice with one built-in speaker, one fixed delivery instruction, and one shared sampling seed across the corpus. The rest of the pipeline treats audio as a compiled form of the Blog: source prose goes in; a versioned MP3 and manifest record come out; a release is complete only after the live asset matches that output.

> Audio is not an attachment to a post. It is another rendering of the post, with its own compiler, artifact lifecycle, and production tests.

## The site exports audio instead of serving a model

The source of truth remains the Markdown or MDX file under `src/content/posts`. A local exporter finds entries whose frontmatter says `section: blog`, derives narration-safe text, synthesizes one MP3 per `postSlug`, and records the source and narrator profile in `public/assets/audio/manifest.json`. Astro maps the Blog route to that static asset.

```text
Blog Markdown or MDX
        |
        v
narration-safe text
        |
        v
sentence-bounded chunks
        |
        v
Qwen3-TTS CustomVoice: Ryan + one fixed delivery profile
        |
        v
24 kHz mono MP3 + source/profile digest
        |
        v
pull request -> GitHub Pages -> live asset verification
```

This split keeps inference off the website. The Apple Silicon machine pays the model-loading and synthesis cost once during export. GitHub Pages serves the resulting MP3 like an image or stylesheet. A reader does not send article text to an API, wait for generation, or inherit whichever speech engine happens to be installed in a browser.

The page component therefore stays small. It wraps a native `<audio>` element, exposes play, pause, seek, and playback speed, and receives an asset URL only for Blog posts. Paper notes and other sections do not get an empty or misleading player.

## A Blog post needs a separate narration view

The first exporter treated Markdown as text with formatting characters removed. That produced predictably bad audio: image descriptions were repeated, figure-source lines interrupted arguments, raw equations became noise, and tables lost the column relationships that made their values useful.

The exporter now assigns each source form a narration contract:

| Source material | Audio treatment | Reason |
| --- | --- | --- |
| Headings and paragraphs | Speak | They carry the argument and keep the listener oriented. |
| Lists | Speak | They often contain the actual criteria or sequence. |
| Tables | Rewrite as labelled rows | Values matter, but column structure must become punctuation. |
| Code and MDX components | Skip | They require inspection rather than linear listening. |
| Images and captions | Skip | The surrounding prose should explain the visual claim. |
| LaTeX | Skip | Symbolic notation needs a prose interpretation. |
| References | Skip | A bibliography is useful for retrieval, not as a closing monologue. |

This contract exposed an editorial constraint. If a transition exists only in a chart, or if an equation carries a conclusion that the prose never states, the page may look complete while the narration sounds disconnected. The fix belongs in the article before it belongs in the speech model. Audio became a second structural review of the Blog.

The extractor preserves headings, removes link destinations while keeping labels, turns table rows into phrases such as `Runtime: MLX; TTFT: 684 milliseconds`, and omits the caption immediately following an image. Only after that transformation does it split the text for synthesis.

## Speaker identity and delivery are different controls

The first Qwen export used [`Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://github.com/QwenLM/Qwen3-TTS). I described the voice I wanted—warm, measured, close-miked, restrained—and sent the same description with every chunk. That preserved a broad style, but not one speaker. Each chunk was another request to design a voice, so timbre and register drifted over a long article.

I then moved to the Base model's in-context-learning path. Every chunk received the same synthetic reference WAV and matching transcript. The reference supplied a concrete identity, which solved the problem VoiceDesign could not: all chunks had an acoustic speaker to imitate rather than a prose description to reinterpret.

It also created a worse failure. The normal ICL decoder reconstructed reference and generated acoustic codes together, then estimated where the reference waveform ended. That cut was imperfect, so the reference sentence—especially “the delivery is warm”—survived at chunk boundaries.

I changed the exporter to the runtime's generated-only streaming decoder and regenerated the corpus. The configuration looked right, and the freshness check proved that the MP3s matched the source and declared profile. Production listening proved something different: the spoken reference was still audible. A manifest can establish provenance, but it cannot establish what a waveform says.

The sequence changed the design criterion:

| Generation strategy | What it solved | Why it was rejected or retained |
| --- | --- | --- |
| VoiceDesign prompt per chunk | Found a useful style | Rejected because speaker identity drifted. |
| Base model with one ICL reference | Anchored speaker identity | Rejected because a spoken reference could enter the output. |
| ICL with generated-only decoding | Removed the obvious decoder splice in code | Rejected after the deployed audio still repeated the reference phrase. |
| CustomVoice with built-in Ryan | Fixes speaker identity without reference audio | Retained as the production narrator. |

The durable fix was to remove the failure source, not keep trimming around it.

## One narrator profile drives every chunk

The production exporter now uses `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` with the built-in English speaker `Ryan`. CustomVoice separates identity from delivery: `Ryan` fixes who is speaking, while one non-spoken instruction fixes how the narration should sound.

```python
VOICE = "Ryan"
VOICE_INSTRUCT = (
    "Use a calm, warm, measured technical-narration delivery. Keep a natural "
    "low-mid register, close-miked clarity, restrained emphasis, and steady pacing."
)

result = model.generate(
    text=chunk,
    voice=VOICE,
    instruct=VOICE_INSTRUCT,
    lang_code="english",
    temperature=0.05,
    top_p=0.9,
    stream=False,
)
```

There is no `ref_audio` and no `ref_text`. Nothing in the conditioning input contains a sentence that can be replayed. Before every chunk, the exporter resets the sampler to the same narrator seed. The words still determine normal sentence rhythm, but speaker selection, delivery instruction, sampling profile, model revision, and encoding settings remain corpus-wide invariants.

Those invariants are part of each manifest digest. A different speaker, instruction, seed, model, chunking rule, or source revision makes the existing MP3 stale. The corpus check also asserts that all Blog entries use the same fixed narrator profile and contain no reference or ICL fields. Cohesion is therefore a release property rather than a convention someone must remember.

## Long posts need bounded, resumable synthesis

A whole technical post is too large for one reliable generation call. The exporter gives each sentence its own synthesis unit and splits a sentence again only when it exceeds 360 characters. I tested larger 720- and 960-character chunks because fewer joins looked attractive, but their failure cost and generation time made a full corpus rerun impractical.

Each chunk also gets a text-derived acoustic-token ceiling: at least 128 tokens, then 1.05 tokens per source character. This fixed a separate failure from the early exporter. A large fixed token floor allowed short headings to continue generating after their text ended, producing long silence or non-speech noise. The bounded ceiling makes a bad stop finite.

The exporter loads the model once, generates chunks sequentially, trims only boundary dead air, and writes temporary WAV files. It inserts 180 ms of silence after each complete sentence, then `ffmpeg` concatenates the files and raises the final tempo to `1.10x`. The same pass converts the result to a mono `24 kHz` MP3 at `64 kbit/s` and collapses residual silence longer than 0.6 seconds to a 60 ms transition. Sentence timing is therefore explicit instead of depending on whichever pause the model happened to leave at a chunk edge.

An MP3 replaces the prior asset atomically only after every chunk for that post succeeds. The manifest entry is written after the replacement. If a corpus run stops, completed posts remain valid and the next run synthesizes only missing or stale entries.

## Deployment starts from the final source revision

The most expensive mistake was generating correct audio from the wrong checkout. A working tree contained Blog edits that had not reached `main`, so its MP3s could be internally fresh while still disagreeing with the deployed article. The repair now begins from a clean worktree at current `origin/main`.

I use the longest Perception post as the canary because its many sections and chunk joins make identity drift, repeated prefaces, runaway tails, and stale source easy to expose. The release sequence is:

1. Generate Perception from the final source revision and listen across several chunk boundaries.
2. Verify its manifest digest and narrator profile.
3. Export the remaining Blog corpus without reloading the model for each post.
4. Fetch `origin/main` again and regenerate anything made stale by concurrent prose changes.
5. Run the corpus-wide audio check, `git diff --check`, and the site's full CI command.
6. Commit only the exporter, manifest, changed Blog source, and regenerated audio assets.
7. Merge through a pull request after GitHub checks pass, then wait for the Pages workflow.
8. Download the live Perception MP3 with a cache-busting query and compare its SHA-256 hash with the committed file.

That last comparison closes a gap the earlier repair left open. An HTTP 200 proves that an MP3 exists. Matching hashes prove that production serves the MP3 that passed local validation. The [reference-free corpus release](https://github.com/arunabh1904/arunabh1904.github.io/pull/204) used that stronger check after replacing all 17 Blog assets.

## The Blog and the audio now share one argument

Narration quality did not come from adding more dramatic instructions. It came from preserving the same reasoning path when visual-only material disappears. The exporter adds a small pause around real pivots such as `However`, `Instead`, `The result is`, or `In practice`, but it does not invent emphasis, rewrite claims, or turn every paragraph into a performance.

That restraint keeps the two versions aligned. The page remains the complete technical artifact, with equations, code, figures, links, and references. The MP3 carries the same sequence of claims in a form that survives linear listening. When an article cannot do that, I revise the prose and regenerate its audio rather than teaching the voice model to compensate for a missing transition.

Static audio still has a cost. A prose edit invalidates one post. A narrator-profile change invalidates the corpus. MP3s increase repository size, and a full local export takes time. I accept those costs because they remain explicit: there is no per-listen API bill, no runtime service, no visitor text upload, and no browser-dependent speaker.

The project started as a player. It ended as a deployment rule: compile narration from the final source, hold one narrator profile constant, and verify the exact artifact that listeners receive.

## References

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- [Astro content collections](https://docs.astro.build/en/guides/content-collections/)
- [Reference-free Blog audio release](https://github.com/arunabh1904/arunabh1904.github.io/pull/204)
