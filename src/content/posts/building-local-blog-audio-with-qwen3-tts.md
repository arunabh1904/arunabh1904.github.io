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
  What it took to turn Qwen3-TTS into a source-pinned, one-voice, under-30-minute
  audio release for every Blog post.
---

# Building Local Blog Audio with Qwen3-TTS

Local Blog audio means that every post has a static MP3 generated on my Mac and served from GitHub Pages. That sounds like a text-to-speech feature. It became a publishing compiler: one system has to decide what the article says aloud, hold a voice steady for thousands of words, reject incomplete output, and prove that the file a listener receives was built from the prose on the page.

I learned that distinction by shipping the wrong thing several times. One release changed voices between chunks. Another repeated part of its voice-reference sentence before nearly every chunk. A later version removed that preface but sounded too excited, flipped register around thirty-five seconds, skipped near forty-two seconds, and pronounced *cyclist* badly. The manifest was fresh. The waveform was not good.

The current system uses Qwen3-TTS Base with one committed synthetic voice anchor: a warm Indian English woman with a low-mid pitch, restrained inflection, controlled energy, and a slightly sombre technical-presenter delivery. It generates one authored heading section at a time, splitting only unusually long sections, decodes generated acoustic codes only, keeps natural timing inside sentences, and applies a small pause only between sections. Eighteen Blog posts now share that profile. Together they contain almost 200 minutes of audio, and the longest file is 18 minutes and 4 seconds.

> The central lesson was uncomfortable: provenance checks can prove which inputs produced a file, but they cannot prove that the file sounds right. Audio needs both compiler invariants and perceptual tests.

## What “local” means

The website never runs a speech model. The source of truth remains the Markdown or MDX file under `src/content/posts`. A laptop-local exporter derives narration-safe text, synthesizes a mono MP3, and records its source and narrator profile in `public/assets/audio/manifest.json`. Astro only renders a player that points at the static asset.

The release path is now explicit. The writing lands first. A clean worktree then reads that exact `origin/main`, compiles the narration, and ships the MP3 in a second pull request. GitHub Pages deploys the audio, and the release ends only when the live route returns successfully and the cache-busted MP3 has the same SHA-256 hash as the committed file.

<div class="compact-flow-diagram"><a href="/assets/images/blog-audio-release-pipeline.svg"><img src="/assets/images/blog-audio-release-pipeline.svg" alt="Compact Blog audio release pipeline from source pull request to clean main, narration compilation with the fixed voice, a focused audio pull request, and GitHub Pages route and hash verification"></a></div>
_The prose PR fixes the source revision. The audio PR compiles from that revision. Production verification closes the loop by checking the exact file listeners receive._

This architecture keeps inference cost and latency away from readers. There is no per-listen API request, visitor text upload, runtime model service, or browser-dependent narrator. My machine pays the synthesis cost once. GitHub Pages serves the result like an image.

The page component stays deliberately boring. It wraps a native `<audio>` element and exposes play, pause, seek, and playback speed. Only Blog posts receive an audio asset. Paper notes, Revision Notes, and Code pages do not render an empty player.

## The article is still the source of truth

The first extractor treated Markdown as text with formatting removed. That was not enough. Image descriptions repeated. Figure-source lines interrupted arguments. Equations became noise. Tables turned into long sequences of cells without their visual relationships. References produced a closing recital of paper titles and URLs.

The compiler now gives each source form a spoken contract. Headings, paragraphs, and useful lists carry the argument, so they are spoken. Code, raw MDX, equations, reference lists, image markup, captions, and table rows are skipped. Links retain their visible labels but lose their destinations. A table's governing comparison must appear in the prose around it.

| Source material | Audio treatment | Obligation on the article |
| --- | --- | --- |
| Headings and paragraphs | Speak | Keep the reasoning path linear and explicit. |
| Lists | Speak when they contain criteria or sequence | Introduce what the listener should retain. |
| Tables | Skip rows | State the governing comparison in prose. |
| Code, MDX, and LaTeX | Skip | Explain the result or decision in words. |
| Images and captions | Skip | Verbalize the mechanism before and interpret it after. |
| References | Skip | Keep citations on the page without reading the bibliography aloud. |

This rule changed how I write. If an argument only works while the reader can inspect a diagram, the prose is incomplete. If a table contains the conclusion but the paragraph after it merely says “as shown above,” the narration has nowhere to go. Audio became a second structural review of every post.

The fix belongs in the article first. I do not ask the voice model to invent a bridge that the prose failed to write.

## The failures changed the architecture

This project did not converge through one parameter sweep. Each failure invalidated an assumption in the design.

The first assumption was that a good voice description creates a stable speaker. I used Qwen3-TTS VoiceDesign and repeated the same request—warm, measured, close-miked, restrained—for every chunk. The broad style survived. The person did not. Each chunk redesigned the voice, so timbre, pitch, register, and energy moved across an article.

The second assumption was that a shared reference waveform solves identity. Qwen3-TTS Base can condition on a reference WAV and its transcript. Giving every chunk the same synthetic sample did anchor one speaker. It also created the most obvious bug in the entire feature: “the delivery is warm” appeared before chunk after chunk.

The normal non-streaming ICL path decoded the reference acoustic codes together with the generated codes, then estimated where to cut the reconstructed waveform. The estimate left part of the reference sentence behind. Trimming harder attacked the symptom and risked deleting the first word of the article.

The third assumption was that generated-only decoding would finish the job. I switched to a streaming path that decodes only newly generated acoustic codes. The code path was right, yet an exported corpus still contained the spoken setup. That taught me to stop treating a manifest and an implementation inspection as an audio test. I needed to transcribe and listen to the artifact itself.

I then removed the reference and used the built-in Ryan CustomVoice speaker. That eliminated reference leakage, but it did not give me the voice I wanted. More importantly, sentence-sized generation still reset prosody constantly. A voice could begin calm, become bright and excited, drop in pitch at the next join, and change timing again thirty seconds later. The identity label remained `Ryan`; the listening experience still sounded stitched together.

Explicit pauses created another trap. Inserting silence after every sentence made technical prose slow and mechanical. A residual-silence filter then tried to compress long gaps across the concatenated file. That filter could eat a legitimate pause or word boundary and produce an audible skip. The skip near forty-two seconds made one rule clear even though the exact boundary failure was difficult to attribute afterward: post-processing must not rewrite time inside speech.

| Failure | Wrong assumption | Architectural correction |
| --- | --- | --- |
| Voice changed between chunks | One VoiceDesign prompt implies one speaker | Commit one concrete synthetic voice anchor. |
| Reference preface repeated | Reference audio can be decoded and trimmed reliably | Decode generated acoustic codes only. |
| Register flipped around joins | A fixed identity makes sentence-sized requests coherent | Keep one streaming context for an authored section. |
| Delivery became too excited | A generic built-in voice is good enough | Select and version the actual narrator as an asset. |
| Sentences sounded slow and rigid | More explicit silence sounds more human | Let punctuation and model context carry local timing. |
| Audio skipped | Silence cleanup is harmless | Trim boundaries only; never rewrite silence inside speech. |
| *Cyclist* sounded wrong | Written spelling always supplies enough phonetic guidance | Apply a tested audio-only pronunciation lexicon. |
| A file ended early | Successful model return implies complete narration | Enforce token, duration-per-word, ending, and ASR checks. |

The table is the compact record. The deeper pattern is that every rejected design tried to recover continuity after splitting the generation into pieces that were too small. The final design moves the continuity boundary up to the article section and makes the remaining joins intentional.

## The narrator is a versioned asset

I eventually generated several voice candidates and chose the one that sounded closest to the Blog: female, Indian English, thoughtful, composed, slightly sombre, and precise without becoming theatrical. That selected sample is 18.6 seconds long and lives at `scripts/assets/blog-narrator-warm-indian-english-reference.mp3`.

The word *reference* can be misleading here. The sample still conditions Qwen3-TTS Base, but the decoder never renders its acoustic codes into the output. It encodes identity; the streaming path decodes generated codes only. The anchor's transcript is an inference input, not prose to be spliced and trimmed from the finished file.

The production profile is fixed:

```python
MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
VOICE = "Warm Indian English"
NARRATOR_SEED = 1904
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.05
SPEED = 1.10
```

The anchor SHA-256, model, sampling settings, seed, speed, decoding policy, sectioning policy, pause policy, pronunciation lexicon, and source digest all enter the manifest profile. Change the article and one post becomes stale. Change the narrator contract and the affected corpus becomes stale.

That distinction matters. The MP3s do not merely happen to sound similar today. The repository can reject a mixed corpus in which one post silently falls back to Ryan, an old chunking rule, or another anchor.

## Authored sections are the continuity unit

A whole long-form article is too large for one reliable generation request. A sentence is too small to carry a stable delivery. The useful boundary is the authored heading section.

The exporter preserves paragraph breaks as structural newlines, groups prose under each heading, and sends up to 4,000 characters in one request. Qwen streams that section in roughly twenty-second decoder pieces, but those pieces share one model context and are concatenated before the section is written. Streaming limits decoder memory; it does not redesign the speaker every twenty seconds.

Every section starts from the same committed anchor and narrator seed. The anchor holds identity across sections. The longer request lets the model hear enough local argument to shape emphasis, connect commas, and vary sentence length without an explicit pause file after every period.

The exporter inserts 650 milliseconds only between completed heading sections. It preserves the model's timing inside each section, trims silence only at generated boundaries, concatenates the sections, and applies `1.10x` tempo once to the final stream. The result is faster than the earlier voice without making each sentence race.

This is the current compromise. One request for an entire post would offer more global continuity but is slow and can fail late. One request per sentence is resumable but sounds assembled. One request per authored section gives the model enough context to sound intentional while keeping failure local and bounded.

## Pronunciation fixes belong to the audio compiler

Speech models occasionally need spelling help, but the Blog should not read like a phonetic script. I do not want to publish *sike-list* to make *cyclist* sound right.

The exporter therefore applies pronunciation hints after extracting the authored prose and before synthesis. The current lexicon maps *cyclist* to `sike-list`, *LiDAR* to `lie-dar`, and *timestamp* to `time stamp`. A narrow prosody rewrite also removes the comma in “timestamps drift, or one sensor degrades,” because the voice treated that comma as a larger break than the sentence intended.

The visible Markdown does not change. Tests assert both sides: the extracted article retains the author's spelling, while the synthesis prompt contains the phonetic form. A new exception needs a targeted test. Otherwise a local fix can become a corpus-wide pronunciation regression.

## Thirty minutes is a compilation limit, not a crop

I wanted a maximum listening time of thirty minutes. Cutting every MP3 at thirty minutes would satisfy a duration check and destroy the argument's ending. Speeding the voice until the file fit would make a dense post harder to follow. Skipping material blindly would move the missing content to an arbitrary point in the article.

The default remains full-source narration. When a post would exceed the limit, I write a reviewed narration sidecar under `src/narrations/blog`. The sidecar is pinned to the exact source SHA-256 and must cover every authored H2 section. It preserves the governing claims, evidence, motive, and conclusion while removing repeated examples, raw visual material, table rows, and detail that the page can carry better than audio.

Three current posts use this section-complete narration view: the VLM progression, omni-model pretraining, and autonomous-driving perception. The other fifteen speak the full extracted source. No exporter is allowed to crop the final MP3. If the generated file exceeds thirty minutes, generation fails and the existing asset remains in place.

The same refusal applies to incomplete synthesis. Each section has a 6,000-token acoustic ceiling. Reaching that ceiling is an error, not a successful result. The exporter also compares delivered duration with word count and rejects a section shorter than a conservative 0.16 seconds per word. Those checks cannot judge prose, but they catch a model that returned early or produced silence while claiming success.

## Freshness is necessary and insufficient

The manifest digest answers a precise question: did this MP3 come from the current source, narration view, extractor, narrator profile, and encoding policy? That prevents a stale post from pretending to be current. It does not answer whether the voice says the right words.

The release gate now has four layers.

First, deterministic checks validate source SHA, narration-sidecar pinning, H2 coverage, narrator profile, forbidden manifest fields, measured duration, and MP3 presence. The compiler refuses a profile mismatch or an asset longer than thirty minutes.

Second, full-file automatic speech recognition audits the spoken artifact. I compare the transcript with the expected beginning and ending, check the ratio of spoken to source words, and search unrelated posts for phrases from the voice anchor. This caught the distinction that source hashes could not: a file may be fresh and still repeat conditioning text.

Third, I listen where the system is most likely to lie. That means the opening, heading joins, the middle of long sections, reported timestamps such as thirty-five and forty-two seconds, difficult pronunciations, and the final paragraph. ASR is useful evidence. It is not an ear.

Fourth, the ordinary site gates still run: targeted exporter tests, `git diff --check`, the full repository CI command, GitHub checks, and the Pages deployment. The live route must return HTTP 200. Each changed MP3 is fetched with a cache-busting query and its SHA-256 must match the committed asset.

For the current 18-post corpus, the mechanical profile and duration checks passed, the ASR audit found every beginning and ending, no unrelated post contained the anchor phrase, every live Blog route returned 200, and every live MP3 hash matched. Those results are release evidence, not a claim that no sentence can ever sound better.

## Blog shipping now means two pull requests

The most expensive stale-audio mistake came from compiling in a worktree whose prose had not reached production. The MP3 matched that checkout perfectly. The website served different words.

A Blog release now has two stages. The first pull request contains the writing and ordinary page assets. After it merges and deploys, a clean worktree starts from the resulting `origin/main` and regenerates audio for each changed `postSlug`. The second pull request contains the MP3, manifest, and a narration sidecar only when the post needs one. Exporter code and tests enter that PR only when the compiler itself changed.

If the compiler changes corpus-wide, I use the autonomous-driving Perception post as a canary. Its length, many headings, diagrams, and pronunciation edge cases expose voice drift, repeated prefaces, bad joins, stale visual narration, early endings, and duration mistakes quickly. Once that canary passes, one foreground exporter handles the remaining stale posts with the model loaded once.

Before the audio PR is committed, the worktree fetches `origin/main` again. A concurrent Blog edit can make a completed MP3 stale during a long corpus run. Only assets whose source and profile digests still match are eligible to ship.

The interval between the two PRs is an incomplete Blog release. It is not a reason to combine prose and generated binaries into one ambiguous review. The publishing workflow now treats the follow-up audio PR, its deployment, and its live hash verification as part of the same task unless I explicitly choose a text-only release.

## Audio became an editorial constraint

The technical work ended up changing the prose more than I expected. A listener cannot glance back at a table, infer that “this” names the left panel, or hold five unexplained acronyms while a sentence detours through a citation. The spoken path makes missing transitions obvious.

That pressure is useful. The page can remain the richer artifact—with code, equations, figures, links, tables, and references—while the prose itself carries a complete argument. The narration compiler removes objects that require vision. It should not remove the reasoning those objects support.

I started with a player. What I actually built is a second rendering contract for the Blog. The source must be final. The narrator must be versioned. The generation boundary must preserve enough context to sound human. Duration limits must reshape the script rather than cut the ending. And production must serve the exact waveform that passed both machine checks and a listener's ear.

That is the part I would carry into any generated-media feature. A model call creates an artifact. A product needs a compiler, a release protocol, and evidence that the artifact people receive is the one you meant to make.

## References

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- [Astro content collections](https://docs.astro.build/en/guides/content-collections/)
- [Initial local read-aloud release](https://github.com/arunabh1904/arunabh1904.github.io/commit/693eb337f81ff93aaa0c998584fffd9bfd2e628f)
- [ICL preface repair](https://github.com/arunabh1904/arunabh1904.github.io/pull/198)
- [Reference-free corpus release](https://github.com/arunabh1904/arunabh1904.github.io/pull/204)
- [Non-truncating 30-minute compiler](https://github.com/arunabh1904/arunabh1904.github.io/pull/223)
- [Bounded narration corpus](https://github.com/arunabh1904/arunabh1904.github.io/pull/228)
- [Selected-voice corpus release](https://github.com/arunabh1904/arunabh1904.github.io/pull/233)
