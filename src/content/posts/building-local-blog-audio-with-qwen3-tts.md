---
title: Building Local Blog Audio That Sounds Human
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
  Why clean waveforms and source hashes were not enough, and how bounded
  synthesis plus chunk-level ASR caught the artifacts listeners could hear.
---

# Building Local Blog Audio That Sounds Human

Local Blog audio means that every post has a static MP3 generated on my Mac and served from GitHub Pages. That sounds like a text-to-speech feature. It became a publishing compiler: one system has to decide what the article says aloud, hold a voice steady for thousands of words, reject incomplete output, and prove that the file a listener receives was built from the prose on the page.

I learned that distinction by shipping the wrong thing several times. One release changed voices between chunks. Another repeated part of its voice-reference sentence before nearly every chunk. A later version removed that preface but sounded too excited, flipped register around thirty-five seconds, skipped near forty-two seconds, and pronounced *cyclist* badly. The next system passed its manifest and waveform checks yet still inserted strange phrases, repeated clauses, and rushed through transitions. The files were technically valid. They did not sound human.

The human narration profile now uses Voxtral 4B TTS with its fixed `casual_female` voice. Adjacent short paragraphs share one bounded request, each request is decoded as one waveform, and the compiler normalizes its edges before assembly. Local speech recognition still audits every request, but a seam audit now checks for suspicious silence inside prose. The VLM and autonomous-driving posts remain the long-form canaries because they expose voice, pacing, pronunciation, and continuity failures quickly.

> The central lesson was uncomfortable: provenance checks can prove which inputs produced a file, and waveform checks can prove that its samples are valid. Neither can prove that the voice said the right words with believable timing. Audio needs compiler invariants, speech-content audits, and a listener's ear.

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

The first assumption was that a good voice description creates a stable speaker. I used Qwen3-TTS VoiceDesign and repeated the same request for every chunk: warm, measured, close-miked, and restrained. The broad style survived. The person did not. Each chunk redesigned the voice, so timbre, pitch, register, and energy moved across an article.

The second assumption was that a shared reference waveform solves identity. Qwen3-TTS Base can condition on a reference WAV and its transcript. Giving every chunk the same synthetic sample did anchor one speaker. It also created the most obvious bug in the entire feature: “the delivery is warm” appeared before chunk after chunk.

The normal non-streaming ICL path decoded the reference acoustic codes together with the generated codes, then estimated where to cut the reconstructed waveform. The estimate left part of the reference sentence behind. Trimming harder attacked the symptom and risked deleting the first word of the article.

The third assumption was that generated-only decoding would finish the job. I switched to a streaming path that decodes only newly generated acoustic codes. The code path was right, yet an exported corpus still contained the spoken setup. That taught me to stop treating a manifest and an implementation inspection as an audio test. I needed to transcribe and listen to the artifact itself.

I then removed the reference and used the built-in Ryan CustomVoice speaker. That eliminated reference leakage, but it did not give me the voice I wanted. More importantly, sentence-sized generation still reset prosody constantly. A voice could begin calm, become bright and excited, drop in pitch at the next join, and change timing again thirty seconds later. The identity label remained `Ryan`; the listening experience still sounded stitched together.

Explicit pauses created another trap. Inserting silence after every sentence made technical prose slow and mechanical. A residual-silence filter then tried to compress long gaps across the concatenated file. That filter could eat a legitimate pause or word boundary and produce an audible skip. The skip near forty-two seconds made one rule clear even though the exact boundary failure was difficult to attribute afterward: post-processing must not rewrite time inside speech.

The Qwen Base profile then exposed a subtler control bug. The exporter declared a detailed `VOICE_INSTRUCT` string, recorded it in the manifest, and never passed it to generation. Even if it had, the official [Qwen3-TTS model matrix](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) lists instruction control for VoiceDesign and CustomVoice, not Base. Base is the voice-cloning checkpoint. A configuration value can look like part of the narrator contract while having no acoustic effect.

The last Qwen exports also ran at `1.10x` tempo. That choice shortened the files, but it compressed the pauses that technical prose uses to separate a claim from its consequence. Long heading-sized requests could still drift or repeat before the duration guard noticed. The fix was not a larger speed penalty or another silence filter. It was a model and generation boundary that matched the delivery I wanted.

| Failure | Wrong assumption | Architectural correction |
| --- | --- | --- |
| Voice changed between chunks | One VoiceDesign prompt implies one speaker | Commit one concrete synthetic voice anchor. |
| Reference preface repeated | Reference audio can be decoded and trimmed reliably | Decode generated acoustic codes only. |
| Register flipped around joins | A fixed identity makes sentence-sized requests coherent | Keep one streaming context for an authored section. |
| Delivery became too excited | A generic built-in voice is good enough | Select and version the actual narrator as an asset. |
| Sentences sounded slow and rigid | More explicit silence sounds more human | Let punctuation and model context carry local timing. |
| Audio skipped | Silence cleanup is harmless | Trim boundaries only; never rewrite silence inside speech. |
| Style settings changed nothing | A declared voice instruction controls every checkpoint | Verify the model variant supports each requested control and that generation receives it. |
| Pauses felt rushed | Final tempo is harmless packaging | Generate at `1.0x`; treat pace as part of the narrator contract. |
| A paragraph invented or repeated speech | Clean samples imply correct speech | Bound generation and audit each chunk against its expected text. |
| *Cyclist* sounded wrong | Written spelling always supplies enough phonetic guidance | Apply a tested audio-only pronunciation lexicon. |
| A file ended early | Successful model return implies complete narration | Enforce token, duration-per-word, ending, and ASR checks. |

The table is the compact record. The deeper pattern changed over time. Early designs split the generation into pieces that were too small, then tried to repair continuity afterward. The section-sized Qwen design moved too far in the other direction: it gave the model enough room to drift before any check could localize the failure. The useful unit is large enough to carry one thought and small enough to reject independently.

## The model contract must match the requested control

Qwen3-TTS Base solved a real problem: one reference clip could anchor a reusable speaker. It did not solve delivery control. VoiceDesign and CustomVoice expose instruction control; Base does not. The lesson is narrower than “use a better model.” Choose the checkpoint whose control surface matches the job, then verify that every declared setting reaches inference.

I tested three preset voices from [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) on the same technical passage. `hi_female` read it accurately, but the `hi` preset carries the Indian-English identity that its name implies. That was the wrong default for this corpus even though the transcript passed. `neutral_female` repeated a phrase for roughly 20 seconds. `casual_female` was quicker, but sounded the most conversational once the compiler supplied the missing structural pauses. Model choice did not remove the need for an artifact gate; it gave the gate a better candidate to work with.

The accepted human profile is fixed:

```python
HUMAN_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
HUMAN_VOICE = "casual_female"
HUMAN_NARRATOR_SEED = 1904
HUMAN_TEMPERATURE = 0.65
HUMAN_TOP_K = 50
HUMAN_TOP_P = 0.9
HUMAN_SPEED = 1.0
HUMAN_PARAGRAPH_PAUSE_SECONDS = 0.24
HUMAN_HEADING_PAUSE_SECONDS = 0.55
```

The model, preset, sampling settings, chunk seeds, speed, pause policy, pronunciation lexicon, and source digest enter the manifest profile. Change the article and that post becomes stale. Change a generation setting and every assigned post becomes stale. A rejected chunk receives a reviewed seed override, so rerunning the exporter reproduces the accepted sample instead of rolling the dice again.

There is also a deployment boundary. Mistral's model card says the supplied reference voices and model inherit CC BY-NC 4.0. That fits this non-commercial personal site. A commercial product would need a model and voice license that permits its use; acoustic quality does not override licensing.

## Paragraph groups are the failure boundary

A whole long-form article is too large for one reliable request. A sentence is too small to carry stable delivery. One request per paragraph still resets the delivery often enough for the joins to become audible. A full heading section, meanwhile, can be long enough for repetition or acoustic drift to hide inside one successful return. The useful boundary is a small group of adjacent authored paragraphs.

The exporter packs adjacent paragraphs up to 1,200 characters while keeping headings separate. That gives the model enough syntax to carry a thought across a paragraph break without giving a failure room to spread across an entire section. Voxtral's fixed preset holds the speaker identity, and newlines inside the request preserve authored structure.

The first Voxtral exporter requested streamed output in eight-second pieces. MLX-Audio includes overlap-aware decoding, but direct concatenation still left audible seams inside some paragraphs. The static-site compiler does not need low-latency playback, so it now asks Voxtral to decode each bounded request as one waveform. The exporter then trims excess edge silence, keeps the speech at `1.0x`, and inserts 240 milliseconds after a paragraph group and 550 milliseconds after a heading. No filter is allowed to search the finished speech for silence and rewrite it.

Short headings were unexpectedly brittle because two or three words give the model little linguistic context. The ASR audit caught headings that began with “man man saw,” “one of these,” or other invented setup. I rerolled only those chunks, recorded their accepted seeds, and left the other accepted waveforms untouched.

This boundary improves both sound and debugging. A failed paragraph group is cheap to identify and replace. A whole-file transcript can say that the average error is acceptable while hiding one repeated clause; a request transcript points to the exact generation that produced it.

## Pronunciation fixes belong to the audio compiler

Speech models occasionally need spelling help, but the Blog should not read like a phonetic script. I do not want to publish *sike-list* to make *cyclist* sound right.

The exporter therefore applies pronunciation hints after extracting the authored prose and before synthesis. The shared lexicon maps *cyclist* to `sike-list`, *LiDAR* to `lie-dar`, and *timestamp* to `time stamp`. The autonomous-driving post adds a local spelling for `BEVDet4D`. It also restores `Lidar encoders` for one short heading because the otherwise useful `lie-dar` spelling became unstable when spoken alone. A narrow prosody rewrite removes the comma in “timestamps drift, or one sensor degrades,” because the voice treated that comma as a larger break than the sentence intended.

The visible Markdown does not change. Tests assert both sides: the extracted article retains the author's spelling, while the synthesis prompt contains the phonetic form. A new exception needs a targeted test. Otherwise a local fix can become a corpus-wide pronunciation regression.

## Thirty minutes is a compilation limit, not a crop

I wanted a maximum listening time of thirty minutes. Cutting every MP3 at thirty minutes would satisfy a duration check and destroy the argument's ending. Speeding the voice until the file fit would make a dense post harder to follow. Skipping material blindly would move the missing content to an arbitrary point in the article.

The default remains full-source narration. A reviewed narration sidecar is available when the editorial decision is to create a shorter audio view. It is pinned to the exact source SHA-256 and must cover every authored H2 section. The VLM progression uses that path. It preserves the governing claims, evidence, motive, and conclusion while removing repeated examples and detail that the page can carry better than audio.

The autonomous-driving survey made the opposite editorial choice: keep the complete 3,840-word narration. Its human-paced render is 31 minutes 28 seconds, so the compiler gives that post a narrow 32-minute cap. This is not a loophole that silently expands every asset. The source, manifest, and per-post limit record the exception. No exporter is allowed to crop the final MP3.

The same refusal applies to incomplete synthesis. Each human-profile chunk has a 1,600-token acoustic ceiling. The exporter rejects chunks that are implausibly short or long for their word count. Those checks cannot judge prose, but they catch silence, early endings, and large continuation loops before assembly.

## Freshness is necessary and insufficient

The manifest digest answers a precise question: did this MP3 come from the current source, narration view, extractor, narrator profile, and encoding policy? That prevents a stale post from pretending to be current. It does not answer whether the voice says the right words.

The release gate now has four layers.

First, deterministic checks validate source SHA, narration-sidecar pinning, H2 coverage, narrator profile, measured duration, chunk count, and MP3 presence. The compiler refuses a profile mismatch or an asset beyond its reviewed limit.

Second, local Whisper transcribes every cached synthesis chunk. The audit compares each transcript with the exact text sent to TTS, normalizes harmless spelled acronyms such as `C L I P`, and then applies three gates: aggregate word error rate must stay below 8%, no invented insertion may exceed two words, and no chunk may exceed 50% error. The insertion rule caught fluent hallucinations that average WER hid. The chunk rule caught short headings whose two important words were both wrong.

Third, waveform checks reject NaNs, infinities, denormals, clipping, and suspicious internal silence that can reveal a decoder seam. These checks remain separate from speech-content QA because a repeated phrase can have a perfectly clean waveform. I then listen at the opening, inside long requests, at paragraph and heading joins, around every rerolled request, at difficult pronunciations, and at the ending. ASR is useful evidence. It is not an ear.

Fourth, the ordinary site gates still run: targeted exporter tests, `git diff --check`, the full repository CI command, GitHub checks, and the Pages deployment. The live route must return HTTP 200. Each changed MP3 is fetched with a cache-busting query and its SHA-256 must match the committed asset.

The first two canaries passed the content and deployment gates: low aggregate WER, no multi-word insertion artifacts, no high-error chunks, and exact production hashes. Listening still rejected them. One used an accent-specific preset that should never have been the unexamined corpus default; both exposed joins inside prose. That result changed the gate itself. A passing transcript and hash are release evidence, not proof that a long-form narration sounds continuous or appropriate.

## Blog shipping now means two pull requests

The most expensive stale-audio mistake came from compiling in a worktree whose prose had not reached production. The MP3 matched that checkout perfectly. The website served different words.

A Blog release now has two stages. The first pull request contains the writing and ordinary page assets. After it merges and deploys, a clean worktree starts from the resulting `origin/main` and regenerates audio for each changed `postSlug`. The second pull request contains the MP3 and manifest, plus a narration sidecar only when the post has an explicitly reviewed shorter audio view. Exporter code and tests enter that PR only when the compiler or human-profile assignment changed.

The VLM and autonomous-driving posts were the migration canaries. The VLM post stresses acronyms and model names. The autonomous-driving survey stresses length, headings, visual omissions, pronunciation, and the duration policy. Existing Qwen assets remain valid until their source changes; every new or changed Blog post moves to the human profile and must pass the chunk-level audit before merge. That incremental boundary avoids regenerating the corpus before the new contract has enough evidence.

Before the audio PR is committed, the worktree fetches `origin/main` again. A concurrent Blog edit can make a completed MP3 stale during a long corpus run. Only assets whose source and profile digests still match are eligible to ship.

The interval between the two PRs is an incomplete Blog release. It is not a reason to combine prose and generated binaries into one ambiguous review. The publishing workflow now treats the follow-up audio PR, its deployment, and its live hash verification as part of the same task unless I explicitly choose a text-only release.

## Audio became an editorial constraint

The technical work ended up changing the prose more than I expected. A listener cannot glance back at a table, infer that “this” names the left panel, or hold five unexplained acronyms while a sentence detours through a citation. The spoken path makes missing transitions obvious.

That pressure is useful. The page can remain the richer artifact, with code, equations, figures, links, tables, and references, while the prose itself carries a complete argument. The narration compiler removes objects that require vision. It should not remove the reasoning those objects support.

I started with a player. What I actually built is a second rendering contract for the Blog. The source must be final. The narrator must be versioned. The generation boundary must preserve enough context to sound human. Duration limits must reshape the script rather than cut the ending. And production must serve the exact waveform that passed both machine checks and a listener's ear.

That is the part I would carry into any generated-media feature. A model call creates an artifact. A product needs a compiler, a release protocol, and evidence that the artifact people receive is the one you meant to make.
