# Tiny-voca on the complete mapping article

Measured November 16, 2026. This is a local candidate evaluation, not a change to the production narrator.

The full M2 recording is at `/assets/audio/benchmarks/tiny-voca-2026-11-16/mapping-m2.mp3`. Opening clips for M2, F2, and published Voxtral sit beside it. The user preferred M2 from the opening samples; full-length perceptual acceptance remains unverified.

## Comparison

| Render | Duration | Phonemization + synthesis | WER | Existing ASR gate |
| --- | --- | --- | --- | --- |
| Published Voxtral casual_female | 35:55 | Not measured | 3.097% | Pass |
| Tiny-voca F2 | 32:51 | 16.900 s | 2.938% | Pass |
| Tiny-voca M2 | 30:32 | 16.398 s | 2.760% | Pass |

Hardware: Apple M5 Max, 64 GiB unified memory, four PyTorch CPU threads. Python 3.12.13; PyTorch 2.15.0; tiny-voca 0.2.0. No GPU was used for synthesis. Whisper used Metal for recognition. Generation time sums the timed `speak_text` calls, including phonemization and peak limiting; it excludes import, model loading, warm-up, disk writes, assembly, encoding, and ASR. One opening request per voice was reused from the identical-profile canary. These are one-run measurements, not repeated latency statistics.

The source was `from-bev-features-to-lane-graphs-and-changing-maps` at website commit `11c42af`, with 4,837 whitespace-delimited spoken words. The ASR normalizer yields 5,037 reference tokens; the difference is tokenization, not omitted text. The same 53 synthesis requests, including the existing pronunciation substitutions, went to all candidates. Tiny-voca's helper split each request into sentences and bounded subclauses, totaling 370 utterances per voice. Both voices used seed 1904 per outer request, speed 1.0, pitch shift 0, pitch range 1, and the upstream 0.99 peak limit. Assembly retained the production 240 ms/550 ms pauses and applied -1 dB before 64 kbps mono MP3 encoding. There was no abridgment, time stretch, training, or quality-driven reroll.

The baseline is the exact published artifact. Its live download, committed MP3, and reconstruction from the cached production requests all have SHA-256 `cad74e3231654f387d8ca4733de026692297718b7b5fa4dc671d7312605f7eba`. Reconstruction used the existing exporter's boundary filter, pauses, gain, and codec. Thus the per-request baseline audit covers the actual speech served by the website, not a fresh Voxtral sample. The baseline runtime is MLX-Audio 0.5.3; the later Voxtral first-frame fix was not benchmarked here.

Every request was transcribed by `mlx-whisper==0.4.3`, using Whisper large-v3-turbo revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`, English, temperature 0, and word timestamps. The repository's existing transcript normalization and limits were unchanged: aggregate WER ≤8%, insertion run ≤2 words, no forbidden fillers, no chunk WER >50%, and no unaligned region >1.5 seconds. All three passed. Tiny-voca had no non-finite, clipped, or denormal samples in saved WAVs and no unknown-phoneme warnings. This checks sample integrity and recognition alignment, not naturalness or every possible acoustic seam.

Recognition discrepancies remain, particularly around names, numbers, and technical terms. M2's highest request WER was 8.51%; F2's was 11.22%. M2 request 45 transcribed “autoregressively” as “at R aggressively”; request 49 confused Fréchet and several model names. These are review locations, not established TTS mispronunciations. Both synthesizer and recognizer can be responsible. The word-error-rate differences are small and are not a significance-tested preference result. Because Voxtral also passed, this experiment cannot claim that an ASR pass resolves the listener's artifact complaint.

## Pinned sources

- [tiny-voca code](https://github.com/n4ze3m/tiny-voca/tree/62e2adef158bc997042cc6234f541f1e0e4cd55d)
- [Weights and phonemizer](https://huggingface.co/n4ze3m/tiny-voca/tree/eac048b7a4392584725af9792460a0211fa1cac7)
- [dizione code](https://github.com/n4ze3m/dizione/tree/8cd889e4de7be6bd1dd2e6891344a0729fa14dd4)
- [Model/data description](https://github.com/n4ze3m/tiny-voca/blob/62e2adef158bc997042cc6234f541f1e0e4cd55d/docs/model.md)

The checkpoint contains 4,537,291 inference parameters, occupies 18,209,221 bytes, and has SHA-256 `ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70`. Model weights, phonemizer data, runtime packages, and working memory are different size measures; the checkpoint size is not total application memory.

## Reproduce

Use an isolated Python 3.12 environment with the pinned tiny-voca checkout and dizione commit, `torch==2.15.0`, `numpy==2.5.5`, `soundfile==0.14.0`, `mlx-whisper==0.4.3`, and `jiwer==4.0.0`. Download the pinned weights and `phonemizer/` data above; ffmpeg is required. The benchmark imports the site's existing exporter and audit normalization.

```sh
export DIZIONE_DATA=/path/to/weights/phonemizer
python scripts/benchmark-tiny-voca.py --stage generate \
  --checkpoint /path/to/weights/tiny-voca.pt --output-dir /tmp/tiny-voca-benchmark
python scripts/benchmark-tiny-voca.py --stage audit \
  --output-dir /tmp/tiny-voca-benchmark --voices M2 F2 \
  --asr-model /path/to/pinned/whisper-large-v3-turbo
```

Add `--limit 1` to both commands for the canary. Add `--baseline` to the audit only when the source-matched production WAV cache is available; the public MP3 alone does not supply those request boundaries. The command refuses a manifest/source mismatch. A newer mapping article is a different benchmark; reproduce this source revision when comparing these numbers. Use a fresh output directory when changing a model, runtime, voice setting, or CPU thread count. Source and waveform hashes protect the per-request caches; ASR caches also include the recognizer path and runtime.

`summary.json` contains the aggregate results and provenance. The `*-generation.json` files contain each tiny-voca request's text, time, waveform hash, and integrity metrics. The `*-audit.json` files retain every expected/recognized text pair and rejection-gate result.

## Retraining decision

The released M2 weights already run on this machine and pass the current objective gate on this post. Retraining is unnecessary for that use. A targeted pronunciation lexicon is the cheaper first experiment for recurring technical words; test changes on held-out passages and retain the unchanged source text.

A new voice would require new paired speech and text, preparation/alignment, and a training evaluation. The released `.pt` file has `model` weights but no `system`, discriminator, or optimizer state. The upstream `--init-from` path expects the full training checkpoint, so this inference file is not a drop-in fine-tuning input. Obtain that checkpoint or implement and validate an explicit initialization path before estimating an adaptation project. No training, cost, or quality improvement from fine-tuning was measured here.
