---
title: Benchmarking Gemma 4 on a 64 GB MacBook Pro
date: '2026-04-04T04:00:00.000Z'
section: blog
blogGroup: local-ai-lab
postSlug: running-gemma-4-locally-on-a-64-gb-macbook-pro
legacyPath: /blog/2026/04/04/running-gemma-4-locally-on-a-64-gb-macbook-pro.html
tags:
  - LLMs
  - Apple Silicon
summary: >-
  A dated comparison of MLX and llama.cpp latency for Gemma 4 on a 64 GB M5
  Max, including why long-prompt prefill changes the recommendation.
---
# Benchmarking Gemma 4 on a 64 GB MacBook Pro

This benchmark compares Gemma 4 inference on a `64 GB` M5 Max using MLX, `llama.cpp`, and Ollama. On the long-prompt suite, MLX reached the first token sooner for every tested model. The `26B A4B` model combined `2.18 s` time to first token with `104.36 tok/s` decode; `31B` took `13.50 s` and decoded at `23.73 tok/s`.

These measurements are a hardware-and-software snapshot from April 4, 2026, not a permanent runtime leaderboard. Google subsequently released Gemma 4 12B Unified on June 3, so that model is outside this benchmark. I also have not rerun the earlier Ollama failure on current releases.

## Model memory requirements

Google's Gemma 4 documentation lists approximate Q4 inference memory requirements of `3.2 GB` for `E2B`, `5 GB` for `E4B`, `15.6 GB` for `26B A4B`, and `17.4 GB` for `31B` ([Google docs](https://ai.google.dev/gemma/docs/core)). Those estimates describe weight/runtime memory, not the full memory and latency cost of a long active context.

All four weight configurations fit this machine's memory budget. The `26B A4B` model activates `4B` parameters per token while retaining the full expert bank in memory; `31B` is dense. Google lists `128K` context for the small models and `256K` for the larger ones ([Google docs](https://ai.google.dev/gemma/docs/core), [Gemma 4 31B card](https://huggingface.co/google/gemma-4-31B-it)). The tests below use much shorter contexts and measure latency, not model quality.

## Benchmark design

I ran everything on:

- `Apple M5 Max`
- `64 GB` unified memory
- `macOS 26.3.1 (a)`

I tested the three obvious Mac paths:

- `llama.cpp` via `llama-server` and the official `ggml-org` GGUF releases
- `MLX` via `mlx-lm` and the `mlx-community` 4-bit conversions
- `Ollama` via native `gemma4:*` tags

I used two text-only suites so the results would reflect inference behavior rather than reasoning verbosity:

- Short suite: `512` input tokens, `192` output tokens
- Long suite: `8192` input tokens, `96` output tokens

The output task was intentionally boring and deterministic: read background text, then print numbered lines. Temperature was `0`. I measured:

- Time to first token
- Decode tokens per second
- Average tokens per second over the full request

One important caveat: this is a fastest-practical-path comparison, not a perfect same-weights lab setup. I used the most direct current artifact for each runtime. That means the `E2B` comparison is not perfectly apples-to-apples: official `llama.cpp` GGUF for `E2B` is `Q8_0`, while the MLX and Ollama paths use 4-bit artifacts.

Thinking mode was disabled where supported, and `llama.cpp` ran without a multimodal projector. Ollama used native `gemma4:*` tags. These choices keep reasoning and image processing outside the timed text-generation path.

## Benchmark results

### Short-context results

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s |
| ----- | ------- | -------- | ---- | ------------ | --------- |
| `E2B` | MLX | `mlx-community/gemma-4-e2b-it-4bit` | `181 ms` | `182.86` | `155.95` |
| `E2B` | llama.cpp | `ggml-org` `Q8_0` GGUF | `127 ms` | `119.46` | `110.73` |
| `E4B` | MLX | `mlx-community/gemma-4-e4b-it-4bit` | `230 ms` | `114.96` | `101.03` |
| `E4B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `391 ms` | `96.54` | `80.69` |
| `26B A4B` | MLX | `mlx-community/gemma-4-26b-a4b-it-4bit` | `422 ms` | `115.80` | `92.31` |
| `26B A4B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `334 ms` | `110.85` | `92.92` |
| `31B` | MLX | `mlx-community/gemma-4-31b-it-4bit` | `906 ms` | `27.50` | `24.34` |
| `31B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `1279 ms` | `24.89` | `21.35` |

### Long-context results

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s |
| ----- | ------- | -------- | ---- | ------------ | --------- |
| `E2B` | MLX | `mlx-community/gemma-4-e2b-it-4bit` | `879 ms` | `175.68` | `67.33` |
| `E2B` | llama.cpp | `ggml-org` `Q8_0` GGUF | `1634 ms` | `114.07` | `38.78` |
| `E4B` | MLX | `mlx-community/gemma-4-e4b-it-4bit` | `1682 ms` | `103.95` | `36.85` |
| `E4B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `3068 ms` | `89.35` | `23.17` |
| `26B A4B` | MLX | `mlx-community/gemma-4-26b-a4b-it-4bit` | `2182 ms` | `104.36` | `30.95` |
| `26B A4B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `3227 ms` | `101.42` | `23.00` |
| `31B` | MLX | `mlx-community/gemma-4-31b-it-4bit` | `13501 ms` | `23.73` | `5.47` |
| `31B` | llama.cpp | `ggml-org` `Q4_K_M` GGUF | `24164 ms` | `20.72` | `3.33` |

MLX wins the small-model tests by a healthy margin. `26B A4B` is the exception that clarifies the comparison: the two runtimes are effectively tied on the short suite, then MLX reaches the first token about a second sooner on the long suite while decode remains close. The runtime choice matters most during prefill, not after generation is underway.

`31B` changes the model choice more than the runtime choice. It fits comfortably, but its long-prompt TTFT is roughly six times the `26B A4B` MLX result and more than seven times the corresponding llama.cpp result. MLX is still better behaved, yet the larger conclusion is that “weights fit” and “this feels good to use” are separate thresholds.

## The measured Ollama failure

In this April 4 run, using the then-current native Gemma 4 tags like `gemma4:e2b-it-q4_K_M`, Ollama `0.20.2` failed before first token with a Metal backend compilation error and returned HTTP `500` from `/api/generate`. The key error was the same `bfloat` vs `half` cooperative tensor mismatch in Metal Performance Primitives that other Apple M5 users have reported upstream ([issue #13460](https://github.com/ollama/ollama/issues/13460), [issue #14432](https://github.com/ollama/ollama/issues/14432), [issue #13867](https://github.com/ollama/ollama/issues/13867)).

MLX and `llama.cpp` both ran on the same hardware. The observed failure was specific to Ollama's Apple/Metal path in this environment.

The linked upstream issue #13460 is now closed, so this result should not be read as a current compatibility claim without a rerun. For reproducing this benchmark snapshot, Ollama was not a viable column. For choosing a runtime now, retest the current Ollama release rather than inheriting that failure.

## What the measurements establish

For these artifacts and prompt lengths, MLX is the faster starting point, particularly during prefill. `26B A4B` offers much lower long-prompt latency than `31B` in both working runtimes. The deterministic output task does not establish which model gives better answers; that requires a separate quality evaluation. Ollama requires a fresh compatibility test because its recorded failure belongs to the April 4 environment.
