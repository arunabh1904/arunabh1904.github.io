---
title: Running Gemma 4 locally on a 64 GB MacBook Pro
date: '2026-04-04T04:00:00.000Z'
section: blog
postSlug: running-gemma-4-locally-on-a-64-gb-macbook-pro
legacyPath: /blog/2026/04/04/running-gemma-4-locally-on-a-64-gb-macbook-pro.html
tags:
  - LLMs
  - Apple Silicon
summary: >-
  Benchmarks for MLX, llama.cpp, and Ollama running Gemma 4 on a 64 GB M5 Max
  MacBook Pro, plus what I would actually use every day.
---
# Running Gemma 4 locally on a 64 GB MacBook Pro

I wanted the answer to a very specific question: on a 64 GB M5 Max MacBook Pro, what is the best Gemma 4 model I can run locally, and what is the fastest way to run it?

The model answer stayed simple.

The runtime answer got much more interesting once I stopped guessing and actually benchmarked it.

The short version, as of April 4, 2026:

- Best model that fits: `Gemma 4 31B`
- Best daily balance: `Gemma 4 26B A4B`
- Fastest usable runtime I tested on this machine: `MLX`
- Most direct low-level path: `llama.cpp`
- Current Ollama status for Gemma 4 on this machine: it crashes before first token

That last one surprised me the most.

## What fits

According to Google's current Gemma docs, approximate Q4 inference memory requirements are `3.2 GB` for `Gemma 4 E2B`, `5 GB` for `Gemma 4 E4B`, `15.6 GB` for `Gemma 4 26B A4B`, and `17.4 GB` for `Gemma 4 31B` ([Google docs](https://ai.google.dev/gemma/docs/core)).

So on a 64 GB machine, all four models are realistic local targets, including the two workstation-class ones that matter most for serious use:

- `Gemma 4 26B A4B`
- `Gemma 4 31B`

The `26B A4B` model is the more interesting one for everyday laptop use. It is MoE, so only `4B` parameters are active per generated token, even though the full model still has to be resident in memory. The `31B` model is the strongest dense option that still makes sense on this machine. Google also lists `128K` context for the small models and `256K` context for the larger ones, which is great capability-wise, but not remotely free from a latency perspective ([Google docs](https://ai.google.dev/gemma/docs/core), [Gemma 4 31B card](https://huggingface.co/google/gemma-4-31B-it)).

So my view did not really change here:

- If you want the best Gemma 4 model that comfortably fits, start with `31B`.
- If you want the model I would actually pay the most attention to for daily use, it is `26B A4B`.

## How I benchmarked it

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

The output task was intentionally boring and deterministic: read background text, then print numbered lines. Temperature was pinned to `0`, and I measured:

- Time to first token
- Decode tokens per second
- Average tokens per second over the full request

One important caveat: this is a fastest-practical-path comparison, not a perfect same-weights lab setup. I used the most direct current artifact for each runtime. That means the `E2B` comparison is not perfectly apples-to-apples: official `llama.cpp` GGUF for `E2B` is `Q8_0`, while the MLX and Ollama paths use 4-bit artifacts.

## Special Things I Had To Do

A few benchmark details ended up mattering more than I expected:

- I disabled Gemma's thinking mode anywhere I could, because otherwise you are partly benchmarking extra reasoning tokens instead of raw runtime behavior.
- I kept the benchmark text-only, which meant running `llama.cpp` without a multimodal projector. That was the cleanest way to measure prompt processing and decode speed instead of image overhead.
- I used a boring deterministic output task and pinned temperature to `0`. That made throughput differences much easier to trust.
- I split the test into short and long prompts on purpose. A runtime can look fine at `512` tokens and then feel much worse once prompt processing climbs into the `8K` range.
- I tried Ollama with native `gemma4:*` tags, not a hacked local import path. I wanted Ollama to get the fairest possible shot before concluding that the current M5 experience is broken.

None of those are exotic tricks, but they were the difference between "the model runs" and "I actually believe this comparison."

## What The Benchmarks Say

I came into this expecting `llama.cpp` to win on raw speed.

That is not what the machine gave me.

For the smaller models, MLX was clearly faster on this M5 Max. For `26B A4B`, the story got more nuanced: `llama.cpp` and MLX were effectively tied on the short prompt, but MLX still pulled ahead once the prompt got long. For `31B`, MLX went back to being the cleaner win, especially on prompt processing and time to first token.

### Short Suite

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

### Long Suite

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

The pattern is pretty clear:

- MLX is winning the small-model tests by a healthy margin.
- `26B A4B` is much closer on short prompts than I expected.
- `31B` is a meaningful step down in responsiveness compared with `26B A4B`, even though it still fits comfortably in memory.
- `31B` also stops being a close runtime contest: MLX is simply better behaved there on this machine.
- On longer prompts, MLX is still more comfortable because time to first token is substantially lower.
- The difference between "weights fit" and "this feels good to use" becomes real very quickly once prompt length goes up.

That last point matters more than people usually admit. Decode speed often holds up decently. What really starts to hurt is prompt processing and time to first token.

## Ollama, Right Now

I wanted a clean Ollama column here.

I could not get one.

On this exact machine, using current native Gemma 4 tags like `gemma4:e2b-it-q4_K_M`, Ollama `0.20.2` failed before first token with a Metal backend compilation error and returned HTTP `500` from `/api/generate`. The key error was the same `bfloat` vs `half` cooperative tensor mismatch in Metal Performance Primitives that other Apple M5 users have reported upstream ([issue #13460](https://github.com/ollama/ollama/issues/13460), [issue #14432](https://github.com/ollama/ollama/issues/14432), [issue #13867](https://github.com/ollama/ollama/issues/13867)).

That matters because it changes the recommendation:

- This is not a general "Gemma 4 cannot run on Apple Silicon" problem.
- It is not even a general "M5 cannot run local models" problem.
- `llama.cpp` and MLX both worked on the same machine.
- The failure appears to be specific to Ollama's current Apple / Metal path on M5-class hardware.

So if your question is "should I start with Ollama because it is easiest," my current answer is no, not for Gemma 4 on this hardware, at least not until that backend issue is fixed.

## What I Would Actually Use

If I cared about the strongest local Gemma 4 model, I would start with `31B`.

If I cared about the model I would actually want to use every day on this machine, I would pay the closest attention to `26B A4B`.

If I cared about the fastest usable runtime on this machine, the answer is no longer "obviously llama.cpp." The current evidence points more toward:

1. `MLX` first
2. `llama.cpp` second
3. `Ollama` later, once the M5 Metal breakage is fixed

And if you do not just want a terminal workflow, this also maps cleanly to a tiny local browser chat app. Both `MLX` and `llama.cpp` are perfectly reasonable backends if what you really want is just to serve Gemma locally and talk to it.

That is a more opinionated answer than I expected going in, but it feels much less hand-wavy now.

And honestly, that was the point of running the benchmarks in the first place.
