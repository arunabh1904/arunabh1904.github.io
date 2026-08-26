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

I wanted one concrete answer: on a 64 GB M5 Max MacBook Pro, which Gemma 4 model should I actually run locally, and through which runtime?

These measurements are a hardware-and-software snapshot from April 4, 2026, not a permanent runtime leaderboard. Google subsequently released Gemma 4 12B Unified on June 3, so that model is outside this benchmark. I also have not rerun the earlier Ollama failure on current releases.

Within the measured snapshot:

- Best model that fits: `Gemma 4 31B`
- Best daily balance: `Gemma 4 26B A4B`
- Fastest usable runtime I tested on this machine: `MLX`
- Most direct low-level path: `llama.cpp`
- Ollama `0.20.2` crashed before first token in this run

## Model memory requirements

Google's Gemma 4 documentation lists approximate Q4 inference memory requirements of `3.2 GB` for `E2B`, `5 GB` for `E4B`, `15.6 GB` for `26B A4B`, and `17.4 GB` for `31B` ([Google docs](https://ai.google.dev/gemma/docs/core)). Those estimates describe weight/runtime memory, not the full memory and latency cost of a long active context.

So on a 64 GB machine, all four models are realistic local targets, including the two workstation-class ones that matter most for serious use:

- `Gemma 4 26B A4B`
- `Gemma 4 31B`

The `26B A4B` model is the more interesting everyday laptop option. It is MoE, so only `4B` parameters are active per generated token, even though the full model still has to sit in memory. The `31B` model is the strongest dense option that still makes sense on this machine. Google also lists `128K` context for the small models and `256K` for the larger ones, but those windows carry a substantial latency cost ([Google docs](https://ai.google.dev/gemma/docs/core), [Gemma 4 31B card](https://huggingface.co/google/gemma-4-31B-it)).

My model recommendation is simple:

- If you want the best Gemma 4 model that comfortably fits, start with `31B`.
- If you want the model I would actually pay the most attention to for daily use, it is `26B A4B`.

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

The remaining controls kept the comparison about runtime behavior rather than hidden work:

- I disabled Gemma's thinking mode anywhere I could, because otherwise you are partly benchmarking extra reasoning tokens instead of raw runtime behavior.
- I kept the benchmark text-only, which meant running `llama.cpp` without a multimodal projector. That was the cleanest way to measure prompt processing and decode speed instead of image overhead.
- I used a boring deterministic output task and pinned temperature to `0`. That made throughput differences much easier to trust.
- I split the test into short and long prompts on purpose. A runtime can look fine at `512` tokens and then feel much worse once prompt processing climbs into the `8K` range.
- I tried Ollama with native `gemma4:*` tags, not a hacked local import path, so the Ollama result reflects the current easy path.

## Benchmark results

I came into this expecting `llama.cpp` to win on raw speed.

That is not what the machine gave me.

For the smaller models, MLX was clearly faster on this M5 Max. For `26B A4B`, the story got more nuanced: `llama.cpp` and MLX were effectively tied on the short prompt, but MLX pulled ahead once the prompt got long. For `31B`, MLX was the cleaner win, especially on prompt processing and time to first token.

The frame-by-frame comparison keeps the model/runtime rows fixed while the input changes from `512` to `8192` tokens. Decode speed moves modestly; time to first token moves enough to change how the model feels.

<div class="architecture-comparison blog-frame-explainer" data-blog-frame-explainer="local-gemma-long-prompt-latency.gif"><div class="blog-frame-explainer__viewport"><a href="/assets/images/blog-explainer-frames/local-gemma-long-prompt-latency/frame-01.webp"><img src="/assets/images/blog-explainer-frames/local-gemma-long-prompt-latency/frame-01.webp" alt="Manual explainer comparing short- and long-prompt time to first token and decode throughput for Gemma 4 on MLX and llama.cpp"></a></div></div>

*Long prompts expose prefill as the practical bottleneck. The `31B` weights fit, but TTFT rises to `13.5 s` on MLX and `24.2 s` on llama.cpp in the measured long suite. `26B A4B` retains roughly `100 tok/s` decode while reaching the first token much sooner. Custom visualization of this post's benchmark tables; measurements are from one 64 GB M5 Max on April 4, 2026.*

This is the distinction the memory table cannot show. Capacity answers whether a model can load. Decode throughput answers how quickly it continues once generation has started. Prefill latency answers whether an `8K` document or agent state feels interactive at all. For daily use, the last quantity makes `26B A4B` a different product from `31B` even though both fit comfortably.

Local inference has three thresholds: load, start, and continue. Weight memory controls the first; prefill controls the second; decode throughput controls the third.

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

I wanted a clean Ollama column here. I could not get one.

On this exact machine, using current native Gemma 4 tags like `gemma4:e2b-it-q4_K_M`, Ollama `0.20.2` failed before first token with a Metal backend compilation error and returned HTTP `500` from `/api/generate`. The key error was the same `bfloat` vs `half` cooperative tensor mismatch in Metal Performance Primitives that other Apple M5 users have reported upstream ([issue #13460](https://github.com/ollama/ollama/issues/13460), [issue #14432](https://github.com/ollama/ollama/issues/14432), [issue #13867](https://github.com/ollama/ollama/issues/13867)).

That matters because it changes the recommendation:

- This is not a general "Gemma 4 cannot run on Apple Silicon" problem.
- It is not even a general "M5 cannot run local models" problem.
- `llama.cpp` and MLX both worked on the same machine.
- The failure was specific to the Ollama Apple/Metal path in this April 4 environment.

The linked upstream issue #13460 is now closed, so this result should not be read as a current compatibility claim without a rerun. For reproducing this benchmark snapshot, Ollama was not a viable column. For choosing a runtime now, retest the current Ollama release rather than inheriting that failure.

## Recommendation

Within this measured snapshot, if I cared about the strongest local Gemma 4 model, I would start with `31B`.

If I cared about the model I would actually want to use every day on this machine, I would pay the closest attention to `26B A4B`.

If I cared about the fastest usable runtime on this machine, the answer is no longer "obviously llama.cpp." These measurements point toward:

1. `MLX` first
2. `llama.cpp` second
3. `Ollama` only after a fresh compatibility run

If you do not want a terminal workflow, this maps cleanly to a tiny local browser chat app. Both `MLX` and `llama.cpp` are reasonable backends if the goal is simply to serve Gemma locally and talk to it.
