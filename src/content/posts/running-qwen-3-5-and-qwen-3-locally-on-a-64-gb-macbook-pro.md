---
title: Running Qwen 3.5 and Qwen 3 locally on a 64 GB MacBook Pro
date: '2026-04-04T04:00:00.000Z'
section: blog
postSlug: running-qwen-3-5-and-qwen-3-locally-on-a-64-gb-macbook-pro
legacyPath: /blog/2026/04/04/running-qwen-3-5-and-qwen-3-locally-on-a-64-gb-macbook-pro.html
tags:
  - LLMs
  - Apple Silicon
summary: >-
  Local Qwen 3.5 and Qwen 3 benchmarks on a 64 GB M5 Max MacBook Pro, plus
  the tradeoffs that actually matter when you are picking a local model.
---
# Running Qwen 3.5 and Qwen 3 locally on a 64 GB MacBook Pro

I wanted the answer to a practical question: on a `64 GB` M5 Max MacBook Pro, which Qwen models can I actually run locally, and which ones still feel good once prompt length stops being toy-sized?

The answer was partly about model size.

It was also about runtime choice, prompt length, and a few very boring setup details that matter a lot more than people usually admit.

One important scope note up front, as of April 4, 2026:

- The official local/open comparison here covers `Qwen 3.5` and `Qwen 3`.
- I did not find an official open local `Qwen 3.6` model listing on the [Qwen Hugging Face organization page](https://huggingface.co/Qwen), so there is no real `Qwen 3.6` local benchmark column in this post.

The short version so far:

- `Qwen 3 4B` is the cleanest fast local baseline I have run on this machine.
- `Qwen 3 14B` fits comfortably, but it is where long-prompt responsiveness stops feeling lightweight.
- `Qwen 3.5 9B` still feels very realistic on a `64 GB` laptop, but it is a clear step down in latency from the `4B` class.
- `Qwen 3.5 4B` is also very usable, but it is a little heavier in memory and a little slower on long prompts in my current setup.
- `MLX` is the first runtime I would reach for on this Mac.
- `4B` is the speed-first choice, but `14B` is where the local quality conversation starts getting more interesting.

## What fits

Qwen's current open model lineup matters here because it immediately narrows the realistic local targets.

On the official [Qwen organization page](https://huggingface.co/Qwen), the newest open local families I found were `Qwen 3.5` and `Qwen 3`, along with the very large `Qwen3.5-122B-A10B`, `Qwen3.5-397B-A17B`, and `Qwen3-235B-A22B` variants that are not sensible first bets for a `64 GB` laptop. So I excluded those from the benchmark matrix and focused on the sizes that have a real chance of being pleasant locally.

That left these practical local targets for this machine:

- `Qwen 3.5 4B`
- `Qwen 3.5 9B`
- `Qwen 3.5 27B`
- `Qwen 3.5 35B A3B`
- `Qwen 3 4B`
- `Qwen 3 14B`
- `Qwen 3 30B A3B`
- `Qwen 3 32B`

The two big family-level differences that matter for local use are:

- [`Qwen 3.5`](https://huggingface.co/Qwen/Qwen3.5-9B) defaults to thinking mode and exposes a `262,144` token default context window, so if you do not explicitly disable thinking you are partly benchmarking chain-of-thought overhead instead of plain inference behavior.
- [`Qwen 3`](https://huggingface.co/Qwen/Qwen3-14B-GGUF) supports both thinking and non-thinking modes in the same model, and the official GGUF releases make `llama.cpp` comparisons much easier for that family.

## How I benchmarked it

I ran everything on:

- `Apple M5 Max`
- `64 GB` unified memory
- `macOS 26.3.1 (a)`

The local software stack for this round was:

- `mlx-lm 0.31.2`
- `mlx-vlm 0.4.4`
- `llama.cpp llama-server 8660`
- `transformers 5.5.0`

I used the same two text-only suites as the Gemma post:

- Short suite: `512` input tokens, `192` output tokens
- Long suite: `8192` input tokens, `96` output tokens

The task was intentionally boring and deterministic: read repeated background text, then emit exactly twelve numbered factual lines. Temperature was pinned to `0`, and I recorded:

- Time to first token
- Decode tokens per second
- Average tokens per second over the whole request
- Peak memory during generation where the runtime exposed it

I also kept the app and the benchmark harness text-only for this pass. No images, no multimodal prompts, and no hidden reasoning tokens if I could turn them off.

## Special Things I Had To Do

A few details ended up mattering more than I expected:

- I forced `Qwen 3.5` into non-thinking mode anywhere I could. Otherwise the benchmark stops being about raw runtime behavior.
- I kept the local chat app text-only even though `Qwen 3.5` ships as an image-text model family. I wanted clean text-generation comparisons first.
- `Qwen 3.5` on MLX still needed `torch` and `torchvision` installed in this environment because the processor stack came in through `mlx-vlm`.
- I used official Qwen GGUF releases for `Qwen 3`, but for `Qwen 3.5` I fell back to pinned community `Q4_K_M` GGUFs because I did not find an official `Qwen 3.5` GGUF release on the Qwen organization page.
- I had to disable Hugging Face Xet downloads on this machine for some official Qwen artifacts, because otherwise a few of the larger MLX downloads would stall on incomplete blobs.

None of that is especially glamorous, but all of it changes whether local benchmarking feels straightforward or mysteriously flaky.

## What The Benchmarks Say

The first finished runs already made one thing clear: `4B` Qwen is not just workable on this machine, it is genuinely pleasant.

`Qwen 3 4B` came out ahead of `Qwen 3.5 4B` in my current MLX setup on both memory footprint and long-prompt responsiveness. The short-prompt throughput story was closer, but even there `Qwen 3 4B` had the better decode speed.

The first bigger model result, `Qwen 3.5 9B`, is useful because it shows where things stop feeling lightweight. It still runs cleanly and it still fits easily, but TTFT and average throughput both move down enough that the gap is noticeable once prompts get longer.

`Qwen 3 14B` pushes that trend further. It still fits comfortably in memory on this machine, but the performance profile changes category: short prompts are still fine, while long prompts become a patience test compared with the 4B models.

The cross-runtime results also sharpened the runtime recommendation. On `Qwen 3.5 9B`, `MLX` beat `llama.cpp` on both suites by a healthy margin, especially once prompt length hit the `8K` range. On `Qwen 3 14B`, the short prompt was much closer, but MLX still pulled ahead clearly on the long prompt where prompt processing dominates the experience.

### Short Suite

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s | Peak memory |
| ----- | ------- | -------- | ---- | ------------ | --------- | ----------- |
| `Qwen 3 14B` | llama.cpp | `Qwen/Qwen3-14B-GGUF` `Q4_K_M` | `568 ms` | `53.36` | `45.17` | `n/a` |
| `Qwen 3 14B` | MLX | `mlx-community/Qwen3-14B-4bit` | `684 ms` | `59.87` | `47.50` | `8.88 GB` |
| `Qwen 3.5 4B` | MLX | `mlx-community/Qwen3.5-4B-MLX-4bit` | `187 ms` | `144.08` | `125.00` | `4.28 GB` |
| `Qwen 3.5 9B` | llama.cpp | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` | `824 ms` | `74.50` | `52.44` | `n/a` |
| `Qwen 3.5 9B` | MLX | `mlx-community/Qwen3.5-9B-MLX-4bit` | `301 ms` | `96.33` | `79.67` | `7.07 GB` |
| `Qwen 3 4B` | MLX | `mlx-community/Qwen3-4B-4bit` | `392 ms` | `176.14` | `128.89` | `3.05 GB` |

### Long Suite

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s | Peak memory |
| ----- | ------- | -------- | ---- | ------------ | --------- | ----------- |
| `Qwen 3 14B` | llama.cpp | `Qwen/Qwen3-14B-GGUF` `Q4_K_M` | `11112 ms` | `44.77` | `7.24` | `n/a` |
| `Qwen 3 14B` | MLX | `mlx-community/Qwen3-14B-4bit` | `4925 ms` | `52.25` | `14.18` | `10.34 GB` |
| `Qwen 3.5 4B` | MLX | `mlx-community/Qwen3.5-4B-MLX-4bit` | `2103 ms` | `131.45` | `34.04` | `5.55 GB` |
| `Qwen 3.5 9B` | llama.cpp | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` | `5158 ms` | `67.28` | `14.58` | `n/a` |
| `Qwen 3.5 9B` | MLX | `mlx-community/Qwen3.5-9B-MLX-4bit` | `2894 ms` | `92.66` | `24.53` | `8.39 GB` |
| `Qwen 3 4B` | MLX | `mlx-community/Qwen3-4B-4bit` | `1742 ms` | `127.76` | `38.41` | `4.24 GB` |

Even from this first slice, the pattern is useful:

- `Qwen 3 4B` is the better low-friction local baseline on this machine.
- `Qwen 3.5 4B` is still very usable, but it carries more memory overhead in my current MLX path.
- `Qwen 3.5 9B` still looks laptop-friendly, but it is where the latency tradeoff starts feeling materially different from the tiny models.
- `Qwen 3.5 9B` also gave the first real runtime verdict: on this hardware, `MLX` was materially better than `llama.cpp`.
- `Qwen 3 14B` looks like the first genuinely stronger dense option that still feels reasonable to keep around locally, but it is no longer fast in the same way.
- `Qwen 3 14B` makes the runtime story more specific: `llama.cpp` is competitive on the short prompt, but MLX is still the better default once prompts get long.
- Long-prompt behavior matters a lot more than short-prompt decode speed if you care about whether a local model actually feels snappy.

## What I Would Actually Use

Right now, my practical recommendation is simple:

1. Start with `Qwen 3 4B` if you want the fastest clean local baseline.
2. Move up to `Qwen 3 14B` if you want a stronger dense model and you can tolerate much slower long-prompt interaction.
3. Keep `Qwen 3.5 9B` in the mix if you specifically care about the 3.5 family behavior and do not mind the extra latency and memory overhead.
4. Prefer `MLX` first on this Mac unless a specific `llama.cpp` model artifact or integration path gives you a reason to switch.

That answer is opinionated, but it is also much less hand-wavy now.

It comes from actually running the models on this machine instead of guessing from parameter counts or release notes.
