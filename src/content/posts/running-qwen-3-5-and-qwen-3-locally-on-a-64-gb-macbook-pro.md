---
title: Benchmarking Qwen 3.5 and Qwen 3 on a 64 GB MacBook Pro
date: '2026-04-04T04:00:00.000Z'
section: blog
blogGroup: local-ai-lab
postSlug: running-qwen-3-5-and-qwen-3-locally-on-a-64-gb-macbook-pro
legacyPath: /blog/2026/04/04/running-qwen-3-5-and-qwen-3-locally-on-a-64-gb-macbook-pro.html
tags:
  - LLMs
  - Apple Silicon
summary: >-
  A dated comparison of Qwen 3.5 and Qwen 3 latency on a 64 GB M5 Max,
  including why long-prompt prefill matters more than whether a model fits.
---
# Benchmarking Qwen 3.5 and Qwen 3 on a 64 GB MacBook Pro

I wanted one practical answer: on a `64 GB` M5 Max MacBook Pro, which Qwen models are pleasant locally once prompt length stops being toy-sized?

One important scope note: the measurements were collected on April 4, 2026.

- The measured comparison covers `Qwen 3.5` and `Qwen 3` only.
- Qwen has since released the official open-weight [`Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B). It was not available for this run, so I do not add an invented or cross-source `Qwen 3.6` column.

Within this benchmark snapshot:

- `Qwen 3 4B` is the cleanest fast local baseline I have run on this machine.
- `Qwen 3 14B` fits comfortably, but it is where long-prompt responsiveness stops feeling lightweight.
- `Qwen 3.5 9B` still feels very realistic on a `64 GB` laptop, but it is a clear step down in latency from the `4B` class.
- `Qwen 3.5 4B` is also very usable, but it is a little heavier in memory and a little slower on long prompts in my current setup.
- `MLX` is the first runtime I would reach for on this Mac.
- `4B` is the speed-first choice, but `14B` is where the local quality conversation starts getting more interesting.

## Models in the snapshot

The open lineup available on the measurement date narrowed the realistic local targets.

On April 4, the newest open local families I found were `Qwen 3.5` and `Qwen 3`, along with very large variants that were not sensible first bets for a `64 GB` laptop. I excluded those and focused on sizes with a realistic chance of being pleasant locally. The later Qwen 3.6 release is a new candidate, not evidence that changes the older measurements.

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

## Benchmark design

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

The task was intentionally boring and deterministic: read repeated background text, then emit exactly twelve numbered factual lines. Temperature was `0`. I recorded:

- Time to first token
- Decode tokens per second
- Average tokens per second over the whole request
- Peak memory during generation where the runtime exposed it

I also kept the app and the benchmark harness text-only for this pass. No images, no multimodal prompts, and no hidden reasoning tokens if I could turn them off.

The remaining controls kept the benchmark from measuring hidden work:

- I forced `Qwen 3.5` into non-thinking mode anywhere I could. Otherwise the benchmark stops being about raw runtime behavior.
- I kept the local chat app text-only even though `Qwen 3.5` ships as an image-text model family. I wanted clean text-generation comparisons first.
- `Qwen 3.5` on MLX still needed `torch` and `torchvision` installed in this environment because the processor stack came in through `mlx-vlm`.
- I used official Qwen GGUF releases for `Qwen 3`, but for `Qwen 3.5` I fell back to pinned community `Q4_K_M` GGUFs because I did not find an official `Qwen 3.5` GGUF release on the Qwen organization page.
- I had to disable Hugging Face Xet downloads for some official Qwen artifacts because a few larger MLX downloads stalled on incomplete blobs.

## Benchmark results

The first finished runs made one thing clear: `4B` Qwen is workable on this machine and genuinely pleasant.

`Qwen 3 4B` came out ahead of `Qwen 3.5 4B` in my current MLX setup on both memory footprint and long-prompt responsiveness. The short-prompt throughput story was closer, but even there `Qwen 3 4B` had the better decode speed.

The first bigger model result, `Qwen 3.5 9B`, is useful because it shows where things stop feeling lightweight. It still runs cleanly and it still fits easily, but TTFT and average throughput both move down enough that the gap is noticeable once prompts get longer.

`Qwen 3 14B` pushes that trend further. It still fits comfortably in memory on this machine, but the performance profile changes category: short prompts are still fine, while long prompts become a patience test compared with the 4B models.

The cross-runtime results sharpened the runtime recommendation. On `Qwen 3.5 9B`, `MLX` beat `llama.cpp` on both suites by a healthy margin, especially once prompt length hit the `8K` range. On `Qwen 3 14B`, the short prompt was much closer, but MLX still pulled ahead on the long prompt where prompt processing dominates the experience.

The frame-by-frame comparison holds every measured model/runtime row fixed and changes only the input length. Long prompts create a much larger time-to-first-token penalty than the decode column alone suggests.

<div class="architecture-comparison blog-frame-explainer" data-blog-frame-explainer="local-qwen-long-prompt-latency.gif"><div class="blog-frame-explainer__viewport"><a href="/assets/images/blog-explainer-frames/local-qwen-long-prompt-latency/frame-01.webp"><img src="/assets/images/blog-explainer-frames/local-qwen-long-prompt-latency/frame-01.webp" alt="Manual explainer comparing short- and long-prompt time to first token and decode throughput for Qwen 3.5 and Qwen 3 on MLX and llama.cpp"></a></div></div>

*The `4B` models remain the interactive tier. `Qwen 3 14B` still fits easily, but TTFT reaches `4.9 s` on MLX and `11.1 s` on llama.cpp in the `8K` suite. Custom visualization of this post's benchmark tables; measurements are from one 64 GB M5 Max on April 4, 2026. Qwen 3.6 is intentionally absent because it was released after the run.*

The figure separates three decisions that parameter count often collapses. Memory determines whether a model loads. Decode throughput determines continuation speed. Prefill determines whether document-scale prompts feel responsive. On this machine, moving from `4B` to `14B` changes the third quantity most sharply, which is why the quality-versus-latency decision should be made with realistic prompt lengths.

A short-prompt benchmark prices generation. An agent workload also prices the history it must reread before every turn. That second bill can dominate the experience.

### Short-context results

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s | Peak memory |
| ----- | ------- | -------- | ---- | ------------ | --------- | ----------- |
| `Qwen 3 14B` | llama.cpp | `Qwen/Qwen3-14B-GGUF` `Q4_K_M` | `568 ms` | `53.36` | `45.17` | `n/a` |
| `Qwen 3 14B` | MLX | `mlx-community/Qwen3-14B-4bit` | `684 ms` | `59.87` | `47.50` | `8.88 GB` |
| `Qwen 3.5 4B` | MLX | `mlx-community/Qwen3.5-4B-MLX-4bit` | `187 ms` | `144.08` | `125.00` | `4.28 GB` |
| `Qwen 3.5 9B` | llama.cpp | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` | `824 ms` | `74.50` | `52.44` | `n/a` |
| `Qwen 3.5 9B` | MLX | `mlx-community/Qwen3.5-9B-MLX-4bit` | `301 ms` | `96.33` | `79.67` | `7.07 GB` |
| `Qwen 3 4B` | MLX | `mlx-community/Qwen3-4B-4bit` | `392 ms` | `176.14` | `128.89` | `3.05 GB` |

### Long-context results

| Model | Runtime | Artifact | TTFT | Decode tok/s | Avg tok/s | Peak memory |
| ----- | ------- | -------- | ---- | ------------ | --------- | ----------- |
| `Qwen 3 14B` | llama.cpp | `Qwen/Qwen3-14B-GGUF` `Q4_K_M` | `11112 ms` | `44.77` | `7.24` | `n/a` |
| `Qwen 3 14B` | MLX | `mlx-community/Qwen3-14B-4bit` | `4925 ms` | `52.25` | `14.18` | `10.34 GB` |
| `Qwen 3.5 4B` | MLX | `mlx-community/Qwen3.5-4B-MLX-4bit` | `2103 ms` | `131.45` | `34.04` | `5.55 GB` |
| `Qwen 3.5 9B` | llama.cpp | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` | `5158 ms` | `67.28` | `14.58` | `n/a` |
| `Qwen 3.5 9B` | MLX | `mlx-community/Qwen3.5-9B-MLX-4bit` | `2894 ms` | `92.66` | `24.53` | `8.39 GB` |
| `Qwen 3 4B` | MLX | `mlx-community/Qwen3-4B-4bit` | `1742 ms` | `127.76` | `38.41` | `4.24 GB` |

`Qwen 3 4B` is the lowest-friction baseline in this snapshot. `Qwen 3.5 4B` remains usable but carries more memory overhead in the measured MLX path, while `Qwen 3.5 9B` is the first size where latency feels materially different from the tiny models. It also gives the cleanest cross-runtime result: MLX reaches the first token substantially sooner than llama.cpp in both suites.

`Qwen 3 14B` is the first stronger dense option I would seriously keep around locally, but it is no longer fast in the same sense. llama.cpp remains competitive on the short prompt; on the `8K` prompt, MLX more than halves TTFT. Long-prompt prefill therefore matters more to the interaction than the short-prompt decode number that usually headlines a local benchmark.

## Recommendation

For this measured snapshot, my practical recommendation is simple:

1. Start with `Qwen 3 4B` if you want the fastest clean local baseline.
2. Move up to `Qwen 3 14B` if you want a stronger dense model and you can tolerate much slower long-prompt interaction.
3. Keep `Qwen 3.5 9B` in the mix if you specifically care about the 3.5 family behavior and do not mind the extra latency and memory overhead.
4. Prefer `MLX` first on this Mac unless a specific `llama.cpp` model artifact or integration path gives you a reason to switch.

That answer comes from this machine, not parameter counts or release notes.
