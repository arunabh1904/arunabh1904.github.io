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

On a `64 GB` M5 Max, the measured Qwen models all fit, but their long-prompt latency differs substantially. With MLX, `Qwen 3 4B` reached the first token in `1.742 s` for an `8,192`-token prompt; `Qwen 3 14B` took `4.925 s`. The same `14B` artifact family took `11.112 s` through `llama.cpp`.

The measurements date to April 4, 2026 and cover Qwen 3 and Qwen 3.5. The later [`Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) release is outside this experiment.

## Models in the snapshot

The capacity shortlist contained:

- `Qwen 3.5 4B`
- `Qwen 3.5 9B`
- `Qwen 3.5 27B`
- `Qwen 3.5 35B A3B`
- `Qwen 3 4B`
- `Qwen 3 14B`
- `Qwen 3 30B A3B`
- `Qwen 3 32B`

This is a candidate list, not a measured leaderboard. The tables below cover only `Qwen 3 4B`, `Qwen 3 14B`, `Qwen 3.5 4B`, and `Qwen 3.5 9B`; the larger entries are fit and usability candidates from the release snapshot, not performance claims from this run.

The two big family-level differences that matter for local use are:

- [`Qwen 3.5`](https://huggingface.co/Qwen/Qwen3.5-9B) defaults to thinking mode and lists a `262,144`-token native context length, so if you do not explicitly disable thinking you are partly benchmarking chain-of-thought overhead instead of plain inference behavior.
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
- Long suite: `8,192` input tokens, `96` output tokens

The task was intentionally boring and deterministic: read repeated background text, then emit exactly twelve numbered factual lines. Temperature was `0`. I recorded:

- Time to first token
- Decode tokens per second
- Average tokens per second over the whole request
- Peak memory during generation where the runtime exposed it

Runtime-specific setup:

- I forced `Qwen 3.5` into non-thinking mode anywhere I could. Otherwise the benchmark stops being about raw runtime behavior.
- I kept the local chat app text-only even though `Qwen 3.5` ships as an image-text model family. I wanted clean text-generation comparisons first.
- `Qwen 3.5` on MLX still needed `torch` and `torchvision` installed in this environment because the processor stack came in through `mlx-vlm`.
- I used official Qwen GGUF releases for `Qwen 3`, but for `Qwen 3.5` I fell back to pinned community `Q4_K_M` GGUFs because I did not find an official `Qwen 3.5` GGUF release on the Qwen organization page.
- I had to disable Hugging Face Xet downloads for some official Qwen artifacts because a few larger MLX downloads stalled on incomplete blobs.

## Benchmark results

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

`Qwen 3 4B` used the least measured memory and had the lowest long-prompt TTFT. Moving to `Qwen 3.5 9B` raised MLX peak memory from `4.24` to `8.39 GB` and TTFT from `1.742` to `2.894 s`. Both fit comfortably; prompt processing separates their responsiveness.

For `Qwen 3 14B`, short-prompt TTFT slightly favored `llama.cpp`, while MLX more than halved TTFT on the `8K` suite. Decode rates stayed much closer. A runtime comparison based only on short-prompt decode would miss the largest measured difference.

## Limits of the comparison

The timed task measures inference behavior, not reasoning, coding, or tool-use quality. It supports `Qwen 3 4B` as the lowest-memory baseline in this set and MLX as the faster long-prompt path on this machine. It does not establish a quality ranking between Qwen generations or justify extrapolating these timings to the larger untested candidates.
