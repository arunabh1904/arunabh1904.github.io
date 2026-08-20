---
title: Benchmarking Muse Glimmer 30B on a 64 GB MacBook Pro
date: '2026-08-13T04:00:00.000Z'
section: blog
blogGroup: local-ai-lab
postSlug: running-muse-glimmer-30b-locally-on-a-64-gb-macbook-pro
legacyPath: /blog/2026/08/13/running-muse-glimmer-30b-locally-on-a-64-gb-macbook-pro.html
tags:
  - LLMs
  - Apple Silicon
  - Inference
summary: >-
  Measured Muse Glimmer 30B latency, memory use, and long-context behavior on
  an M5 Max.
---
# Benchmarking Muse Glimmer 30B on a 64 GB MacBook Pro

`Muse Glimmer 30B` fits comfortably on a `64 GB` M5 Max MacBook Pro and serves through `llama.cpp`. The official `Q4_K_M` GGUF occupies `16.76 GB`. With all layers explicitly placed on Metal, it generated at `27.9 tokens/s`; adding Meta's official `1.63 GB` DFlash drafter raised short-prompt decode to `48.3 tokens/s`.

The first run was much slower—about `10 tokens/s`—because I had left GPU-layer placement on the runtime's automatic setting. The controlled reruns make the practical lesson clearer than the initial number: use full Metal offload, then enable DFlash. That gets close to Meta's reported M5 Max speed on short generation. Long prompts remain the constraint: the `8K` DFlash run needed `16.4 s` to begin reasoning and `17.9 s` to show the answer.

This is a hardware-and-software snapshot from August 13, 2026. Muse Glimmer and its `llama.cpp` support are new enough that runtime releases may change the result materially.

## Why the model fits

Meta's [`Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a `29.6B` dense vision-language model, including an approximately `1.8B` perception encoder, with a `131K` context window. Unlike a mixture-of-experts model, it does not reduce decode compute by activating only a small expert subset. Quantization is what makes the laptop deployment practical.

The official [GGUF release](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) offers two clear Mac targets:

| Official artifact | File size | Published target | `64 GB` verdict |
| --- | ---: | ---: | --- |
| `K-Quant-17GB-Q4_K_M` | `16.76 GB` | `24 GB` VRAM | Comfortable |
| `Dynamic-20GB-Q4_K_XL` | `19.65 GB` | `32 GB` VRAM | Comfortable |

The vision projector adds about `1.4 GB`; Meta's DFlash speculative draft model adds about `1.6 GB`. Even the `20 GB` quant plus both auxiliaries leaves a credible memory budget on this machine. Context still consumes KV-cache capacity, so “supports 131K” should not be read as “131K will feel interactive.”

I tested the smaller official quant because it is the obvious starting point. It is an official Meta artifact rather than an untracked community conversion, and its exact size is `16,756,683,904` bytes.

## Benchmark setup

I ran the model on:

- `Apple M5 Max`
- `64 GB` unified memory
- `macOS 26.5.2`
- `llama.cpp` build `10360` (`48d22e295`)
- Official `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
- Official `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` for the speculative run

Glimmer support requires `llama.cpp` build `10353` or newer, so I upgraded the local Homebrew build from `8660` to `10360` before testing. I used `llama-server` with one slot, Metal acceleration, flash attention, a `16K` allocated context, and the model's Jinja chat template.

The benchmark was text-only, so I did not load the vision projector. I ran three serving configurations: automatic layer placement without a drafter, explicit full Metal offload, and full Metal offload plus DFlash. I used two deterministic suites:

- Short: `512` input tokens, at most `192` completion tokens
- Long: `8193` input tokens, at most `96` completion tokens

The task asked the model to print zero-padded integers, which it followed until each completion cap. Temperature was `0`, top-p was `1`, top-k was `1`, and the seed was fixed. Glimmer reasons by default, so I set low reasoning strength and a `32`-token reasoning budget rather than pretending hidden generation did not exist.

I report two first-token measurements. *First generated token* includes hidden reasoning. *First answer token* is when visible content begins. Decode throughput comes from all completion tokens and the time after the first streamed token, while average throughput includes prompt processing and generation.

## Results

| Configuration | Input | First generated | First answer | Decode tok/s | End-to-end average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auto placement | `512` | `2.40 s` | `9.77 s` | `9.92` | `8.83 tok/s` |
| Full Metal | `512` | `0.95 s` | `3.50 s` | `27.90` | `24.52 tok/s` |
| Full Metal + DFlash | `512` | `1.01 s` | `2.67 s` | `48.31` | `38.54 tok/s` |
| Auto placement | `8193` | `32.26 s` | `37.41 s` | `10.07` | `2.30 tok/s` |
| Full Metal | `8193` | `17.10 s` | `19.01 s` | `26.62` | `4.64 tok/s` |
| Full Metal + DFlash | `8193` | `16.41 s` | `17.94 s` | `35.47` | `5.02 tok/s` |

The largest correction was explicit placement. `--gpu-layers all` lifted the short decode rate from `9.92` to `27.90 tok/s`, a `2.8×` change before speculation entered the comparison. It also cut the long prompt's first generated token from `32.26 s` to `17.10 s`. A model that fits in unified memory can still run slowly if the engine chooses a conservative CPU/GPU split.

DFlash then changed decode rather than fit. With `--spec-type draft-dflash` active, the server accepted `87.9%` of proposed short-suite draft tokens and `79.8%` on the long suite. Short decode rose from `27.90` to `48.31 tok/s`; long decode rose from `26.62` to `35.47 tok/s`. The speedup is smaller at `8K` because prompt processing is unchanged by speculative generation and dominates more of the request.

Reasoning creates a second latency boundary. In the short DFlash suite, the server began hidden generation at `1.01 s`, while the first visible integer arrived at `2.67 s`. A client that reports only transport-level time to first token still makes the chat experience look faster than it feels, although the gap is no longer severe.

Meta reports `26.6 tok/s` without DFlash and `50.2 tok/s` with DFlash on an M5 Max, using ExecuTorch, batch size one, and greedy decoding. My short `llama.cpp` results of `27.90` and `48.31 tok/s` are remarkably close, but they remain separate measurements from a different runtime and harness. The agreement is useful evidence that the optimized local path is working; it is not a cross-runtime benchmark victory.

## Serving Glimmer locally

The text-only OpenAI-compatible server I validated is:

```bash
llama-server \
  --model Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  --alias muse-glimmer-30b \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 16384 \
  --parallel 1 \
  --gpu-layers all \
  --flash-attn on \
  --model-draft dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  --gpu-layers-draft all \
  --spec-type draft-dflash \
  --jinja \
  --reasoning-budget 32 \
  --chat-template-kwargs '{"reasoning_strength":"low"}'
```

Keeping the host on `127.0.0.1` matters because `llama-server` otherwise needs deliberate authentication and network policy. Once it is listening, an OpenAI-compatible client can use `http://127.0.0.1:8080/v1` as its base URL.

For vision, download the matching projector from the official GGUF repository and add `--mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf`. That adds image input but was not part of this text benchmark. On `llama.cpp` build `10360`, merely loading the DFlash file did not activate speculation: the server log showed draft loading but no accepted proposals. Adding `--spec-type draft-dflash` produced explicit acceptance statistics and the measured speedup, so I would keep that flag rather than assume autodetection.

The reusable harness for this run is in [`scripts/bench_llama_server_local.py`](https://github.com/arunabh1904/arunabh1904.github.io/blob/main/scripts/bench_llama_server_local.py). It refuses to run on an occupied port so that a stale local server cannot silently contaminate the measurements.

## Recommendation

> The `17 GB` Q4_K_M quant makes Glimmer useful on a `64 GB` Mac because it leaves headroom for the application and context, not because the advertised maximum context is automatically usable. Add the projector only when vision is needed, then measure the workload before expanding context.

I would use the full-Metal DFlash configuration for rapid local chat. Nearly `48 tok/s` on the short suite is fluid, and the model still leaves ample memory headroom. I would remain careful with agent loops that repeatedly inject `8K` of state: speculative decoding accelerates generation, not the entire prefill, so the long suite still took almost `18 s` to expose an answer. The next optimization target is prompt reuse or a smaller active context, not a larger quant.

For the broader laptop decision, see [which current open-weight models fit on a 64 GB Mac](/blog/2026/08/13/open-weight-models-that-fit-on-a-64-gb-macbook-pro.html). Glimmer is the most interesting new model in that list because it fits with room to spare. DeepSeek V4 Flash is the opposite case: its official checkpoint is about `167 GB`, so [serving it from this Mac means using an API](/blog/2026/08/13/running-deepseek-v4-flash-0731-on-a-64-gb-macbook-pro.html), not finding a cleverer local runtime.
