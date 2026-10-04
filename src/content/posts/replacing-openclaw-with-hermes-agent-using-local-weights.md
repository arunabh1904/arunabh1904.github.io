---
title: Running Hermes Agent with a Local GGUF
date: '2026-04-04T17:59:45.000Z'
section: blog
blogGroup: projects
postSlug: replacing-openclaw-with-hermes-agent-using-local-weights
legacyPath: /blog/2026/04/04/replacing-openclaw-with-hermes-agent-using-local-weights.html
tags:
  - Agents
  - LLMs
  - Apple Silicon
summary: >-
  How Hermes Agent connects to a local OpenAI-compatible endpoint backed by
  llama.cpp and an already-downloaded Gemma GGUF.
---
# Running Hermes Agent with a Local GGUF

I replaced OpenClaw with [Hermes Agent](https://github.com/nousresearch/hermes-agent) while keeping inference local and using existing model files. The working April 4, 2026 setup connected Hermes to `llama-server`, which loaded a cached Gemma GGUF.

## Separate the agent from inference

Hermes manages tools, filesystem and terminal access, sessions, and skills.

Hermes does not force one inference path. It works with hosted providers, but it can also point at any OpenAI-compatible local endpoint. That separation let the agent framework stay fixed while the model runtime changed underneath it.

Hermes calls an HTTP API; the serving process owns model loading, the KV cache, and token generation; the GGUF is inert model data on disk.

This separation also changes how to debug. If Hermes cannot reach `/v1/chat/completions`, inspect the endpoint and configuration. If the endpoint returns HTTP `500` while loading a model, inspect the runtime, artifact, and hardware path. If text is generated but tools appear as plain text, inspect the server's chat template and tool-call support. Treating those as three different contracts avoids reinstalling the wrong layer.

## Install Hermes

The install command was:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
```

That bootstrapped:

- `uv`
- Python `3.11`
- the Hermes repo under `~/.hermes/hermes-agent`
- the `hermes` CLI symlink in `~/.local/bin`
- the default config in `~/.hermes/config.yaml`

## Why Ollama failed

Hermes could see the local endpoint. Ollama listed local Gemma models. But actual inference failed with HTTP `500` during model load on this Apple Silicon setup. In other words, the Hermes install was fine, but the runtime below it was not stable enough for the job.

## Serve the local GGUF

The machine already had local Gemma GGUF artifacts in the Hugging Face cache, including:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/...
```

`llama-server` was already installed through Homebrew.

I started `llama-server` directly against the cached GGUF. The command below records the setup that worked on April 4, 2026:

```bash
llama-server \
  --model ~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/<revision>/gemma-4-e4b-it-Q4_K_M.gguf \
  --no-mmproj \
  --reasoning off \
  --host 127.0.0.1 \
  --port 18080 \
  --ctx-size 32768 \
  --parallel 1 \
  --flash-attn on
```

A few details mattered in that run:

- `--ctx-size 32768` was necessary because Hermes sends a large system prompt and `8192` was not enough.
- `--parallel 1` kept the memory footprint reasonable while still leaving enough room for the larger context window.
- `--reasoning off` matched what I wanted anyway: no extra thinking overhead for a local smoke test.

The context value is the main dated part of this recipe. `32768` was enough for this April 4 smoke test, but current Hermes documentation requires at least `64,000` tokens for agent use with tools because the system prompt, schemas, and working conversation already consume substantial context. A new setup should therefore size both Hermes and `llama-server` consistently, typically at `65536` or higher if the model and available memory support it, instead of copying the older `32768` value blindly.

Once that server was up, it exposed the OpenAI-compatible endpoint Hermes wanted at:

```text
http://127.0.0.1:18080/v1
```

## Point Hermes at localhost

The current Hermes setup path is `hermes model`, then **Custom endpoint**. I originally pointed Hermes at `llama-server` by editing `~/.hermes/config.yaml` directly:

```yaml
model:
  default: gemma-4-e4b-it-Q4_K_M.gguf
  provider: custom
  base_url: http://127.0.0.1:18080/v1
```

One small gotcha: on my machine, `hermes config set model ...` collapsed the whole `model:` block into a plain string, so editing the YAML directly was more reliable for this local-endpoint setup.

After that, `hermes status --deep` showed exactly what I wanted:

- model set to the local GGUF-backed model
- provider set to `Custom endpoint`
- no cloud API keys required

## Verify the full path

I checked the complete inference path with:

```bash
hermes chat -q 'Reply with exactly READY and nothing else.' -Q --max-turns 1
```

And it returned:

```text
READY
```

The unresolved part is agent behavior: structured tool calls, long histories, and recovery after a server restart. The one-turn `READY` test establishes that Hermes can obtain a completion from the local weights; it does not measure those capabilities.
