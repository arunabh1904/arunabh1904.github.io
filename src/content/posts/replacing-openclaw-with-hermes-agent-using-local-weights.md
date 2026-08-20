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
  How to run Hermes Agent locally with llama.cpp and an existing GGUF.
---
# Running Hermes Agent with a Local GGUF

I wanted a specific outcome: replace OpenClaw with [Hermes Agent](https://github.com/nousresearch/hermes-agent) while keeping inference fully local. That meant two constraints:

1. I did not want to fall back to OpenRouter, Anthropic, or anything else cloud-hosted.
2. I only wanted to use model artifacts that were already on disk.

On this machine, the artifacts I could verify quickly were local Gemma GGUFs, so that is the path I got working end to end. I did not see my Qwen artifacts in the usual cache locations during setup, but the same `llama.cpp` pattern should apply to local Qwen GGUFs too.

## Separate the agent from inference

Hermes is a much more opinionated agent shell than a bare local chat loop. It has the things I actually care about when I say "agent" instead of "chatbot":

- tool use
- filesystem access
- terminal execution
- sessions
- skills
- multiple provider backends

Hermes does not force one inference path. It works with hosted providers, but it can also point at any OpenAI-compatible local endpoint. That separation let the agent framework stay fixed while the model runtime changed underneath it.

The figure shows the boundary that resolved the setup. A prompt does not travel from Hermes directly into a weight file. Hermes calls an HTTP API; the serving process owns model loading, the KV cache, and token generation; the GGUF is inert model data on disk.

[![Animation showing Hermes Agent calling a localhost OpenAI-compatible endpoint backed by llama-server and an on-disk GGUF](/assets/images/blog-hermes-local-stack.gif)](/assets/images/blog-hermes-local-stack.gif)

*Hermes owns the agent loop, tools, sessions, and skills. The custom endpoint is the interface. `llama-server` owns inference, and the GGUF supplies weights and tokenizer data. A model-load error below the API boundary can therefore be fixed without replacing the agent shell. Custom explanatory diagram, checked against the current [Hermes provider documentation](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md).*

This separation also changes how to debug. If Hermes cannot reach `/v1/chat/completions`, inspect the endpoint and configuration. If the endpoint returns HTTP `500` while loading a model, inspect the runtime, artifact, and hardware path. If text is generated but tools appear as plain text, inspect the server's chat template and tool-call support. Treating those as three different contracts avoids reinstalling the wrong layer.

## Installing Hermes Agent

The Hermes install was not the hard part:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
```

That bootstrapped:

- `uv`
- Python `3.11`
- the Hermes repo under `~/.hermes/hermes-agent`
- the `hermes` CLI symlink in `~/.local/bin`
- the default config in `~/.hermes/config.yaml`

The real question was what local model server Hermes should talk to.

## Why Ollama did not work

Since I already had Ollama installed and local Gemma tags visible, I tried the most obvious route first.

Hermes could see the local endpoint. Ollama listed local Gemma models. But actual inference failed with HTTP `500` during model load on this Apple Silicon setup. In other words, the Hermes install was fine, but the runtime below it was not stable enough for the job.

The key realization: Hermes was not the problem. My local model server choice was.

## `llama-server` with a local Gemma GGUF

The machine already had local Gemma GGUF artifacts in the Hugging Face cache, including:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/...
```

`llama-server` was already installed via Homebrew, and it turned out to be the cleanest fully local setup.

I started `llama-server` directly against the cached GGUF. This command records the setup that worked on April 4, 2026:

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

The context value is the main dated part of this recipe. Current Hermes documentation requires at least `64,000` tokens for agent use with tools because the system prompt, schemas, and working conversation already consume substantial context. A new setup should therefore size both Hermes and `llama-server` consistently—typically `65536` or higher if the model and available memory support it—instead of copying the older `32768` value blindly.

Once that server was up, it exposed the OpenAI-compatible endpoint Hermes wanted at:

```text
http://127.0.0.1:18080/v1
```

That split worked: Hermes stayed as the agent shell, and `llama.cpp` handled local serving.

## Point Hermes at the local API

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

## Verification

The test I cared about was extremely boring on purpose:

```bash
hermes chat -q 'Reply with exactly READY and nothing else.' -Q --max-turns 1
```

And it returned:

```text
READY
```

That was enough proof: Hermes was running locally against weights already on disk.

## What the test establishes

The `READY` response proves the inference path, not full agent quality. A useful next pass should separately test model loading, ordinary chat, structured tool calls, long-context behavior, and recovery when the local server restarts. Those checks locate regressions at the same boundaries shown in the figure.

The remaining operational work is straightforward:

- keep `llama-server` running behind a LaunchAgent or small wrapper script
- point Hermes at a stronger local model if I want better tool-use quality
- wire in the local Qwen artifacts the same way, once I decide which exact GGUF or local server path I want to standardize on

The architecture is now clean:

- Hermes for the agent layer
- `llama.cpp` for the local serving layer
- existing local weights for inference

> **Deep insight.** Separating the agent layer, serving layer, and model artifacts prevents one tool's lifecycle from determining all three. It makes an agent replacement, runtime change, or weight update independently testable.
