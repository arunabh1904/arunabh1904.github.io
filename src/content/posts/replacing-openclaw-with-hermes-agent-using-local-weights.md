---
title: Replacing OpenClaw with Hermes Agent using local weights
date: '2026-04-04T17:59:45.000Z'
section: blog
postSlug: replacing-openclaw-with-hermes-agent-using-local-weights
legacyPath: /blog/2026/04/04/replacing-openclaw-with-hermes-agent-using-local-weights.html
tags:
  - Agents
  - LLMs
  - Apple Silicon
summary: >-
  How I switched from OpenClaw to Hermes Agent, kept inference fully local, and
  got it working with an already-downloaded Gemma GGUF instead of pulling new
  models.
---
# Replacing OpenClaw with Hermes Agent using local weights

I wanted a pretty specific outcome: stop using OpenClaw, switch over to [Hermes Agent](https://github.com/nousresearch/hermes-agent), and keep the whole thing local.

Not "local except for the model."

Actually local.

That meant two constraints:

1. I did not want to fall back to OpenRouter, Anthropic, or anything else cloud-hosted.
2. I only wanted to use model artifacts that were already on disk.

On this machine, the artifacts I could verify quickly were my local Gemma ones, so that is the path I ended up getting working end to end. I did not see my Qwen artifacts in the usual local cache locations during setup, but the same `llama.cpp` pattern should apply to local Qwen GGUFs too.

## Why I wanted Hermes instead

Hermes is a much more opinionated agent shell than a bare local chat loop. It has the things I actually care about when I say "agent" instead of "chatbot":

- tool use
- filesystem access
- terminal execution
- sessions
- skills
- multiple provider backends

What I liked immediately is that Hermes does not force one inference path. It is perfectly happy with hosted providers, but it also lets me point it at any OpenAI-compatible local endpoint. That makes it much easier to keep the agent framework and swap the model runtime underneath it.

That separation ended up mattering a lot.

## The install itself was easy

The Hermes install was not the hard part. This worked fine:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
```

That bootstrapped:

- `uv`
- Python `3.11`
- the Hermes repo under `~/.hermes/hermes-agent`
- the `hermes` CLI symlink in `~/.local/bin`
- the default config in `~/.hermes/config.yaml`

So far, so good.

The real question was: what local model server should Hermes talk to?

## Ollama was the obvious first try, but it was the wrong one here

Since I already had Ollama installed and local Gemma tags visible, I tried the most obvious route first.

That looked promising for about five minutes.

Hermes could see the local endpoint. Ollama listed local Gemma models. But actual inference failed with HTTP `500` during model load on this Apple Silicon setup. In other words, the Hermes install was fine, but the runtime below it was not stable enough for the job.

That was the key realization:

- Hermes was not the problem.
- My local model server choice was the problem.

Once I stopped treating those two things as the same layer, the path got much cleaner.

## What actually worked: `llama-server` plus an existing Gemma GGUF

The machine already had local Gemma GGUF artifacts in the Hugging Face cache, including:

```text
~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/...
```

And `llama-server` was already installed via Homebrew.

That turned out to be the cleanest fully local setup.

I started `llama-server` directly against the cached GGUF:

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

A few details here mattered:

- `--ctx-size 32768` was necessary because Hermes sends a large system prompt and `8192` was not enough.
- `--parallel 1` kept the memory footprint reasonable while still leaving enough room for the larger context window.
- `--reasoning off` matched what I wanted anyway: no extra thinking overhead for a local smoke test.

Once that server was up, it exposed the OpenAI-compatible endpoint Hermes wanted at:

```text
http://127.0.0.1:18080/v1
```

That was the turning point.

## The Hermes config I ended up using

I pointed Hermes at the local `llama-server` endpoint by editing `~/.hermes/config.yaml` to this:

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

## The actual proof that it was running

The test I cared about was extremely boring on purpose:

```bash
hermes chat -q 'Reply with exactly READY and nothing else.' -Q --max-turns 1
```

And it returned:

```text
READY
```

That was enough.

At that point Hermes was no longer "installed." It was actually running, locally, against weights that were already on disk.

## What I would do next

This setup is already useful, but there are a few obvious next steps:

- keep `llama-server` running behind a LaunchAgent or small wrapper script
- point Hermes at a stronger local model if I want better tool-use quality
- wire in the local Qwen artifacts the same way, once I decide which exact GGUF or local server path I want to standardize on

The important part is that the architecture is now right:

- Hermes for the agent layer
- `llama.cpp` for the local serving layer
- existing local weights for inference

That is a much cleaner split than trying to make one tool do all three jobs at once.

And honestly, that was the real unlock here.
