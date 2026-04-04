#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


COMMON_TOKENIZER = "google/gemma-4-E4B-it"
SYSTEM_PROMPT = "You are a concise assistant. Do not use markdown. Answer directly."
OLLAMA_HOST = "127.0.0.1:11435"
OLLAMA_URL = f"http://{OLLAMA_HOST}"
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 18080
LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
REQUEST_TIMEOUT = 60 * 60

CORPUS = """
Running Gemma 4 locally on a 64 GB MacBook Pro is really a question about two
separate bottlenecks. The first is whether the weights fit in memory. The
second is whether the runtime still feels fast enough to use once the prompt
gets long and the KV cache starts growing. The interesting decision is not just
which model loads, but which one feels snappy enough to become a daily model.

Gemma 4 offers four practical sizes for local experiments on Apple Silicon: E2B,
E4B, 26B A4B, and 31B. The small models are easier to fit and easier to serve,
but the larger models are meaningfully stronger on reasoning and coding tasks.
The 26B A4B model is especially interesting because it is a mixture-of-experts
model with only a small active parameter set per generated token, even though
all parameters still need to be resident for fast routing.

For a laptop benchmark, the honest question is not whether a model can run at
all. It is whether the prompt processing feels reasonable, whether time to first
token stays humane, and whether decode throughput is high enough for interactive
work. Long context is attractive in a model card, but every extra token shows up
in latency. That means a runtime comparison should separate short-prompt
interactive behavior from longer prompt stress tests.

Llama.cpp, MLX, and Ollama all represent viable local paths on Apple Silicon.
Llama.cpp gives direct control and tends to be the first stop for squeezing out
raw performance from GGUF releases. MLX is the most Apple-native path, which can
matter on newer Macs where the framework keeps getting faster. Ollama is the
convenience layer: easy model tags, easy local API, and a simple integration
story, but not automatically the fastest stack.

When the goal is to benchmark runtimes rather than model quality, the output
task should be easy and repeatable. That way, differences in throughput mostly
reflect inference behavior instead of chain-of-thought length, reasoning depth,
or response verbosity. It is also useful to run a longer prompt variant, because
prompt processing often dominates the experience long before generation speed
becomes the main bottleneck.
""".strip()


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    official_id: str
    llama_repo: str
    mlx_repo: str
    ollama_tag: str
    llama_quant: str
    llama_file: str


MODELS: dict[str, ModelSpec] = {
    "e2b": ModelSpec(
        slug="e2b",
        label="Gemma 4 E2B",
        official_id="google/gemma-4-E2B-it",
        llama_repo="ggml-org/gemma-4-E2B-it-GGUF",
        mlx_repo="mlx-community/gemma-4-e2b-it-4bit",
        ollama_tag="gemma4:e2b-it-q4_K_M",
        llama_quant="Q8_0",
        llama_file="gemma-4-e2b-it-Q8_0.gguf",
    ),
    "e4b": ModelSpec(
        slug="e4b",
        label="Gemma 4 E4B",
        official_id="google/gemma-4-E4B-it",
        llama_repo="ggml-org/gemma-4-E4B-it-GGUF",
        mlx_repo="mlx-community/gemma-4-e4b-it-4bit",
        ollama_tag="gemma4:e4b-it-q4_K_M",
        llama_quant="Q4_K_M",
        llama_file="gemma-4-e4b-it-Q4_K_M.gguf",
    ),
    "26b": ModelSpec(
        slug="26b",
        label="Gemma 4 26B A4B",
        official_id="google/gemma-4-26B-A4B-it",
        llama_repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
        mlx_repo="mlx-community/gemma-4-26b-a4b-it-4bit",
        ollama_tag="gemma4:26b-a4b-it-q4_K_M",
        llama_quant="Q4_K_M",
        llama_file="gemma-4-26B-A4B-it-Q4_K_M.gguf",
    ),
    "31b": ModelSpec(
        slug="31b",
        label="Gemma 4 31B",
        official_id="google/gemma-4-31B-it",
        llama_repo="ggml-org/gemma-4-31B-it-GGUF",
        mlx_repo="mlx-community/gemma-4-31b-it-4bit",
        ollama_tag="gemma4:31b-it-q4_K_M",
        llama_quant="Q4_K_M",
        llama_file="gemma-4-31B-it-Q4_K_M.gguf",
    ),
}

SUITES = {
    "short": {"input_tokens": 512, "max_new_tokens": 192},
    "long": {"input_tokens": 8192, "max_new_tokens": 96},
}


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
    return session


def get_tokenizer():
    log(f"Loading tokenizer: {COMMON_TOKENIZER}")
    return AutoTokenizer.from_pretrained(COMMON_TOKENIZER)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_background(tokenizer, target_tokens: int) -> tuple[str, int]:
    corpus_tokens = tokenizer.encode(CORPUS, add_special_tokens=False)
    if target_tokens <= 0:
        return "", 0
    repeats = (target_tokens // len(corpus_tokens)) + 2
    combined_tokens = (corpus_tokens * repeats)[:target_tokens]
    background = tokenizer.decode(combined_tokens, skip_special_tokens=True)
    actual = count_tokens(tokenizer, background)
    return background, actual


def build_user_prompt(tokenizer, target_tokens: int, lines: int) -> tuple[str, int]:
    instruction = (
        f"Read the background notes below. Then print the integers 1 through {lines}, "
        "one per line, zero-padded to three digits, and nothing else.\n\n"
        "Background notes:\n"
    )
    instruction_tokens = count_tokens(tokenizer, instruction)
    background, background_tokens = build_background(tokenizer, max(target_tokens - instruction_tokens, 1))
    prompt = f"{instruction}{background}"
    actual_tokens = count_tokens(tokenizer, prompt)
    return prompt, actual_tokens


def cleanup_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        proc.wait(timeout=20)


def ensure_local_gguf(spec: ModelSpec) -> str:
    log(f"Downloading GGUF artifact from {spec.llama_repo}/{spec.llama_file}")
    return hf_hub_download(repo_id=spec.llama_repo, filename=spec.llama_file)


def wait_for_http(url: str, attempts: int = 900, delay_s: float = 2.0) -> None:
    session = make_session()
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = session.get(url, timeout=10)
            if response.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(delay_s)
    raise RuntimeError(f"Timed out waiting for {url}") from last_error


def llama_request(session: requests.Session, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": "gemma4",
        "messages": messages,
        "stream": True,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "seed": 123,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    response = session.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    first_token_at: float | None = None
    pieces: list[str] = []

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        if not raw_line.startswith(b"data: "):
            continue
        chunk = raw_line[6:].decode("utf-8")
        if chunk == "[DONE]":
            break
        data = json.loads(chunk)
        delta = data["choices"][0].get("delta", {}).get("content", "")
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(delta)

    elapsed = time.perf_counter() - start
    ttft = None if first_token_at is None else first_token_at - start
    return {
        "text": "".join(pieces),
        "elapsed_s": elapsed,
        "ttft_s": ttft,
    }


def start_llama_server(spec: ModelSpec, ctx_size: int, model_path: str) -> tuple[subprocess.Popen[Any], Path]:
    log(f"Starting llama.cpp server for {spec.label} ({spec.llama_quant})")
    log_path = Path("/private/tmp") / f"llama-server-{spec.slug}.log"
    log_file = log_path.open("w")
    cmd = [
        "llama-server",
        "--model",
        model_path,
        "--no-mmproj",
        "--chat-template-kwargs",
        '{"enable_thinking": false}',
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        str(ctx_size),
        "--flash-attn",
        "on",
        "--metrics",
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        wait_for_http(f"{LLAMA_URL}/health")
    except Exception:
        cleanup_process(proc)
        raise
    return proc, log_path


def benchmark_llama(spec: ModelSpec, tokenizer, suites: list[str]) -> dict[str, Any]:
    ctx_size = 16384
    model_path = ensure_local_gguf(spec)
    proc, log_path = start_llama_server(spec, ctx_size=ctx_size, model_path=model_path)
    session = make_session()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Print OK and nothing else."},
    ]
    try:
        log(f"Warming up llama.cpp for {spec.label}")
        llama_request(session, messages, max_tokens=8)

        suite_results: dict[str, Any] = {}
        for suite_name in suites:
            config = SUITES[suite_name]
            prompt, prompt_tokens = build_user_prompt(
                tokenizer,
                target_tokens=config["input_tokens"],
                lines=config["max_new_tokens"],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            log(f"Measuring llama.cpp {spec.label} [{suite_name}]")
            result = llama_request(session, messages, max_tokens=config["max_new_tokens"])
            output_tokens = count_tokens(tokenizer, result["text"])
            decode_window = None
            decode_tps = None
            if result["ttft_s"] is not None and result["elapsed_s"] > result["ttft_s"]:
                decode_window = result["elapsed_s"] - result["ttft_s"]
                if output_tokens > 0:
                    decode_tps = output_tokens / decode_window
            avg_tps = output_tokens / result["elapsed_s"] if output_tokens > 0 else None
            suite_results[suite_name] = {
                "target_input_tokens": config["input_tokens"],
                "actual_input_tokens": prompt_tokens,
                "max_new_tokens": config["max_new_tokens"],
                "output_tokens": output_tokens,
                "ttft_ms": None if result["ttft_s"] is None else round(result["ttft_s"] * 1000, 2),
                "elapsed_ms": round(result["elapsed_s"] * 1000, 2),
                "decode_toks_per_s": None if decode_tps is None else round(decode_tps, 2),
                "avg_toks_per_s": None if avg_tps is None else round(avg_tps, 2),
            }

        return {
            "runtime": "llama.cpp",
            "runtime_artifact": model_path,
            "notes": "Text-only benchmark using llama-server with flash attention on and no multimodal projector.",
            "server_log": str(log_path),
            "suites": suite_results,
        }
    finally:
        cleanup_process(proc)


def ensure_ollama_daemon() -> tuple[subprocess.Popen[Any] | None, bool]:
    session = make_session()
    try:
        response = session.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.ok:
            return None, False
    except Exception:  # noqa: BLE001
        pass

    flash_attention = os.environ.get("OLLAMA_FLASH_ATTENTION")
    if flash_attention:
        log(f"Starting dedicated Ollama daemon with OLLAMA_FLASH_ATTENTION={flash_attention}")
    else:
        log("Starting dedicated Ollama daemon with default settings")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    log_path = Path("/private/tmp") / "ollama-gemma4-bench.log"
    log_file = log_path.open("w")
    proc = subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=subprocess.STDOUT, env=env)
    wait_for_http(f"{OLLAMA_URL}/api/tags")
    return proc, True


def ollama_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    return subprocess.run(
        ["ollama", *args],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_ollama_model(spec: ModelSpec) -> str:
    log(f"Pulling Ollama model {spec.ollama_tag}")
    ollama_cmd("pull", spec.ollama_tag)
    return spec.ollama_tag


def ollama_request(session: requests.Session, model_tag: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "keep_alive": "15m",
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 123,
            "num_predict": max_tokens,
        },
    }
    start = time.perf_counter()
    response = session.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    first_token_at: float | None = None
    pieces: list[str] = []
    final_chunk: dict[str, Any] | None = None

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        data = json.loads(raw_line.decode("utf-8"))
        content = data.get("response", "")
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(content)
        if data.get("done"):
            final_chunk = data
            break

    elapsed = time.perf_counter() - start
    ttft = None if first_token_at is None else first_token_at - start
    return {
        "text": "".join(pieces),
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "final_chunk": final_chunk or {},
    }


def benchmark_ollama(spec: ModelSpec, tokenizer, suites: list[str]) -> dict[str, Any]:
    daemon_proc, started_here = ensure_ollama_daemon()
    session = make_session()
    model_name = ensure_ollama_model(spec)
    try:
        log(f"Warming up Ollama for {spec.label}")
        warm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Print OK and nothing else."},
        ]
        warm_prompt = tokenizer.apply_chat_template(
            warm_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ollama_request(session, model_name, warm_prompt, max_tokens=8)

        suite_results: dict[str, Any] = {}
        for suite_name in suites:
            config = SUITES[suite_name]
            prompt, prompt_tokens = build_user_prompt(
                tokenizer,
                target_tokens=config["input_tokens"],
                lines=config["max_new_tokens"],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            log(f"Measuring Ollama {spec.label} [{suite_name}]")
            result = ollama_request(session, model_name, rendered_prompt, max_tokens=config["max_new_tokens"])
            output_tokens = count_tokens(tokenizer, result["text"])
            decode_window = None
            decode_tps = None
            if result["ttft_s"] is not None and result["elapsed_s"] > result["ttft_s"]:
                decode_window = result["elapsed_s"] - result["ttft_s"]
                if output_tokens > 0:
                    decode_tps = output_tokens / decode_window
            avg_tps = output_tokens / result["elapsed_s"] if output_tokens > 0 else None
            final_chunk = result["final_chunk"]
            suite_results[suite_name] = {
                "target_input_tokens": config["input_tokens"],
                "actual_input_tokens": prompt_tokens,
                "max_new_tokens": config["max_new_tokens"],
                "output_tokens": output_tokens,
                "ttft_ms": None if result["ttft_s"] is None else round(result["ttft_s"] * 1000, 2),
                "elapsed_ms": round(result["elapsed_s"] * 1000, 2),
                "decode_toks_per_s": None if decode_tps is None else round(decode_tps, 2),
                "avg_toks_per_s": None if avg_tps is None else round(avg_tps, 2),
                "ollama_prompt_eval_count": final_chunk.get("prompt_eval_count"),
                "ollama_prompt_eval_duration_ms": (
                    None
                    if final_chunk.get("prompt_eval_duration") is None
                    else round(final_chunk["prompt_eval_duration"] / 1_000_000, 2)
                ),
                "ollama_eval_count": final_chunk.get("eval_count"),
                "ollama_eval_duration_ms": (
                    None
                    if final_chunk.get("eval_duration") is None
                    else round(final_chunk["eval_duration"] / 1_000_000, 2)
                ),
            }

        details = ollama_cmd("show", model_name, "--modelfile").stdout
        flash_attention = os.environ.get("OLLAMA_FLASH_ATTENTION")
        runtime_note = "Text-only benchmark using a dedicated Ollama daemon and a local GGUF import."
        if flash_attention:
            runtime_note = (
                "Text-only benchmark using a dedicated Ollama daemon, "
                f"OLLAMA_FLASH_ATTENTION={flash_attention}, and a local GGUF import."
            )
        return {
            "runtime": "ollama",
            "runtime_artifact": model_name,
            "notes": f"{runtime_note} Native Ollama tag benchmark, not a local GGUF import.",
            "ollama_modelfile": details,
            "suites": suite_results,
        }
    finally:
        if started_here:
            cleanup_process(daemon_proc)


def benchmark_mlx(spec: ModelSpec, tokenizer, suites: list[str]) -> dict[str, Any]:
    log(f"Loading MLX model {spec.mlx_repo}")
    model, model_tokenizer = load(spec.mlx_repo)
    sampler = make_sampler(temp=0.0, top_k=1, top_p=1.0)
    try:
        warm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Print OK and nothing else."},
        ]
        warm_prompt = model_tokenizer.apply_chat_template(
            warm_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        log(f"Warming up MLX for {spec.label}")
        for _ in stream_generate(model, model_tokenizer, warm_prompt, max_tokens=8, sampler=sampler):
            pass

        suite_results: dict[str, Any] = {}
        for suite_name in suites:
            config = SUITES[suite_name]
            prompt, prompt_tokens = build_user_prompt(
                tokenizer,
                target_tokens=config["input_tokens"],
                lines=config["max_new_tokens"],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            rendered_prompt = model_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            log(f"Measuring MLX {spec.label} [{suite_name}]")
            start = time.perf_counter()
            first_token_at: float | None = None
            pieces: list[str] = []
            for chunk in stream_generate(
                model,
                model_tokenizer,
                rendered_prompt,
                max_tokens=config["max_new_tokens"],
                sampler=sampler,
            ):
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(chunk.text)
            elapsed = time.perf_counter() - start
            ttft = None if first_token_at is None else first_token_at - start
            text = "".join(pieces)
            output_tokens = count_tokens(tokenizer, text)
            decode_window = None
            decode_tps = None
            if ttft is not None and elapsed > ttft:
                decode_window = elapsed - ttft
                if output_tokens > 0:
                    decode_tps = output_tokens / decode_window
            avg_tps = output_tokens / elapsed if output_tokens > 0 else None
            suite_results[suite_name] = {
                "target_input_tokens": config["input_tokens"],
                "actual_input_tokens": prompt_tokens,
                "max_new_tokens": config["max_new_tokens"],
                "output_tokens": output_tokens,
                "ttft_ms": None if ttft is None else round(ttft * 1000, 2),
                "elapsed_ms": round(elapsed * 1000, 2),
                "decode_toks_per_s": None if decode_tps is None else round(decode_tps, 2),
                "avg_toks_per_s": None if avg_tps is None else round(avg_tps, 2),
            }

        return {
            "runtime": "mlx-lm",
            "runtime_artifact": spec.mlx_repo,
            "notes": "Text-only benchmark using mlx-lm Python API and 4-bit MLX community weights.",
            "suites": suite_results,
        }
    finally:
        del model
        del model_tokenizer
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 local runtimes on Apple Silicon.")
    parser.add_argument("--runtime", choices=["llama", "mlx", "ollama"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS.keys()), required=True)
    parser.add_argument(
        "--suites",
        default="short,long",
        help="Comma-separated list drawn from: short,long",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_names = [name.strip() for name in args.suites.split(",") if name.strip()]
    unknown = [name for name in suite_names if name not in SUITES]
    if unknown:
        raise SystemExit(f"Unknown suites: {', '.join(unknown)}")

    spec = MODELS[args.model]
    tokenizer = get_tokenizer()

    if args.runtime == "llama":
        result = benchmark_llama(spec, tokenizer, suite_names)
    elif args.runtime == "mlx":
        result = benchmark_mlx(spec, tokenizer, suite_names)
    else:
        result = benchmark_ollama(spec, tokenizer, suite_names)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "hostname": os.uname().nodename,
            "platform": sys.platform,
            "cpu": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
            "memory_gb": round(
                int(
                    subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        check=True,
                        text=True,
                        capture_output=True,
                    ).stdout.strip()
                )
                / (1024**3),
                1,
            ),
        },
        "model": {
            "slug": spec.slug,
            "label": spec.label,
            "official_id": spec.official_id,
        },
        **result,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
