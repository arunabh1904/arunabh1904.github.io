#!/usr/bin/env python3

"""Benchmark one local GGUF through llama-server on short and long prompts.

The harness reports both time to the first generated token and time to the
first answer token. Those differ for models that stream hidden reasoning before
visible content. It intentionally measures runtime behavior, not answer quality.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOST = "127.0.0.1"
PORT = 18081
BASE_URL = f"http://{HOST}:{PORT}"
REQUEST_TIMEOUT_S = 60 * 60

CORPUS = """
Running a language model locally is a question about capacity and latency. The
weights must fit in memory, but that only establishes that inference can start.
Prompt processing determines how long a document-scale request waits before the
first generated token, and decode throughput determines whether the continuation
feels interactive. A useful benchmark therefore holds the output task fixed and
changes prompt length rather than mixing model quality with runtime behavior.

Apple Silicon exposes one unified memory pool to the CPU and GPU. That removes a
separate host-to-device copy, but the operating system, model weights, runtime
workspace, and key-value cache still compete for the same finite capacity. A
quantized checkpoint that occupies twenty gigabytes has very different headroom
from a sixty-gigabyte checkpoint on a machine with sixty-four gigabytes total.

Serving adds another distinction. A command-line generation can prove that a
model loads, while an OpenAI-compatible server tests the interface used by chat
applications and agent frameworks. For daily use, time to first visible answer
can matter more than time to the first hidden reasoning token.
""".strip()

SUITES = {
    "short": {"input_tokens": 512, "max_new_tokens": 192},
    "long": {"input_tokens": 8192, "max_new_tokens": 96},
}


@dataclass(frozen=True)
class Measurement:
    target_input_tokens: int
    actual_input_tokens: int
    max_new_tokens: int
    completion_tokens: int | None
    reasoning_characters: int
    answer_characters: int
    reasoning_preview: str
    answer_preview: str
    first_generated_token_ms: float | None
    first_answer_token_ms: float | None
    elapsed_ms: float
    decode_tokens_per_s: float | None
    average_tokens_per_s: float | None


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen[str], attempts: int = 900) -> None:
    for _ in range(attempts):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        try:
            request_json("/health")
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
    raise RuntimeError("Timed out waiting for llama-server")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def ensure_port_is_free(host: str, port: int) -> None:
    """Fail before launch instead of benchmarking an unrelated stale server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(f"Refusing to use occupied benchmark port {host}:{port}")


def token_count(text: str) -> int:
    response = request_json("/tokenize", {"content": text, "add_special": False})
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"Unexpected /tokenize response: {response}")
    return len(tokens)


def build_prompt(target_tokens: int, lines: int) -> tuple[str, int]:
    instruction = (
        f"Read the background notes. Then print the integers 1 through {lines}, "
        "one per line, zero-padded to three digits, and nothing else.\n\n"
        "Background notes:\n"
    )
    corpus_words = CORPUS.split()
    low = 1
    high = max(target_tokens * 2, len(corpus_words))
    best_prompt = instruction
    best_count = token_count(best_prompt)

    while low <= high:
        word_count = (low + high) // 2
        repeated = [corpus_words[index % len(corpus_words)] for index in range(word_count)]
        candidate = f"{instruction}{' '.join(repeated)}"
        candidate_count = token_count(candidate)
        if abs(candidate_count - target_tokens) < abs(best_count - target_tokens):
            best_prompt = candidate
            best_count = candidate_count
        if candidate_count < target_tokens:
            low = word_count + 1
        elif candidate_count > target_tokens:
            high = word_count - 1
        else:
            break

    return best_prompt, best_count


def stream_chat(
    model_alias: str,
    prompt: str,
    max_tokens: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Measurement:
    payload = {
        "model": model_alias,
        "messages": [
            {
                "role": "system",
                "content": "Be concise. Follow the requested output format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": 123,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    first_generated_at: float | None = None
    first_answer_at: float | None = None
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    completion_tokens: int | None = None

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            event = json.loads(chunk)
            usage = event.get("usage") or {}
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            if reasoning or content:
                first_generated_at = first_generated_at or time.perf_counter()
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                first_answer_at = first_answer_at or time.perf_counter()
                answer_parts.append(content)

    finished_at = time.perf_counter()
    elapsed_s = finished_at - started_at
    decode_s = None if first_generated_at is None else finished_at - first_generated_at
    decode_tps = None
    average_tps = None
    if completion_tokens is not None and completion_tokens > 0:
        average_tps = completion_tokens / elapsed_s
        if decode_s is not None and decode_s > 0:
            decode_tps = completion_tokens / decode_s

    return Measurement(
        target_input_tokens=0,
        actual_input_tokens=0,
        max_new_tokens=max_tokens,
        completion_tokens=completion_tokens,
        reasoning_characters=len("".join(reasoning_parts)),
        answer_characters=len("".join(answer_parts)),
        reasoning_preview="".join(reasoning_parts)[:240],
        answer_preview="".join(answer_parts)[:240],
        first_generated_token_ms=(
            None if first_generated_at is None else round((first_generated_at - started_at) * 1000, 2)
        ),
        first_answer_token_ms=(
            None if first_answer_at is None else round((first_answer_at - started_at) * 1000, 2)
        ),
        elapsed_ms=round(elapsed_s * 1000, 2),
        decode_tokens_per_s=None if decode_tps is None else round(decode_tps, 2),
        average_tokens_per_s=None if average_tps is None else round(average_tps, 2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--draft-path",
        type=Path,
        help="Optional speculative-decoding draft model, such as Glimmer DFlash.",
    )
    parser.add_argument(
        "--spec-type",
        default="draft-dflash",
        help="llama.cpp speculative decoder type used when --draft-path is set.",
    )
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-alias", default="local-model")
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--reasoning-budget", type=int, default=32)
    parser.add_argument("--reasoning-strength", default="low")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--suites", default="short,long")
    return parser.parse_args()


def main() -> int:
    global BASE_URL

    args = parse_args()
    if not args.model_path.is_file():
        raise SystemExit(f"Model file does not exist: {args.model_path}")
    if args.draft_path is not None and not args.draft_path.is_file():
        raise SystemExit(f"Draft model file does not exist: {args.draft_path}")
    suite_names = [name.strip() for name in args.suites.split(",") if name.strip()]
    unknown_suites = [name for name in suite_names if name not in SUITES]
    if unknown_suites:
        raise SystemExit(f"Unknown suites: {', '.join(unknown_suites)}")
    if not 1 <= args.port <= 65535:
        raise SystemExit(f"Port must be between 1 and 65535: {args.port}")
    ensure_port_is_free(HOST, args.port)
    BASE_URL = f"http://{HOST}:{args.port}"

    log_path = Path("/private/tmp") / f"llama-server-{args.model_alias}.log"
    command = [
        "llama-server",
        "--model",
        str(args.model_path),
        "--alias",
        args.model_alias,
        "--host",
        HOST,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--parallel",
        "1",
        "--flash-attn",
        "on",
        "--gpu-layers",
        "all",
        "--jinja",
        "--metrics",
        "--reasoning-budget",
        str(args.reasoning_budget),
        "--chat-template-kwargs",
        json.dumps({"reasoning_strength": args.reasoning_strength}),
    ]
    if args.draft_path is not None:
        command.extend(
            [
                "--model-draft",
                str(args.draft_path),
                "--gpu-layers-draft",
                "all",
                "--spec-type",
                args.spec_type,
            ]
        )

    process: subprocess.Popen[str] | None = None
    try:
        log(f"Starting llama-server for {args.model_label}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wait_for_server(process)

        log("Warming up")
        stream_chat(
            args.model_alias,
            "Reply with OK and nothing else.",
            max_tokens=40,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        measurements: dict[str, dict[str, Any]] = {}
        for suite_name in suite_names:
            suite = SUITES[suite_name]
            prompt, input_tokens = build_prompt(suite["input_tokens"], suite["max_new_tokens"])
            log(f"Measuring {suite_name} suite with {input_tokens} input tokens")
            measurement = stream_chat(
                args.model_alias,
                prompt,
                suite["max_new_tokens"],
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            result = asdict(measurement)
            result["target_input_tokens"] = suite["input_tokens"]
            result["actual_input_tokens"] = input_tokens
            measurements[suite_name] = result

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "machine": {
                "platform": sys.platform,
                "cpu": subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip(),
                "memory_gib": round(
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
            "runtime": subprocess.run(
                ["llama-server", "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.strip(),
            "model": {
                "label": args.model_label,
                "alias": args.model_alias,
                "path": str(args.model_path),
                "size_bytes": args.model_path.stat().st_size,
                "draft_path": None if args.draft_path is None else str(args.draft_path),
                "draft_size_bytes": (
                    None if args.draft_path is None else args.draft_path.stat().st_size
                ),
            },
            "controls": {
                "ctx_size": args.ctx_size,
                "parallel": 1,
                "gpu_layers": "all",
                "flash_attention": True,
                "speculative_type": (
                    None if args.draft_path is None else args.spec_type
                ),
                "reasoning_budget": args.reasoning_budget,
                "reasoning_strength": args.reasoning_strength,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "server_log": str(log_path),
            "suites": measurements,
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
