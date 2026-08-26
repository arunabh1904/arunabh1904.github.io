#!/usr/bin/env python3
"""Audit generated Blog narration against its source with local ASR.

Run from the repository root:

    uv run --with mlx-whisper --with jiwer \
      python scripts/audit-blog-audio.py --post <post-slug>

The audit is deliberately stricter about multi-word insertions than ordinary
word error rate. A plausible proper-name transcription can raise WER, while a
fluent clause invented by the speech model is a release-blocking artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate-blog-audio.py"
DEFAULT_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MAX_WER = 0.08
DEFAULT_MAX_INSERTION_WORDS = 2
DEFAULT_MAX_CHUNK_WER = 0.50


def load_generator() -> Any:
    """Load the hyphenated generator script as an importable module."""

    spec = importlib.util.spec_from_file_location("blog_audio_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_transcript(text: str) -> str:
    """Return a punctuation-insensitive, word-aligned ASR comparison string."""

    words = re.findall(r"[a-z0-9]+", text.casefold())
    normalized: list[str] = []
    index = 0
    while index < len(words):
        end = index
        while end < len(words) and len(words[end]) == 1 and words[end].isalpha():
            end += 1
        run_length = end - index
        # Whisper may spell an acronym that the source writes as one token.
        # Do not collapse long runs: those are often audible decoder loops.
        if 2 <= run_length <= 8:
            normalized.append("".join(words[index:end]))
            index = end
            continue
        normalized.append(words[index])
        index += 1
    return " ".join(normalized)


def audit_post(
    post: dict[str, Any],
    *,
    asr_model: str,
    max_wer: float,
    max_insertion_words: int,
    max_chunk_wer: float,
) -> dict[str, Any]:
    """Transcribe cached synthesis chunks and return release-gate metrics."""

    import jiwer
    import mlx_whisper

    chunk_cache = (
        Path(tempfile.gettempdir()) / "blog-audio-chunks" / post["digest"]
    )
    chunk_paths = [
        chunk_cache / f"chunk-{index:03d}.wav"
        for index in range(len(post["narration_chunks"]))
    ]
    missing = [str(path) for path in chunk_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing synthesis chunks; generate this post before auditing:\n"
            + "\n".join(missing)
        )

    references: list[str] = []
    hypotheses: list[str] = []
    long_insertions: list[dict[str, Any]] = []
    high_error_chunks: list[dict[str, Any]] = []
    for index, (chunk, chunk_path) in enumerate(
        zip(post["narration_chunks"], chunk_paths)
    ):
        transcription = mlx_whisper.transcribe(
            str(chunk_path),
            path_or_hf_repo=asr_model,
            language="en",
        )
        reference = normalize_transcript(chunk.text)
        hypothesis = normalize_transcript(transcription["text"])
        references.append(reference)
        hypotheses.append(hypothesis)
        chunk_comparison = jiwer.process_words(reference, hypothesis)
        hypothesis_words = chunk_comparison.hypotheses[0]
        for alignment in chunk_comparison.alignments[0]:
            if alignment.type != "insert":
                continue
            inserted_words = hypothesis_words[
                alignment.hyp_start_idx : alignment.hyp_end_idx
            ]
            if len(inserted_words) > max_insertion_words:
                long_insertions.append(
                    {
                        "chunk": index,
                        "word_count": len(inserted_words),
                        "text": " ".join(inserted_words),
                    }
                )
        if chunk_comparison.wer > max_chunk_wer:
            high_error_chunks.append(
                {
                    "chunk": index,
                    "wer": round(chunk_comparison.wer, 6),
                    "expected": chunk.text,
                    "recognized": transcription["text"].strip(),
                }
            )

    comparison = jiwer.process_words(references, hypotheses)
    passed = (
        comparison.wer <= max_wer
        and not long_insertions
        and not high_error_chunks
    )
    return {
        "slug": post["slug"],
        "audio": str(post["output"].relative_to(ROOT)),
        "asr_model": asr_model,
        "audited_chunks": len(chunk_paths),
        "reference_words": sum(len(words) for words in comparison.references),
        "recognized_words": sum(len(words) for words in comparison.hypotheses),
        "wer": round(comparison.wer, 6),
        "substitutions": comparison.substitutions,
        "deletions": comparison.deletions,
        "insertions": comparison.insertions,
        "long_insertions": long_insertions,
        "high_error_chunks": high_error_chunks,
        "limits": {
            "max_wer": max_wer,
            "max_insertion_words": max_insertion_words,
            "max_chunk_wer": max_chunk_wer,
        },
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post",
        action="append",
        required=True,
        help="Blog postSlug to audit; repeat for multiple posts.",
    )
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--max-wer", type=float, default=DEFAULT_MAX_WER)
    parser.add_argument(
        "--max-insertion-words",
        type=int,
        default=DEFAULT_MAX_INSERTION_WORDS,
    )
    parser.add_argument("--max-chunk-wer", type=float, default=DEFAULT_MAX_CHUNK_WER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = load_generator()
    posts_by_slug = {post["slug"]: post for post in generator.discover_posts()}
    missing = [slug for slug in args.post if slug not in posts_by_slug]
    if missing:
        raise ValueError(f"Unknown Blog postSlug(s): {', '.join(missing)}")

    reports = [
        audit_post(
            posts_by_slug[slug],
            asr_model=args.asr_model,
            max_wer=args.max_wer,
            max_insertion_words=args.max_insertion_words,
            max_chunk_wer=args.max_chunk_wer,
        )
        for slug in args.post
    ]
    print(json.dumps(reports, indent=2))
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
