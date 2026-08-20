#!/usr/bin/env python3
"""Generate local, static read-aloud audio for Blog posts.

The script intentionally keeps model execution out of the Astro build. Run it
on an Apple Silicon Mac, commit the generated MP3s, and let GitHub Pages serve
them as ordinary static assets.

Usage:
    uv run --with mlx-audio \
      python scripts/generate-blog-audio.py
    uv run --with mlx-audio \
      python scripts/generate-blog-audio.py --check

The model is loaded once per invocation. Existing files whose source hash and
voice settings are unchanged are skipped, so the command is safe to rerun
after adding or editing posts. The fixed synthetic ICL reference anchors one
narrator; low-temperature sampling with a stable per-chunk seed keeps runs
reproducible without relying on greedy decoding's unreliable long-form stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "src" / "content" / "posts"
AUDIO_DIR = ROOT / "public" / "assets" / "audio" / "blog"
MANIFEST_PATH = ROOT / "public" / "assets" / "audio" / "manifest.json"
MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
VOICE = "synthetic-clean-warm-icl"
LANGUAGE = "English"
LANGUAGE_CODE = "english"
# This is a deliberately synthetic narrator reference, not a recording of a person.
# The matching transcript is required by Qwen3-TTS in-context learning (ICL).
REFERENCE_AUDIO_PATH = ROOT / "scripts" / "audio-assets" / "clean-warm-synthetic-reference.wav"
REFERENCE_TEXT = (
    "I will guide you through technical ideas with calm confidence. "
    "The delivery is warm, clear, and measured."
)
TEMPERATURE = 0.05
TOP_P = 0.9
SAMPLING_SEED_VERSION = "per-post-chunk-v1"
# The ordinary ICL decode path reconstructs reference and generated codes together,
# then estimates a waveform cut.  That estimate can leave the reference phrase at
# the start of every article chunk.  The streaming decoder reconstructs generated
# codes only; the deliberately high interval still returns one final result here.
ICL_DECODE_MODE = "generated-only-streaming-v1"
ICL_STREAMING_INTERVAL = 120.0
SPEED = 1.0
BITRATE = "64k"
MAX_CHARS = 360  # Proven reliable bound; low-temperature seeded sampling prevents drift.
SILENCE_THRESHOLD = 0.008  # -42 dBFS; matches the verification threshold below.
BOUNDARY_SILENCE_SAMPLES = 1_200  # Retain 50 ms at 24 kHz, without a synthetic gap.
LONG_SILENCE_DURATION = 0.6
RETAINED_SILENCE_DURATION = 0.06
SILENCE_FILTER = (
    "silenceremove="
    f"start_periods=1:start_duration=0.05:start_threshold={SILENCE_THRESHOLD}:"
    "start_silence=0.05:"
    f"stop_periods=-1:stop_duration={LONG_SILENCE_DURATION}:"
    f"stop_threshold={SILENCE_THRESHOLD}:stop_silence={RETAINED_SILENCE_DURATION}"
)
EXTRACTION_VERSION = "markdown-prose-v5-narration"
DEFAULT_BATCH_SIZE = 1
MIN_GENERATION_TOKENS = 128
MAX_TOKENS_PER_CHAR = 1.05


def parse_frontmatter(source: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---\n", source, flags=re.DOTALL)
    if not match:
        return {}

    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip().strip("'\"")
    return values


def reference_audio_sha256() -> str:
    """Return a stable identity for the fixed synthetic narrator reference."""

    if not REFERENCE_AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Missing narrator reference: {REFERENCE_AUDIO_PATH}. "
            "Restore the committed synthetic reference before exporting audio."
        )
    return hashlib.sha256(REFERENCE_AUDIO_PATH.read_bytes()).hexdigest()


def _table_sentences(rows: list[str]) -> list[str]:
    """Convert a Markdown table into labelled sentences for natural narration."""

    parsed = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    parsed = [row for row in parsed if row and any(row)]
    if not parsed:
        return []

    has_header = len(parsed) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in parsed[1])
    headers = parsed[0] if has_header else []
    data_rows = parsed[2:] if has_header else parsed
    sentences: list[str] = []
    for row in data_rows:
        if headers:
            fields = [
                f"{headers[index]}: {value}"
                for index, value in enumerate(row)
                if value and index < len(headers) and headers[index]
            ]
            sentence = "; ".join(fields)
        else:
            sentence = "; ".join(value for value in row if value)
        if sentence:
            sentences.append(f"{sentence.rstrip('.')}.")
    return sentences


def clean_markdown(source: str) -> str:
    """Extract only coherent blog prose, excluding code, images, and captions.

    Headings and list items remain because they orient a listener. Tables are
    rewritten as labelled rows. Display equations, image alt text, figure
    captions, and the References bibliography are omitted because their raw
    Markdown/LaTeX forms do not make useful spoken content.
    """

    body = re.sub(r"\A---\n.*?\n---\n", "", source, count=1, flags=re.DOTALL)
    lines = body.splitlines()
    output: list[str] = []
    table_rows: list[str] = []
    in_fence = False
    in_display_math = False
    skip_caption = False
    skip_references = False

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            output.extend(_table_sentences(table_rows))
            table_rows = []

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith(("```", "~~~")):
            flush_table()
            in_fence = not in_fence
            continue
        if in_fence or skip_references:
            continue
        if line in {"$$", r"\[", r"\]"}:
            flush_table()
            in_display_math = not in_display_math if line != r"\]" else False
            continue
        if in_display_math:
            continue
        if not line:
            flush_table()
            continue
        if line.startswith(("import ", "export ")) or (line.startswith("<") and line.endswith(">")):
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            flush_table()
            heading_text = heading.group(1).strip()
            if heading_text.casefold() == "references":
                skip_references = True
                continue
            output.append(heading_text.rstrip(".!?") + ".")
            continue
        if (line.startswith("![") or line.startswith("[![")) and "](" in line:
            flush_table()
            skip_caption = True
            continue
        if skip_caption and re.fullmatch(r"(?:\*.*\*|_.*_)", line):
            skip_caption = False
            continue
        skip_caption = False
        if line.startswith("|"):
            table_rows.append(line)
            continue
        flush_table()
        if re.fullmatch(r"[-*_]{3,}", line):
            continue
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"\$\$.*?\$\$", "", line)
        line = re.sub(r"\$[^$\n]+\$", "", line)
        line = re.sub(r"\\\(.*?\\\)", "", line)
        line = re.sub(r"\\\[.*?\\\]", "", line)
        line = re.sub(r"(`{1,3}|\*\*|__|[*_~])", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            output.append(line)

    flush_table()
    text = " ".join(output)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\.{2,}", ".", text).strip()


def shape_narration(text: str) -> str:
    """Add restrained, content-led delivery cues without changing visible prose."""

    # Qwen3-TTS Base has no reliable SSML-style emphasis channel. These cues
    # mark real argument turns without adding claims, caps, or fake intensity.
    text = re.sub(r"\b(But|However|Instead|Yet|Still),?\s+", r"\1 — ", text)
    text = re.sub(
        r"\b(The point|The result|The consequence|The lesson|The trade-off) is\b",
        r"\1 is:",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(That means|In practice|Put differently),?\s+", r"\1 — ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_for_tts(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on sentence/paragraph boundaries while keeping model inputs bounded."""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            words = sentence.split()
            sentence_parts: list[str] = []
            part = ""
            for word in words:
                candidate = f"{part} {word}".strip()
                if part and len(candidate) > max_chars:
                    sentence_parts.append(part)
                    part = word
                else:
                    part = candidate
            if part:
                sentence_parts.append(part)
        else:
            sentence_parts = [sentence]

        for part in sentence_parts:
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def discover_posts() -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    reference_hash = reference_audio_sha256()
    for path in sorted(POSTS_DIR.glob("*.md")) + sorted(POSTS_DIR.glob("*.mdx")):
        source = path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(source)
        if frontmatter.get("section") != "blog":
            continue
        post_slug = frontmatter.get("postSlug")
        title = frontmatter.get("title", path.stem)
        if not post_slug:
            raise ValueError(f"Blog post is missing postSlug: {path}")
        text = shape_narration(clean_markdown(source))
        digest = hashlib.sha256(
            json.dumps(
                {
                    "text": text,
                    "model": MODEL,
                    "voice": VOICE,
                    "language": LANGUAGE,
                    "language_code": LANGUAGE_CODE,
                    "voice_mode": "in-context-learning",
                    "reference_audio_sha256": reference_hash,
                    "reference_text": REFERENCE_TEXT,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sampling_seed_version": SAMPLING_SEED_VERSION,
                    "icl_decode_mode": ICL_DECODE_MODE,
                    "icl_streaming_interval": ICL_STREAMING_INTERVAL,
                    "speed": SPEED,
                    "bitrate": BITRATE,
                    "max_chars": MAX_CHARS,
                    "silence_threshold": SILENCE_THRESHOLD,
                    "boundary_silence_samples": BOUNDARY_SILENCE_SAMPLES,
                    "long_silence_duration": LONG_SILENCE_DURATION,
                    "retained_silence_duration": RETAINED_SILENCE_DURATION,
                    "min_generation_tokens": MIN_GENERATION_TOKENS,
                    "max_tokens_per_char": MAX_TOKENS_PER_CHAR,
                    "extraction_version": EXTRACTION_VERSION,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        posts.append(
            {
                "path": path,
                "slug": post_slug,
                "title": title,
                "text": text,
                "digest": digest,
                "output": AUDIO_DIR / f"{post_slug}.mp3",
            }
        )
    return posts


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST_PATH)


def verify_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required; install it with `brew install ffmpeg`.")


def trim_boundary_silence(audio: Any) -> Any:
    """Remove model-generated dead air while preserving a tiny natural release.

    VoiceDesign can occasionally emit seconds of trailing silence after a chunk.
    Concatenating those tails makes a static export sound broken, so we only
    inspect the leading and trailing waveform boundaries. Deliberate pauses
    inside a spoken chunk are left untouched.
    """

    samples = np.asarray(audio)
    active = np.flatnonzero(np.abs(samples) >= SILENCE_THRESHOLD)
    if active.size == 0:
        raise RuntimeError("Model generated a silent audio chunk.")

    start = max(0, int(active[0]) - BOUNDARY_SILENCE_SAMPLES)
    end = min(samples.size, int(active[-1]) + 1 + BOUNDARY_SILENCE_SAMPLES)
    return samples[start:end]


def seed_chunk_generation(post_slug: str, chunk_index: int) -> None:
    """Make low-temperature sampling reproducible without forcing greedy EOS behavior."""

    import mlx.core as mx

    digest = hashlib.sha256(f"{post_slug}:{chunk_index}".encode("utf-8")).digest()
    mx.random.seed(int.from_bytes(digest[:4], byteorder="big"))


def generate_post(
    model: Any,
    post: dict[str, Any],
    batch_size: int,
    chunk_progress: tqdm,
) -> None:
    from mlx_audio.audio_io import write as audio_write

    chunks = split_for_tts(post["text"])
    if not chunks:
        raise ValueError(f"No speakable text found in {post['path']}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tts-{post['slug']}-") as temporary_dir:
        temporary = Path(temporary_dir)
        manifest_file = temporary / "concat.txt"
        chunk_paths: list[Path] = []
        with manifest_file.open("w", encoding="utf-8") as listing:
            for batch_start in range(0, len(chunks), batch_size):
                chunk_batch = chunks[batch_start : batch_start + batch_size]
                results_by_index: dict[int, tuple[Any, int]] = {}
                if len(chunk_batch) == 1:
                    seed_chunk_generation(post["slug"], batch_start)
                    max_tokens = max(
                        MIN_GENERATION_TOKENS,
                        math.ceil(len(chunk_batch[0]) * MAX_TOKENS_PER_CHAR),
                    )
                    results = list(
                        model.generate(
                            text=chunk_batch[0],
                            ref_audio=str(REFERENCE_AUDIO_PATH),
                            ref_text=REFERENCE_TEXT,
                            lang_code=LANGUAGE_CODE,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            repetition_penalty=1.5,
                            split_pattern="",
                            max_tokens=max_tokens,
                            stream=True,
                            streaming_interval=ICL_STREAMING_INTERVAL,
                        )
                    )
                    if not results or any(
                        not getattr(result, "is_streaming_chunk", False) for result in results
                    ):
                        raise RuntimeError(
                            "Qwen ICL did not return generated-only streaming audio; refusing "
                            "to export a reference-prefixed chunk."
                        )
                    results_by_index[0] = (
                        np.concatenate(
                            [trim_boundary_silence(result.audio) for result in results], axis=0
                        ),
                        results[0].sample_rate,
                    )
                else:
                    for result in model.batch_generate(
                        texts=chunk_batch,
                        voices=[None] * len(chunk_batch),
                        ref_audio=str(REFERENCE_AUDIO_PATH),
                        ref_text=REFERENCE_TEXT,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        repetition_penalty=1.5,
                        lang_code=LANGUAGE_CODE,
                        max_tokens=max(
                            max(MIN_GENERATION_TOKENS, math.ceil(len(chunk) * MAX_TOKENS_PER_CHAR))
                            for chunk in chunk_batch
                        ),
                    ):
                        # Non-streaming batch generation yields one complete audio
                        # result per sequence. Current mlx-audio marks these events
                        # as non-streaming rather than setting is_final_chunk.
                        results_by_index[result.sequence_idx] = (result.audio, result.sample_rate)

                missing = set(range(len(chunk_batch))) - results_by_index.keys()
                if missing:
                    missing_chunks = ", ".join(str(batch_start + index + 1) for index in sorted(missing))
                    raise RuntimeError(
                        f"Model returned no final audio for chunk(s) {missing_chunks} of {post['slug']}"
                    )

                for batch_index in range(len(chunk_batch)):
                    index = batch_start + batch_index
                    audio, sample_rate = results_by_index[batch_index]
                    audio = trim_boundary_silence(audio)
                    chunk_path = temporary / f"chunk-{index:05d}.wav"
                    audio_write(str(chunk_path), audio, sample_rate, format="wav")
                    chunk_paths.append(chunk_path)
                    listing.write(f"file '{chunk_path.as_posix()}'\n")
                    chunk_progress.update(1)

        temporary_mp3 = temporary / "audio.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_file),
                "-af",
                SILENCE_FILTER,
                "-ac",
                "1",
                "-ar",
                "24000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                BITRATE,
                str(temporary_mp3),
            ],
            check=True,
        )
        post["output"].parent.mkdir(parents=True, exist_ok=True)
        temporary_mp3.replace(post["output"])


def select_posts(posts: list[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    if not requested:
        return posts
    requested_set = set(requested)
    selected = [post for post in posts if post["slug"] in requested_set]
    missing = requested_set - {post["slug"] for post in selected}
    if missing:
        raise ValueError(f"Unknown postSlug(s): {', '.join(sorted(missing))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify every Blog post has current audio.")
    parser.add_argument("--force", action="store_true", help="Regenerate selected audio even when unchanged.")
    parser.add_argument("--post", action="append", default=[], help="Only process this postSlug; repeatable.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Must remain 1: generated-only Qwen ICL decoding is direct-only.",
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("--batch-size must be 1: generated-only Qwen ICL decoding is direct-only")

    posts = select_posts(discover_posts(), args.post)
    manifest = read_manifest()
    stale: list[dict[str, Any]] = []
    for post in posts:
        record = manifest.get(post["slug"], {})
        if record.get("digest") != post["digest"] or not post["output"].exists():
            stale.append(post)

    if args.check:
        if stale:
            print("Missing or stale Blog audio:", file=sys.stderr)
            for post in stale:
                print(f"  {post['slug']}", file=sys.stderr)
            return 1
        print(f"Audio check passed for {len(posts)} Blog posts.")
        return 0

    if not stale and not args.force:
        print(f"Audio is current for {len(posts)} Blog posts.")
        return 0

    verify_ffmpeg()
    from mlx_audio.tts.utils import load_model

    targets = posts if args.force else stale
    print(
        f"Loading {MODEL} once for {len(targets)} post(s) "
        f"with the fixed synthetic narrator reference..."
    )
    model = load_model(MODEL)
    with tqdm(total=len(targets), desc="Blogs", unit="post", position=0) as blog_progress:
        for index, post in enumerate(targets, start=1):
            chunks = split_for_tts(post["text"])
            blog_progress.set_postfix_str(f"{index}/{len(targets)} {post['title'][:42]}")
            with tqdm(
                total=len(chunks),
                desc="  Chunks",
                unit="chunk",
                position=1,
                leave=False,
            ) as chunk_progress:
                generate_post(model, post, args.batch_size, chunk_progress)
            manifest[post["slug"]] = {
                "digest": post["digest"],
                "model": MODEL,
                "voice": VOICE,
                "language": LANGUAGE,
                "language_code": LANGUAGE_CODE,
                "voice_mode": "in-context-learning",
                "reference_audio_sha256": reference_audio_sha256(),
                "reference_text": REFERENCE_TEXT,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "sampling_seed_version": SAMPLING_SEED_VERSION,
                "icl_decode_mode": ICL_DECODE_MODE,
                "icl_streaming_interval": ICL_STREAMING_INTERVAL,
                "speed": SPEED,
                "bitrate": BITRATE,
                "max_chars": MAX_CHARS,
                "silence_threshold": SILENCE_THRESHOLD,
                "boundary_silence_samples": BOUNDARY_SILENCE_SAMPLES,
                "long_silence_duration": LONG_SILENCE_DURATION,
                "retained_silence_duration": RETAINED_SILENCE_DURATION,
                "min_generation_tokens": MIN_GENERATION_TOKENS,
                "max_tokens_per_char": MAX_TOKENS_PER_CHAR,
                "file": f"/assets/audio/blog/{post['slug']}.mp3",
            }
            write_manifest(manifest)
            blog_progress.update(1)
    print(f"Generated {len(targets)} Blog audio file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
