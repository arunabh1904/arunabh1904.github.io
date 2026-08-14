#!/usr/bin/env python3
"""Generate local, static read-aloud audio for Blog posts.

The script intentionally keeps model execution out of the Astro build. Run it
on an Apple Silicon Mac, commit the generated MP3s, and let GitHub Pages serve
them as ordinary static assets.

Usage:
    uv run --with mlx-audio --with 'misaki[en]' \
      python scripts/generate-blog-audio.py
    uv run --with mlx-audio --with 'misaki[en]' \
      python scripts/generate-blog-audio.py --check

The model is loaded once per invocation. Existing files whose source hash and
voice settings are unchanged are skipped, so the command is safe to rerun
after adding or editing posts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "src" / "content" / "posts"
AUDIO_DIR = ROOT / "public" / "assets" / "audio" / "blog"
MANIFEST_PATH = ROOT / "public" / "assets" / "audio" / "manifest.json"
MODEL = "mlx-community/Kokoro-82M-bf16"
VOICE = "af_heart"
LANG_CODE = "a"
SPEED = 1.0
BITRATE = "48k"
MAX_CHARS = 900
PAUSE_SAMPLES = 8_400  # 350 ms at Kokoro's 24 kHz output rate.


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


def clean_markdown(source: str) -> str:
    """Turn Markdown/MDX into speech-friendly prose without reading markup."""

    body = re.sub(r"\A---\n.*?\n---\n", "", source, count=1, flags=re.DOTALL)
    body = re.sub(r"^import .*?$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^export .*?$", "", body, flags=re.MULTILINE)
    body = re.sub(r"```[\s\S]*?```", "", body)
    body = re.sub(r"~~~[\s\S]*?~~~", "", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", body)
    body = re.sub(r"^\s*[-*_]{3,}\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s*[-+*]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s*\d+[.)]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s*[|: -]+\s*$", "", body, flags=re.MULTILINE)
    body = body.replace("|", ". ")
    body = re.sub(r"^\s{0,3}#{1,6}\s*", "", body, flags=re.MULTILINE)
    body = re.sub(r"(`{1,3}|\*\*|__|[*_~])", "", body)
    body = re.sub(r"\$\$([\s\S]*?)\$\$", r". Equation: \1. ", body)
    body = re.sub(r"\$([^$\n]+)\$", r" \1 ", body)
    body = re.sub(r"\\\((.*?)\\\)", r" \1 ", body)
    body = re.sub(r"\\\[(.*?)\\\]", r". Equation: \1. ", body, flags=re.DOTALL)
    body = re.sub(r"\s+", " ", body)
    return re.sub(r"\s+([,.!?;:])", r"\1", body).strip()


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
    for path in sorted(POSTS_DIR.glob("*.md")) + sorted(POSTS_DIR.glob("*.mdx")):
        source = path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(source)
        if frontmatter.get("section") != "blog":
            continue
        post_slug = frontmatter.get("postSlug")
        title = frontmatter.get("title", path.stem)
        if not post_slug:
            raise ValueError(f"Blog post is missing postSlug: {path}")
        text = clean_markdown(source)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "text": text,
                    "model": MODEL,
                    "voice": VOICE,
                    "lang_code": LANG_CODE,
                    "speed": SPEED,
                    "bitrate": BITRATE,
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


def generate_post(model: Any, post: dict[str, Any]) -> None:
    import mlx.core as mx
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
            for index, chunk in enumerate(chunks):
                results = list(
                    model.generate(
                        text=chunk,
                        voice=VOICE,
                        speed=SPEED,
                        lang_code=LANG_CODE,
                    )
                )
                if not results:
                    raise RuntimeError(f"Model returned no audio for chunk {index} of {post['slug']}")
                audio = mx.concatenate([result.audio for result in results], axis=0)
                # A short pause between chunks prevents sentence boundaries from
                # sounding clipped after the files are concatenated.
                if index < len(chunks) - 1:
                    audio = mx.concatenate(
                        [audio, mx.zeros((PAUSE_SAMPLES,), dtype=audio.dtype)], axis=0
                    )
                chunk_path = temporary / f"chunk-{index:05d}.wav"
                audio_write(str(chunk_path), audio, results[0].sample_rate, format="wav")
                chunk_paths.append(chunk_path)
                listing.write(f"file '{chunk_path.as_posix()}'\n")
                print(f"  {index + 1}/{len(chunks)} chunks", flush=True)

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
    args = parser.parse_args()

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

    print(f"Loading {MODEL} once for {len(stale if not args.force else posts)} post(s)...")
    model = load_model(MODEL)
    targets = posts if args.force else stale
    for index, post in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {post['title']}")
        generate_post(model, post)
        manifest[post["slug"]] = {
            "digest": post["digest"],
            "model": MODEL,
            "voice": VOICE,
            "speed": SPEED,
            "file": f"/assets/audio/blog/{post['slug']}.mp3",
        }
        write_manifest(manifest)
    print(f"Generated {len(targets)} Blog audio file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
