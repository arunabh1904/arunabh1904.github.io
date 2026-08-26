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
after adding or editing posts. A fixed generated voice anchor preserves the
same narrator identity across posts. Each authored heading section is decoded
as generated audio only, preventing the anchor phrase from entering the output.
Streaming bounds decoder memory without resetting the voice inside a section.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "src" / "content" / "posts"
NARRATION_DIR = ROOT / "src" / "narrations" / "blog"
AUDIO_DIR = ROOT / "public" / "assets" / "audio" / "blog"
MANIFEST_PATH = ROOT / "public" / "assets" / "audio" / "manifest.json"
MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
VOICE = "Warm Indian English"
LANGUAGE = "English"
LANGUAGE_CODE = "english"
# Do not use Qwen ICL for static narration. Its spoken reference can be replayed
# by a faulty non-streaming decode path. The chosen VoiceDesign sample is used
# only as an encoder anchor. Streaming ICL decodes generated codes alone, while
# section-scoped requests bound long-form generation without changing identity.
VOICE_MODE = "voice-anchor-section-streaming-v1"
VOICE_INSTRUCT = (
    "An English-speaking Indian woman with a warm, thoughtful technical-presenter voice "
    "and a low-mid pitch. Composed and slightly sombre, with controlled energy, subtle "
    "natural inflection, precise diction, and a conversational pace. Never theatrical. "
    "Keep commas connected and articulate every word."
)
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.05
NARRATOR_SEED = 1904
SAMPLING_SEED_VERSION = "voice-anchor-sections-v1"
SPEED = 1.10
BITRATE = "64k"
SENTENCE_PAUSE_SECONDS = 0.0
PARAGRAPH_PAUSE_SECONDS = 0.0
HEADING_PAUSE_SECONDS = 0.0
SENTENCE_PAUSE_POLICY = "model-natural-structural-newlines-v4"
CHUNKING_POLICY = "source-heading-sections-with-continuous-stream-v1"
STREAMING_INTERVAL_SECONDS = 20.0
SECTION_MAX_CHARS = 4_000
SECTION_PAUSE_SECONDS = 0.65
SECTION_POLICY = "source-heading-sections-v1"
TABLE_POLICY = "skip-rows-require-spoken-takeaway-v1"
MAX_AUDIO_SECONDS = 30 * 60
MAX_ABRIDGED_WORDS = 3_200
HEADING_START = "[[BLOG_HEADING]]"
HEADING_END = "[[/BLOG_HEADING]]"
PARAGRAPH_BREAK = "[[BLOG_PARAGRAPH]]"
SILENCE_THRESHOLD = 0.008  # -42 dBFS; matches the verification threshold below.
BOUNDARY_SILENCE_SECONDS = 0.1
# Trim only the beginning and end of the completed stream. Reversing for the
# second pass avoids touching any natural silence inside the narration.
BOUNDARY_TRIM_FILTER = (
    "silenceremove="
    f"start_periods=1:start_duration=0.05:start_threshold={SILENCE_THRESHOLD}:"
    f"start_silence={BOUNDARY_SILENCE_SECONDS},"
    "areverse,"
    "silenceremove="
    f"start_periods=1:start_duration=0.05:start_threshold={SILENCE_THRESHOLD}:"
    f"start_silence={BOUNDARY_SILENCE_SECONDS},"
    "areverse"
)
AUDIO_FILTER = f"{BOUNDARY_TRIM_FILTER},atempo={SPEED}"
EXTRACTION_VERSION = "markdown-prose-v8-continuous-stream-audio"
MAX_GENERATION_TOKENS = 6_000
MIN_SECONDS_PER_WORD = 0.16
VOICE_ANCHOR_PATH = ROOT / "scripts" / "assets" / "blog-narrator-warm-indian-english-reference.mp3"
VOICE_ANCHOR_TEXT = (
    "That clean description breaks as soon as evidence conflicts, an actor is occluded, "
    "time stamps drift or one sensor degrades. A distant sike-list at dusk may occupy a few "
    "image pixels, two or three lie-dar returns, and one noisy radar detection with radial velocity."
)
VOICE_ANCHOR_SHA256 = hashlib.sha256(VOICE_ANCHOR_PATH.read_bytes()).hexdigest()
DECODE_POLICY = "generated-codes-only-streaming-v1"
PRONUNCIATION_LEXICON = {
    "cyclists": "sike-lists",
    "cyclist": "sike-list",
    "timestamps": "time stamps",
    "timestamp": "time stamp",
    "lidar": "lie-dar",
}
AUDIO_PROSODY_REWRITES = {
    "time stamps drift, or one sensor": "time stamps drift or one sensor",
}
PRONUNCIATION_POLICY = "audio-only-phonetic-and-prosody-lexicon-v2"
FORBIDDEN_REFERENCE_FIELDS = {
    "reference_audio_sha256",
    "reference_text",
    "icl_decode_mode",
    "icl_streaming_interval",
}

# The first two posts on the human-narration profile are deliberate canaries.
# Keeping the assignment explicit lets us validate long-form quality before a
# corpus-wide migration, without making every existing MP3 stale at once.
HUMAN_NARRATION_POSTS = frozenset(
    {
        "from-seeing-to-doing-the-evolution-of-vision-language-models",
        "how-unified-sensor-models-are-built-for-autonomous-driving",
    }
)
HUMAN_MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
HUMAN_VOICE = "hi_female"
HUMAN_VOICE_MODE = "voxtral-fixed-preset-paragraph-streaming-v1"
HUMAN_TEMPERATURE = 0.65
HUMAN_TOP_K = 50
HUMAN_TOP_P = 0.9
HUMAN_NARRATOR_SEED = 1904
HUMAN_SAMPLING_SEED_VERSION = "voxtral-human-canary-v1"
# Chunk-level ASR rejected the first deterministic samples below for invented
# clauses, repetitions, or badly slurred headings. Keep reviewed rerolls
# reproducible instead of relying on whichever sample happens to run next.
HUMAN_CHUNK_SEED_OVERRIDES = {
    "from-seeing-to-doing-the-evolution-of-vision-language-models": {
        "2": 1905,
        "4": 1905,
        "21": 1905,
        "24": 1905,
        "26": 1906,
        "40": 1905,
        "55": 1905,
    },
    "how-unified-sensor-models-are-built-for-autonomous-driving": {
        "5": 1905,
        "11": 1906,
        "31": 1905,
        "39": 1905,
        "55": 1905,
        "61": 1906,
        "63": 1905,
        "70": 1905,
    },
}
HUMAN_POST_PRONUNCIATION_LEXICONS = {
    "how-unified-sensor-models-are-built-for-autonomous-driving": {
        "BEVDet4D": "B E V Det four D",
        # The shared phonetic spelling is helpful in prose but unstable as a
        # two-word Voxtral prompt. Restore the written form for this heading.
        "lie-dar encoders": "Lidar encoders",
    },
}
# Full-source coverage was an explicit editorial decision for this post. Its
# accepted human-paced rendering is 31:36, so it gets a narrow 32-minute cap.
HUMAN_MAX_AUDIO_SECONDS = {
    "how-unified-sensor-models-are-built-for-autonomous-driving": 32 * 60,
}
HUMAN_SPEED = 1.0
HUMAN_PARAGRAPH_PAUSE_SECONDS = 0.35
HUMAN_HEADING_PAUSE_SECONDS = 0.65
HUMAN_SENTENCE_PAUSE_POLICY = "model-natural-with-explicit-structure-v1"
HUMAN_CHUNKING_POLICY = "source-paragraphs-with-bounded-sentence-groups-v1"
HUMAN_CHUNK_MAX_CHARS = 1_200
HUMAN_STREAMING_INTERVAL_SECONDS = 8.0
HUMAN_MAX_GENERATION_TOKENS = 1_600
HUMAN_MIN_SECONDS_PER_WORD = 0.24
HUMAN_MAX_SECONDS_PER_WORD = 0.72
HUMAN_DECODE_POLICY = "overlap-context-streaming-v1"
HUMAN_EXTRACTION_VERSION = "markdown-prose-v9-human-paragraph-audio"


@dataclass(frozen=True)
class NarrationChunk:
    """A bounded synthesis unit and its explicit structural pause."""

    text: str
    pause_seconds: float
    kind: str


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
    """Extract only coherent blog prose, excluding code, images, and captions.

    Headings and list items remain because they orient a listener. Table rows,
    display equations, image alt text, figure captions, and the References
    bibliography are omitted because their raw forms do not make useful spoken
    content. All authored headings and prose remain in source order.
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
            if output and output[-1] != PARAGRAPH_BREAK:
                output.append(PARAGRAPH_BREAK)
            continue
        if line.startswith(("import ", "export ")):
            continue
        if line.startswith("<") and line.endswith(">"):
            # Raw HTML image wrappers are visual-only. Mark their immediately
            # following italic line as a caption just like Markdown images.
            if "<img" in line or "compact-flow-diagram" in line:
                skip_caption = True
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            flush_table()
            heading_text = heading.group(1).strip()
            if heading_text.casefold() == "references":
                skip_references = True
                continue
            spoken_heading = heading_text.rstrip(".!?") + "."
            if output and output[-1] != PARAGRAPH_BREAK:
                output.append(PARAGRAPH_BREAK)
            output.append(f"{HEADING_START}{spoken_heading}{HEADING_END}")
            output.append(PARAGRAPH_BREAK)
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
    while output and output[-1] == PARAGRAPH_BREAK:
        output.pop()
    text = " ".join(output)
    text = re.sub(
        rf"(?:\s*{re.escape(PARAGRAPH_BREAK)}\s*)+",
        f" {PARAGRAPH_BREAK} ",
        text,
    )
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


def apply_pronunciation_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Apply one audio-only phonetic lexicon without changing authored prose."""

    for written, spoken in lexicon.items():
        text = re.sub(rf"\b{re.escape(written)}\b", spoken, text, flags=re.IGNORECASE)
    return text


def apply_pronunciation_hints(text: str) -> str:
    """Apply shared audio-only phonetic and prosody hints."""

    text = apply_pronunciation_lexicon(text, PRONUNCIATION_LEXICON)
    for written, spoken in AUDIO_PROSODY_REWRITES.items():
        text = re.sub(re.escape(written), spoken, text, flags=re.IGNORECASE)
    return text


def render_for_tts(text: str) -> str:
    """Render structural markers as one continuous, human-readable prompt.

    Newlines give Qwen paragraph and heading context without creating paragraph-
    or sentence-level requests. Heading sections remain long enough for natural
    prosody while the fixed anchor preserves narrator identity between them.
    """

    text = text.replace(HEADING_START, "").replace(HEADING_END, "")
    text = re.sub(
        rf"\s*{re.escape(PARAGRAPH_BREAK)}\s*",
        "\n",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return apply_pronunciation_hints(text.strip())


def section_prompts_for_tts(text: str) -> list[str]:
    """Group narration by authored headings, bounding unusually long sections."""

    raw_sections = re.split(rf"(?={re.escape(HEADING_START)})", text)
    sections: list[str] = []
    for raw_section in raw_sections:
        rendered = render_for_tts(raw_section)
        if not rendered:
            continue
        current = ""
        for paragraph in rendered.splitlines():
            candidate = f"{current}\n{paragraph}".strip()
            if current and len(candidate) > SECTION_MAX_CHARS:
                sections.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            sections.append(current)
    return sections


def split_bounded_sentences(text: str, max_chars: int) -> list[str]:
    """Group complete sentences without exceeding the model's safe prompt size."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # Technical prose can contain an unusually long semicolon-heavy
            # sentence. Fall back to word boundaries without dropping text.
            words = sentence.split()
            parts: list[str] = []
            part = ""
            for word in words:
                candidate = f"{part} {word}".strip()
                if part and len(candidate) > max_chars:
                    parts.append(part)
                    part = word
                else:
                    part = candidate
            if part:
                parts.append(part)
        else:
            parts = [sentence]

        for part in parts:
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_chars:
                groups.append(current)
                current = part
            else:
                current = candidate
    if current:
        groups.append(current)
    return groups


def paragraph_chunks_for_tts(text: str) -> list[NarrationChunk]:
    """Compile headings and paragraphs into stable, human-paced requests.

    A fixed Voxtral voice embedding keeps speaker identity stable when the
    generation resets. Paragraph-sized requests bound acoustic drift, while
    explicit pauses make heading and paragraph timing deterministic.
    """

    chunks: list[NarrationChunk] = []
    blocks = re.split(rf"\s*{re.escape(PARAGRAPH_BREAK)}\s*", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        is_heading = block.startswith(HEADING_START) and block.endswith(HEADING_END)
        rendered = render_for_tts(block)
        groups = split_bounded_sentences(rendered, HUMAN_CHUNK_MAX_CHARS)
        for index, group in enumerate(groups):
            is_last = index == len(groups) - 1
            pause = 0.0
            if is_last:
                pause = (
                    HUMAN_HEADING_PAUSE_SECONDS
                    if is_heading
                    else HUMAN_PARAGRAPH_PAUSE_SECONDS
                )
            chunks.append(
                NarrationChunk(
                    text=group,
                    pause_seconds=pause,
                    kind="heading" if is_heading else "paragraph",
                )
            )
    if chunks:
        final = chunks[-1]
        chunks[-1] = NarrationChunk(
            text=final.text,
            pause_seconds=0.0,
            kind=final.kind,
        )
    return chunks


def normalize_heading(value: str) -> str:
    """Normalize a heading for explicit sidecar coverage checks."""

    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[`*_~]", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def source_section_headings(source: str) -> set[str]:
    """Return every non-reference section heading that an abridgement must cover."""

    body = re.sub(r"\A---\n.*?\n---\n", "", source, count=1, flags=re.DOTALL)
    headings: set[str] = set()
    in_fence = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{2,6}\s+(.+)$", line)
        if not match:
            continue
        heading = normalize_heading(match.group(1))
        if heading == "references":
            break
        headings.add(heading)
    return headings


def abridgement_coverage(source: str, narration_source: str) -> set[str]:
    """Return source headings missing from an abridged narration's coverage map."""

    covered = source_section_headings(narration_source)
    for marker in re.findall(
        r"<!--\s*covers:\s*(.*?)\s*-->", narration_source, flags=re.IGNORECASE | re.DOTALL
    ):
        covered.update(normalize_heading(item) for item in marker.split("|") if item.strip())
    return source_section_headings(source) - covered


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
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        narration_path = NARRATION_DIR / f"{post_slug}.md"
        narration_mode = "full-source"
        narration_source = source
        if narration_path.exists():
            narration_source = narration_path.read_text(encoding="utf-8")
            narration_frontmatter = parse_frontmatter(narration_source)
            if narration_frontmatter.get("postSlug") != post_slug:
                raise ValueError(f"Narration sidecar postSlug does not match {post_slug}")
            if narration_frontmatter.get("sourceSha256") != source_sha256:
                raise ValueError(
                    f"Narration sidecar has not been reviewed against current source: {post_slug}"
                )
            missing_headings = abridgement_coverage(source, narration_source)
            if missing_headings:
                raise ValueError(
                    f"Narration sidecar omits source sections for {post_slug}: "
                    + ", ".join(sorted(missing_headings))
                )
            narration_mode = "section-complete-abridgement"

        text = shape_narration(clean_markdown(narration_source))
        uses_human_profile = post_slug in HUMAN_NARRATION_POSTS
        narration_chunks = paragraph_chunks_for_tts(text) if uses_human_profile else []
        post_pronunciations = HUMAN_POST_PRONUNCIATION_LEXICONS.get(post_slug, {})
        if narration_chunks and post_pronunciations:
            narration_chunks = [
                NarrationChunk(
                    text=apply_pronunciation_lexicon(chunk.text, post_pronunciations),
                    pause_seconds=chunk.pause_seconds,
                    kind=chunk.kind,
                )
                for chunk in narration_chunks
            ]
        tts_sections = [] if uses_human_profile else section_prompts_for_tts(text)
        tts_text = (
            "\n".join(chunk.text for chunk in narration_chunks)
            if uses_human_profile
            else "\n".join(tts_sections)
        )
        narration_word_count = len(tts_text.split())
        if narration_mode != "full-source" and narration_word_count > MAX_ABRIDGED_WORDS:
            raise ValueError(
                f"Narration sidecar exceeds {MAX_ABRIDGED_WORDS} words for {post_slug}: "
                f"{narration_word_count}"
            )
        if uses_human_profile:
            digest_payload = {
                "source_sha256": source_sha256,
                "text": text,
                "narration_mode": narration_mode,
                "model": HUMAN_MODEL,
                "voice": HUMAN_VOICE,
                "language": LANGUAGE,
                "voice_mode": HUMAN_VOICE_MODE,
                "temperature": HUMAN_TEMPERATURE,
                "top_k": HUMAN_TOP_K,
                "top_p": HUMAN_TOP_P,
                "narrator_seed": HUMAN_NARRATOR_SEED,
                "sampling_seed_version": HUMAN_SAMPLING_SEED_VERSION,
                "chunk_seed_overrides": HUMAN_CHUNK_SEED_OVERRIDES.get(post_slug, {}),
                "speed": HUMAN_SPEED,
                "paragraph_pause_seconds": HUMAN_PARAGRAPH_PAUSE_SECONDS,
                "heading_pause_seconds": HUMAN_HEADING_PAUSE_SECONDS,
                "sentence_pause_policy": HUMAN_SENTENCE_PAUSE_POLICY,
                "chunking_policy": HUMAN_CHUNKING_POLICY,
                "chunk_max_chars": HUMAN_CHUNK_MAX_CHARS,
                "streaming_interval_seconds": HUMAN_STREAMING_INTERVAL_SECONDS,
                "max_generation_tokens": HUMAN_MAX_GENERATION_TOKENS,
                "min_seconds_per_word": HUMAN_MIN_SECONDS_PER_WORD,
                "max_seconds_per_word": HUMAN_MAX_SECONDS_PER_WORD,
                "decode_policy": HUMAN_DECODE_POLICY,
                "narration_chunks": [asdict(chunk) for chunk in narration_chunks],
                "pronunciation_policy": PRONUNCIATION_POLICY,
                "pronunciation_lexicon": PRONUNCIATION_LEXICON,
                "audio_prosody_rewrites": AUDIO_PROSODY_REWRITES,
                "table_policy": TABLE_POLICY,
                "max_audio_seconds": HUMAN_MAX_AUDIO_SECONDS.get(
                    post_slug, MAX_AUDIO_SECONDS
                ),
                "max_abridged_words": MAX_ABRIDGED_WORDS,
                "bitrate": BITRATE,
                "silence_threshold": SILENCE_THRESHOLD,
                "boundary_silence_seconds": BOUNDARY_SILENCE_SECONDS,
                "extraction_version": HUMAN_EXTRACTION_VERSION,
            }
            if post_pronunciations:
                digest_payload["post_pronunciation_lexicon"] = post_pronunciations
        else:
            # Keep the legacy digest byte-for-byte compatible so a two-post
            # canary does not force an unrelated corpus regeneration.
            digest_payload = {
                    "source_sha256": source_sha256,
                    "text": text,
                    "narration_mode": narration_mode,
                    "model": MODEL,
                    "voice": VOICE,
                    "language": LANGUAGE,
                    "language_code": LANGUAGE_CODE,
                    "voice_mode": VOICE_MODE,
                    "voice_instruct": VOICE_INSTRUCT,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repetition_penalty": REPETITION_PENALTY,
                    "narrator_seed": NARRATOR_SEED,
                    "sampling_seed_version": SAMPLING_SEED_VERSION,
                    "speed": SPEED,
                    "sentence_pause_seconds": SENTENCE_PAUSE_SECONDS,
                    "paragraph_pause_seconds": PARAGRAPH_PAUSE_SECONDS,
                    "heading_pause_seconds": HEADING_PAUSE_SECONDS,
                    "sentence_pause_policy": SENTENCE_PAUSE_POLICY,
                    "chunking_policy": CHUNKING_POLICY,
                    "section_policy": SECTION_POLICY,
                    "section_max_chars": SECTION_MAX_CHARS,
                    "section_pause_seconds": SECTION_PAUSE_SECONDS,
                    "voice_anchor_sha256": VOICE_ANCHOR_SHA256,
                    "decode_policy": DECODE_POLICY,
                    "pronunciation_policy": PRONUNCIATION_POLICY,
                    "pronunciation_lexicon": PRONUNCIATION_LEXICON,
                    "audio_prosody_rewrites": AUDIO_PROSODY_REWRITES,
                    "table_policy": TABLE_POLICY,
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                    "max_abridged_words": MAX_ABRIDGED_WORDS,
                    "bitrate": BITRATE,
                    "silence_threshold": SILENCE_THRESHOLD,
                    "boundary_silence_seconds": BOUNDARY_SILENCE_SECONDS,
                    "streaming_interval_seconds": STREAMING_INTERVAL_SECONDS,
                    "max_generation_tokens": MAX_GENERATION_TOKENS,
                    "min_seconds_per_word": MIN_SECONDS_PER_WORD,
                    "extraction_version": EXTRACTION_VERSION,
            }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        posts.append(
            {
                "path": path,
                "slug": post_slug,
                "title": title,
                "text": text,
                "tts_text": tts_text,
                "tts_sections": tts_sections,
                "narration_chunks": narration_chunks,
                "synthesis_profile": "human" if uses_human_profile else "legacy",
                "source_sha256": source_sha256,
                "narration_mode": narration_mode,
                "narration_word_count": narration_word_count,
                "narration_path": narration_path if narration_path.exists() else None,
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


def narrator_profile_errors(posts: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    """Return manifest violations that could make the Blog sound multi-speaker."""

    legacy_expected = {
        "model": MODEL,
        "voice": VOICE,
        "voice_mode": VOICE_MODE,
        "voice_instruct": VOICE_INSTRUCT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "narrator_seed": NARRATOR_SEED,
        "sampling_seed_version": SAMPLING_SEED_VERSION,
        "speed": SPEED,
        "sentence_pause_seconds": SENTENCE_PAUSE_SECONDS,
        "paragraph_pause_seconds": PARAGRAPH_PAUSE_SECONDS,
        "heading_pause_seconds": HEADING_PAUSE_SECONDS,
        "sentence_pause_policy": SENTENCE_PAUSE_POLICY,
        "chunking_policy": CHUNKING_POLICY,
        "section_policy": SECTION_POLICY,
        "section_max_chars": SECTION_MAX_CHARS,
        "section_pause_seconds": SECTION_PAUSE_SECONDS,
        "voice_anchor_sha256": VOICE_ANCHOR_SHA256,
        "decode_policy": DECODE_POLICY,
        "pronunciation_policy": PRONUNCIATION_POLICY,
        "pronunciation_lexicon": PRONUNCIATION_LEXICON,
        "audio_prosody_rewrites": AUDIO_PROSODY_REWRITES,
        "table_policy": TABLE_POLICY,
    }
    human_expected = {
        "model": HUMAN_MODEL,
        "voice": HUMAN_VOICE,
        "voice_mode": HUMAN_VOICE_MODE,
        "temperature": HUMAN_TEMPERATURE,
        "top_k": HUMAN_TOP_K,
        "top_p": HUMAN_TOP_P,
        "narrator_seed": HUMAN_NARRATOR_SEED,
        "sampling_seed_version": HUMAN_SAMPLING_SEED_VERSION,
        "speed": HUMAN_SPEED,
        "paragraph_pause_seconds": HUMAN_PARAGRAPH_PAUSE_SECONDS,
        "heading_pause_seconds": HUMAN_HEADING_PAUSE_SECONDS,
        "sentence_pause_policy": HUMAN_SENTENCE_PAUSE_POLICY,
        "chunking_policy": HUMAN_CHUNKING_POLICY,
        "chunk_max_chars": HUMAN_CHUNK_MAX_CHARS,
        "streaming_interval_seconds": HUMAN_STREAMING_INTERVAL_SECONDS,
        "max_generation_tokens": HUMAN_MAX_GENERATION_TOKENS,
        "min_seconds_per_word": HUMAN_MIN_SECONDS_PER_WORD,
        "max_seconds_per_word": HUMAN_MAX_SECONDS_PER_WORD,
        "decode_policy": HUMAN_DECODE_POLICY,
        "pronunciation_policy": PRONUNCIATION_POLICY,
        "pronunciation_lexicon": PRONUNCIATION_LEXICON,
        "audio_prosody_rewrites": AUDIO_PROSODY_REWRITES,
        "table_policy": TABLE_POLICY,
    }
    errors: list[str] = []
    for post in posts:
        record = manifest.get(post["slug"], {})
        expected = (
            {
                **human_expected,
                "chunk_seed_overrides": HUMAN_CHUNK_SEED_OVERRIDES.get(
                    post["slug"], {}
                ),
                "post_pronunciation_lexicon": HUMAN_POST_PRONUNCIATION_LEXICONS.get(
                    post["slug"], {}
                ),
                "max_audio_seconds": HUMAN_MAX_AUDIO_SECONDS.get(
                    post["slug"], MAX_AUDIO_SECONDS
                ),
            }
            if post["synthesis_profile"] == "human"
            else legacy_expected
        )
        mismatched = [key for key, value in expected.items() if record.get(key) != value]
        forbidden = sorted(FORBIDDEN_REFERENCE_FIELDS.intersection(record))
        if mismatched:
            errors.append(f"{post['slug']}: narrator fields differ: {', '.join(mismatched)}")
        if forbidden:
            errors.append(f"{post['slug']}: forbidden reference fields: {', '.join(forbidden)}")
    return errors


def verify_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} required; install the ffmpeg package with "
            "`brew install ffmpeg`."
        )


def audio_duration_seconds(path: Path) -> float:
    """Return an audio file's container duration using ffprobe."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def seed_narrator_generation(seed: int = NARRATOR_SEED) -> None:
    """Seed the one continuous narrator generation for a post."""

    import mlx.core as mx

    mx.random.seed(seed)


def generate_legacy_post(
    model: Any,
    post: dict[str, Any],
    stream_progress: Any,
) -> float:
    import numpy as np
    from mlx_audio.audio_io import write as audio_write

    sections = post["tts_sections"]
    if not sections:
        raise ValueError(f"No speakable text found in {post['path']}")
    if not VOICE_ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing narrator voice anchor: {VOICE_ANCHOR_PATH}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tts-{post['slug']}-") as temporary_dir:
        temporary = Path(temporary_dir)
        manifest_file = temporary / "concat.txt"
        stream_parts = 0
        pause_path: Path | None = None
        with manifest_file.open("w", encoding="utf-8") as listing:
            for section_index, text in enumerate(sections):
                generated_tokens = 0
                pieces: list[Any] = []
                sample_rate = 0
                seed_narrator_generation()
                for result in model.generate(
                    text=text,
                    ref_audio=str(VOICE_ANCHOR_PATH),
                    ref_text=VOICE_ANCHOR_TEXT,
                    lang_code=LANGUAGE_CODE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                    split_pattern="",
                    max_tokens=MAX_GENERATION_TOKENS,
                    stream=True,
                    streaming_interval=STREAMING_INTERVAL_SECONDS,
                ):
                    pieces.append(np.asarray(result.audio))
                    sample_rate = result.sample_rate
                    generated_tokens += result.token_count
                    stream_parts += 1
                    stream_progress.update(1)

                if not pieces:
                    raise RuntimeError(
                        f"Model generated no audio for section {section_index + 1} "
                        f"of {post['slug']}"
                    )
                if generated_tokens >= MAX_GENERATION_TOKENS:
                    raise RuntimeError(
                        f"Model hit its {MAX_GENERATION_TOKENS}-token safety cap for "
                        f"section {section_index + 1} of {post['slug']}; refusing a "
                        "potentially truncated narration."
                    )

                audio = np.concatenate(pieces)
                active = np.flatnonzero(np.abs(audio) >= SILENCE_THRESHOLD)
                if active.size == 0:
                    raise RuntimeError(
                        f"Model generated silence for section {section_index + 1} "
                        f"of {post['slug']}"
                    )
                retained = round(sample_rate * BOUNDARY_SILENCE_SECONDS)
                start = max(0, int(active[0]) - retained)
                end = min(audio.size, int(active[-1]) + 1 + retained)
                audio = audio[start:end]
                delivered_seconds = audio.size / sample_rate / SPEED
                minimum_seconds = len(text.split()) * MIN_SECONDS_PER_WORD
                if delivered_seconds < minimum_seconds:
                    raise RuntimeError(
                        f"Section {section_index + 1} of {post['slug']} is implausibly "
                        f"short ({delivered_seconds:.1f}s for {len(text.split())} words); "
                        "refusing possible early EOS truncation."
                    )

                section_path = temporary / f"section-{section_index:03d}.wav"
                audio_write(str(section_path), audio, sample_rate, format="wav")
                listing.write(f"file '{section_path.as_posix()}'\n")
                if section_index < len(sections) - 1:
                    if pause_path is None:
                        pause_path = temporary / "section-pause.wav"
                        pause_samples = round(
                            sample_rate * SECTION_PAUSE_SECONDS * SPEED
                        )
                        audio_write(
                            str(pause_path),
                            np.zeros(pause_samples, dtype=np.float32),
                            sample_rate,
                            format="wav",
                        )
                    listing.write(f"file '{pause_path.as_posix()}'\n")

        if stream_parts == 0:
            raise RuntimeError(f"Model generated no audio for {post['slug']}")

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
                AUDIO_FILTER,
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
        duration_seconds = audio_duration_seconds(temporary_mp3)
        if duration_seconds > MAX_AUDIO_SECONDS:
            raise RuntimeError(
                f"Generated audio for {post['slug']} is {duration_seconds / 60:.1f} minutes; "
                f"shorten its section-complete narration instead of truncating at "
                f"{MAX_AUDIO_SECONDS / 60:.0f} minutes."
            )
        post["output"].parent.mkdir(parents=True, exist_ok=True)
        temporary_mp3.replace(post["output"])
        return duration_seconds


def generate_human_post(
    model: Any,
    post: dict[str, Any],
    stream_progress: Any,
) -> float:
    """Generate a fixed-voice, paragraph-bounded Voxtral narration."""

    import numpy as np
    from mlx_audio.audio_io import write as audio_write

    chunks: list[NarrationChunk] = post["narration_chunks"]
    if not chunks:
        raise ValueError(f"No speakable text found in {post['path']}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    chunk_cache = (
        Path(tempfile.gettempdir()) / "blog-audio-chunks" / post["digest"]
    )
    chunk_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tts-{post['slug']}-") as temporary_dir:
        temporary = Path(temporary_dir)
        manifest_file = temporary / "concat.txt"
        pause_paths: dict[tuple[int, float], Path] = {}
        stream_parts = 0

        with manifest_file.open("w", encoding="utf-8") as listing:
            for chunk_index, chunk in enumerate(chunks):
                chunk_path = temporary / f"chunk-{chunk_index:03d}.wav"
                cached_chunk = chunk_cache / chunk_path.name
                if cached_chunk.exists():
                    shutil.copy2(cached_chunk, chunk_path)
                    sample_rate = 24_000
                    stream_progress.update(1)
                    listing.write(f"file '{chunk_path.as_posix()}'\n")
                    if chunk.pause_seconds > 0 and chunk_index < len(chunks) - 1:
                        pause_key = (sample_rate, chunk.pause_seconds)
                        pause_path = pause_paths.get(pause_key)
                        if pause_path is None:
                            pause_path = temporary / (
                                f"pause-{sample_rate}-{round(chunk.pause_seconds * 1000)}ms.wav"
                            )
                            pause_samples = round(sample_rate * chunk.pause_seconds)
                            audio_write(
                                str(pause_path),
                                np.zeros(pause_samples, dtype=np.float32),
                                sample_rate,
                                format="wav",
                            )
                            pause_paths[pause_key] = pause_path
                        listing.write(f"file '{pause_path.as_posix()}'\n")
                    continue

                pieces: list[Any] = []
                generated_tokens = 0
                sample_rate = 0
                chunk_seed = HUMAN_CHUNK_SEED_OVERRIDES.get(post["slug"], {}).get(
                    str(chunk_index), HUMAN_NARRATOR_SEED
                )
                seed_narrator_generation(chunk_seed)
                for result in model.generate(
                    text=chunk.text,
                    voice=HUMAN_VOICE,
                    temperature=HUMAN_TEMPERATURE,
                    top_k=HUMAN_TOP_K,
                    top_p=HUMAN_TOP_P,
                    max_tokens=HUMAN_MAX_GENERATION_TOKENS,
                    stream=True,
                    streaming_interval=HUMAN_STREAMING_INTERVAL_SECONDS,
                ):
                    audio_piece = np.asarray(result.audio)
                    if audio_piece.size:
                        pieces.append(audio_piece)
                    sample_rate = result.sample_rate
                    generated_tokens += result.token_count
                    stream_parts += 1
                    stream_progress.update(1)

                if not pieces:
                    raise RuntimeError(
                        f"Model generated no audio for chunk {chunk_index + 1} "
                        f"of {post['slug']}"
                    )
                if generated_tokens >= HUMAN_MAX_GENERATION_TOKENS:
                    raise RuntimeError(
                        f"Model hit its {HUMAN_MAX_GENERATION_TOKENS}-token safety cap "
                        f"for chunk {chunk_index + 1} of {post['slug']}"
                    )

                audio = np.concatenate(pieces)
                active = np.flatnonzero(np.abs(audio) >= SILENCE_THRESHOLD)
                if active.size == 0:
                    raise RuntimeError(
                        f"Model generated silence for chunk {chunk_index + 1} "
                        f"of {post['slug']}"
                    )
                retained = round(sample_rate * BOUNDARY_SILENCE_SECONDS)
                start = max(0, int(active[0]) - retained)
                end = min(audio.size, int(active[-1]) + 1 + retained)
                audio = audio[start:end]

                delivered_seconds = audio.size / sample_rate
                word_count = len(chunk.text.split())
                minimum_seconds = word_count * HUMAN_MIN_SECONDS_PER_WORD
                # Short headings need a fixed allowance. This coarse ceiling
                # catches only major runaways; the post-generation ASR audit
                # distinguishes expressive slow delivery from extra speech.
                fixed_allowance = 20.0 if chunk.kind == "heading" else 12.0
                maximum_seconds = max(
                    fixed_allowance,
                    word_count * HUMAN_MAX_SECONDS_PER_WORD + 2.0,
                )
                if delivered_seconds < minimum_seconds:
                    raise RuntimeError(
                        f"Chunk {chunk_index + 1} of {post['slug']} is implausibly "
                        f"short ({delivered_seconds:.1f}s for {word_count} words)"
                    )
                if delivered_seconds > maximum_seconds:
                    raise RuntimeError(
                        f"Chunk {chunk_index + 1} of {post['slug']} is implausibly "
                        f"long ({delivered_seconds:.1f}s for {word_count} words); "
                        "refusing a likely continuation or repetition artifact"
                    )

                audio_write(str(chunk_path), audio, sample_rate, format="wav")
                shutil.copy2(chunk_path, cached_chunk)
                listing.write(f"file '{chunk_path.as_posix()}'\n")

                if chunk.pause_seconds > 0 and chunk_index < len(chunks) - 1:
                    pause_key = (sample_rate, chunk.pause_seconds)
                    pause_path = pause_paths.get(pause_key)
                    if pause_path is None:
                        pause_path = temporary / (
                            f"pause-{sample_rate}-{round(chunk.pause_seconds * 1000)}ms.wav"
                        )
                        pause_samples = round(sample_rate * chunk.pause_seconds)
                        audio_write(
                            str(pause_path),
                            np.zeros(pause_samples, dtype=np.float32),
                            sample_rate,
                            format="wav",
                        )
                        pause_paths[pause_key] = pause_path
                    listing.write(f"file '{pause_path.as_posix()}'\n")

        missing_chunks = [
            index
            for index in range(len(chunks))
            if not (temporary / f"chunk-{index:03d}.wav").exists()
        ]
        if missing_chunks:
            raise RuntimeError(
                f"Missing generated chunks for {post['slug']}: {missing_chunks}"
            )

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
                BOUNDARY_TRIM_FILTER,
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
        duration_seconds = audio_duration_seconds(temporary_mp3)
        max_audio_seconds = HUMAN_MAX_AUDIO_SECONDS.get(
            post["slug"], MAX_AUDIO_SECONDS
        )
        if duration_seconds > max_audio_seconds:
            raise RuntimeError(
                f"Generated audio for {post['slug']} is {duration_seconds / 60:.1f} "
                f"minutes; shorten its narration instead of truncating it"
            )
        post["output"].parent.mkdir(parents=True, exist_ok=True)
        temporary_mp3.replace(post["output"])
        return duration_seconds


def generate_post(
    model: Any,
    post: dict[str, Any],
    stream_progress: Any,
) -> float:
    if post["synthesis_profile"] == "human":
        return generate_human_post(model, post, stream_progress)
    return generate_legacy_post(model, post, stream_progress)


def select_posts(posts: list[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    if not requested:
        return posts
    requested_set = set(requested)
    selected = [post for post in posts if post["slug"] in requested_set]
    missing = requested_set - {post["slug"] for post in selected}
    if missing:
        raise ValueError(f"Unknown postSlug(s): {', '.join(sorted(missing))}")
    return selected


def manifest_record(post: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
    common = {
        "digest": post["digest"],
        "language": LANGUAGE,
        "pronunciation_policy": PRONUNCIATION_POLICY,
        "pronunciation_lexicon": PRONUNCIATION_LEXICON,
        "audio_prosody_rewrites": AUDIO_PROSODY_REWRITES,
        "table_policy": TABLE_POLICY,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "narration_mode": post["narration_mode"],
        "narration_word_count": post["narration_word_count"],
        "source_sha256": post["source_sha256"],
        "duration_seconds": duration_seconds,
        "bitrate": BITRATE,
        "silence_threshold": SILENCE_THRESHOLD,
        "boundary_silence_seconds": BOUNDARY_SILENCE_SECONDS,
        "file": f"/assets/audio/blog/{post['slug']}.mp3",
    }
    if post["synthesis_profile"] == "human":
        return {
            **common,
            "synthesis_profile": "human",
            "model": HUMAN_MODEL,
            "voice": HUMAN_VOICE,
            "voice_mode": HUMAN_VOICE_MODE,
            "temperature": HUMAN_TEMPERATURE,
            "top_k": HUMAN_TOP_K,
            "top_p": HUMAN_TOP_P,
            "narrator_seed": HUMAN_NARRATOR_SEED,
            "sampling_seed_version": HUMAN_SAMPLING_SEED_VERSION,
            "chunk_seed_overrides": HUMAN_CHUNK_SEED_OVERRIDES.get(post["slug"], {}),
            "post_pronunciation_lexicon": HUMAN_POST_PRONUNCIATION_LEXICONS.get(
                post["slug"], {}
            ),
            "narration_chunk_count": len(post["narration_chunks"]),
            "max_audio_seconds": HUMAN_MAX_AUDIO_SECONDS.get(
                post["slug"], MAX_AUDIO_SECONDS
            ),
            "speed": HUMAN_SPEED,
            "paragraph_pause_seconds": HUMAN_PARAGRAPH_PAUSE_SECONDS,
            "heading_pause_seconds": HUMAN_HEADING_PAUSE_SECONDS,
            "sentence_pause_policy": HUMAN_SENTENCE_PAUSE_POLICY,
            "chunking_policy": HUMAN_CHUNKING_POLICY,
            "chunk_max_chars": HUMAN_CHUNK_MAX_CHARS,
            "streaming_interval_seconds": HUMAN_STREAMING_INTERVAL_SECONDS,
            "max_generation_tokens": HUMAN_MAX_GENERATION_TOKENS,
            "min_seconds_per_word": HUMAN_MIN_SECONDS_PER_WORD,
            "max_seconds_per_word": HUMAN_MAX_SECONDS_PER_WORD,
            "decode_policy": HUMAN_DECODE_POLICY,
        }
    return {
        **common,
        "model": MODEL,
        "voice": VOICE,
        "language_code": LANGUAGE_CODE,
        "voice_mode": VOICE_MODE,
        "voice_instruct": VOICE_INSTRUCT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "narrator_seed": NARRATOR_SEED,
        "sampling_seed_version": SAMPLING_SEED_VERSION,
        "speed": SPEED,
        "sentence_pause_seconds": SENTENCE_PAUSE_SECONDS,
        "paragraph_pause_seconds": PARAGRAPH_PAUSE_SECONDS,
        "heading_pause_seconds": HEADING_PAUSE_SECONDS,
        "sentence_pause_policy": SENTENCE_PAUSE_POLICY,
        "chunking_policy": CHUNKING_POLICY,
        "section_policy": SECTION_POLICY,
        "section_max_chars": SECTION_MAX_CHARS,
        "section_pause_seconds": SECTION_PAUSE_SECONDS,
        "voice_anchor_sha256": VOICE_ANCHOR_SHA256,
        "decode_policy": DECODE_POLICY,
        "streaming_interval_seconds": STREAMING_INTERVAL_SECONDS,
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "min_seconds_per_word": MIN_SECONDS_PER_WORD,
    }


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
        profile_errors = narrator_profile_errors(posts, manifest)
        if profile_errors:
            print("Blog narrator profile is inconsistent:", file=sys.stderr)
            for error in profile_errors:
                print(f"  {error}", file=sys.stderr)
            return 1
        duration_errors: list[str] = []
        for post in posts:
            actual_duration = audio_duration_seconds(post["output"])
            recorded_duration = manifest[post["slug"]].get("duration_seconds")
            max_audio_seconds = (
                HUMAN_MAX_AUDIO_SECONDS.get(post["slug"], MAX_AUDIO_SECONDS)
                if post["synthesis_profile"] == "human"
                else MAX_AUDIO_SECONDS
            )
            if actual_duration > max_audio_seconds:
                duration_errors.append(
                    f"{post['slug']}: {actual_duration / 60:.1f} minutes exceeds "
                    f"{max_audio_seconds / 60:.0f} minutes"
                )
            if not isinstance(recorded_duration, (int, float)) or abs(
                actual_duration - recorded_duration
            ) > 1.0:
                duration_errors.append(f"{post['slug']}: manifest duration is missing or stale")
        if duration_errors:
            print("Blog audio duration checks failed:", file=sys.stderr)
            for error in duration_errors:
                print(f"  {error}", file=sys.stderr)
            return 1
        print(
            f"Audio, narrator-profile, and duration checks passed for {len(posts)} Blog posts."
        )
        return 0

    if not stale and not args.force:
        print(f"Audio is current for {len(posts)} Blog posts.")
        return 0

    verify_ffmpeg()
    from mlx_audio.tts.utils import load_model
    from tqdm.auto import tqdm

    targets = posts if args.force else stale
    loaded_models: dict[str, Any] = {}
    with tqdm(total=len(targets), desc="Blogs", unit="post", position=0) as blog_progress:
        for index, post in enumerate(targets, start=1):
            model_name = (
                HUMAN_MODEL if post["synthesis_profile"] == "human" else MODEL
            )
            model = loaded_models.get(model_name)
            if model is None:
                print(f"Loading {model_name} for the {post['synthesis_profile']} profile...")
                model = load_model(model_name)
                loaded_models[model_name] = model
            blog_progress.set_postfix_str(f"{index}/{len(targets)} {post['title'][:42]}")
            with tqdm(
                total=None,
                desc="  Section streams",
                unit="part",
                position=1,
                leave=False,
            ) as stream_progress:
                duration_seconds = generate_post(model, post, stream_progress)
            manifest[post["slug"]] = manifest_record(post, duration_seconds)
            write_manifest(manifest)
            blog_progress.update(1)
    print(f"Generated {len(targets)} Blog audio file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
