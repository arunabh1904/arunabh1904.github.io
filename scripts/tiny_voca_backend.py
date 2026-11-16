"""Pinned CPU narrator. Importing this module requires only the standard library."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path

PROFILE = {
    "synthesis_profile": "tiny-voca",
    "model": "n4ze3m/tiny-voca",
    "model_revision": "eac048b7a4392584725af9792460a0211fa1cac7",
    "checkpoint_sha256": "ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70",
    "code_revision": "62e2adef158bc997042cc6234f541f1e0e4cd55d",
    "code_sha256": "0f190da92106d722f830b232c509f89dee29fe2cabc15299e370f5f0b38edb7b",
    "frontend_revision": "8cd889e4de7be6bd1dd2e6891344a0729fa14dd4",
    "frontend_code_sha256": "3240cb3d6a2badcfa41c35d5f83917327b01f88d4a92ba1026f743f850fd92c2",
    "frontend_data_sha256": "5999cdb138a9561342cbd87a165433fb9f60bbf3720b7e706e166d170fe407ce",
    "inference_runtime": {"tiny-voca": "0.2.0", "dizione": "0.1.0", "torch": "2.15.0",
                          "numpy": "2.5.5", "soundfile": "0.14.0"},
    "device": "cpu",
    "threads": 4,
    "voice": "M2",
    "voice_mode": "fixed-preset",
    "frontend": "own",
    "speed": 1.0,
    "pitch_shift": 0.0,
    "pitch_range": 1.0,
    "narrator_seed": 1904,
    "sampling_seed_version": "request-seed-plus-utterance-index-v1",
    "decode_policy": "upstream-sentences-recursive-clause-split-fail-on-unreadable-v1",
    "utterance_fade_seconds": 0.005,
    "utterance_fade_policy": "half-cosine-both-edges-no-trim-v1",
    "peak_limit": 0.99,
    "sample_rate": 24000,
    "cache_encoding": "PCM_16",
    "min_seconds_per_word": 0.24,
    "max_seconds_per_word": 0.72,
    "output_gain_db": -1,
}


def tree_hash(root: Path, pattern: str = "*") -> str:
    entries = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob(pattern))
               if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def load(weights: Path):
    """Validate the actual installed runtime and data before allowing cache writes."""
    for package, expected in PROFILE["inference_runtime"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Unreviewed {package} version: {actual}; expected {expected}")
    checkpoint = weights / "tiny-voca.pt"
    checks = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "frontend_data_sha256": tree_hash(weights / "phonemizer"),
    }
    for package, field in [("tiny_voca", "code_sha256"), ("dizione", "frontend_code_sha256")]:
        spec = importlib.util.find_spec(package)
        checks[field] = tree_hash(Path(spec.origin).parent, "*.py")
    for field, actual in checks.items():
        if actual != PROFILE[field]:
            raise RuntimeError(f"Unreviewed tiny-voca {field}: {actual}")

    os.environ["DIZIONE_DATA"] = str((weights / "phonemizer").resolve())
    import torch
    from tiny_voca.infer import load_model
    from tiny_voca.text import set_frontend
    torch.set_num_threads(PROFILE["threads"])
    set_frontend(PROFILE["frontend"])
    model = load_model(checkpoint, PROFILE["device"])
    if model.config.sample_rate != PROFILE["sample_rate"]:
        raise RuntimeError("Unexpected tiny-voca sample rate")
    return model


def fade_edges(audio, sample_rate: int):
    """Preserve length and interior samples; bring only the two edges to zero."""
    result = audio.copy()
    count = min(round(sample_rate * PROFILE["utterance_fade_seconds"]), len(result) // 2)
    for index in range(count):
        gain = 0.5 - 0.5 * math.cos(math.pi * index / (count - 1)) if count > 1 else 0
        result[index] *= gain
        result[-1 - index] *= gain
    return result


def synthesize(model, text: str):
    """Keep upstream timing/seeds, fade every utterance, and never skip unreadable text."""
    import numpy as np
    from tiny_voca.infer import finished, sentences, shorter, speak
    from tiny_voca.model import UtteranceTooLong
    from tiny_voca.text import phonemize

    waves, records = [], []
    limit = model.config.max_tokens - 12

    def read(piece):
        phones = phonemize(piece)  # NothingToPronounce is deliberately a release failure.
        if len(phones) <= limit:
            try:
                wave, phones, _ = speak(
                    model, piece, PROFILE["voice"], speed=PROFILE["speed"],
                    pitch_shift=PROFILE["pitch_shift"], pitch_range=PROFILE["pitch_range"],
                    seed=PROFILE["narrator_seed"] + len(waves),
                )
            except UtteranceTooLong:
                pass
            else:
                data = wave.numpy()
                if not len(data) or not np.isfinite(data).all():
                    raise RuntimeError("Empty or non-finite tiny-voca utterance")
                records.append({"text": piece, "samples": len(data),
                                "unknown_phonemes": list(phones.unknown)})
                waves.append(fade_edges(data, PROFILE["sample_rate"]))
                return
        pieces = shorter(piece)
        if not pieces:
            raise ValueError(f"One word is too long to speak: {piece[:40]!r}")
        for part in pieces:
            read(part)

    for sentence in sentences(finished(text)):
        read(sentence)
    if not waves:
        raise ValueError("No speakable tiny-voca utterances")
    data = np.concatenate(waves)
    peak = float(np.max(np.abs(data)))
    if peak > PROFILE["peak_limit"]:
        data *= PROFILE["peak_limit"] / peak
    validate_audio(data, PROFILE["sample_rate"], text)
    return data, records


def validate_audio(data, sample_rate: int, text: str):
    import numpy as np
    if sample_rate != PROFILE["sample_rate"] or data.ndim != 1 or not len(data):
        raise RuntimeError("Invalid tiny-voca waveform format")
    if not np.isfinite(data).all() or np.max(np.abs(data)) >= 1 or np.max(np.abs(data)) < 0.008:
        raise RuntimeError("Non-finite, clipped, or silent tiny-voca request")
    seconds, words = len(data) / sample_rate, len(text.split())
    if not words * PROFILE["min_seconds_per_word"] <= seconds <= max(12, words * PROFILE["max_seconds_per_word"] + 2):
        raise RuntimeError(f"Implausible tiny-voca duration: {seconds:.2f}s for {words} words")
