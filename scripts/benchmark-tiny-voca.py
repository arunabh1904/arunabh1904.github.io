#!/usr/bin/env python3
"""Evaluate tiny-voca without changing production narration or its manifest.

Install tiny-voca (including its own frontend), soundfile, jiwer, and
mlx-whisper in an isolated environment. Set DIZIONE_DATA to the downloaded
phonemizer directory. Use --limit 1 for a canary, then omit it for the full post.
The audit uses the existing repository normalization and rejection thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def assemble(paths, chunks, out, gain):
    import numpy as np
    import soundfile as sf
    waves = []
    for i, (path, chunk) in enumerate(zip(paths, chunks)):
        wave, sr = sf.read(path, dtype="float32")
        assert sr == 24000 and wave.ndim == 1
        waves.append(wave)
        if i < len(chunks) - 1:
            waves.append(np.zeros(round(chunk.pause_seconds * sr), dtype=np.float32))
    wav = out.with_suffix('.wav')
    sf.write(wav, np.concatenate(waves), 24000, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                    "-af", f"volume={gain}dB", "-ac", "1", "-ar", "24000",
                    "-c:a", "libmp3lame", "-b:a", "64k", str(out)], check=True)


def generate(args, post, chunks, gen):
    import numpy as np
    import soundfile as sf
    import torch
    from tiny_voca.infer import load_model, speak_text
    from tiny_voca.text import set_frontend
    torch.set_num_threads(args.threads)
    set_frontend("own")
    started = time.perf_counter()
    model = load_model(args.checkpoint, "cpu")
    load_seconds = time.perf_counter() - started
    # Warm-up is recorded separately and never counted as measured synthesis.
    started = time.perf_counter()
    speak_text(model, "This is the warm up sentence.", "F2")
    warmup_seconds = time.perf_counter() - started
    profile = {
        "site_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "post_slug": post["slug"], "source_sha256": post["source_sha256"],
        "narration_digest": post["digest"], "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "threads": args.threads,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "tiny-voca", "dizione", "numpy"]},
        "speed": 1.0, "pitch_shift": 0.0, "pitch_range": 1.0,
        "seed": 1904, "peak_limit": 0.99, "output_gain_db": -1,
        "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
        "note": "Exact production synthesis text; upstream sentence splitting within each request; no rerolls or training.",
    }
    for voice in args.voices:
        directory = args.output_dir / voice
        directory.mkdir(exist_ok=True)
        previous_profile = directory / 'generation.json'
        if previous_profile.exists():
            old = json.loads(previous_profile.read_text())
            for field in ['checkpoint_sha256', 'versions', 'speed', 'pitch_shift',
                          'pitch_range', 'seed', 'peak_limit', 'threads']:
                if old[field] != profile[field]:
                    raise RuntimeError(f"Changed {field}; use a fresh output directory")
        records = []
        for index, chunk in enumerate(chunks):
            path = directory / f"chunk-{index:03d}.wav"
            meta = path.with_suffix('.json')
            text_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            if path.exists() and meta.exists():
                record = json.loads(meta.read_text())
                if record['text_sha256'] != text_hash or record['checkpoint_sha256'] != profile['checkpoint_sha256']:
                    raise RuntimeError(f"Stale benchmark cache: {path}; use a fresh output directory")
                if record['wav_sha256'] != sha(path):
                    raise RuntimeError(f"Changed waveform: {path}")
            else:
                started = time.perf_counter()
                wave, utterances = speak_text(model, chunk.text, voice, seed=1904)
                elapsed = time.perf_counter() - started
                data = wave.numpy()
                sf.write(path, data, 24000, subtype="FLOAT")
                record = {
                    "index": index, "text": chunk.text, "text_sha256": text_hash,
                    "checkpoint_sha256": profile['checkpoint_sha256'], "wav_sha256": sha(path),
                    "generation_seconds": elapsed, "audio_seconds": len(data) / 24000,
                    "utterances": len(utterances), "peak": float(np.max(np.abs(data))),
                    "rms": float(np.sqrt(np.mean(data ** 2))),
                    "nonfinite_samples": int(np.count_nonzero(~np.isfinite(data))),
                    "clipped_samples": int(np.count_nonzero(np.abs(data) >= 1)),
                    "denormal_samples": int(np.count_nonzero((np.abs(data)>0)&(np.abs(data)<np.finfo(np.float32).tiny))),
                    "unknown_phonemes": [list(p.unknown) for p in utterances if p.unknown],
                }
                save(meta, record)
            records.append(record)
            print(f"{voice} {index+1}/{len(chunks)}: {record['audio_seconds']:.1f}s audio, {record['generation_seconds']:.3f}s generation", flush=True)
        paths = [directory / f"chunk-{i:03d}.wav" for i in range(len(chunks))]
        assemble(paths, chunks, args.output_dir / f"mapping-{voice}.mp3", -1)
        seconds = sum(r['audio_seconds'] for r in records)
        elapsed = sum(r['generation_seconds'] for r in records)
        save(directory / 'generation.json', {**profile, "voice": voice, "chunks": records,
             "audio_seconds_without_pauses": seconds, "generation_seconds": elapsed,
             "real_time_factor": elapsed / seconds, "times_real_time": seconds / elapsed})


def audit(args, post, chunks, gen, gate):
    import jiwer
    import mlx_whisper
    import soundfile as sf
    if args.baseline:
        manifest = gen.read_manifest()[post['slug']]
        if manifest['digest'] != post['digest'] or manifest['source_sha256'] != post['source_sha256']:
            raise RuntimeError('Published baseline has different source/profile; choose a matched post')
    variants = list(args.voices) + (["Voxtral-published-cache"] if args.baseline else [])
    for variant in variants:
        directory = args.output_dir / variant
        directory.mkdir(exist_ok=True)
        records, refs, hyps = [], [], []
        for index, chunk in enumerate(chunks):
            path = gen.human_chunk_cache_path(post, index) if variant == "Voxtral-published-cache" else directory / f"chunk-{index:03d}.wav"
            if not path.exists():
                raise FileNotFoundError(path)
            key = {"wav_sha256": sha(path), "asr_model": args.asr_model,
                   "mlx_whisper": importlib.metadata.version('mlx-whisper'),
                   "language": "en", "temperature": 0.0, "word_timestamps": True}
            cache = directory / f"asr-{index:03d}.json"
            if cache.exists() and json.loads(cache.read_text())["key"] == key:
                transcription = json.loads(cache.read_text())["transcription"]
            else:
                transcription = mlx_whisper.transcribe(str(path), path_or_hf_repo=args.asr_model,
                    language="en", temperature=0.0, word_timestamps=True)
                save(cache, {"key": key, "transcription": transcription})
            ref = gate.normalize_transcript(chunk.text)
            hyp = gate.normalize_transcript(transcription['text'])
            refs.append(ref); hyps.append(hyp)
            comparison = jiwer.process_words(ref, hyp)
            insertions = []
            for alignment in comparison.alignments[0]:
                if alignment.type == 'insert':
                    words = comparison.hypotheses[0][alignment.hyp_start_idx:alignment.hyp_end_idx]
                    insertions.append({"text": " ".join(words), "count": len(words),
                       "forbidden": any(gate.is_disallowed_insertion(w) for w in words) or " ".join(words) in gate.DISALLOWED_INSERTION_PHRASES})
            info = sf.info(path)
            regions = gate.unaligned_regions(transcription, audio_seconds=info.duration,
                                            max_unaligned_seconds=gate.DEFAULT_MAX_UNALIGNED_SECONDS)
            records.append({"index": index, "expected": chunk.text, "recognized": transcription['text'],
                            "wer": comparison.wer, "insertions": insertions, "unaligned_regions": regions,
                            "audio_seconds": info.duration, "wav_sha256": key['wav_sha256']})
            print(f"ASR {variant} {index+1}/{len(chunks)} WER {comparison.wer:.3%}", flush=True)
        score = jiwer.process_words(refs, hyps)
        long_insertions = [{"index": r['index'], **i} for r in records for i in r['insertions'] if i['count'] > gate.DEFAULT_MAX_INSERTION_WORDS]
        forbidden = [{"index": r['index'], **i} for r in records for i in r['insertions'] if i['forbidden']]
        high = [r['index'] for r in records if r['wer'] > gate.DEFAULT_MAX_CHUNK_WER]
        gaps = [{"index": r['index'], **g} for r in records for g in r['unaligned_regions']]
        save(directory / 'audit.json', {"variant": variant, "asr_model": args.asr_model,
             "source_sha256": post['source_sha256'], "narration_digest": post['digest'],
             "reference_words": sum(len(r) for r in score.references), "wer": score.wer,
             "substitutions": score.substitutions, "deletions": score.deletions, "insertions": score.insertions,
             "long_insertions": long_insertions, "forbidden_insertions": forbidden,
             "high_error_chunks": high, "unaligned_regions": gaps, "chunks": records,
             "passed": score.wer <= gate.DEFAULT_MAX_WER and not (long_insertions or forbidden or high or gaps),
             "perceptual_review": "Not established by ASR; listen to complete narration and joins before adoption."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post', default='from-bev-features-to-lane-graphs-and-changing-maps')
    parser.add_argument('--stage', choices=['generate', 'audit'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--voices', nargs='+', choices=['F2', 'M2'], default=['F2', 'M2'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--asr-model', default='mlx-community/whisper-large-v3-turbo')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve(); args.output_dir.mkdir(parents=True, exist_ok=True)
    gen = load('benchmark_generator', ROOT / 'scripts/generate-blog-audio.py')
    gate = load('benchmark_gate', ROOT / 'scripts/audit-blog-audio.py')
    post = next(p for p in gen.discover_posts() if p['slug'] == args.post)
    chunks = post['narration_chunks'][:args.limit]
    if args.stage == 'generate':
        if args.checkpoint is None:
            parser.error('--checkpoint is required for generation')
        generate(args, post, chunks, gen)
    else:
        audit(args, post, chunks, gen, gate)


if __name__ == '__main__':
    main()
