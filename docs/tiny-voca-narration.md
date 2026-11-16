# Tiny-voca narration

The mapping post and audio-compiler article use the M2 preset from tiny-voca.
Other posts retain their existing profile. Generation is offline CPU inference
after setup; it does not require Voxtral, MLX-Audio, a GPU, or training.
The tested environment is Python 3.12 on an Apple M5 Max, four CPU threads.
The website continues to serve static MP3 files.

## Setup and export

```sh
uv venv --python 3.12 .venv-tiny-voca
uv pip install --python .venv-tiny-voca/bin/python -r scripts/requirements-tiny-voca.txt
uv tool run --from huggingface-hub hf download n4ze3m/tiny-voca \
  --revision eac048b7a4392584725af9792460a0211fa1cac7 \
  --local-dir /path/to/tiny-voca-weights
.venv-tiny-voca/bin/python scripts/generate-blog-audio.py \
  --post from-bev-features-to-lane-graphs-and-changing-maps \
  --tiny-voca-weights /path/to/tiny-voca-weights
python3 scripts/generate-blog-audio.py --check \
  --post from-bev-features-to-lane-graphs-and-changing-maps
uv run --with mlx-whisper --with jiwer python scripts/audit-blog-audio.py \
  --post from-bev-features-to-lane-graphs-and-changing-maps
```

The final command uses the existing Apple Silicon ASR gate, which is independent
of synthesis. It does not use Voxtral. Pin/cache the ASR model for a recorded
benchmark. Inspect waveform integrity and listen to difficult names, joins,
openings, and endings before treating an automated pass as a quality judgment.
The runtime loader verifies weight, frontend-data, package-source, and dependency
pins. Changing those pins requires reevaluation, not bypassing the check.

## Export contract

- Full source and existing pronunciation transforms; heading-plus-prose audit
  requests remain bounded to 900 characters.
- M2, speed 1, pitch shift 0, pitch range 1, seed 1904 plus utterance index inside
  each request. No training and no Voxtral sampling or reroll controls.
- Upstream sentence splitting and recursive clause/word fallback for model
  length limits. Unpronounceable sentences fail the export instead of disappearing.
- Each utterance gets a 5 ms half-cosine fade at each edge. No samples are removed,
  no speech is accelerated, and no silence filter rewrites interior speech.
- Peak limit 0.99 per request; PCM16 cached requests; 240 ms paragraph and 550 ms
  section gaps; -1 dB output gain; mono 24 kHz, 64 kbit/s MP3.
- Content-addressed cache in the system temporary directory under
  `blog-audio-chunks/tiny-voca/`. Keys include the synthesis text and full profile.
  A JSON sidecar records each utterance and the cached waveform hash.
- The manifest records the source SHA, profile digest, exact model/runtime pins,
  duration, and controls. Existing profile digests are unchanged by this migration.

For a source edit, merge and deploy the prose first, then generate from a clean
checkout of that revision. Run the unchanged ASR thresholds, site CI, and MP3
integrity checks before merging the audio follow-up. Verify live MP3 hashes.

The original [November benchmark](benchmarks/tiny-voca-2026-11-16/README.md)
remains a historical comparison without the new utterance fades. Its baseline
reproduction requires the source/profile revision recorded in that report.
