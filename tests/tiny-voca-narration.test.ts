import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
function run(body: string) {
  const result = spawnSync('python3', ['-c', `
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('g', 'scripts/generate-blog-audio.py')
g = importlib.util.module_from_spec(spec)
sys.modules['g'] = g
spec.loader.exec_module(g)
${body}
`], { cwd: root, encoding: 'utf8' });
  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return JSON.parse(result.stdout);
}

describe('tiny-voca production contract', () => {
  it('spells the technical ICL acronym instead of allowing a chat expansion', () => {
    expect(run(`
import re
p = next(p for p in g.discover_posts() if p['slug'].startswith('building-local'))
print(json.dumps({
    'written_acronym_retained': bool(re.search(r'\\bICL\\b', p['text'])),
    'spoken_letters': 'I C L' in p['tts_text'],
    'unexpanded_acronym': bool(re.search(r'\\bICL\\b', p['tts_text'])),
}))
`)).toEqual({ written_acronym_retained: true, spoken_letters: true, unexpanded_acronym: false });
  });

  it('keeps exact benchmark synthesis text and imports no model runtime for checks', () => {
    expect(run(`
p = next(p for p in g.discover_posts() if p['slug'].startswith('from-bev'))
baseline = json.loads(Path('docs/benchmarks/tiny-voca-2026-11-16/M2-generation.json').read_text())
print(json.dumps({
    'same_text': [c.text for c in p['narration_chunks']] == [c['text'] for c in baseline['chunks']],
    'profile': p['synthesis_profile'], 'full_source': p['narration_mode'],
    'loaded_runtime': any(n in sys.modules for n in ['torch', 'mlx_audio', 'tiny_voca', 'numpy']),
}))
`)).toEqual({ same_text: true, profile: 'tiny-voca', full_source: 'full-source', loaded_runtime: false });
  });

  it('invalidates waveform caches for voice changes and rejects misleading manifest controls', () => {
    expect(run(`
p = next(p for p in g.discover_posts() if p['slug'].startswith('from-bev'))
original = g.narration_chunk_cache_path(p, 0)
record = g.manifest_record(p, 1831)
valid = not g.narrator_profile_errors([p], {p['slug']: record})
g.tiny_voca_backend.PROFILE['voice'] = 'F2'
changed = g.narration_chunk_cache_path(p, 0)
print(json.dumps({
    'valid': valid, 'cache_changed': original != changed,
    'separate_backend': original != g.human_chunk_cache_path(p, 0),
    'wrong_voice_rejected': bool(g.narrator_profile_errors([p], {p['slug']: record})),
    'fake_controls': any(k in record for k in ['temperature', 'top_k', 'top_p', 'chunk_seed_overrides']),
}))
`)).toEqual({ valid: true, cache_changed: true, separate_backend: true, wrong_voice_rejected: true, fake_controls: false });
  });

  it('eliminates boundary steps without shortening audio or changing interior speech', () => {
    expect(run(`
audio = [0.5] * 1000
faded = g.tiny_voca_backend.fade_edges(audio, 24000)
print(json.dumps({
    'same_length': len(faded) == len(audio),
    'zero_edges': faded[0] == 0 and faded[-1] == 0,
    'interior_unchanged': faded[120:-120] == audio[120:-120],
    'source_unchanged': audio == [0.5] * 1000,
    'no_overshoot': all(0 <= x <= 0.5 for x in faded),
}))
`)).toEqual({ same_length: true, zero_edges: true, interior_unchanged: true, source_unchanged: true, no_overshoot: true });
  });
});
