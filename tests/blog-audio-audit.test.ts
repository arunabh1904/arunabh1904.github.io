import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const auditPath = path.join(projectRoot, 'scripts', 'audit-blog-audio.py');

function normalizeTranscript(text: string) {
  const program = [
    'import importlib.util, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio_audit", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'print(module.normalize_transcript(sys.stdin.read()))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, auditPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: text,
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return result.stdout.trim();
}

function classifyInsertedWords(words: string[]) {
  const program = [
    'import importlib.util, json, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio_audit", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'print(json.dumps([module.is_disallowed_insertion(word) for word in json.loads(sys.stdin.read())]))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, auditPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: JSON.stringify(words),
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return JSON.parse(result.stdout) as boolean[];
}

function findUnalignedRegions() {
  const transcription = {
    segments: [
      {
        words: [
          { word: 'Heading', start: 0.1, end: 0.5 },
          { word: 'prose', start: 2.2, end: 2.6 },
        ],
      },
    ],
  };
  const program = [
    'import importlib.util, json, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio_audit", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'transcription = json.loads(sys.stdin.read())',
    'print(json.dumps(module.unaligned_regions(transcription, audio_seconds=4.4, max_unaligned_seconds=1.5)))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, auditPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: JSON.stringify(transcription),
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return JSON.parse(result.stdout) as Array<{ position: string; seconds: number }>;
}

describe('Blog audio artifact audit', () => {
  it('normalizes harmless ASR variants for technical units and identifiers', () => {
    expect(normalizeTranscript('30BK 17GB at 48 tokens/s on C++')).toBe(
      normalizeTranscript('30 B K, 17 gigabytes at 48 tokens per second on C P P'),
    );
  });

  it('rejects inserted fillers and non-lexical vocalizations', () => {
    expect(
      classifyInsertedWords(['um', 'Yeah', 'mmm', 'mmaaa', 'laughs', 'sensor']),
    ).toEqual([true, true, true, true, true, false]);
  });

  it('rejects long internal and trailing regions with no aligned word', () => {
    expect(findUnalignedRegions()).toEqual([
      { position: 'internal', seconds: 1.7, context: 'Heading | prose' },
      { position: 'suffix', seconds: 1.8, context: 'prose' },
    ]);
  });
});
