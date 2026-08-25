import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const exporterPath = path.join(projectRoot, 'scripts', 'generate-blog-audio.py');

function cleanMarkdown(source: string) {
  const program = [
    'import importlib.util, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'print(module.clean_markdown(sys.stdin.read()))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, exporterPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: source,
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return result.stdout.trim();
}

function renderForTts(source: string) {
  const program = [
    'import importlib.util, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'cleaned = module.shape_narration(module.clean_markdown(sys.stdin.read()))',
    'print(module.render_for_tts(cleaned))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, exporterPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: source,
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return result.stdout.trim();
}

function sectionPromptsForTts(source: string) {
  const program = [
    'import importlib.util, json, pathlib, sys',
    'path = pathlib.Path(sys.argv[1])',
    'spec = importlib.util.spec_from_file_location("blog_audio", path)',
    'module = importlib.util.module_from_spec(spec)',
    'sys.modules[spec.name] = module',
    'spec.loader.exec_module(module)',
    'cleaned = module.shape_narration(module.clean_markdown(sys.stdin.read()))',
    'print(json.dumps(module.section_prompts_for_tts(cleaned)))',
  ].join('; ');
  const result = spawnSync('python3', ['-c', program, exporterPath], {
    cwd: projectRoot,
    encoding: 'utf8',
    input: source,
  });

  expect(result.stderr).toBe('');
  expect(result.status).toBe(0);
  return JSON.parse(result.stdout) as string[];
}

describe('Blog audio extraction', () => {
  it('preserves authored prose in order while omitting visual-only material', () => {
    const source = [
      '# Evidence',
      '',
      'The first paragraph introduces the argument.',
      '',
      '| Sensor | Contribution |',
      '| --- | --- |',
      '| Camera | Semantics |',
      '',
      'The second paragraph develops it after the table.',
      '',
      '![A comparison of fusion architectures](/fusion.svg)',
      '_The figure contrasts point, query, and dense fusion._',
      '',
      '$$',
      'x = y + z',
      '$$',
      '',
      'The final paragraph completes the argument.',
      '',
      '## References',
      '',
      '- A paper that should not be narrated.',
    ].join('\n');

    const narration = cleanMarkdown(source);
    expect(narration).toBe([
      '[[BLOG_HEADING]]Evidence.[[/BLOG_HEADING]]',
      '[[BLOG_PARAGRAPH]]',
      'The first paragraph introduces the argument.',
      '[[BLOG_PARAGRAPH]]',
      'The second paragraph develops it after the table.',
      '[[BLOG_PARAGRAPH]]',
      'The final paragraph completes the argument.',
    ].join(' '));
  });

  it('skips raw HTML diagrams and their captions while preserving the spoken path', () => {
    const source = [
      '---',
      'section: blog',
      '---',
      '# Perception',
      '',
      'Preserve native evidence before fusion.',
      '',
      '<div class="compact-flow-diagram"><a href="/flow.svg"><img src="/flow.svg" alt="Sensor evidence to planning"></a></div>',
      '_The stages name visual information obligations._',
      '',
      'Prediction and planning act under a deadline.',
    ].join('\n');

    const narration = cleanMarkdown(source);
    expect(narration).toContain('Perception.');
    expect(narration).toContain('Preserve native evidence before fusion.');
    expect(narration).toContain('Prediction and planning act under a deadline.');
    expect(narration).not.toContain('compact-flow-diagram');
    expect(narration).not.toContain('Sensor evidence to planning');
    expect(narration).not.toContain('visual information obligations');
  });

  it('renders the entire post as one structured synthesis prompt', () => {
    const prompt = renderForTts([
      '# Perception',
      '',
      'The evidence is partial. The world model still has to become coherent.',
      '',
      'Planning consumes that state.',
    ].join('\n'));

    expect(prompt).toBe([
      'Perception.',
      'The evidence is partial. The world model still has to become coherent.',
      'Planning consumes that state.',
    ].join('\n'));
    expect(prompt).not.toContain('[[BLOG_');
  });

  it('uses audio-only pronunciation hints without rewriting extracted prose', () => {
    const source = 'A cyclist aligns LiDAR evidence after timestamps drift.';

    expect(cleanMarkdown(source)).toBe(source);
    expect(renderForTts(source)).toBe(
      'A sike-list aligns lie-dar evidence after time stamps drift.',
    );
  });

  it('keeps the reported sensor failure clause connected in audio only', () => {
    const source = 'Timestamps drift, or one sensor degrades.';

    expect(cleanMarkdown(source)).toBe(source);
    expect(renderForTts(source)).toBe('time stamps drift or one sensor degrades.');
  });

  it('bounds synthesis at authored heading sections rather than prose paragraphs', () => {
    const sections = sectionPromptsForTts([
      '# Perception',
      '',
      'First paragraph. Second sentence.',
      '',
      'Second paragraph stays in the same request.',
      '',
      '## Geometry',
      '',
      'A new heading starts the next request.',
    ].join('\n'));

    expect(sections).toEqual([
      'Perception.\nFirst paragraph. Second sentence.\nSecond paragraph stays in the same request.',
      'Geometry.\nA new heading starts the next request.',
    ]);
  });
});
