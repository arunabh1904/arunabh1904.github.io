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

describe('Blog audio extraction', () => {
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
});
