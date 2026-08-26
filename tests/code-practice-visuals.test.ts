import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { codePracticeProblems } from '../src/lib/code-practice';
import visualSpecs from '../src/lib/code-practice-visuals.json';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const imageDir = path.join(projectRoot, 'public', 'assets', 'images');

describe('Code Practice at-a-glance visuals', () => {
  it('gives every problem one problem-specific visual', () => {
    const problemIds = codePracticeProblems.map((problem) => problem.id).sort();
    const visualIds = visualSpecs.map((visual) => visual.id).sort();

    expect(visualIds).toEqual(problemIds);
    expect(new Set(visualIds).size).toBe(visualIds.length);

    for (const problem of codePracticeProblems) {
      expect(problem.visual?.src, problem.id).toBe(
        `/assets/images/code-glance-${problem.id}.svg`,
      );
      expect(problem.visual?.alt.trim(), problem.id).not.toBe('');
      expect(problem.visual?.caption.trim(), problem.id).not.toBe('');
    }
  });

  it('keeps every visual compact, static, and readable in one glance', async () => {
    for (const spec of visualSpecs) {
      expect(spec.steps, spec.id).toHaveLength(3);
      expect(spec.headline.length, spec.id).toBeLessThanOrEqual(64);
      expect(spec.caption.length, spec.id).toBeLessThanOrEqual(140);

      const filePath = path.join(imageDir, `code-glance-${spec.id}.svg`);
      const [source, file] = await Promise.all([readFile(filePath, 'utf8'), stat(filePath)]);

      expect(source, spec.id).toContain('width="720" height="300"');
      expect(source, spec.id).toContain('<title id="title">');
      expect(source, spec.id).not.toMatch(/<(?:animate|animateTransform|set)\b/);
      expect(file.size, spec.id).toBeLessThan(8_000);
    }
  });
});
