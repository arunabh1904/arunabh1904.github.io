import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { CALM_BLOG_STORIES } from '../scripts/calm-blog-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const authoredSvgFiles = [
  'blog-audio-release-pipeline.svg',
  'autonomous-driving-perception-system.svg',
  'autonomous-driving-two-speed-stack.svg',
  'robot-post-training-loop.svg',
];

// These phrases were removed because they compress several mechanisms into a slogan
// or metaphor. Keep the regression list narrow enough to permit technical terms when
// the visual defines them explicitly.
const bannedPhrases = [
  'two-speed perception',
  'one checked trajectory',
  'design synthesis',
  'memory carrier',
  'what survives',
  'training pressure',
  'evidence path',
  'shared parameters feel',
  'cleaner objective',
];

describe('authored Blog visual language', () => {
  it('uses literal mechanism labels instead of the removed slogan phrases', async () => {
    const sources = await Promise.all([
      readFile(path.join(projectRoot, 'scripts', 'calm-blog-gifs.mjs'), 'utf8'),
      ...authoredSvgFiles.map((filename) =>
        readFile(path.join(projectRoot, 'public', 'assets', 'images', filename), 'utf8')),
    ]);
    const visibleTextSource = sources.join('\n').toLowerCase();

    for (const phrase of bannedPhrases) {
      expect(visibleTextSource, phrase).not.toContain(phrase);
    }
  });

  it('keeps every GIF scene title within one readable line', () => {
    for (const [filename, story] of Object.entries(CALM_BLOG_STORIES)) {
      for (const step of story.steps) {
        expect(step.title.length, `${filename}: ${step.title}`).toBeLessThanOrEqual(72);
      }
    }
  });
});
