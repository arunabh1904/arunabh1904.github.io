import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import matter from 'gray-matter';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import {
  CALM_BLOG_STORIES,
  CALM_BLOG_GIFS,
  DEFAULT_BUILD_SECONDS,
  DEFAULT_TRANSITION_SECONDS,
  HEIGHT,
  WIDTH,
} from '../scripts/calm-blog-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const postsDir = path.join(projectRoot, 'src', 'content', 'posts');
const imageDir = path.join(projectRoot, 'public', 'assets', 'images');
type TimedStep = {
  title: string;
  seconds: number;
  buildSeconds?: number;
  transitionSeconds?: number;
};
const timedStories = CALM_BLOG_STORIES as Record<string, { steps: TimedStep[] }>;

describe('Blog GIF visual system', () => {
  it('covers every GIF referenced by a Blog post', async () => {
    const postFiles = await fg('**/*.{md,mdx}', { cwd: postsDir, absolute: true });
    const referenced = new Set<string>();

    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      const parsed = matter(source);
      if (parsed.data.section !== 'blog') continue;

      for (const match of source.matchAll(/\/assets\/images\/([^\s)"'<>]+\.gif)/g)) {
        referenced.add(match[1]);
      }
    }

    expect([...referenced].sort()).toEqual([...CALM_BLOG_GIFS].sort());
  });

  it('keeps every Blog GIF slow, legible, and bounded in size', async () => {
    for (const filename of CALM_BLOG_GIFS) {
      const filePath = path.join(imageDir, filename);
      const story = timedStories[filename];
      const [metadata, file] = await Promise.all([
        sharp(filePath, { animated: true }).metadata(),
        stat(filePath),
      ]);

      const durations = story.steps.map((step) => step.seconds);
      const expectedDurationMs = durations.reduce((sum, duration) => sum + duration, 0) * 1_000;

      expect(durations.every((duration) => typeof duration === 'number'), filename).toBe(true);
      expect(new Set(durations).size, filename).toBeGreaterThan(1);
      for (const step of story.steps) {
        const transitionSeconds = step.transitionSeconds ?? DEFAULT_TRANSITION_SECONDS;
        const buildSeconds = step.buildSeconds ?? DEFAULT_BUILD_SECONDS;
        const completedStateSeconds = step.seconds - transitionSeconds - 0.8 * buildSeconds;
        expect(completedStateSeconds, `${filename}: ${step.title}`).toBeGreaterThanOrEqual(3);
      }

      expect(metadata.width, filename).toBe(WIDTH);
      expect(metadata.pageHeight, filename).toBe(HEIGHT);
      expect(metadata.pages, filename).toBeGreaterThanOrEqual(180);
      const renderedDurationMs = (metadata.delay ?? []).reduce((sum, delay) => sum + delay, 0);
      expect(Math.abs(renderedDurationMs - expectedDurationMs), filename).toBeLessThanOrEqual(150);
      expect(file.size, filename).toBeLessThan(2_000_000);
    }
  });
});
