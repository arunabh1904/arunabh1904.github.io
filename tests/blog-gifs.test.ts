import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import matter from 'gray-matter';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import {
  CALM_BLOG_GIFS,
  HEIGHT,
  WIDTH,
} from '../scripts/calm-blog-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const postsDir = path.join(projectRoot, 'src', 'content', 'posts');
const imageDir = path.join(projectRoot, 'public', 'assets', 'images');

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
      const [metadata, file] = await Promise.all([
        sharp(filePath, { animated: true }).metadata(),
        stat(filePath),
      ]);

      expect(metadata.width, filename).toBe(WIDTH);
      expect(metadata.pageHeight, filename).toBe(HEIGHT);
      expect(metadata.pages, filename).toBeGreaterThanOrEqual(180);
      expect(
        (metadata.delay ?? []).reduce((sum, delay) => sum + delay, 0),
        filename,
      ).toBeGreaterThanOrEqual(20_000);
      expect(file.size, filename).toBeLessThan(2_000_000);
    }
  });
});
