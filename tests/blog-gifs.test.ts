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
  HEIGHT,
  WIDTH,
} from '../scripts/calm-blog-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const postsDir = path.join(projectRoot, 'src', 'content', 'posts');
const imageDir = path.join(projectRoot, 'public', 'assets', 'images');
type StoryStep = {
  title: string;
};
const stories = CALM_BLOG_STORIES as Record<string, { steps: StoryStep[] }>;
type ExplainerManifest = Record<string, { frames: Array<{ src: string; title: string; description: string }> }>;

describe('Blog GIF visual system', () => {
  it('replaces every Blog GIF with a manual frame explainer', async () => {
    const postFiles = await fg('**/*.{md,mdx}', { cwd: postsDir, absolute: true });
    const referenced = new Set<string>();

    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      const parsed = matter(source);
      if (parsed.data.section !== 'blog') continue;

      expect(source).not.toMatch(/<(?:img|source)[^>]+src=["'][^"']+\.gif/i);
      for (const match of source.matchAll(/data-blog-frame-explainer=["']([^"']+\.gif)["']/g)) {
        referenced.add(match[1]);
      }
    }

    expect([...referenced].sort()).toEqual([...CALM_BLOG_GIFS].sort());
  });

  it('renders one legible, bounded image for every complete storyboard state', async () => {
    const manifest = JSON.parse(
      await readFile(path.join(imageDir, 'blog-explainer-frames', 'manifest.json'), 'utf8'),
    ) as ExplainerManifest;

    expect(Object.keys(manifest).sort()).toEqual([...CALM_BLOG_GIFS].sort());

    for (const filename of CALM_BLOG_GIFS) {
      const story = stories[filename];
      const frames = manifest[filename].frames;
      expect(frames).toHaveLength(story.steps.length);

      for (const [index, frame] of frames.entries()) {
        expect(frame.title, `${filename}: frame ${index + 1}`).toBe(story.steps[index].title);
        const filePath = path.join(projectRoot, 'public', frame.src.replace(/^\//, ''));
        const [metadata, file] = await Promise.all([sharp(filePath).metadata(), stat(filePath)]);
        expect(metadata.width, frame.src).toBe(WIDTH);
        expect(metadata.height, frame.src).toBe(HEIGHT);
        expect(file.size, frame.src).toBeLessThan(100_000);
      }
    }
  });
});
