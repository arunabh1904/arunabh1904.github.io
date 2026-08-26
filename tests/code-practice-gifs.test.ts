import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import {
  CODE_PRACTICE_GIFS,
  CODE_PRACTICE_GIF_TIMING,
  FPS,
  HEIGHT,
  TRANSITION_SECONDS,
  WIDTH,
} from '../scripts/generate-code-practice-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const imageDir = path.join(projectRoot, 'public', 'assets', 'images');

describe('Code Practice teaching GIFs', () => {
  it('covers every GIF referenced by a Code Practice problem', async () => {
    const source = await readFile(path.join(projectRoot, 'src', 'lib', 'code-practice.ts'), 'utf8');
    const referenced = new Set(
      [...source.matchAll(/\/assets\/images\/(code-[^'"\s]+\.gif)/g)].map((match) => match[1]),
    );

    expect([...referenced].sort()).toEqual([...CODE_PRACTICE_GIFS].sort());
  });

  it('holds each concept long enough to read and dissolves between slides', async () => {
    expect(TRANSITION_SECONDS).toBeGreaterThanOrEqual(0.5);
    expect(TRANSITION_SECONDS).toBeLessThanOrEqual(1);

    for (const filename of CODE_PRACTICE_GIFS) {
      const timing = CODE_PRACTICE_GIF_TIMING[filename];
      const filePath = path.join(imageDir, filename);
      const [metadata, file] = await Promise.all([
        sharp(filePath, { animated: true }).metadata(),
        stat(filePath),
      ]);
      const expectedDuration = timing.slideCount * timing.secondsPerSlide * 1000;
      const actualDuration = (metadata.delay ?? []).reduce((sum, delay) => sum + delay, 0);

      expect(timing.secondsPerSlide, filename).toBeGreaterThanOrEqual(6.5);
      expect(metadata.width, filename).toBe(WIDTH);
      expect(metadata.pageHeight, filename).toBe(HEIGHT);
      expect(metadata.pages, filename).toBeGreaterThanOrEqual((expectedDuration / 1000) * FPS - 1);
      expect(actualDuration, filename).toBeGreaterThanOrEqual(expectedDuration - 150);
      expect(actualDuration, filename).toBeLessThanOrEqual(expectedDuration + 150);
      expect(file.size, filename).toBeLessThan(1_500_000);
    }
  });
});
