import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { CALM_BLOG_STORIES, PERCEPTION_BLOG_GIFS } from '../scripts/calm-blog-gifs.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
type VisualStep = { title: string; seconds: number };
const perceptionStories = CALM_BLOG_STORIES as Record<string, { steps: VisualStep[] }>;
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
  'StreamPETR carries recurrent actor instances',
  'StreamPETR transforms recurrent instances',
  'Temporal memory must align motion',
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

  it('grounds every perception GIF in observable road events', () => {
    const concreteRoadTerms = /cyclist|lead car|van|curb|road|crosswalk|rain|camera|LiDAR|radar/i;

    for (const filename of PERCEPTION_BLOG_GIFS) {
      const story = perceptionStories[filename];
      const groundedSteps = story.steps.filter((step) => concreteRoadTerms.test(step.title));
      expect(groundedSteps.length, filename).toBeGreaterThanOrEqual(3);
      expect(new Set(story.steps.map((step) => step.seconds)).size, filename).toBeGreaterThan(1);
    }
  });

  it('draws one encoder lane per driving sensor before fusion', async () => {
    const source = await readFile(
      path.join(projectRoot, 'public', 'assets', 'images', 'autonomous-driving-two-speed-stack.svg'),
      'utf8',
    );

    for (const label of ['Image encoder', 'Range encoder', 'Motion encoder']) {
      expect(source).toContain(label);
    }
    for (const visualCue of ['Candidate paths', 'Learned scorer', 'Independent checks']) {
      expect(source).toContain(visualCue);
    }
    expect(source).toContain('top-k maneuvers');
    expect(source).not.toContain('three maneuvers');
    expect(source).not.toContain('safety · progress · comfort');
    expect(source).toContain('id="measured-state-to-validator"');
    expect(source).toContain('id="learned-features-to-generator"');
    expect(source).not.toContain('id="learned-features-to-validator"');
  });
});
