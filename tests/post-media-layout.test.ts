import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const stylesheetPath = path.join(projectRoot, 'public', 'css', 'override.css');

describe('post media layout', () => {
  it('uses one aligned media column for every authored post section', async () => {
    const stylesheet = await readFile(stylesheetPath, 'utf8');

    expect(stylesheet).toContain('[data-post-section] p:has(> img)');
    expect(stylesheet).toContain('[data-post-section] p:has(> a > img)');
    expect(stylesheet).toContain('.page-content--narrow:has([data-post-section])');
    expect(stylesheet).not.toContain('width: min(68rem, calc(100vw - 20rem))');
  });

  it('keeps text-bearing GIF and SVG media locally scrollable on narrow screens', async () => {
    const stylesheet = await readFile(stylesheetPath, 'utf8');

    expect(stylesheet).toContain('[data-post-section] p:has(> a[href$=".gif"])');
    expect(stylesheet).toContain('[data-post-section] p:has(> a[href$=".svg"])');
    expect(stylesheet).toContain('[data-post-section] p:has(> img[src$=".gif"])');
    expect(stylesheet).toContain('[data-post-section] p:has(> img[src$=".svg"])');
  });
});
