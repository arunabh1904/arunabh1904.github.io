import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const manifestPath = path.join(projectRoot, 'src', 'content', 'migration-manifest.json');

type ManifestEntry = {
  legacyPath: string;
  section: string;
};

async function loadManifest() {
  return JSON.parse(await readFile(manifestPath, 'utf8')) as ManifestEntry[];
}

describe('migration manifest', () => {
  it('contains all migrated posts', async () => {
    const manifest = await loadManifest();
    expect(manifest).toHaveLength(23);
  });

  it('has unique legacy routes', async () => {
    const manifest = await loadManifest();
    const routes = manifest.map((entry) => entry.legacyPath);
    expect(new Set(routes).size).toBe(routes.length);
  });

  it('assigns each post to a known section', async () => {
    const manifest = await loadManifest();
    const sections = new Set(
      manifest.map((entry) => entry.section),
    );
    expect(sections).toEqual(
      new Set([
        'paper-shorts',
        'ponderings',
        'revision-notes',
        'my-journey',
      ]),
    );
  });
});
