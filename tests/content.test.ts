import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  groupPostsByField,
  groupPostsByTag,
  sortPostsAscending,
  sortPostsDescending,
  toStaticPostSlug,
} from '../src/lib/post-utils';

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

function createPost(
  date: string,
  legacyPath: string,
  tags: string[],
  field?: string,
) {
  return {
    data: {
      date: new Date(date),
      legacyPath,
      tags,
      field,
    },
  };
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

describe('content helpers', () => {
  const posts = [
    createPost('2024-06-03', '/gamma.html', ['zeta', 'ml'], 'Optimization'),
    createPost('2024-01-15', '/alpha.html', ['ml'], 'Vision'),
    createPost('2024-03-20', '/beta.html', []),
  ];

  it('sorts posts in both directions', () => {
    expect(sortPostsAscending(posts).map((post) => post.data.legacyPath)).toEqual([
      '/alpha.html',
      '/beta.html',
      '/gamma.html',
    ]);
    expect(sortPostsDescending(posts).map((post) => post.data.legacyPath)).toEqual([
      '/gamma.html',
      '/beta.html',
      '/alpha.html',
    ]);
  });

  it('groups posts by field and tag with stable fallbacks', () => {
    expect(groupPostsByField(posts)).toEqual([
      { field: 'Optimization', posts: [posts[0]] },
      { field: 'Vision', posts: [posts[1]] },
      { field: 'Other', posts: [posts[2]] },
    ]);

    expect(groupPostsByTag(posts)).toEqual([
      { tag: 'ml', items: [posts[1], posts[0]] },
      { tag: 'Other', items: [posts[2]] },
      { tag: 'zeta', items: [posts[0]] },
    ]);
  });

  it('normalizes legacy paths into static slugs', () => {
    expect(toStaticPostSlug('/paper shorts/2024/06/03/gamma.html')).toBe(
      'paper shorts/2024/06/03/gamma',
    );
  });
});
