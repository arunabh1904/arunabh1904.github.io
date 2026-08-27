import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import {
  groupPostsByField,
  groupPostsByTag,
  sortPostsAscending,
  sortPostsDescending,
  toStaticPostSlug,
} from '../src/lib/post-utils';
import { groupBlogPosts } from '../src/lib/blog-index';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const manifestPath = path.join(projectRoot, 'src', 'content', 'migration-manifest.json');
const postsDir = path.join(projectRoot, 'src', 'content', 'posts');

const paperNotePattern = /^section:\s*['"]?paper-shorts['"]?\s*$/m;
const localImagePattern = /^!\[[^\]]*\]\((\/assets\/images\/[^)]+)\)$/;
const figureCaptionPattern = /^\*Fig (\d+): (.+) \| source: \[([^\]]+)\]\((https?:\/\/[^)]+)\)\*$/;

function differenceHash(buffer: Buffer, width: number, height: number) {
  let hash = 0n;
  let bit = 1n;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      if (buffer[y * width + x] > buffer[y * width + x + 1]) {
        hash |= bit;
      }
      bit <<= 1n;
    }
  }
  return hash;
}

function averageHash(buffer: Buffer) {
  const mean = buffer.reduce((sum, value) => sum + value, 0) / buffer.length;
  let hash = 0n;
  let bit = 1n;
  for (const value of buffer) {
    if (value >= mean) {
      hash |= bit;
    }
    bit <<= 1n;
  }
  return hash;
}

function hammingDistance(left: bigint, right: bigint) {
  let value = left ^ right;
  let distance = 0;
  while (value) {
    distance += Number(value & 1n);
    value >>= 1n;
  }
  return distance;
}

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
        'blog',
        'revision-notes',
      ]),
    );
  });
});

describe('markdown authoring', () => {
  it('uses remark-math compatible delimiters in posts', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const offenders: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (/\\\(|\\\)|\\\[|\\\]/.test(source)) {
        offenders.push(path.relative(projectRoot, filePath));
      }
    }

    expect(
      offenders,
      'Use $...$ or $$...$$ instead of \\(...\\) or \\[...\\] in markdown posts.',
    ).toEqual([]);
  });

  it('keeps paper-note summaries marked as callouts', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const offenders: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (
        /^section:\s*['"]?paper-shorts['"]?\s*$/m.test(source) &&
        !/PAPER RADAR DRAFT/.test(source) &&
        !/^## Summary\n\n> /m.test(source)
      ) {
        offenders.push(path.relative(projectRoot, filePath));
      }
    }

    expect(offenders, 'Paper-note summaries should render as callouts.').toEqual([]);
  });

  it('keeps one to three valid local images in every published paper note', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const countOffenders: string[] = [];
    const missingAssets: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (
        !/^section:\s*['"]?paper-shorts['"]?\s*$/m.test(source) ||
        /PAPER RADAR DRAFT/.test(source)
      ) {
        continue;
      }

      const imagePaths = Array.from(
        source.matchAll(/^!\[[^\]]*\]\((\/assets\/images\/[^)]+)\)$/gm),
        (match) => match[1],
      );
      if (imagePaths.length < 1 || imagePaths.length > 3) {
        countOffenders.push(
          `${path.relative(projectRoot, filePath)} (${imagePaths.length} images)`,
        );
      }

      for (const imagePath of imagePaths) {
        const assetPath = path.join(projectRoot, 'public', imagePath);
        try {
          await readFile(assetPath);
        } catch {
          missingAssets.push(
            `${path.relative(projectRoot, filePath)} -> ${imagePath}`,
          );
        }
      }
    }

    expect(
      countOffenders,
      'Published paper notes must contain one to three local explanatory images.',
    ).toEqual([]);
    expect(missingAssets, 'Every paper-note image must exist in public/.').toEqual([]);
  });

  it('uses explanatory, sequential captions for every paper-note image', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const offenders: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (!paperNotePattern.test(source) || /PAPER RADAR DRAFT/.test(source)) {
        continue;
      }

      const lines = source.split(/\r?\n/);
      let expectedFigure = 1;
      for (let index = 0; index < lines.length; index += 1) {
        if (!localImagePattern.test(lines[index])) {
          continue;
        }

        let captionIndex = index + 1;
        while (captionIndex < lines.length && lines[captionIndex].trim() === '') {
          captionIndex += 1;
        }
        const caption = lines[captionIndex]?.trim() ?? '';
        const match = caption.match(figureCaptionPattern);
        const location = `${path.relative(projectRoot, filePath)}:${captionIndex + 1}`;
        if (!match) {
          offenders.push(`${location} malformed or missing caption`);
          continue;
        }

        const [, figureNumber, explanation, sourceLabel] = match;
        const wordCount = explanation.split(/\s+/).filter(Boolean).length;
        if (Number(figureNumber) !== expectedFigure) {
          offenders.push(`${location} expected Fig ${expectedFigure}, found Fig ${figureNumber}`);
        }
        if (wordCount < 8 || wordCount > 60) {
          offenders.push(`${location} explanation has ${wordCount} words`);
        }
        if (
          !sourceLabel.trim() ||
          /https?:|\[[^\]]+\]\([^)]+\)|org\/abs/i.test(explanation) ||
          /\(\s*\)|\{\s*,?\s*\}|,,/.test(explanation)
        ) {
          offenders.push(`${location} contains source or extraction residue`);
        }
        expectedFigure += 1;
      }
    }

    expect(
      offenders,
      'Use `Fig <n>: <explanation> | source: [paper](URL)` with note-local numbering and a concise explanation.',
    ).toEqual([]);
  });

  it('does not repeat exact or perceptually duplicate images within a paper note', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });
    const fingerprintCache = new Map<string, {
      contentHash: string;
      differenceHash: bigint;
      averageHash: bigint;
    }>();
    const offenders: string[] = [];
    // These KTO figures share a plot template but compare different objectives and scales.
    const visuallyDistinctNearMatch = new Set([
      [
        '/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-2.webp',
        '/assets/images/kto-model-alignment-as-prospect-theoretic-optimization-source-figure-3.webp',
      ].sort().join('|'),
    ]);

    async function fingerprint(imagePath: string) {
      const cached = fingerprintCache.get(imagePath);
      if (cached) {
        return cached;
      }
      const assetPath = path.join(projectRoot, 'public', imagePath);
      const file = await readFile(assetPath);
      const differencePixels = await sharp(file, { animated: false, pages: 1 })
        .flatten({ background: '#ffffff' })
        .grayscale()
        .resize(17, 16, { fit: 'fill' })
        .raw()
        .toBuffer();
      const averagePixels = await sharp(file, { animated: false, pages: 1 })
        .flatten({ background: '#ffffff' })
        .grayscale()
        .resize(16, 16, { fit: 'fill' })
        .raw()
        .toBuffer();
      const result = {
        contentHash: createHash('sha256').update(file).digest('hex'),
        differenceHash: differenceHash(differencePixels, 17, 16),
        averageHash: averageHash(averagePixels),
      };
      fingerprintCache.set(imagePath, result);
      return result;
    }

    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (!paperNotePattern.test(source) || /PAPER RADAR DRAFT/.test(source)) {
        continue;
      }
      const imagePaths = source
        .split(/\r?\n/)
        .map((line) => line.match(localImagePattern)?.[1])
        .filter((imagePath): imagePath is string => Boolean(imagePath));
      const fingerprints = await Promise.all(imagePaths.map(fingerprint));

      for (let left = 0; left < imagePaths.length; left += 1) {
        for (let right = left + 1; right < imagePaths.length; right += 1) {
          const pairKey = [imagePaths[left], imagePaths[right]].sort().join('|');
          const exactDuplicate =
            imagePaths[left] === imagePaths[right] ||
            fingerprints[left].contentHash === fingerprints[right].contentHash;
          const perceptualDuplicate =
            hammingDistance(
              fingerprints[left].differenceHash,
              fingerprints[right].differenceHash,
            ) <= 40 &&
            hammingDistance(
              fingerprints[left].averageHash,
              fingerprints[right].averageHash,
            ) <= 40 &&
            !visuallyDistinctNearMatch.has(pairKey);
          if (exactDuplicate || perceptualDuplicate) {
            offenders.push(
              `${path.relative(projectRoot, filePath)} -> ${imagePaths[left]} <> ${imagePaths[right]}`,
            );
          }
        }
      }
    }

    expect(
      offenders,
      'Paper notes should not repeat the same figure through alternate filenames, formats, or crops.',
    ).toEqual([]);
  }, 30_000);

  it('keeps Blog callouts sparse instead of enforcing a quota', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const offenders: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (!/^section: blog$/m.test(source)) {
        continue;
      }

      const calloutCount = source
        .split(/\n{2,}/)
        .filter((block) => block.trimStart().startsWith('> ')).length;

      if (calloutCount > 3) {
        offenders.push(
          `${path.relative(projectRoot, filePath)} (${calloutCount} callouts)`,
        );
      }
    }

    expect(
      offenders,
      'Blog callouts should remain sparse and earned.',
    ).toEqual([]);
  });

  it('renders multi-stage Blog processes as diagrams instead of arrow chains', async () => {
    const postFiles = await fg('**/*.{md,mdx}', {
      cwd: postsDir,
      absolute: true,
    });

    const offenders: string[] = [];
    for (const filePath of postFiles) {
      const source = await readFile(filePath, 'utf8');
      if (!/^section:\s*['"]?blog['"]?\s*$/m.test(source)) {
        continue;
      }

      let inFence = false;
      source.split('\n').forEach((line, index) => {
        if (/^\s*```/.test(line)) {
          inFence = !inFence;
          return;
        }
        if (inFence || /^\s*\|/.test(line)) {
          return;
        }

        const arrowCount = line.match(/(?:→|←|↔|⇄|⇒|⇐|⟶|->|<-|=>)/g)?.length ?? 0;
        if (arrowCount >= 3) {
          offenders.push(`${path.relative(projectRoot, filePath)}:${index + 1}`);
        }
      });
    }

    expect(
      offenders,
      'Processes with four or more named stages should use a compact diagram, not an inline arrow chain.',
    ).toEqual([]);
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

describe('blog index', () => {
  it('groups posts into fixed reading paths and preserves any ungrouped post', () => {
    const posts = [
      { data: { blogGroup: 'projects' as const } },
      { data: { blogGroup: 'research-guides' as const } },
      { data: {} },
    ];

    expect(groupBlogPosts(posts).map((group) => [group.id, group.posts.length])).toEqual([
      ['research-guides', 1],
      ['projects', 1],
      ['other', 1],
    ]);
  });
});
