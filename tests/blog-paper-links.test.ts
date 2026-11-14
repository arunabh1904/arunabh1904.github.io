import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import matter from 'gray-matter';
import { expect, it } from 'vitest';

const postsDir = fileURLToPath(new URL('../src/content/posts/', import.meta.url));

it('routes Blog paper citations through existing Arxiv Notes', async () => {
  const files = await fg('*.{md,mdx}', { cwd: postsDir });
  const posts = await Promise.all(files.map(async (file) => ({
    file,
    ...matter(await readFile(path.join(postsDir, file), 'utf8')),
  })));
  const noteRoutes = new Set(posts
    .filter(({ data }) => data.section === 'paper-shorts')
    .map(({ data }) => data.legacyPath));
  const errors: string[] = [];

  for (const { file, data, content } of posts) {
    if (data.section !== 'blog') continue;
    for (const match of content.matchAll(/https?:\/\/(?:www\.)?(?:arxiv\.org\/(?:abs|pdf|html)\/|ar5iv\.labs\.arxiv\.org\/html\/|doi\.org\/10\.48550\/arXiv\.)[^\s)<>"']+/gi)) {
      errors.push(`${file}: direct paper citation ${match[0]}; create or link its Arxiv Note`);
    }
    for (const match of content.matchAll(/(?:\]\(|href=["'])(\/paper(?:%20| )shorts\/[^)"']+)/g)) {
      const route = decodeURIComponent(match[1].split(/[?#]/)[0]);
      if (!noteRoutes.has(route)) errors.push(`${file}: missing Arxiv Note ${route}`);
    }
  }

  expect(errors, 'Follow AGENTS.md: Blog paper references belong in the Arxiv Notes reading path.').toEqual([]);
});
