import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import matter from 'gray-matter';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const sourceRoot = process.env.JEKYLL_SOURCE
  ? path.resolve(process.env.JEKYLL_SOURCE)
  : path.resolve(projectRoot, '../arunabh1904.github.io');

const sectionByCategory = {
  'Paper Shorts': 'paper-shorts',
  Ponderings: 'blog',
  'Build Intuition': 'build-intuition',
  'Revision Notes': 'revision-notes',
  'Machine Learning Deep-Dives': 'ai-generated-reports',
  'My Journey So Far': 'blog',
};

function deriveSummary(content) {
  const collapsed = content
    .replace(/^```[\s\S]*?^```/gm, '')
    .replace(/^#+\s+/gm, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .split(/\n{2,}/)
    .map((chunk) => chunk.trim())
    .find((chunk) => chunk && !chunk.startsWith('---'));

  if (!collapsed) return undefined;
  return collapsed.replace(/\s+/g, ' ').slice(0, 220);
}

function normalizeTags(tags) {
  if (Array.isArray(tags) && tags.length > 0) {
    return tags.map(String);
  }
  return ['Other'];
}

function compactObject(value) {
  return Object.fromEntries(
    Object.entries(value).filter(([, entryValue]) => entryValue !== undefined),
  );
}

function computeFallbackPath(category, dateValue, slug) {
  const date = new Date(dateValue);
  const year = String(date.getUTCFullYear());
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  return `/${category.toLowerCase()}/${year}/${month}/${day}/${slug}.html`;
}

async function copyDirContents(sourceDir, destinationDir) {
  await mkdir(destinationDir, { recursive: true });
  const entries = await readdir(sourceDir);
  for (const entry of entries) {
    await cp(path.join(sourceDir, entry), path.join(destinationDir, entry), {
      recursive: true,
      force: true,
    });
  }
}

async function main() {
  const postsDir = path.join(sourceRoot, '_posts');
  const builtSiteDir = path.join(sourceRoot, '_site');
  const assetsDir = path.join(sourceRoot, 'assets');
  const cssDir = path.join(sourceRoot, 'css');

  const outputPostsDir = path.join(projectRoot, 'src', 'content', 'posts');
  const outputAssetsDir = path.join(projectRoot, 'public', 'assets');
  const outputCssDir = path.join(projectRoot, 'public', 'css');
  const manifestPath = path.join(projectRoot, 'src', 'content', 'migration-manifest.json');

  const legacyBySlug = new Map();
  const builtHtmlFiles = await fg('**/*.html', {
    cwd: builtSiteDir,
    onlyFiles: true,
  });

  for (const file of builtHtmlFiles) {
    if (!file.includes('/')) continue;
    const slug = path.basename(file, '.html');
    legacyBySlug.set(slug, `/${file}`);
  }

  const postFiles = await fg('*.md', {
    cwd: postsDir,
    absolute: true,
    onlyFiles: true,
  });

  await rm(outputPostsDir, { recursive: true, force: true });
  await mkdir(outputPostsDir, { recursive: true });
  await copyDirContents(assetsDir, outputAssetsDir);
  await mkdir(outputCssDir, { recursive: true });
  await cp(path.join(cssDir, 'override.css'), path.join(outputCssDir, 'override.css'), {
    force: true,
  });

  const manifest = [];

  for (const file of postFiles) {
    const raw = await readFile(file, 'utf8');
    const parsed = matter(raw);
    const category = Array.isArray(parsed.data.categories)
      ? parsed.data.categories[0]
      : parsed.data.categories;

    if (!category || !(category in sectionByCategory)) {
      throw new Error(`Unknown or missing category in ${path.basename(file)}`);
    }

    const postSlug = path.basename(file, '.md').replace(/^\d{4}-\d{2}-\d{2}-/, '');
    const legacyPath =
      legacyBySlug.get(postSlug) ?? computeFallbackPath(category, parsed.data.date, postSlug);

    const frontmatter = compactObject({
      title: String(parsed.data.title),
      date: new Date(parsed.data.date).toISOString(),
      section: sectionByCategory[category],
      postSlug,
      legacyPath,
      tags: normalizeTags(parsed.data.tags),
      field: parsed.data.field ? String(parsed.data.field) : undefined,
      summary: deriveSummary(parsed.content),
    });

    const content = matter.stringify(parsed.content.trimStart(), frontmatter);
    await writeFile(path.join(outputPostsDir, `${postSlug}.md`), content);

    manifest.push({
      title: frontmatter.title,
      section: frontmatter.section,
      postSlug,
      legacyPath,
      date: frontmatter.date,
      field: frontmatter.field ?? null,
    });
  }

  manifest.sort((left, right) => left.legacyPath.localeCompare(right.legacyPath));
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  console.log(`Migrated ${manifest.length} posts from ${sourceRoot}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
