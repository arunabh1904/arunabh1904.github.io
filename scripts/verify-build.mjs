import { readFile } from 'node:fs/promises';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';
import matter from 'gray-matter';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');
const manifestPath = path.join(projectRoot, 'src', 'content', 'migration-manifest.json');
const postsDir = path.join(projectRoot, 'src', 'content', 'posts');

const criticalPages = [
  'index.html',
  'archive.html',
  'blog.html',
  'build_intuition.html',
  'revision_notes.html',
  'ai_generated_reports.html',
  'paper_summaries.html',
  'paper_summaries_field.html',
  'music.html',
  'feed.xml',
];

function assertExists(targetPath) {
  if (!fs.existsSync(targetPath)) {
    throw new Error(`Expected build output missing: ${targetPath}`);
  }
}

async function getMathPostRoutes() {
  const postFiles = await fg('**/*.{md,mdx}', {
    cwd: postsDir,
    absolute: true,
  });

  const routes = [];
  for (const filePath of postFiles) {
    const source = await readFile(filePath, 'utf8');
    if (!/\$\$|\$[^$\n]+\$/.test(source)) {
      continue;
    }

    const { data } = matter(source);
    if (typeof data.legacyPath !== 'string') {
      throw new Error(`Expected legacyPath in frontmatter for ${path.relative(projectRoot, filePath)}.`);
    }

    routes.push(data.legacyPath.trim());
  }

  return routes;
}

async function main() {
  for (const page of criticalPages) {
    assertExists(path.join(distDir, page));
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  for (const entry of manifest) {
    const relativePath = entry.legacyPath.replace(/^\//, '');
    assertExists(path.join(distDir, relativePath));
  }

  const attentionPagePath = path.join(
    distDir,
    'build intuition',
    '2025',
    '05',
    '25',
    'attention-mechanisms-demystified.html',
  );
  const attentionHtml = await readFile(attentionPagePath, 'utf8');
  if (!attentionHtml.includes('katex-display')) {
    throw new Error('Expected the attention mechanisms post to contain rendered KaTeX output.');
  }
  if (attentionHtml.includes('<h1 id="operatornameattentionq-k-v">[')) {
    throw new Error('Attention mechanisms post still contains raw markdown-mangled math output.');
  }

  const mathRoutes = await getMathPostRoutes();
  for (const route of mathRoutes) {
    const relativePath = route.replace(/^\//, '');
    const builtPath = path.join(distDir, relativePath);
    const html = await readFile(builtPath, 'utf8');
    if (!html.includes('class="katex"')) {
      throw new Error(`Expected rendered KaTeX output in ${relativePath}.`);
    }
  }

  console.log(`Verified ${criticalPages.length} critical pages and ${manifest.length} post routes.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
