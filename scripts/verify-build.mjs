import { readFile } from 'node:fs/promises';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');
const manifestPath = path.join(projectRoot, 'src', 'content', 'migration-manifest.json');

const criticalPages = [
  'index.html',
  'archive.html',
  'ponderings.html',
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
    'ponderings',
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

  console.log(`Verified ${criticalPages.length} critical pages and ${manifest.length} post routes.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
