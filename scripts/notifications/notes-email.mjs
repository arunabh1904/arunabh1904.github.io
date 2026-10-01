import { execFileSync } from 'node:child_process';
import { writeFileSync, appendFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import matter from 'gray-matter';

const site = 'https://arunabh1904.github.io';
const git = (...args) => execFileSync('git', args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });

export function plainText(text) {
  return text.replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<!--[^]*?-->/g, '').replace(/<[^>]*>/g, '')
    .replace(/^[>#]+\s*/gm, '').replace(/[*`]/g, '').replace(/\s+/g, ' ').trim();
}

export function noteFromSource(source) {
  const { data, content } = matter(source);
  if (data.section !== 'paper-shorts' || /PAPER RADAR DRAFT|unreviewed metadata draft/i.test(source)) return null;
  if (!data.title || !data.legacyPath) throw new Error('A note is missing its title or canonical route');
  const url = new URL(data.legacyPath, site);
  if (url.origin !== site) throw new Error('Note route must belong to the website');
  const summaryBlock = content.match(/^## Summary\s*\n([^]*?)(?=^## |$(?![^]))/m)?.[1]
    || content.match(/\*\*Summary:\*\*\s*([^]*?)(?=\n\s*\n|$)/)?.[1]
    || data.summary || '';
  const summary = plainText(summaryBlock).split(/(?<=[.!?])\s+/).slice(0, 2).join(' ');
  const paper = content.match(/https:\/\/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})(?:v\d+)?/)
    || content.match(/https:\/\/doi\.org\/[^\s)]+/);
  return {
    title: plainText(data.title), field: data.field || 'Arxiv Notes',
    url: url.href, paperUrl: paper?.[1] ? `https://arxiv.org/abs/${paper[1]}` : paper?.[0],
    summary, content: content.trim(),
  };
}

export function changedNotes(before, after) {
  const byUrl = new Map(before.map(n => [n.url, n]));
  const byPaper = new Map(before.filter(n => n.paperUrl).map(n => [n.paperUrl, n]));
  return after.flatMap(note => {
    const old = byUrl.get(note.url) || (note.paperUrl && byPaper.get(note.paperUrl));
    // Metadata-only recategorization and route changes do not need an email.
    if (old && old.content === note.content && old.title === note.title) return [];
    if (!note.paperUrl) throw new Error(`Missing original paper link: ${note.title}`);
    return [{ ...note, change: old ? 'Updated' : 'Added' }];
  });
}

function notesAt(ref) {
  return git('ls-tree', '-r', '--name-only', ref, '--', 'src/content/posts').trim().split('\n')
    .filter(path => /\.mdx?$/.test(path))
    .map(path => noteFromSource(git('show', `${ref}:${path}`))).filter(Boolean);
}

export function buildPayload(base, head) {
  for (const ref of [base, head]) {
    if (!/^[a-f0-9]{40}$/.test(ref)) throw new Error('Expected full commit SHA');
    git('cat-file', '-e', `${ref}^{commit}`);
  }
  git('merge-base', '--is-ancestor', base, head);
  git('merge-base', '--is-ancestor', head, 'origin/main');
  return { head, compareUrl: `https://github.com/arunabh1904/arunabh1904.github.io/compare/${base}...${head}`, notes: changedNotes(notesAt(base), notesAt(head)) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const payload = buildPayload(process.env.NOTES_BASE_SHA, process.env.NOTES_HEAD_SHA);
  writeFileSync('notes-email.json', JSON.stringify(payload, null, 2));
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `count=${payload.notes.length}\n`);
  console.log(`${payload.notes.length} added or updated notes`);
}
