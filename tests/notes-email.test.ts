import { describe, it, expect } from 'vitest';
import { spawnSync } from 'node:child_process';
import { noteFromSource, changedNotes } from '../scripts/notifications/notes-email.mjs';

function source(body = '## Summary\n\n> A geometry prior improves the map. The controlled comparison tests connectivity.\n\n## Method\n\nMethod details.\n\n**Paper:** [paper](https://arxiv.org/abs/2505.12246v2)', field = 'Mapping') {
  return `---\ntitle: 'Maps & topology'\nsection: paper-shorts\nlegacyPath: /paper shorts/2025/05/18/map.html\nfield: '${field}'\nsummary: '2025 – Maps & topology'\n---\n${body}`;
}

describe('merge notification selection', () => {
  it('uses the authored summary, canonical note URL, and original paper', () => {
    const note = noteFromSource(source());
    expect(note).toMatchObject({ title: 'Maps & topology', field: 'Mapping',
      url: 'https://arunabh1904.github.io/paper%20shorts/2025/05/18/map.html',
      paperUrl: 'https://arxiv.org/abs/2505.12246',
      summary: 'A geometry prior improves the map. The controlled comparison tests connectivity.' });
    expect(changedNotes([], [note])[0].change).toBe('Added');
  });
  it('does not notify about unfinished scaffolds or other categories', () => {
    expect(noteFromSource(source('<!-- PAPER RADAR DRAFT -->'))).toBeNull();
    expect(noteFromSource(source().replace('section: paper-shorts', 'section: blog'))).toBeNull();
  });
  it('ignores category-only changes, removed notes, and renames', () => {
    const old = noteFromSource(source())!;
    expect(changedNotes([old], [noteFromSource(source(undefined, 'BEV Perception'))])).toEqual([]);
    expect(changedNotes([old], [])).toEqual([]);
    expect(changedNotes([old], [{ ...old, url: old.url.replace('map.html', 'renamed.html') }])).toEqual([]);
  });
  it('includes substantive revisions as updates', () => {
    const old = noteFromSource(source())!;
    expect(changedNotes([old], [noteFromSource(source().replace('Method details.', 'Revised method evidence.'))])[0].change).toBe('Updated');
  });
  it('refuses a missing original paper link instead of inventing one', () => {
    expect(() => changedNotes([], [noteFromSource(source('## Summary\n\nA complete note without a source.'))])).toThrow('Missing original paper link');
  });
  it('rejects routes to another website', () => {
    expect(() => noteFromSource(source().replace('/paper shorts/2025/05/18/map.html', 'https://example.com'))).toThrow('belong to the website');
  });
  it('covers SMTP deduplication, failures, and safe email rendering', () => {
    const result = spawnSync('python3', ['scripts/notifications/test_send_notes_email.py'], { encoding: 'utf8' });
    expect(result.status, result.stdout + result.stderr).toBe(0);
  });
});
