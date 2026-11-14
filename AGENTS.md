# Agent Notes

## Website Writing Skills

- Apply the global `writing-style` skill to every prose change in this repository, including shelf copy and Code practice text.
- Also apply exactly the matching category overlay: `write-paper-note`, `write-blog-post`, `write-revision-notes`, or `write-code-practice-problem`. The Blog overlay covers essays, intuition-first technical tutorials, and multi-paper surveys.
- Use `publish-website-writing` when the request includes adding, publishing, or shipping content. It owns routing and validation; category skills own format; `writing-style` alone owns shared topic, continuity, and prose rules.
- Do not copy shared prose rules into category skills. Update `writing-style` when a rule should apply everywhere and update an overlay only when the category genuinely differs.

## A Coherent Writing Corpus

- Before drafting or revising a Blog, search the existing Blog and Arxiv Notes corpus for concepts and papers already covered. Give each substantial explanation a clear home; use a brief local recap and a contextual section link elsewhere instead of repeating a primer.
- Make callbacks part of the argument: explain what the earlier post established and what the current post adds. Use “earlier” only when chronologically accurate; call newer work a follow-up. Preserve enough context for each post and its narration to stand alone.
- Paper references in Blogs must link to the corresponding Arxiv Note (`section: paper-shorts`), including paper references in tables, reading lists, and figure captions. Resolve the exact `legacyPath` from frontmatter; never invent a route or redirect a paper citation to an unrelated note.
- If a cited paper has no note, create one with `write-paper-note` before publishing the citing Blog. Read the canonical source, follow the note format and source-figure requirements, and verify the paper identity. A placeholder or title-only stub is not a completed note.
- Keep canonical external paper, project, code, and source-figure links in the Arxiv Note so readers can reach the primary evidence. An Arxiv Note's own source links must remain external; cross-paper discussion should link to the relevant existing site note when available.
- Preserve precise provenance when moving a Blog citation into a note: retain named paper sections, tables, figure numbers, and versions in the note, using a verified note anchor when helpful. Leave non-paper documentation, model cards, repositories, and author commentary linked to their actual sources.
- Verify internal routes and section anchors after edits. Prefer meaningful callbacks over a link quota or a generic related-post list, and remove repeated explanations without erasing the author's voice, experimental controls, or evidence boundaries.

## Math In Posts

- Use `$...$` for inline math and `$$...$$` for display math in markdown posts.
- Do not use `\(...\)` or `\[...\]` in markdown posts. This repo's math pipeline and tests are built around dollar-style delimiters.
- After math-heavy content edits, run `npm run ci`. The test suite rejects unsupported delimiters and the build verification checks that math posts emit KaTeX markup.

## KaTeX Assets

- If you update the KaTeX CDN URL or version in `src/layouts/BaseLayout.astro`, update the stylesheet SRI hash too.
- A bad KaTeX CSS integrity hash causes browsers to reject the stylesheet, which makes equations render as a broken mix of MathML and visible KaTeX HTML.

## Post Images

- Prefer post diagrams that are immediately grokkable at a glance: minimal text, minimal visual noise, and one clear idea per image.
- Favor simple hand-drawn or sketch-style visuals with clean black strokes, lots of whitespace, and restrained color accents, similar to an Excalidraw-style explainer.
- For animated explanatory visuals, keep motion subtle and instructional rather than flashy. The animation should make the concept easier to follow, not add decoration.

## Code Practice

- The Code practice workspace uses CodeMirror, not a plain `textarea`. Preserve the editor-based workflow on `/code/<problem-id>.html` unless the user explicitly asks to redesign the editing experience.
- Keep the current keyboard affordances intact when touching the editor: `Tab` indents, `Shift+Tab` outdents, and `Cmd/Ctrl + /` toggles comments.
- After changing the Code practice editor or its styling, run `npm run test -- tests/code-editor.test.ts tests/code-practice-lab.test.tsx`, `npm run check`, and `npm run build`.
