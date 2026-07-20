# Agent Notes

## Website Writing Skills

- Apply the global `writing-style` skill to every prose change in this repository, including shelf copy and Code practice text.
- Also apply exactly the matching category overlay: `write-paper-note`, `write-blog-post`, `write-revision-notes`, or `write-code-practice-problem`. The Blog overlay covers essays, intuition-first technical tutorials, and multi-paper surveys.
- Use `publish-website-writing` when the request includes adding, publishing, or shipping content. It owns routing and validation; category skills own format; `writing-style` alone owns shared topic, continuity, and prose rules.
- Do not copy shared prose rules into category skills. Update `writing-style` when a rule should apply everywhere and update an overlay only when the category genuinely differs.

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
