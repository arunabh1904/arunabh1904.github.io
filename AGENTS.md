# Agent Notes

## Math In Posts

- Use `$...$` for inline math and `$$...$$` for display math in markdown posts.
- Do not use `\(...\)` or `\[...\]` in markdown posts. This repo's math pipeline and tests are built around dollar-style delimiters.
- After math-heavy content edits, run `npm run ci`. The test suite rejects unsupported delimiters and the build verification checks that math posts emit KaTeX markup.

## KaTeX Assets

- If you update the KaTeX CDN URL or version in `src/layouts/BaseLayout.astro`, update the stylesheet SRI hash too.
- A bad KaTeX CSS integrity hash causes browsers to reject the stylesheet, which makes equations render as a broken mix of MathML and visible KaTeX HTML.
