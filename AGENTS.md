# Agent Notes

## Math In Posts

- For markdown posts, use `$...$` for inline math and `$$...$$` for display math.
- Do not use `\(...\)` or `\[...\]` in markdown posts. This repo's `remark-math` pipeline is validated against dollar-style delimiters.
- Before shipping math-heavy post edits, run `npm run ci`. The test suite and build verification now check for unsupported delimiters and for rendered KaTeX output on the attention post.

## KaTeX Assets

- If you change the KaTeX CDN URL or version in `src/layouts/BaseLayout.astro`, update the stylesheet SRI hash at the same time.
- A bad KaTeX CSS integrity hash causes browsers to reject the stylesheet, which makes equations show both MathML and KaTeX HTML at once.
