# arunabh1904.github.io

Personal blog rebuilt with Astro, MDX, and light React islands for interactive components.

## Stack

- Astro
- MDX
- React islands
- Vitest
- GitHub Actions CI

## Commands

```bash
npm install
npm run dev
npm run ci
```

## Notes

- Legacy post URLs from the previous Jekyll blog are preserved.
- Content lives in `src/content/posts`.
- Static assets live in `public/assets`.
- Build verification checks both critical pages and migrated post routes.

## Structure

- `src/layouts` contains the shared page shells.
- `src/components` contains reusable UI building blocks.
- `src/lib/content.ts` contains shared post querying and date formatting helpers.
- `src/pages/[...slug].astro` renders migrated post routes from content frontmatter.
