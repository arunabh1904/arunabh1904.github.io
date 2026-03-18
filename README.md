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
nvm use
npm ci
npm run validate
npm run build
npm run verify:build
npm run dev
```

## Notes

- Legacy post URLs from the previous Jekyll blog are preserved.
- Content lives in `src/content/posts`.
- Static assets live in `public/assets`.
- Build verification checks both critical pages and migrated post routes.
- `.nvmrc` pins the Node version used by local development and GitHub Actions.
