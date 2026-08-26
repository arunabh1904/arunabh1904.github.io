import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const specs = JSON.parse(
  readFileSync(join(root, 'src/lib/code-practice-visuals.json'), 'utf8'),
);
const outputDir = join(root, 'public/assets/images');

const WIDTH = 720;
const HEIGHT = 300;
const colors = {
  background: '#090d13',
  panel: '#101722',
  border: '#253247',
  text: '#f4f7fb',
  muted: '#9aa9bc',
  accent: '#f0c95a',
  blue: '#67b7ff',
  green: '#77d99a',
  pink: '#f28da7',
};

const esc = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;');

function splitLines(value, maxLength = 24) {
  const explicitLines = String(value).split('\n');
  const lines = [];
  for (const explicit of explicitLines) {
    const words = explicit.split(' ');
    let line = '';
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (candidate.length > maxLength && line) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    }
    if (line) lines.push(line);
  }
  return lines.slice(0, 3);
}

function textLines(x, y, value, size, fill, weight, lineHeight, maxLength = 24) {
  return splitLines(value, maxLength)
    .map((line, index) => (
      `<text x="${x}" y="${y + index * lineHeight}" fill="${fill}" `
      + `font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="${size}" `
      + `font-weight="${weight}">${esc(line)}</text>`
    ))
    .join('');
}

function arrow(x1, x2, y) {
  return `<path d="M${x1} ${y} H${x2 - 9}" stroke="${colors.accent}" stroke-width="2.5" stroke-linecap="round"/>`
    + `<path d="M${x2 - 9} ${y - 5} L${x2} ${y} L${x2 - 9} ${y + 5}" fill="none" stroke="${colors.accent}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function motif(kind) {
  const common = `fill="none" stroke-width="2" opacity="0.82"`;
  if (kind === 'boxes') {
    return `<g ${common} stroke="${colors.blue}"><rect x="627" y="24" width="34" height="26"/><rect x="644" y="34" width="34" height="26" stroke="${colors.pink}"/></g>`;
  }
  if (kind === 'geometry') {
    return `<g ${common} stroke="${colors.blue}"><circle cx="651" cy="42" r="23"/><path d="M651 42 L671 29"/><circle cx="671" cy="29" r="3" fill="${colors.accent}" stroke="none"/></g>`;
  }
  if (kind === 'ranking') {
    return `<g fill="${colors.blue}" opacity="0.82"><rect x="627" y="47" width="8" height="13"/><rect x="641" y="35" width="8" height="25"/><rect x="655" y="22" width="8" height="38"/><rect x="669" y="42" width="8" height="18"/></g>`;
  }
  if (kind === 'sequence') {
    return `<g fill="${colors.panel}" stroke="${colors.blue}" stroke-width="1.8">${[0, 1, 2, 3].map((index) => `<rect x="${617 + index * 18}" y="32" width="14" height="22" rx="3"/>`).join('')}</g>`;
  }
  if (kind === 'network' || kind === 'attention') {
    return `<g ${common} stroke="${colors.blue}"><path d="M622 29 L651 42 L680 25 M622 55 L651 42 L680 58"/><circle cx="622" cy="29" r="4" fill="${colors.blue}"/><circle cx="622" cy="55" r="4" fill="${colors.blue}"/><circle cx="651" cy="42" r="5" fill="${colors.accent}" stroke="none"/><circle cx="680" cy="25" r="4" fill="${colors.green}"/><circle cx="680" cy="58" r="4" fill="${colors.green}"/></g>`;
  }
  if (kind === 'patches' || kind === 'masks' || kind === 'matrix') {
    return `<g fill="none" stroke="${colors.blue}" stroke-width="1.4" opacity="0.82">${[0, 1, 2].flatMap((row) => [0, 1, 2].map((col) => `<rect x="${632 + col * 14}" y="${22 + row * 14}" width="12" height="12"${row === col ? ` fill="${colors.accent}" fill-opacity="0.45"` : ''}/>`)).join('')}</g>`;
  }
  return `<path d="M620 56 C636 56 638 26 652 26 C666 26 669 56 684 56" fill="none" stroke="${colors.blue}" stroke-width="2.5" opacity="0.82"/>`;
}

function render(spec) {
  const columnX = [36, 270, 504];
  const headlineSize = spec.headline.length > 48 ? 20 : spec.headline.length > 40 ? 22 : 25;
  const columns = spec.steps.map(([label, main, detail], index) => {
    const x = columnX[index];
    return `<g>`
      + `<text x="${x}" y="112" fill="${colors.accent}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" font-weight="750" letter-spacing="1.3">${esc(label)}</text>`
      + textLines(x, 145, main, 17, colors.text, 680, 23, 20)
      + textLines(x, 231, detail, 13, colors.muted, 450, 19, 28)
      + `</g>`;
  }).join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title desc">`
    + `<title id="title">${esc(spec.headline)}</title>`
    + `<desc id="desc">${esc(spec.caption)}</desc>`
    + `<rect width="${WIDTH}" height="${HEIGHT}" rx="18" fill="${colors.background}"/>`
    + `<rect x="1" y="1" width="${WIDTH - 2}" height="${HEIGHT - 2}" rx="17" fill="none" stroke="${colors.border}" stroke-width="2"/>`
    + `<text x="36" y="52" fill="${colors.text}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="${headlineSize}" font-weight="720">${esc(spec.headline)}</text>`
    + motif(spec.kind)
    + `<path d="M252 96 V267 M486 96 V267" stroke="${colors.border}" stroke-width="1"/>`
    + columns
    + arrow(231, 263, 177)
    + arrow(465, 497, 177)
    + `</svg>`;
}

mkdirSync(outputDir, { recursive: true });
for (const spec of specs) {
  writeFileSync(join(outputDir, `code-glance-${spec.id}.svg`), render(spec));
}

console.log(`generated ${specs.length} compact Code Practice visuals`);
