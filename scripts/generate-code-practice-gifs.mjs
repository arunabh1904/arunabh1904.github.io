import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import sharp from 'sharp';

const root = resolve(new URL('..', import.meta.url).pathname);
const output = join(root, 'public/assets/images');
const frames = join(root, '.tmp-code-practice-gif-frames');
mkdirSync(output, { recursive: true });
rmSync(frames, { recursive: true, force: true });
mkdirSync(frames, { recursive: true });

const colors = {
  background: '#101820',
  panel: '#172633',
  text: '#f7f3df',
  muted: '#b9c6c9',
  accent: '#ffd166',
  blue: '#75c9ff',
  green: '#8ee3a1',
  pink: '#ff9fb2',
};

const esc = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x, y, value, size, fill = colors.text, weight = 400, anchor = 'start') =>
  `<text x="${x}" y="${y}" fill="${fill}" font-family="Arial, sans-serif" font-size="${size}px" font-weight="${weight}" text-anchor="${anchor}">${esc(value)}</text>`;
const rect = (x, y, width, height, fill = colors.panel, stroke = 'none', radius = 12) =>
  `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="3"/>`;
const line = (x1, y1, x2, y2, stroke = colors.accent, width = 4) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}"/>`;
const arrow = (x1, y1, x2, y2, label) =>
  `${line(x1, y1, x2, y2)}<polygon points="${x2},${y2} ${x2 - 14},${y2 - 9} ${x2 - 14},${y2 + 9}" fill="${colors.accent}"/>${text((x1 + x2) / 2, y1 - 20, label, 17, colors.accent, 700, 'middle')}`;

function shell(title, subtitle, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620"><rect width="1000" height="620" fill="${colors.background}"/>${text(42, 58, title, 34, colors.text, 700)}${text(44, 100, subtitle, 22, colors.muted)}${body}</svg>`;
}

function grid(x, y, rows, cols, cell, values, stroke, label) {
  let body = rect(x - 12, y - 45, cols * cell + 24, rows * cell + 57);
  body += text(x, y - 14, label, 24, stroke, 700);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const left = x + col * cell;
      const top = y + row * cell;
      body += `<rect x="${left}" y="${top}" width="${cell}" height="${cell}" fill="none" stroke="${stroke}" stroke-width="2"/>`;
      body += text(left + cell / 2, top + cell / 2 + 7, values?.[row]?.[col] ?? '', 17, colors.text, 400, 'middle');
    }
  }
  return body;
}

function tensorFrames() {
  return [
    shell('Tensor shapes: the four moves you keep seeing', 'Every frame shows the shape change before the values are computed.', [
      grid(80, 210, 3, 1, 72, [['a'], ['b'], ['c']], colors.blue, '(3, 1)'),
      grid(615, 210, 1, 4, 72, [['1', '2', '3', '4']], colors.green, '(1, 4)'),
      text(500, 455, 'broadcasting compares compatible axes', 24, colors.text, 700, 'middle'),
      text(500, 505, '(3, 1) × (1, 4)  →  (3, 4)', 22, colors.accent, 400, 'middle'),
    ].join('')),
    shell('Broadcasting and expand', 'A singleton axis acts like a handle: it can be viewed across the larger axis.', [
      grid(70, 195, 3, 1, 74, [['a'], ['b'], ['c']], colors.blue, '(3, 1)'),
      grid(610, 195, 3, 4, 58, [['a', 'a', 'a', 'a'], ['b', 'b', 'b', 'b'], ['c', 'c', 'c', 'c']], colors.green, '(3, 4)'),
      arrow(235, 315, 590, 315, 'expand view'),
      text(500, 485, 'expand changes the view/shape; it does not invent new values', 19, colors.text, 400, 'middle'),
    ].join('')),
    shell('torch.cat: join along an existing axis', 'Concatenation keeps the number of dimensions and grows one chosen axis.', [
      grid(65, 205, 2, 2, 68, [['a', 'b'], ['c', 'd']], colors.blue, '(2, 2)'),
      grid(665, 205, 2, 1, 68, [['e'], ['f']], colors.pink, '(2, 1)'),
      arrow(250, 315, 645, 315, 'cat(dim=1)'),
      grid(330, 445, 2, 3, 55, [['a', 'b', 'e'], ['c', 'd', 'f']], colors.accent, '(2, 3)'),
    ].join('')),
    shell('torch.stack: add a new axis', 'Stacking requires matching shapes and places each tensor in a new slice.', [
      grid(90, 220, 2, 2, 68, [['a', 'b'], ['c', 'd']], colors.blue, 'A: (2, 2)'),
      grid(600, 220, 2, 2, 68, [['e', 'f'], ['g', 'h']], colors.green, 'B: (2, 2)'),
      arrow(270, 315, 575, 315, 'stack(dim=0)'),
      rect(315, 435, 375, 95, colors.panel, colors.accent),
      text(502, 477, '(2, 2) + (2, 2)', 19, colors.text, 400, 'middle'),
      text(502, 515, '→ (2, 2, 2)', 24, colors.accent, 700, 'middle'),
    ].join('')),
  ];
}

function patchFrames() {
  const values = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', '16']];
  const patchColors = [[colors.blue, colors.blue, colors.green, colors.green], [colors.blue, colors.blue, colors.green, colors.green], [colors.pink, colors.pink, colors.accent, colors.accent], [colors.pink, colors.pink, colors.accent, colors.accent]];
  let image = rect(68, 148, 4 * 65 + 24, 4 * 65 + 57) + text(80, 179, 'image (1, 4, 4)', 24, colors.text, 700);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      const x = 80 + col * 65;
      const y = 190 + row * 65;
      image += `<rect x="${x}" y="${y}" width="65" height="65" fill="none" stroke="${patchColors[row][col]}" stroke-width="4"/>`;
      image += text(x + 32, y + 40, values[row][col], 17, colors.text, 400, 'middle');
    }
  }
  const forward = [
    arrow(390, 320, 570, 320, 'patchify'),
    ...['[1,2,5,6]', '[3,4,7,8]', '[9,10,13,14]', '[11,12,15,16]'].map((value, index) => {
      const y = 170 + index * 75;
      return rect(610, y, 315, 55, colors.panel, [colors.blue, colors.green, colors.pink, colors.accent][index]) + text(767, y + 36, `token ${index}: ${value}`, 19, colors.text, 400, 'middle');
    }),
  ].join('');
  const reverse = [
    arrow(570, 320, 390, 320, 'unpatchify'),
    text(600, 260, 'tokens are reshaped to', 19, colors.muted),
    text(600, 300, '(grid_h, grid_w, C, P, P)', 24, colors.accent, 700),
    text(600, 360, 'then permuted to', 19, colors.muted),
    text(600, 400, '(C, grid_h, P, grid_w, P)', 24, colors.accent, 700),
    text(600, 475, 'grid axes + local pixel axes', 19, colors.text),
    text(600, 510, 'collapse back to (C, H, W)', 19, colors.text),
  ].join('');
  return [
    shell('Patch layout: reshape exposes the hidden axes', 'A 4×4, one-channel image with P=2 becomes four row-major tokens.', image + forward),
    shell('Patch layout: reshape exposes the hidden axes', 'A 4×4, one-channel image with P=2 becomes four row-major tokens.', image + reverse),
  ];
}

function bceFrames() {
  return [
    shell('A logit becomes a probability', 'Start by naming the input. z is unbounded; p must lie between 0 and 1.', [
      text(92, 205, 'z = raw model output', 28, colors.blue, 700),
      arrow(355, 196, 625, 196, 'sigmoid'),
      text(690, 205, 'p = 1 / (1 + exp(-z))', 26, colors.green, 700),
      line(120, 480, 880, 480, colors.muted, 2),
      line(500, 140, 500, 505, colors.muted, 2),
      '<line x1="120" y1="310" x2="880" y2="310" stroke="#526573" stroke-width="2" stroke-dasharray="9 9"/>',
      '<path d="M135 450 C390 450 390 170 865 170" fill="none" stroke="#75c9ff" stroke-width="9" stroke-linecap="round"/>',
      '<circle cx="500" cy="310" r="10" fill="#ffd166"/>',
      text(525, 318, 'z = 0  →  p = 0.5', 22, colors.accent, 700),
    ].join('')),
    shell('The target selects one log term', 'The full equation is easier to remember as two label cases.', [
      rect(90, 180, 365, 260, '#122b24', colors.green, 20),
      text(125, 235, 'target y = 1', 27, colors.green, 700),
      text(125, 310, 'loss = -log(p)', 36, colors.text, 700),
      text(125, 370, 'high p  →  small loss', 23, colors.muted),
      rect(545, 180, 365, 260, '#302018', colors.pink, 20),
      text(580, 235, 'target y = 0', 27, colors.pink, 700),
      text(580, 310, 'loss = -log(1 - p)', 36, colors.text, 700),
      text(580, 370, 'low p  →  small loss', 23, colors.muted),
      text(500, 510, 'BCE = -[y log(p) + (1-y) log(1-p)]', 29, colors.accent, 700, 'middle'),
    ].join('')),
    shell('Probability-input BCE', 'This function receives p. Do not apply sigmoid a second time.', [
      rect(105, 170, 790, 300, colors.panel, colors.accent, 22),
      text(150, 235, '1', 26, colors.accent, 700),
      text(205, 235, 'validate 0 ≤ p ≤ 1 and y ∈ {0, 1}', 27, colors.text, 700),
      text(150, 315, '2', 26, colors.accent, 700),
      text(205, 315, 'p_safe = clip(p, eps, 1 - eps)', 27, colors.text, 700),
      text(150, 395, '3', 26, colors.accent, 700),
      text(205, 395, 'evaluate the two log terms, then mean', 27, colors.text, 700),
      text(500, 535, 'clipping protects log(0); it does not convert logits', 23, colors.green, 700, 'middle'),
    ].join('')),
    shell('Logits-input BCE uses a different path', 'Keep the algebra stable instead of forming sigmoid → clamp → log.', [
      rect(100, 165, 800, 305, colors.panel, colors.blue, 22),
      text(500, 235, 'z may be any real number', 27, colors.blue, 700, 'middle'),
      text(500, 325, 'loss = max(z, 0) - z·y', 34, colors.text, 700, 'middle'),
      text(500, 385, '+ log(1 + exp(-|z|))', 34, colors.text, 700, 'middle'),
      text(500, 530, 'PyTorch: binary_cross_entropy_with_logits', 25, colors.accent, 700, 'middle'),
    ].join('')),
  ];
}

async function rasterize(svg, path) {
  await sharp(Buffer.from(svg)).png().toFile(path);
}

async function saveGif(svgs, filename, durationMs) {
  const prefix = join(frames, filename.replace('.gif', ''));
  const pngPaths = [];
  for (const [index, svg] of svgs.entries()) {
    const path = `${prefix}-${String(index).padStart(2, '0')}.png`;
    await rasterize(svg, path);
    pngPaths.push(path);
  }
  const outputPath = join(output, filename);
  execFileSync('ffmpeg', ['-y', '-framerate', String(1000 / durationMs), '-i', `${prefix}-%02d.png`, '-vf', 'split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=sierra2_4a', '-loop', '0', outputPath], { stdio: 'inherit' });
  return pngPaths;
}

const target = process.argv[2] ?? 'all';
if (target === 'all' || target === 'tensor') {
  await saveGif(tensorFrames(), 'code-tensor-ops-broadcasting.gif', 1800);
}
if (target === 'all' || target === 'patch') {
  await saveGif(patchFrames(), 'code-patchify-layout.gif', 2200);
}
if (target === 'all' || target === 'bce') {
  await saveGif(bceFrames(), 'code-bce-probabilities-vs-logits.gif', 2300);
}
rmSync(frames, { recursive: true, force: true });
