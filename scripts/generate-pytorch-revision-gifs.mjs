import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import sharp from 'sharp';

const WIDTH = 960;
const HEIGHT = 540;
const FPS = 10;
const FRAMES = 50;
const ROOT = process.cwd();
const OUTPUT = join(ROOT, 'public/assets/images');
const SCRATCH = join(ROOT, '.tmp-pytorch-revision-gifs');

const C = {
  bg: '#071019', panel: '#101f2d', grid: '#294052', ink: '#f2f7f8', muted: '#91a8b7',
  cyan: '#59d8ff', amber: '#ffc565', pink: '#ff75ae', green: '#76e0a4', violet: '#b49dff', red: '#ff7b76',
};

const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
const mix = (a, b, t) => a + (b - a) * t;
const ease = (value) => { const t = clamp(value); return t * t * (3 - 2 * t); };
const phase = (t, start, end) => ease((t - start) / (end - start));
const alpha = (value) => clamp(value).toFixed(3);
const cycle = (frame) => frame / FRAMES;

function esc(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function rect(x, y, w, h, fill, radius = 14, opacity = 1, stroke = 'none', sw = 1) {
  const resolved = fill === C.panel ? 'url(#panel-fill)' : fill;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${resolved}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function line(x1, y1, x2, y2, stroke, sw = 2, opacity = 1, dash = '') {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}" stroke-opacity="${alpha(opacity)}" stroke-linecap="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
}

function circle(x, y, r, fill, opacity = 1, stroke = 'none', sw = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function label(value, x, y, size = 14, fill = C.ink, anchor = 'start', weight = 500, opacity = 1, spacing = 0) {
  return `<text x="${x}" y="${y}" fill="${fill}" fill-opacity="${alpha(opacity)}" font-family="SF Pro Display, SF Pro Text, Arial, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" letter-spacing="${spacing}">${esc(value)}</text>`;
}

function arrow(x1, y1, x2, y2, color, sw = 2, opacity = 1) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const len = 9 + sw;
  const wing = 4 + sw * 0.5;
  const ax = x2 - Math.cos(angle) * len;
  const ay = y2 - Math.sin(angle) * len;
  const p1 = `${ax + Math.cos(angle + Math.PI / 2) * wing},${ay + Math.sin(angle + Math.PI / 2) * wing}`;
  const p2 = `${ax + Math.cos(angle - Math.PI / 2) * wing},${ay + Math.sin(angle - Math.PI / 2) * wing}`;
  return `${line(x1, y1, x2, y2, color, sw, opacity)}<polygon points="${x2},${y2} ${p1} ${p2}" fill="${color}" fill-opacity="${alpha(opacity)}"/>`;
}

function glow(x, y, r, color, opacity = 1) {
  return `${circle(x, y, r, color, opacity * 0.13)}${circle(x, y, Math.max(3, r * 0.27), color, opacity)}`;
}

function flow(x1, y1, x2, y2, progress, color, opacity = 1) {
  const p = clamp(progress);
  return glow(mix(x1, x2, p), mix(y1, y2, p), 11, color, opacity);
}

function header(index, kicker, title) {
  return `${label(index, 46, 48, 12, C.green, 'start', 650, 1, 2.2)}${label(kicker.toUpperCase(), 84, 48, 12, C.muted, 'start', 650, 1, 2.1)}${label(title, 46, 88, 29, C.ink, 'start', 620)}`;
}

function panel(x, y, w, h, title, accent = C.grid) {
  return `${rect(x, y, w, h, C.panel, 18, 0.96, accent, 1)}${line(x + 18, y + 1, x + 78, y + 1, accent, 1.6, 0.72)}${label(title, x + 17, y + 29, 11, C.muted, 'start', 650, 1, 1.1)}`;
}

function svg(body, description) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${esc(description)}">
  <defs>
    <radialGradient id="bg-glow" cx="50%" cy="0%" r="96%"><stop offset="0%" stop-color="#17344a"/><stop offset="56%" stop-color="${C.bg}"/><stop offset="100%" stop-color="#050a10"/></radialGradient>
    <linearGradient id="panel-fill" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#142838"/><stop offset="52%" stop-color="#0f1e2b"/><stop offset="100%" stop-color="#0a151f"/></linearGradient>
    <pattern id="micro-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 L 0 0 0 32" fill="none" stroke="#8cb5c8" stroke-width="0.5" stroke-opacity="0.10"/></pattern>
  </defs>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#bg-glow)"/>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#micro-grid)" opacity="0.18"/>
  ${body}
  </svg>`;
}

function storageCell(x, y, value, color, active = 0) {
  return `${rect(x, y, 54, 48, color, 8, 0.04 + 0.12 * active, color, 0.8)}${label(value, x + 27, y + 30, 13, C.ink, 'middle', 650)}${active > 0 ? glow(x + 27, y + 24, 16, color, active * 0.65) : ''}`;
}

function stridesFrame(frame) {
  const t = cycle(frame);
  const step = Math.min(5, Math.floor(phase(t, 0.04, 0.82) * 6));
  const row = Math.floor(step / 3);
  const col = step % 3;
  const storage = row * 3 + col;
  let body = header('01', 'SHAPE · STRIDES · STORAGE OFFSET', 'Metadata maps logical indices onto storage');
  body += panel(38, 122, 430, 326, 'LOGICAL TENSOR · SHAPE (2, 3)', C.cyan);
  body += panel(492, 122, 430, 326, 'ONE STORAGE · STRIDES (3, 1)', C.violet);
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      const index = r * 3 + c;
      body += storageCell(106 + c * 92, 194 + r * 92, `${index}`, C.cyan, index === step ? 1 : 0);
      body += label(`[${r},${c}]`, 133 + c * 92, 258 + r * 92, 9, index === step ? C.cyan : C.muted, 'middle', 600);
    }
  }
  body += label(`logical index [${row}, ${col}]`, 253, 397, 12, C.ink, 'middle', 620);
  for (let i = 0; i < 6; i++) body += storageCell(525 + i * 61, 226, `${i}`, C.violet, i === storage ? 1 : 0);
  body += arrow(253, 417, 707, 328, C.green, 1.7, 0.72);
  body += label(`${row} × 3 + ${col} × 1 = storage[${storage}]`, 707, 363, 13, C.green, 'middle', 650);
  body += rect(548, 390, 318, 34, C.violet, 14, 0.04, C.violet, 0.8);
  body += label('transpose changes shape and strides—not these six values', 707, 412, 10, C.muted, 'middle', 560);
  body += label('A view changes the map; a copy creates another allocation.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'A logical two-by-three tensor walks through indices. Shape and strides map each logical index to one position in a six-element storage allocation; metadata can change this mapping without moving stored values.');
}

function aliasingFrame(frame) {
  const t = cycle(frame);
  const mutate = phase(t, 0.28, 0.62);
  const reveal = phase(t, 0.58, 0.84);
  const sharedValue = mutate > 0.5 ? '99' : '2';
  let body = header('02', 'VIEW · DETACH · CLONE · IN-PLACE', 'Graph independence is not storage independence');
  body += panel(38, 122, 884, 326, 'ONE MUTATION · THREE ALIASING OUTCOMES', C.pink);
  body += label('BASE STORAGE', 480, 172, 10, C.muted, 'middle', 650, 1, 1);
  for (let i = 0; i < 5; i++) body += storageCell(315 + i * 66, 191, i === 2 ? sharedValue : `${i}`, i === 2 ? C.pink : C.grid, i === 2 ? mutate : 0);
  body += label(mutate > 0.5 ? 'base[2] = 99' : 'base = [0, 1, 2, 3, 4]', 480, 266, 12, mutate > 0.5 ? C.pink : C.ink, 'middle', 650);

  const branches = [
    { x: 92, title: 'SLICE VIEW', sub: 'shares storage + graph', value: sharedValue, color: C.cyan },
    { x: 369, title: 'DETACH', sub: 'shares storage; no incoming edge', value: sharedValue, color: C.amber },
    { x: 646, title: 'CLONE', sub: 'new storage; graph connected', value: '2', color: C.violet },
  ];
  branches.forEach((branch, index) => {
    body += arrow(480, 274, branch.x + 111, 318, branch.color, 1.4, 0.45);
    body += rect(branch.x, 320, 222, 94, branch.color, 13, 0.04 + 0.05 * reveal, branch.color, 1);
    body += label(branch.title, branch.x + 111, 345, 10, branch.color, 'middle', 700, 1, 1);
    body += label(`observes ${branch.value}`, branch.x + 111, 374, 15, C.ink, 'middle', 650, 0.5 + 0.5 * reveal);
    body += label(branch.sub, branch.x + 111, 398, 9, C.muted, 'middle', 520);
    if (index < 2 && mutate > 0.5) body += glow(branch.x + 111, 371, 18, branch.color, reveal * 0.7);
  });
  body += label('detach cuts an autograd edge; clone breaks the storage alias.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'An in-place mutation changes the base tensor. A slice view and a detached tensor both observe the new value because they share storage, while a cloned tensor retains its copied value. Detach changes autograd connectivity, whereas clone changes storage ownership.');
}

function graphNode(x, y, w, title, subtitle, color, active = 0) {
  return `${rect(x, y, w, 62, color, 12, 0.04 + 0.09 * active, color, 1)}${label(title, x + w / 2, y + 25, 11, C.ink, 'middle', 650)}${label(subtitle, x + w / 2, y + 45, 9, color, 'middle', 600)}`;
}

function autogradFrame(frame) {
  const t = cycle(frame);
  const forward = phase(t, 0.04, 0.42);
  const backward = phase(t, 0.45, 0.88);
  const nodes = [
    { x: 55, y: 240, w: 100, title: 'X', sub: 'input', color: C.muted },
    { x: 55, y: 330, w: 100, title: 'w, b', sub: 'leaf parameters', color: C.cyan },
    { x: 233, y: 278, w: 138, title: 'Xw + b', sub: 'AddBackward', color: C.violet },
    { x: 449, y: 278, w: 138, title: 'ŷ − y', sub: 'SubBackward', color: C.amber },
    { x: 665, y: 278, w: 138, title: 'mean(square)', sub: 'MeanBackward', color: C.pink },
  ];
  let body = header('03', 'DYNAMIC GRAPH · VECTOR-JACOBIAN PRODUCTS', 'Forward records; backward traverses in reverse');
  body += panel(36, 122, 888, 326, t < 0.45 ? 'FORWARD · RECORD EXECUTED OPERATIONS' : 'BACKWARD · APPLY LOCAL DERIVATIVES', t < 0.45 ? C.violet : C.green);
  nodes.forEach((node, i) => { body += graphNode(node.x, node.y, node.w, node.title, node.sub, node.color, forward > i / 5 ? 1 : 0.2); });
  const edges = [
    [155, 271, 233, 300], [155, 361, 233, 318], [371, 309, 449, 309], [587, 309, 665, 309],
  ];
  edges.forEach((edge, i) => {
    body += arrow(...edge, C.violet, 1.6, 0.48);
    body += flow(edge[0], edge[1], edge[2], edge[3], clamp(forward * 1.5 - i * 0.16), C.violet, 0.75);
    const reverseProgress = clamp(backward * 1.5 - (edges.length - 1 - i) * 0.16);
    body += flow(edge[2], edge[3], edge[0], edge[1], reverseProgress, C.green, 0.8);
  });
  body += graphNode(832, 278, 66, 'L', 'scalar', C.red, forward);
  body += arrow(803, 309, 832, 309, C.pink, 1.5, 0.55);
  body += flow(803, 309, 832, 309, forward, C.pink, 0.8);
  body += flow(832, 309, 803, 309, backward, C.green, 0.8);
  body += label(t < 0.45 ? 'values move toward the scalar loss' : 'an upstream gradient of 1 moves toward the leaves', 480, 190, 12, t < 0.45 ? C.violet : C.green, 'middle', 650);
  body += label(backward > 0.75 ? 'w.grad and b.grad accumulate here' : 'each node computes one local VJP', 105, 418, 10, backward > 0.75 ? C.green : C.muted, 'middle', 620);
  body += label('Autograd composes local backward rules; it does not build one symbolic derivative.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'The forward pass records executed tensor operations from inputs and leaf parameters to a scalar loss. Backward starts with an upstream gradient of one and traverses the graph in reverse, composing local vector-Jacobian products and accumulating gradients on leaf parameters.');
}

function accumulationFrame(frame) {
  const t = cycle(frame);
  const stage = Math.min(4, Math.floor(t * 5));
  const stageProgress = (t * 5) % 1;
  const names = ['MICROBATCH 1', 'MICROBATCH 2', 'MICROBATCH 3', 'OPTIMIZER STEP', 'ZERO GRAD'];
  const colors = [C.cyan, C.amber, C.pink, C.green, C.violet];
  let body = header('04', 'GRADIENT ACCUMULATION · OPTIMIZER STATE', 'Several backwards can feed one parameter update');
  body += panel(38, 122, 884, 326, names[stage], colors[stage]);
  body += label('PARAMETER θ', 122, 182, 10, C.muted, 'middle', 650, 1, 1);
  body += rect(72, 202, 100, 74, C.cyan, 13, 0.05, C.cyan, 1);
  body += label(stage < 3 ? 'θ₀' : stage === 3 ? 'θ₀ → θ₁' : 'θ₁', 122, 247, 19, C.ink, 'middle', 700);
  body += arrow(181, 239, 266, 239, colors[stage], 1.7, 0.45 + 0.45 * stageProgress);

  body += label('.grad BUFFER', 480, 182, 10, C.muted, 'middle', 650, 1, 1);
  body += rect(282, 202, 396, 74, C.grid, 13, 0.05, C.grid, 1);
  for (let i = 0; i < 3; i++) {
    const filled = stage > i || (stage === i && stageProgress > 0.5);
    body += rect(302 + i * 120, 221, 102, 36, colors[i], 8, filled ? 0.15 : 0.025, colors[i], 0.8);
    body += label(filled ? `+ g${i + 1}/3` : 'empty', 353 + i * 120, 244, 11, filled ? C.ink : C.muted, 'middle', 650);
  }
  body += label(stage < 3 ? 'backward adds into the existing buffer' : stage === 3 ? 'optimizer reads g₁/3 + g₂/3 + g₃/3' : 'set_to_none=True removes the buffer', 480, 303, 11, colors[stage], 'middle', 620);

  const timelineX = [100, 290, 480, 670, 860];
  body += line(100, 366, 860, 366, C.grid, 2, 0.65);
  timelineX.forEach((x, i) => {
    const active = i === stage;
    body += circle(x, 366, active ? 12 : 7, colors[i], active ? 0.95 : 0.28, colors[i], 1);
    body += label(i < 3 ? `backward ${i + 1}` : i === 3 ? 'step' : 'clear', x, 397, 10, active ? colors[i] : C.muted, 'middle', active ? 700 : 520);
  });
  if (stage === 3) body += flow(678, 239, 754, 239, stageProgress, C.green, 0.85);
  body += label('The accumulation boundary defines the effective batch and update frequency.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'Three microbatches each run backward and add a scaled gradient into the same parameter gradient buffer. After the third backward, one optimizer step updates the parameter, then zero-grad removes the buffer before the next accumulation window.');
}

const animations = [
  ['pytorch-tensor-storage-strides.gif', stridesFrame],
  ['pytorch-tensor-aliasing.gif', aliasingFrame],
  ['pytorch-autograd-forward-backward.gif', autogradFrame],
  ['pytorch-gradient-accumulation.gif', accumulationFrame],
];

async function renderAnimation(filename, renderer) {
  const key = filename.replace(/\.gif$/, '');
  const frameDir = join(SCRATCH, key);
  rmSync(frameDir, { recursive: true, force: true });
  mkdirSync(frameDir, { recursive: true });
  for (let frame = 0; frame < FRAMES; frame++) {
    const stem = `frame-${String(frame).padStart(3, '0')}`;
    const frameSvg = renderer(frame);
    writeFileSync(join(frameDir, `${stem}.svg`), frameSvg);
    await sharp(Buffer.from(frameSvg)).png().toFile(join(frameDir, `${stem}.png`));
  }
  const outputPath = join(OUTPUT, filename);
  const filter = '[0:v]split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle';
  const result = spawnSync('ffmpeg', [
    '-hide_banner', '-loglevel', 'error', '-y', '-framerate', String(FPS),
    '-i', join(frameDir, 'frame-%03d.png'), '-filter_complex', filter, '-loop', '0', outputPath,
  ], { stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`ffmpeg failed for ${filename}`);
  console.log(`generated ${outputPath}`);
}

mkdirSync(OUTPUT, { recursive: true });
mkdirSync(SCRATCH, { recursive: true });
for (const [filename, renderer] of animations) await renderAnimation(filename, renderer);
rmSync(SCRATCH, { recursive: true, force: true });
