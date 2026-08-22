import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let sharp;
try {
  sharp = require('sharp');
} catch {
  const fallback = process.env.BLOG_FIGURE_SHARP_PATH;
  if (!fallback) throw new Error('Install dependencies or set BLOG_FIGURE_SHARP_PATH.');
  sharp = require(fallback);
}

const WIDTH = 960;
const HEIGHT = 540;
const FPS = 10;
const FRAMES = 50;
const ROOT = process.cwd();
const OUTPUT = join(ROOT, 'public/assets/images');
const SCRATCH = join(ROOT, '.tmp-blog-intuition-gifs');

const C = {
  bg: '#071019',
  bg2: '#0b1722',
  panel: '#101f2d',
  panel2: '#142636',
  grid: '#294052',
  ink: '#f2f7f8',
  muted: '#91a8b7',
  dim: '#526b7b',
  cyan: '#59d8ff',
  amber: '#ffc565',
  pink: '#ff75ae',
  green: '#76e0a4',
  violet: '#b49dff',
  red: '#ff7b76',
};

const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const mix = (a, b, t) => a + (b - a) * t;
const ease = (v) => { const t = clamp(v); return t * t * (3 - 2 * t); };
const phase = (t, start, end) => ease((t - start) / (end - start));
const pulse = (t) => 0.5 - 0.5 * Math.cos(t * Math.PI * 2);
const alpha = (v) => clamp(v).toFixed(3);
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

function path(d, stroke, sw = 2, opacity = 1, fill = 'none', dash = '') {
  return `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" stroke-opacity="${alpha(opacity)}" stroke-linecap="round" stroke-linejoin="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
}

function circle(x, y, r, fill, opacity = 1, stroke = 'none', sw = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

function label(value, x, y, size = 14, fill = C.ink, anchor = 'start', weight = 500, opacity = 1, spacing = 0) {
  return `<text x="${x}" y="${y}" fill="${fill}" fill-opacity="${alpha(opacity)}" font-family="SF Pro Display, SF Pro Text, Arial, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" letter-spacing="${spacing}">${esc(value)}</text>`;
}

function glow(x, y, r, color, opacity = 1) {
  return `${circle(x, y, r, color, opacity * 0.14)}${circle(x, y, Math.max(2, r * 0.28), color, opacity)}`;
}

function arrow(x1, y1, x2, y2, color, sw = 2, opacity = 1) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const len = 9 + sw;
  const wing = 4 + sw * 0.5;
  const ax = x2 - Math.cos(a) * len;
  const ay = y2 - Math.sin(a) * len;
  const p1 = `${ax + Math.cos(a + Math.PI / 2) * wing},${ay + Math.sin(a + Math.PI / 2) * wing}`;
  const p2 = `${ax + Math.cos(a - Math.PI / 2) * wing},${ay + Math.sin(a - Math.PI / 2) * wing}`;
  return `${line(x1, y1, x2, y2, color, sw, opacity)}<polygon points="${x2},${y2} ${p1} ${p2}" fill="${color}" fill-opacity="${alpha(opacity)}"/>`;
}

function flow(x1, y1, x2, y2, p, color, opacity = 1) {
  const q = clamp(p);
  return glow(mix(x1, x2, q), mix(y1, y2, q), 11, color, opacity);
}

function header(index, kicker, title) {
  return `${label(index, 46, 48, 12, C.green, 'start', 650, 1, 2.2)}${label(kicker.toUpperCase(), 84, 48, 12, C.muted, 'start', 650, 1, 2.1)}${label(title, 46, 88, 29, C.ink, 'start', 620)}`;
}

function panel(x, y, w, h, title, accent = C.grid) {
  return `${rect(x, y, w, h, C.panel, 18, 0.96, accent, 1)}${path(`M ${x + 18} ${y + 1} L ${x + 78} ${y + 1}`, accent, 1.6, 0.72)}${label(title, x + 17, y + 29, 11, C.muted, 'start', 650, 1, 1.1)}`;
}

function token(x, y, text, color, opacity = 1, w = 42) {
  return `${rect(x, y, w, 28, color, 7, 0.06 + 0.07 * opacity, color, 1)}${label(text, x + w / 2, y + 19, 10, C.ink, 'middle', 560, opacity)}`;
}

function dotGrid(x, y, cols, rows, color, active, cell = 14) {
  let out = '';
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      out += rect(x + i * cell, y + j * cell, cell - 3, cell - 3, k < active ? color : C.grid, 3, k < active ? 0.22 : 0.12, k < active ? color : C.grid, 0.6);
    }
  }
  return out;
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
  <circle cx="860" cy="72" r="210" fill="${C.cyan}" fill-opacity="0.025"/>
  <circle cx="70" cy="520" r="220" fill="${C.green}" fill-opacity="0.020"/>
  ${body}
  </svg>`;
}

function attentionFrame(frame) {
  const t = cycle(frame);
  const write = phase(t, 0.05, 0.55);
  const retrieve = phase(t, 0.48, 0.80);
  const shimmer = 0.55 + 0.45 * pulse(t * 1.4);
  let b = header('01', 'MHA · GQA · MLA · GATED DELTANET', 'Attention variants change what the model stores');
  const xs = [28, 260, 492, 724];
  const names = ['MHA', 'GQA', 'MLA · DEEPSEEK-V2', 'GATED DELTANET'];
  const colors = [C.cyan, C.amber, C.violet, C.green];
  xs.forEach((x, i) => { b += panel(x, 123, 208, 316, names[i], colors[i]); });
  const words = ['the', 'key', 'was', 'blue'];
  xs.forEach((x, c) => words.forEach((w, i) => { b += token(x + 14 + i * 45, 166, w, colors[c], 0.55 + 0.45 * write, 40); }));

  // MHA: every head stores a full key and value for every token.
  for (let h = 0; h < 4; h++) {
    b += label(`h${h + 1}`, 44, 241 + h * 37, 9, C.muted, 'start', 600);
    for (let i = 0; i < 4; i++) {
      b += rect(66 + i * 37, 226 + h * 37, 30, 24, C.cyan, 5, 0.04 + 0.14 * write, C.cyan, 0.8);
      if (i === Math.floor(write * 4)) b += glow(81 + i * 37, 238 + h * 37, 10, C.cyan, 0.7);
    }
  }
  b += label('one K/V history per head', 132, 398, 11, C.ink, 'middle', 600);
  b += label('exact, largest cache', 132, 418, 10, C.muted, 'middle', 500);

  // GQA: query heads share fewer stored histories.
  for (let g = 0; g < 2; g++) {
    b += label(`KV group ${g + 1}`, 281, 255 + g * 76, 9, C.muted, 'start', 600);
    for (let i = 0; i < 4; i++) b += rect(282 + i * 39, 267 + g * 76, 31, 32, C.amber, 6, 0.04 + 0.15 * write, C.amber, 0.9);
    for (let q = 0; q < 2; q++) b += line(300 + q * 18, 238 + g * 76, 324 + q * 46, 267 + g * 76, C.amber, 1, 0.30 + 0.36 * write);
  }
  b += label('query heads share K/V', 364, 398, 11, C.ink, 'middle', 600);
  b += label('fewer full histories', 364, 418, 10, C.muted, 'middle', 500);

  // MLA: all token state is projected into a smaller latent cache.
  for (let i = 0; i < 4; i++) {
    b += flow(526 + i * 43, 194, 596, 286 + i * 12, write, C.violet, 0.74);
  }
  b += rect(553, 273, 86, 94, C.violet, 13, 0.06 + 0.13 * write, C.violet, 1.2);
  b += dotGrid(568, 290, 4, 4, C.violet, Math.max(1, Math.floor(write * 16)), 15);
  b += arrow(640, 320, 672, 320, C.violet, 1.5, 0.34 + 0.45 * retrieve);
  b += label('reconstruct', 672, 306, 9, C.muted, 'middle', 550);
  b += label('cache a learned latent', 596, 398, 11, C.ink, 'middle', 600);
  b += label('compressed token history', 596, 418, 10, C.muted, 'middle', 500);

  // Gated DeltaNet: token histories are superposed in one fixed-size state.
  b += rect(767, 256, 122, 105, C.green, 13, 0.04, C.green, 1.1);
  for (let j = 0; j < 5; j++) for (let i = 0; i < 6; i++) {
    const on = ((i + j * 2 + Math.floor(write * 5)) % 5) < 2;
    b += rect(780 + i * 17, 270 + j * 16, 12, 11, on ? C.green : C.grid, 2, on ? 0.12 + 0.12 * shimmer : 0.08);
  }
  words.forEach((_, i) => b += flow(748 + i * 45, 194, 828, 308, clamp(write * 1.15 - i * 0.12), C.green, 0.72));
  b += label('fixed state', 828, 381, 9, C.green, 'middle', 650);
  b += label('update one fixed state', 828, 398, 11, C.ink, 'middle', 600);
  b += label('bounded, lossy memory', 828, 418, 10, C.muted, 'middle', 500);

  b += rect(157, 462, 646, 44, C.bg2, 20, 0.9, C.grid, 1);
  b += label('HYBRIDS', 181, 489, 10, C.violet, 'start', 700, 1, 1.2);
  for (let i = 0; i < 8; i++) b += rect(258 + i * 48, 474, 34, 18, i === 3 || i === 7 ? C.cyan : C.green, 5, 0.10 + (i === Math.floor(t * 8) ? 0.16 : 0.04), i === 3 || i === 7 ? C.cyan : C.green, 0.8);
  b += label('periodic explicit retrieval', 780, 489, 9, C.muted, 'end', 550);
  return svg(b, 'The same context is stored as separate full key-value histories in multi-head attention, fewer shared histories in grouped-query attention, a compressed latent in MLA, or one fixed recurrent state in Gated DeltaNet; hybrids periodically restore explicit retrieval.');
}

function scene(x, y, color, detailed = 1) {
  let out = rect(x, y, 178, 128, C.bg2, 12, 0.96, C.grid, 1);
  out += rect(x + 18, y + 91, 142, 14, C.grid, 5, 0.28);
  out += rect(x + 96, y + 55, 42, 38, C.cyan, 7, 0.10, C.cyan, 1);
  out += path(`M ${x + 138} ${y + 66} C ${x + 159} ${y + 63} ${x + 160} ${y + 88} ${x + 138} ${y + 87}`, C.cyan, 2, 0.72);
  out += rect(x + 28, y + 66, 48, 28, C.amber, 7, 0.08, C.amber, 1);
  out += label('A7', x + 52, y + 85, 9, C.amber, 'middle', 700, detailed);
  out += circle(x + 117, y + 77, 5, color, 0.34 * detailed);
  return out;
}

function vlmFrame(frame) {
  const t = cycle(frame);
  const encode = phase(t, 0.04, 0.38);
  const preserve = phase(t, 0.34, 0.76);
  const shimmer = 0.55 + 0.45 * pulse(t * 1.4);
  let b = header('02', 'IMAGE-TEXT · MULTIMODAL LLM · GROUNDING · VLA', 'The output changes what vision must preserve');
  const xs = [26, 260, 494, 728];
  const colors = [C.cyan, C.violet, C.amber, C.green];
  const names = ['CLIP · IMAGE-TEXT', 'LLAVA · MULTIMODAL LLM', 'MOLMO · GROUNDING', 'PI0 · VLA'];
  xs.forEach((x, i) => { b += panel(x, 123, 206, 340, names[i], colors[i]); b += scene(x + 14, 166, colors[i], 1); });

  // CLIP collapses the scene to a global semantic vector; location is not required by the objective.
  b += arrow(129, 300, 129, 326, C.cyan, 1.6, 0.72);
  for (let i = 0; i < 7; i++) b += rect(67 + i * 18, 340, 12, 54 * (0.35 + ((i * 7) % 5) / 8), C.cyan, 3, 0.06 + 0.18 * encode, C.cyan, 0.6);
  b += label('“mug” ↔ image', 129, 413, 11, C.ink, 'middle', 620);
  b += label('image-text similarity', 129, 438, 10, C.muted, 'middle', 500);

  // LLaVA projects a grid of visual features into an autoregressive language model.
  b += arrow(363, 300, 363, 322, C.violet, 1.6, 0.72);
  b += dotGrid(314, 334, 7, 3, C.violet, Math.max(1, Math.floor(encode * 21)), 14);
  b += arrow(410, 355, 438, 355, C.violet, 1.5, 0.32 + 0.44 * preserve);
  ['put', 'the', 'mug'].forEach((w, i) => { b += token(298 + i * 53, 389, w, C.violet, 0.55 + 0.45 * preserve, 47); });
  b += label('visual tokens → text', 363, 438, 10, C.muted, 'middle', 500);

  // Point supervision forces an inspectable spatial binding.
  b += arrow(597, 300, 597, 326, C.amber, 1.6, 0.72);
  b += rect(541, 338, 112, 62, C.bg2, 10, 0.96, C.amber, 1);
  b += path('M 553 388 L 638 350', C.grid, 1, 0.5);
  b += circle(622, 359, 11 + 4 * shimmer, C.amber, 0.07, C.amber, 1.4);
  b += glow(622, 359, 11, C.amber, 0.35 + 0.55 * preserve);
  b += label('(x, y)', 622, 386, 9, C.amber, 'middle', 650);
  b += label('phrase → point', 597, 438, 10, C.muted, 'middle', 500);

  // A VLA must preserve state through time and emit an embodiment-specific action.
  b += arrow(831, 300, 831, 323, C.green, 1.6, 0.72);
  for (let k = 0; k < 3; k++) {
    b += rect(754 + k * 44, 337, 36, 48, C.green, 7, 0.04 + 0.11 * preserve, C.green, 0.8);
    b += circle(766 + k * 44, 361 - k * 4, 4, C.cyan, 0.7);
    if (k < 2) b += arrow(792 + k * 44, 361 - k * 4, 796 + k * 44, 357 - (k + 1) * 4, C.green, 1.2, 0.65);
  }
  b += path('M 770 412 C 802 390 843 390 877 410', C.green, 2.2, 0.35 + 0.45 * preserve);
  b += circle(mix(770, 877, preserve), 412 - 20 * Math.sin(preserve * Math.PI), 5, C.green, 0.9);
  b += label('observations → actions', 831, 438, 10, C.muted, 'middle', 500);

  b += label('The task decides which visual evidence training must preserve.', 480, 501, 13, C.green, 'middle', 620);
  return svg(b, 'The same mug-and-tray scene produces image-text similarity in CLIP, generated text in LLaVA, a grounded point in Molmo, and robot actions in Pi0; each task requires different visual evidence.');
}

function pretrainingFrame(frame) {
  const t = cycle(frame);
  const expand = phase(t, 0.06, 0.48);
  const update = phase(t, 0.42, 0.78);
  const colors = [C.cyan, C.violet, C.pink, C.green];
  const names = ['TEXT', 'IMAGE', 'VIDEO', 'ACTION'];
  const units = [8, 20, 42, 28];
  const flops = [0.18, 0.42, 0.94, 0.61];
  let b = header('03', 'MM1 · MIXED-MODAL SCALING · PI0', 'Example percentages are not gradient budgets');
  b += panel(34, 124, 246, 335, '1 · SAMPLED EXAMPLES', C.cyan);
  b += panel(304, 124, 304, 335, '2 · EXPAND INTO TRAINING UNITS', C.violet);
  b += panel(632, 124, 294, 335, '3 · REACH SHARED PARAMETERS', C.green);

  names.forEach((name, i) => {
    const y = 178 + i * 62;
    b += rect(58, y, 196, 42, colors[i], 10, 0.05 + 0.05 * (i === Math.floor(t * 4)), colors[i], 1);
    b += circle(77, y + 21, 5, colors[i], 0.9);
    b += label(name, 91, y + 18, 10, C.ink, 'start', 650);
    b += label('25 examples', 91, y + 33, 9, C.muted, 'start', 500);

    const unitY = y + 4;
    b += flow(254, y + 21, 334, unitY + 17, clamp(expand - i * 0.07), colors[i], 0.72);
    b += label(name.toLowerCase(), 330, unitY + 20, 9, colors[i], 'start', 650);
    const shown = Math.max(1, Math.floor(units[i] * expand));
    b += dotGrid(376, unitY, 12, 2, colors[i], shown, 12);
    b += label(`${units[i]} predicted units`, 580, unitY + 20, 9, C.muted, 'end', 500);
  });

  b += label('same count', 156, 432, 11, C.cyan, 'middle', 650);
  b += label('different sequence expansion', 456, 432, 11, C.violet, 'middle', 650);

  const trunkX = 792;
  b += rect(739, 196, 106, 148, C.green, 16, 0.05, C.green, 1.2);
  for (let j = 0; j < 5; j++) b += rect(755, 213 + j * 24, 74, 14, C.green, 5, 0.05 + 0.06 * update, C.green, 0.6);
  b += label('shared trunk', trunkX, 369, 11, C.ink, 'middle', 620);
  names.forEach((_, i) => {
    const sy = 180 + i * 60;
    const width = 15 + flops[i] * 76 * update;
    b += path(`M 644 ${sy} C 686 ${sy} 704 ${240 + i * 18} 738 ${244 + i * 18}`, colors[i], 2 + flops[i] * 4, 0.18 + 0.55 * update);
    b += rect(856, sy - 8, width, 16, colors[i], 5, 0.10 + 0.12 * update, colors[i], 0.7);
    b += label('update norm', 914, sy + 4, 8, C.muted, 'end', 500);
  });
  b += label('video can dominate at equal sample share', 779, 420, 10, C.pink, 'middle', 600);
  b += label('Track examples · units · FLOPs · update norm', 480, 501, 13, C.green, 'middle', 620);
  return svg(b, 'Equal example percentages for text, images, video, and actions expand into different numbers of predicted units, consume different FLOPs, and apply different update norms to shared parameters; four ledgers are needed to understand the mixture.');
}

function trajectoryRow(x, y, color, activeStart, activeEnd, p, labels = false) {
  let out = line(x + 16, y + 14, x + 316, y + 14, C.grid, 2, 0.8);
  for (let i = 0; i < 10; i++) {
    const active = i >= activeStart && i <= activeEnd;
    const revealed = i <= Math.floor(p * 10);
    out += circle(x + 18 + i * 33, y + 14, active ? 8 : 5, active ? color : C.grid, revealed ? (active ? 0.88 : 0.34) : 0.12, active ? color : C.grid, active ? 1 : 0.5);
    if (labels && (i === 6 || i === 7)) out += label(i === 6 ? 'miss' : 'takeover', x + 18 + i * 33, y - 5, 8, i === 6 ? C.red : C.amber, 'middle', 600);
  }
  return out;
}

function posttrainingFrame(frame) {
  const t = cycle(frame);
  const reveal = phase(t, 0.04, 0.38);
  const assign = phase(t, 0.34, 0.72);
  let b = header('04', 'DPO · APO · VLAC · RIPT-VLA', 'Feedback should claim only what the rollout reveals');
  b += panel(34, 123, 892, 346, 'ONE ROLLOUT · SAME FAILURE', C.red);
  b += label('approach', 90, 178, 9, C.muted, 'middle', 550);
  b += label('contact', 349, 178, 9, C.muted, 'middle', 550);
  b += label('recovery', 655, 178, 9, C.muted, 'middle', 550);
  b += trajectoryRow(72, 195, C.red, 6, 6, reveal, true);
  b += label('camera missed handle at step 7; human takes over at step 8', 480, 241, 11, C.ink, 'middle', 620);

  const rows = [276, 331, 386];
  const names = ['EPISODE OUTCOME', 'ACTION PREFERENCE OPTIMIZATION', 'PROCESS / INTERACTIVE FEEDBACK'];
  const sub = ['failure bit copied across the whole trajectory', 'intervention supports a local undesirable / corrective label', 'progress or group reward is useful only on comparable states'];
  const colors = [C.red, C.amber, C.green];
  rows.forEach((y, i) => {
    b += label(names[i], 72, y + 18, 10, colors[i], 'start', 680, 1, 0.6);
    b += trajectoryRow(388, y, colors[i], i === 0 ? 0 : (i === 1 ? 6 : 4), i === 0 ? 9 : (i === 1 ? 7 : 7), assign, false);
    b += label(sub[i], 895, y + 18, 9, C.muted, 'end', 500);
  });
  b += line(605, 316, 605, 370, C.amber, 1.2, 0.46 + 0.35 * assign, '4 4');
  b += label('matched state?', 605, 374, 8, C.amber, 'middle', 600);
  b += label('A cleaner objective cannot manufacture a counterfactual the robot never observed.', 480, 504, 13, C.green, 'middle', 620);
  return svg(b, 'A single robot failure is assigned at three granularities: an episode outcome labels everything, Action Preference Optimization uses the local intervention and correction, and process or interactive methods require progress judgments or comparable rollout groups.');
}

function rlFrame(frame) {
  const t = cycle(frame);
  const sample = phase(t, 0.04, 0.42);
  const signal = phase(t, 0.36, 0.78);
  const xs = [26, 260, 494, 728];
  const colors = [C.cyan, C.pink, C.amber, C.green];
  const names = ['PPO', 'DPO', 'GRPO · DEEPSEEKMATH', 'GKD · ON-POLICY KD'];
  let b = header('05', 'PPO · DPO · GRPO · GKD', 'Each method builds a different contrast');
  xs.forEach((x, i) => { b += panel(x, 123, 206, 336, names[i], colors[i]); b += token(x + 52, 163, 'prompt x', colors[i], 0.9, 102); });

  // PPO: a critic estimates a baseline along one sampled trajectory.
  const ppoX = 129;
  b += arrow(ppoX, 192, ppoX, 218, C.cyan, 1.5, 0.75);
  ['a₁', 'a₂', 'a₃', 'a₄'].forEach((a, i) => { b += token(47 + i * 41, 235, a, C.cyan, 0.5 + 0.5 * sample, 36); });
  [0.2, -0.1, 0.7, 0.35].forEach((a, i) => {
    const h = Math.abs(a) * 70 * signal;
    b += rect(57 + i * 41, a >= 0 ? 350 - h : 350, 16, h, a >= 0 ? C.green : C.red, 3, 0.18, a >= 0 ? C.green : C.red, 0.7);
  });
  b += line(48, 350, 211, 350, C.grid, 1, 0.7);
  b += label('critic estimates A_t', ppoX, 403, 11, C.ink, 'middle', 620);
  b += label('current rollout + value baseline', ppoX, 428, 9, C.muted, 'middle', 500);

  // DPO: the contrast already exists as a fixed preference pair.
  b += arrow(363, 192, 363, 218, C.pink, 1.5, 0.75);
  b += token(286, 239, 'chosen y⁺', C.green, 0.55 + 0.45 * sample, 154);
  b += token(286, 290, 'rejected y⁻', C.red, 0.55 + 0.45 * sample, 154);
  b += label('>', 363, 283, 18, C.ink, 'middle', 700, signal);
  b += line(314, 348, 412, 348, C.grid, 3, 0.6);
  b += circle(mix(340, 402, signal), 348, 8, C.pink, 0.9);
  b += label('reference-relative margin', 363, 403, 11, C.ink, 'middle', 620);
  b += label('fixed pair; no exploration', 363, 428, 9, C.muted, 'middle', 500);

  // GRPO: current-policy completions are normalized within one prompt group.
  b += arrow(597, 192, 597, 214, C.amber, 1.5, 0.75);
  const rewards = [0, 1, 1, 0];
  rewards.forEach((r, i) => {
    b += token(518 + (i % 2) * 82, 230 + Math.floor(i / 2) * 61, `y${i + 1} · r=${r}`, r ? C.green : C.red, 0.48 + 0.48 * sample, 72);
    const ay = 335 + (r ? -20 : 20) * signal;
    b += circle(536 + (i % 2) * 82, ay, 6, r ? C.green : C.red, 0.75);
  });
  b += line(526, 335, 668, 335, C.grid, 1, 0.7);
  b += label('group mean is the baseline', 597, 403, 11, C.ink, 'middle', 620);
  b += label('no variance → no gradient', 597, 428, 9, C.muted, 'middle', 500);

  // GKD: the student samples its own prefix, then receives a dense teacher distribution there.
  b += arrow(831, 192, 831, 216, C.green, 1.5, 0.75);
  ['s₁', 's₂', 'mistake'].forEach((a, i) => { b += token(747 + i * 56, 236, a, i === 2 ? C.red : C.green, 0.5 + 0.48 * sample, 51); });
  b += path('M 849 277 C 894 291 894 337 846 347', C.green, 1.6, 0.26 + 0.5 * signal, 'none', '4 5');
  for (let i = 0; i < 5; i++) {
    const h = [0.22, 0.62, 0.35, 0.82, 0.45][i] * 58 * signal;
    b += rect(765 + i * 27, 360 - h, 17, h, C.green, 3, 0.14, C.green, 0.7);
  }
  b += label('teacher logits at student states', 831, 403, 11, C.ink, 'middle', 620);
  b += label('dense imitation; no return', 831, 428, 9, C.muted, 'middle', 500);
  b += label('The acronym matters less than the sampled state, feedback unit, and baseline.', 480, 503, 13, C.green, 'middle', 620);
  return svg(b, 'PPO compares actions with a learned value baseline, DPO compares a fixed chosen and rejected pair, GRPO compares current-policy completions within one prompt, and GKD asks a teacher for dense next-token targets on student-generated prefixes.');
}

function hermesFrame(frame) {
  const t = cycle(frame);
  const request = phase(t, 0.03, 0.46);
  const response = phase(t, 0.50, 0.88);
  const blocks = [
    { x: 54, w: 196, title: 'HERMES AGENT', sub: 'tools · sessions · skills', color: C.violet },
    { x: 284, w: 176, title: 'LOCAL API', sub: '127.0.0.1:18080/v1', color: C.cyan },
    { x: 494, w: 176, title: 'LLAMA-SERVER', sub: 'sampling · KV cache', color: C.green },
    { x: 704, w: 202, title: 'GGUF ON DISK', sub: 'weights · tokenizer', color: C.amber },
  ];
  let b = header('06', 'HERMES AGENT · LLAMA.CPP', 'The agent shell and model runtime are separate');
  blocks.forEach((q) => {
    b += panel(q.x, 178, q.w, 194, q.title, q.color);
    b += rect(q.x + 24, 235, q.w - 48, 62, q.color, 12, 0.05 + 0.06 * pulse(t * 1.4), q.color, 1);
    b += label(q.sub, q.x + q.w / 2, 269, 10, C.ink, 'middle', 580);
  });
  for (let i = 0; i < blocks.length - 1; i++) {
    const a = blocks[i]; const n = blocks[i + 1];
    b += arrow(a.x + a.w, 275, n.x - 6, 275, C.cyan, 1.5, 0.42);
    b += flow(a.x + a.w, 275, n.x - 6, 275, clamp(request * 1.45 - i * 0.24), C.cyan, 0.88);
    b += flow(n.x - 6, 319, a.x + a.w, 319, clamp(response * 1.45 - (2 - i) * 0.24), C.green, 0.86);
  }
  b += label('request', 365, 244, 9, C.cyan, 'middle', 650);
  b += label('tokens', 576, 244, 9, C.cyan, 'middle', 650);
  b += label('READY', 365, 344, 9, C.green, 'middle', 650);
  b += label('generated text', 576, 344, 9, C.green, 'middle', 650);
  b += rect(275, 405, 410, 50, C.red, 22, 0.035, C.red, 1);
  b += label('A model-load error belongs below the API boundary.', 480, 436, 12, C.red, 'middle', 620);
  b += label('Hermes can stay fixed while the local server or weights change.', 480, 501, 13, C.green, 'middle', 620);
  return svg(b, 'A request moves from Hermes Agent through a localhost OpenAI-compatible API to llama-server and an on-disk GGUF, while generated text returns through the same boundary; a load failure belongs to the serving layer, not the agent shell.');
}

const gemmaRows = [
  ['E2B · MLX', 181, 879, 182.86, 175.68, C.cyan],
  ['E2B · llama.cpp', 127, 1634, 119.46, 114.07, C.amber],
  ['E4B · MLX', 230, 1682, 114.96, 103.95, C.cyan],
  ['E4B · llama.cpp', 391, 3068, 96.54, 89.35, C.amber],
  ['26B A4B · MLX', 422, 2182, 115.80, 104.36, C.cyan],
  ['26B A4B · llama.cpp', 334, 3227, 110.85, 101.42, C.amber],
  ['31B · MLX', 906, 13501, 27.50, 23.73, C.cyan],
  ['31B · llama.cpp', 1279, 24164, 24.89, 20.72, C.amber],
];

const qwenRows = [
  ['Qwen 3 4B · MLX', 392, 1742, 176.14, 127.76, C.cyan],
  ['Qwen 3.5 4B · MLX', 187, 2103, 144.08, 131.45, C.violet],
  ['Qwen 3.5 9B · MLX', 301, 2894, 96.33, 92.66, C.violet],
  ['Qwen 3.5 9B · llama.cpp', 824, 5158, 74.50, 67.28, C.amber],
  ['Qwen 3 14B · MLX', 684, 4925, 59.87, 52.25, C.cyan],
  ['Qwen 3 14B · llama.cpp', 568, 11112, 53.36, 44.77, C.amber],
];

function benchmarkFrame(frame, spec) {
  const t = cycle(frame);
  const long = phase(t, 0.32, 0.58);
  const settle = phase(t, 0.02, 0.26);
  let b = header(spec.index, spec.kicker, 'Long prompts move the bottleneck to prefill');
  b += panel(34, 122, 892, 345, 'MEASURED ON ONE 64 GB M5 MAX · APRIL 4, 2026', C.cyan);
  b += rect(690, 144, 202, 34, C.bg2, 17, 0.95, C.grid, 1);
  b += rect(692 + long * 99, 146, 99, 30, long > 0.5 ? C.violet : C.cyan, 15, 0.13, long > 0.5 ? C.violet : C.cyan, 1);
  b += label('512 TOKENS', 742, 167, 9, long < 0.5 ? C.ink : C.dim, 'middle', 650);
  b += label('8192 TOKENS', 840, 167, 9, long > 0.5 ? C.ink : C.dim, 'middle', 650);
  b += label('TTFT', 449, 196, 10, C.muted, 'middle', 650, 1, 1.1);
  b += label('DECODE TOK/S', 823, 196, 10, C.muted, 'middle', 650, 1, 1.1);
  b += line(257, 205, 652, 205, C.grid, 1, 0.7);
  b += line(720, 205, 895, 205, C.grid, 1, 0.7);
  spec.rows.forEach((r, i) => {
    const y = 225 + i * spec.rowGap;
    const ttft = mix(r[1], r[2], long);
    const decode = mix(r[3], r[4], long);
    const ttftW = Math.max(2, 370 * ttft / spec.maxTtft) * settle;
    const decodeW = 160 * decode / spec.maxDecode * settle;
    b += label(r[0], 64, y + 10, 10, C.ink, 'start', 560);
    b += rect(257, y, ttftW, 16, r[5], 5, 0.10 + 0.09 * long, r[5], 0.7);
    b += label(ttft >= 1000 ? `${(ttft / 1000).toFixed(1)} s` : `${Math.round(ttft)} ms`, 640, y + 11, 9, C.muted, 'end', 520);
    b += rect(720, y, decodeW, 16, r[5], 5, 0.08, r[5], 0.7);
    b += label(decode.toFixed(1), 902, y + 11, 9, C.muted, 'end', 520);
  });
  b += label(spec.footer, 480, 502, 13, C.green, 'middle', 620);
  return svg(b, spec.description);
}

const animations = [
  ['blog-attention-memory.gif', attentionFrame],
  ['blog-vlm-evidence-contract.gif', vlmFrame],
  ['blog-multimodal-gradient-budget.gif', pretrainingFrame],
  ['blog-vla-feedback-attribution.gif', posttrainingFrame],
  ['blog-rl-learning-signals.gif', rlFrame],
  ['blog-hermes-local-stack.gif', hermesFrame],
  ['local-gemma-long-prompt-latency.gif', (frame) => benchmarkFrame(frame, {
    index: '07', kicker: 'GEMMA 4 · MLX VS LLAMA.CPP', rows: gemmaRows, rowGap: 29, maxTtft: 25000, maxDecode: 190,
    footer: 'Weights fit in memory before the interaction feels responsive.',
    description: 'Measured Gemma 4 time to first token increases sharply when the prompt grows from 512 to 8192 tokens, especially for 31B, while decode speed changes much less; MLX generally keeps prefill latency lower than llama.cpp in this snapshot.',
  })],
  ['local-qwen-long-prompt-latency.gif', (frame) => benchmarkFrame(frame, {
    index: '08', kicker: 'QWEN 3.5 AND QWEN 3 · MLX VS LLAMA.CPP', rows: qwenRows, rowGap: 36, maxTtft: 12000, maxDecode: 180,
    footer: 'The 4B models stay interactive; 14B pays mainly in prefill.',
    description: 'Measured Qwen time to first token increases as the prompt grows from 512 to 8192 tokens; the 4B models remain responsive, while Qwen 3 14B pays a much larger prefill cost and MLX outperforms llama.cpp on the long prompt in this snapshot.',
  })],
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
    '-hide_banner', '-loglevel', 'error', '-y',
    '-framerate', String(FPS),
    '-i', join(frameDir, 'frame-%03d.png'),
    '-filter_complex', filter,
    '-loop', '0',
    outputPath,
  ], { stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`ffmpeg failed for ${filename}`);
  console.log(`generated ${outputPath}`);
}

mkdirSync(OUTPUT, { recursive: true });
mkdirSync(SCRATCH, { recursive: true });
for (const [filename, renderer] of animations) await renderAnimation(filename, renderer);
rmSync(SCRATCH, { recursive: true, force: true });
