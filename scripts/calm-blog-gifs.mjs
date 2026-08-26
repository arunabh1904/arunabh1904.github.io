import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

const sharp = createRequire(import.meta.url)('sharp');

export const WIDTH = 960;
export const HEIGHT = 540;
export const FPS = 8;
export const DEFAULT_STEP_SECONDS = 8;
export const DEFAULT_BUILD_SECONDS = 3.2;
export const DEFAULT_TRANSITION_SECONDS = 0.65;

export const GENERAL_BLOG_GIFS = [
  'blog-attention-memory.gif',
  'blog-vlm-evidence-contract.gif',
  'blog-multimodal-gradient-budget.gif',
  'blog-vla-feedback-attribution.gif',
  'blog-rl-learning-signals.gif',
  'blog-hermes-local-stack.gif',
  'local-gemma-long-prompt-latency.gif',
  'local-qwen-long-prompt-latency.gif',
];

export const PERCEPTION_BLOG_GIFS = [
  'autonomous-perception-camera-encoder.gif',
  'autonomous-perception-lidar-encoder.gif',
  'autonomous-perception-radar-encoder.gif',
  'autonomous-perception-camera-lifting.gif',
  'autonomous-perception-fusion-granularity.gif',
  'autonomous-perception-modality-dropout.gif',
  'autonomous-perception-temporal-memory.gif',
  'autonomous-perception-lidar-training-contracts.gif',
];

export const CALM_BLOG_GIFS = [...GENERAL_BLOG_GIFS, ...PERCEPTION_BLOG_GIFS];

const C = {
  bg: '#050505',
  ink: '#f4f1ec',
  muted: '#aaa49a',
  faint: '#34322f',
  teal: '#78c9c3',
  amber: '#efa66f',
  blue: '#93a9e8',
  rose: '#e58c9c',
  green: '#86c89a',
  red: '#df786f',
};

const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const mix = (a, b, t) => a + (b - a) * t;
const ease = (x) => {
  const t = clamp(x);
  return t * t * (3 - 2 * t);
};
const esc = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;');
const alpha = (value) => clamp(value).toFixed(3);

function text(value, x, y, size = 16, fill = C.ink, anchor = 'start', weight = 520, opacity = 1, spacing = 0) {
  return `<text x="${x}" y="${y}" fill="${fill}" fill-opacity="${alpha(opacity)}" font-family="SF Pro Display, SF Pro Text, Helvetica Neue, Arial, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" letter-spacing="${spacing}">${esc(value)}</text>`;
}

function rect(x, y, w, h, fill = C.bg, radius = 10, opacity = 1, stroke = 'none', sw = 1) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-opacity="${alpha(opacity)}" stroke-width="${sw}"/>`;
}

function line(x1, y1, x2, y2, stroke = C.faint, sw = 2, opacity = 1, dash = '') {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-opacity="${alpha(opacity)}" stroke-width="${sw}" stroke-linecap="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
}

function path(d, stroke = C.faint, sw = 2, opacity = 1, fill = 'none', dash = '') {
  return `<path d="${d}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-opacity="${alpha(opacity)}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
}

function circle(x, y, r, fill = C.ink, opacity = 1, stroke = 'none', sw = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-opacity="${alpha(opacity)}" stroke-width="${sw}"/>`;
}

function ring(x, y, r, stroke, opacity = 1, sw = 1, dash = '') {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${stroke}" stroke-opacity="${alpha(opacity)}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`;
}

function arrow(x1, y1, x2, y2, stroke = C.teal, sw = 2, opacity = 1) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const len = 10;
  const wing = 4;
  const bx = x2 - Math.cos(angle) * len;
  const by = y2 - Math.sin(angle) * len;
  const p1 = `${bx + Math.cos(angle + Math.PI / 2) * wing},${by + Math.sin(angle + Math.PI / 2) * wing}`;
  const p2 = `${bx + Math.cos(angle - Math.PI / 2) * wing},${by + Math.sin(angle - Math.PI / 2) * wing}`;
  return `${line(x1, y1, x2, y2, stroke, sw, opacity)}<polygon points="${x2},${y2} ${p1} ${p2}" fill="${stroke}" fill-opacity="${alpha(opacity)}"/>`;
}

function token(label, x, y, opacity = 1, accent = C.amber, width = 92) {
  return `${rect(x, y, width, 42, C.bg, 9, opacity, accent, 1.2)}${text(label, x + width / 2, y + 27, 15, C.ink, 'middle', 560, opacity)}`;
}

function top(index, total, title, opacity) {
  const titleSize = title.length > 62 ? 24 : title.length > 50 ? 26 : 30;
  return `${text(`${String(index).padStart(2, '0')} / ${String(total).padStart(2, '0')}`, 62, 65, 12, C.amber, 'start', 650, opacity, 1.9)}${text(title, 480, 86, titleSize, C.ink, 'middle', 620, opacity)}`;
}

function footer(value, opacity, color = C.muted) {
  return text(value, 480, 492, 16, color, 'middle', 560, opacity);
}

function labelPair(left, right, y, opacity, color = C.teal) {
  return `${text(left, 160, y, 16, color, 'start', 650, opacity)}${text(right, 390, y, 16, C.ink, 'start', 520, opacity)}${line(112, y + 17, 848, y + 17, C.faint, 1, opacity)}`;
}

function reveal(local, start = 0.12, end = 0.62) {
  return ease((local - start) / (end - start));
}

function sequenceTokens(y, opacity, accent = C.amber, labels = ['the', 'key', 'was', 'blue']) {
  const width = labels.length > 4 ? 76 : 92;
  const gap = labels.length > 4 ? 18 : 28;
  const total = labels.length * width + (labels.length - 1) * gap;
  const start = (WIDTH - total) / 2;
  return labels.map((label, i) => token(label, start + i * (width + gap), y, opacity, accent, width)).join('');
}

function grid(x, y, cols, rows, cell, color, fillAmount, opacity = 1) {
  let out = '';
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const k = row * cols + col;
      const on = k < Math.round(fillAmount * cols * rows);
      out += rect(x + col * cell, y + row * cell, cell - 5, cell - 5, on ? color : C.faint, 4, opacity * (on ? 0.56 : 0.38));
    }
  }
  return out;
}

function spatialGrid(x, y, w, h, cols, rows, opacity = 1, accent = C.teal, marker = null) {
  let out = rect(x, y, w, h, C.bg, 9, opacity, accent, 1.2);
  for (let col = 1; col < cols; col++) out += line(x + w * col / cols, y, x + w * col / cols, y + h, C.faint, 1, opacity);
  for (let row = 1; row < rows; row++) out += line(x, y + h * row / rows, x + w, y + h * row / rows, C.faint, 1, opacity);
  if (marker) {
    const cellW = w / cols;
    const cellH = h / rows;
    const markerX = x + marker.col * cellW;
    const markerY = y + marker.row * cellH;
    out += rect(markerX + 2, markerY + 2, cellW - 4, cellH - 4, marker.color ?? accent, 3, opacity * 0.28, marker.color ?? accent, 1.2);
    if (marker.crossed) {
      out += line(markerX + 5, markerY + 5, markerX + cellW - 5, markerY + cellH - 5, marker.color ?? C.red, 2, opacity);
      out += line(markerX + cellW - 5, markerY + 5, markerX + 5, markerY + cellH - 5, marker.color ?? C.red, 2, opacity);
    } else {
      out += circle(markerX + cellW / 2, markerY + cellH / 2, Math.min(cellW, cellH) * 0.15, marker.color ?? accent, opacity);
    }
  }
  return out;
}

function bev(x, y, w, h, opacity = 1, accent = C.teal) {
  let out = rect(x, y, w, h, C.bg, 12, opacity, accent, 1.2);
  for (let i = 1; i < 7; i++) out += line(x + (w * i) / 7, y, x + (w * i) / 7, y + h, C.faint, 1, opacity);
  for (let i = 1; i < 5; i++) out += line(x, y + (h * i) / 5, x + w, y + (h * i) / 5, C.faint, 1, opacity);
  out += rect(x + w * 0.47, y + h * 0.65, 20, 36, C.ink, 6, 0.25 * opacity, C.ink, 1);
  return out;
}

function cameraView(x, y, w, h, opacity = 1) {
  let out = rect(x, y, w, h, C.bg, 12, opacity, C.faint, 1.2);
  out += path(`M ${x + 20} ${y + h} L ${x + w * 0.42} ${y + 42} L ${x + w * 0.58} ${y + 42} L ${x + w - 20} ${y + h}`, C.faint, 2, opacity);
  out += rect(x + w * 0.58, y + h * 0.55, 38, 26, C.bg, 5, opacity, C.amber, 1.2);
  out += circle(x + w * 0.30, y + h * 0.45, 6, C.teal, opacity);
  return out;
}

function bars(rows, x, y, maxWidth, maxValue, opacity, color = C.teal, suffix = '') {
  let out = '';
  rows.forEach(([name, value], i) => {
    const rowY = y + i * 54;
    out += text(name, x, rowY + 16, 14, C.ink, 'start', 540, opacity);
    out += rect(x + 195, rowY, maxWidth * value / maxValue, 20, color, 5, opacity * 0.34, color, 0.8);
    out += text(`${value}${suffix}`, x + 205 + maxWidth, rowY + 16, 14, C.muted, 'end', 520, opacity);
  });
  return out;
}

function summary(rows, opacity, footerText) {
  let out = '';
  rows.forEach((row, i) => { out += labelPair(row[0], row[1], 178 + i * 61, opacity, row[2] ?? C.teal); });
  out += footer(footerText, opacity, C.amber);
  return out;
}

function drawAttention(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'intro') {
    return `${sequenceTokens(205, opacity * p)}${arrow(480, 265, 480, 322, C.teal, 2, opacity * reveal(local, 0.38, 0.58))}${text('Which past tokens remain individually retrievable?', 480, 372, 18, C.muted, 'middle', 520, opacity * reveal(local, 0.48, 0.68))}`;
  }
  if (mode === 'mha') {
    let out = '';
    for (let row = 0; row < 4; row++) {
      const y = 150 + row * 65;
      out += text(`head ${row + 1}`, 170, y + 27, 13, C.muted, 'end', 540, opacity);
      ['the', 'key', 'was', 'blue'].forEach((label, col) => { out += token(label, 198 + col * 132, y, opacity * (0.18 + 0.82 * ease(p * 5 - col)), C.teal, 108); });
    }
    return out + footer('4 heads × 4 token records · exact retrieval, largest cache', opacity * reveal(local, 0.62, 0.76), C.teal);
  }
  if (mode === 'gqa') {
    let out = '';
    for (let i = 0; i < 4; i++) {
      const y = 158 + i * 60;
      out += token(`query ${i + 1}`, 112, y, opacity, C.blue, 98);
      out += arrow(214, y + 21, 330, i < 2 ? 211 : 345, C.blue, 1.5, opacity * p);
    }
    [0, 1].forEach((group) => {
      const y = group ? 286 : 151;
      out += text(`KV group ${group + 1}`, 352, y + 18, 13, C.muted, 'start', 600, opacity);
      ['the', 'key', 'was', 'blue'].forEach((label, col) => { out += token(label, 352 + col * 112, y + 30, opacity * reveal(local, 0.26 + col * 0.04, 0.42 + col * 0.04), C.teal, 94); });
    });
    return out + footer('Fewer K/V histories are stored; every token keeps its index.', opacity * reveal(local, 0.64, 0.78), C.teal);
  }
  if (mode === 'mla') {
    let out = '';
    ['the', 'key', 'was', 'blue'].forEach((label, i) => {
      const y = 142 + i * 72;
      out += token(label, 112, y, opacity, C.amber, 82);
      for (let d = 0; d < 6; d++) out += rect(236 + d * 18, y + 33 - (10 + ((i * 5 + d * 7) % 23)), 10, 10 + ((i * 5 + d * 7) % 23), C.blue, 3, opacity * 0.65);
      out += arrow(350, y + 21, 568, y + 21, C.teal, 1.8, opacity * p);
      out += rect(596, y + 2, 66, 38, C.bg, 8, opacity * p, C.teal, 1.3);
      out += text(`z${i + 1}`, 629, y + 27, 15, C.teal, 'middle', 650, opacity * p);
      out += text(`token ${i + 1}`, 700, y + 26, 13, C.muted, 'start', 520, opacity * p);
    });
    return out + footer('Each token keeps an index; its key and value are reconstructed from zᵢ.', opacity * reveal(local, 0.64, 0.78), C.teal);
  }
  if (mode === 'delta') {
    let out = '';
    ['the', 'key', 'was', 'blue'].forEach((label, i) => {
      const q = ease(p * 5 - i);
      out += token(label, mix(104 + i * 113, 515, q), mix(220, 264, q), opacity * (1 - 0.72 * q), C.amber, 82);
    });
    out += arrow(532, 285, 646, 285, C.teal, 2, opacity * p);
    out += rect(676, 185, 170, 170, C.bg, 16, opacity, C.teal, 1.4);
    out += grid(697, 207, 5, 5, 28, C.teal, p, opacity);
    out += text('Sₜ', 761, 390, 18, C.teal, 'middle', 650, opacity);
    return out + footer('State size stays fixed. Individual token records disappear.', opacity * reveal(local, 0.66, 0.80), C.teal);
  }
  let out = '';
  const rows = [
    ['MHA', C.teal, '4 separate K/V histories'],
    ['GQA', C.blue, '2 shared K/V histories'],
    ['MLA', C.amber, 'one zᵢ per token'],
    ['DeltaNet', C.green, 'one fixed state Sₜ'],
  ];
  rows.forEach(([name, color, result], row) => {
    const y = 142 + row * 72;
    out += text(name, 92, y + 26, 15, color, 'start', 700, opacity);
    ['the', 'key', 'was', 'blue'].forEach((label, col) => { out += token(label, 188 + col * 67, y, opacity, color, 55); });
    out += arrow(462, y + 21, 530, y + 21, color, 1.6, opacity);
    if (name === 'DeltaNet') {
      out += grid(556, y - 2, 5, 2, 24, color, 0.7, opacity);
    } else {
      const count = name === 'MHA' ? 4 : name === 'GQA' ? 2 : 4;
      for (let i = 0; i < count; i++) out += rect(558 + i * 34, y + 2, 24, 34, C.bg, 5, opacity, color, 1.2);
    }
    out += text(result, 742, y + 26, 14, C.ink, 'start', 580, opacity);
  });
  return out + footer('Only MHA, GQA, and MLA keep every token individually addressable.', opacity, C.amber);
}

function sameScene(opacity) {
  return `${cameraView(116, 158, 316, 238, opacity)}${text('same mug-and-tray scene', 274, 427, 14, C.muted, 'middle', 520, opacity)}`;
}

function drawVlm(mode, local, opacity) {
  const p = reveal(local);
  let out = sameScene(opacity);
  if (mode === 'intro') return out + arrow(452, 275, 542, 275, C.teal, 2, opacity * p) + text('The required output changes which details are retained.', 684, 268, 18, C.ink, 'middle', 600, opacity * p) + text('identity · language · location · action', 684, 309, 15, C.muted, 'middle', 520, opacity * p);
  if (mode === 'clip') {
    out += arrow(452, 275, 574, 275, C.teal, 2, opacity * p);
    out += grid(609, 215, 8, 3, 25, C.teal, p, opacity);
    out += text('one global embedding', 710, 327, 17, C.teal, 'middle', 620, opacity);
    return out + footer('CLIP needs features for image-text matching, not exact object coordinates.', opacity * reveal(local, 0.62, 0.77));
  }
  if (mode === 'llava') {
    out += arrow(452, 275, 548, 275, C.blue, 2, opacity * p);
    out += grid(575, 210, 6, 3, 25, C.blue, p, opacity);
    out += arrow(730, 250, 796, 250, C.blue, 2, opacity * p);
    out += text('“put the mug', 812, 252, 16, C.ink, 'middle', 560, opacity * p);
    out += text('on the tray”', 812, 281, 16, C.ink, 'middle', 560, opacity * p);
    return out + footer('LLaVA retains image features used to generate the answer text.', opacity * reveal(local, 0.62, 0.77));
  }
  if (mode === 'molmo') {
    out += arrow(452, 275, 610, 275, C.amber, 2, opacity * p);
    out += circle(699, 245, 46, C.amber, 0.05 * opacity, C.amber, 1.5);
    out += circle(699, 245, 7, C.amber, opacity * p);
    out += line(699, 184, 699, 306, C.amber, 1, opacity * p);
    out += line(638, 245, 760, 245, C.amber, 1, opacity * p);
    out += text('phrase → point', 699, 338, 17, C.amber, 'middle', 620, opacity);
    return out + footer('Molmo must bind a phrase to an inspectable location.', opacity * reveal(local, 0.62, 0.77));
  }
  if (mode === 'pi0') {
    out += arrow(452, 275, 562, 275, C.green, 2, opacity * p);
    out += path('M 594 333 C 640 240 716 215 820 175', C.green, 4, opacity * p);
    for (let i = 0; i < 5; i++) out += circle(mix(594, 820, i / 4), mix(333, 175, i / 4) - Math.sin(i / 4 * Math.PI) * 24, 6, C.green, opacity * p);
    out += text('action trajectory', 708, 369, 17, C.green, 'middle', 620, opacity);
    return out + footer('π0 retains scene state across time and predicts robot actions.', opacity * reveal(local, 0.62, 0.77));
  }
  out += arrow(452, 275, 500, 275, C.teal, 2, opacity);
  const cards = [
    [520, 154, 'CLIP', 'image ↔ text score', C.teal],
    [704, 154, 'LLaVA', 'generated text', C.blue],
    [520, 292, 'Molmo', 'mug point (x, y)', C.amber],
    [704, 292, 'π0', 'robot action path', C.green],
  ];
  cards.forEach(([x, y, name, output, color]) => {
    out += rect(x, y, 164, 108, C.bg, 10, opacity, color, 1.2);
    out += text(name, x + 82, y + 31, 14, color, 'middle', 700, opacity);
    out += text(output, x + 82, y + 72, 14, C.ink, 'middle', 580, opacity);
  });
  out += circle(602, 377, 6, C.amber, opacity);
  out += path('M 734 375 C 770 352 816 350 850 326', C.green, 2.5, opacity);
  return out + footer('The same scene supports different models only if the representation preserves the required output.', opacity, C.amber);
}

function modalityRows(opacity, values, unit) {
  let out = '';
  const names = ['text', 'image', 'video', 'action'];
  const colors = [C.teal, C.blue, C.rose, C.green];
  names.forEach((name, i) => {
    const y = 157 + i * 66;
    out += text(name.toUpperCase(), 130, y + 17, 14, colors[i], 'start', 650, opacity);
    const count = values[i];
    for (let j = 0; j < count; j++) out += rect(280 + j * 18, y, 12, 24, colors[i], 3, opacity * 0.45);
    out += text(`${count} ${unit}`, 790, y + 17, 14, C.muted, 'end', 520, opacity);
  });
  return out;
}

function drawBudget(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'examples') return modalityRows(opacity, [8, 8, 8, 8], 'examples') + footer('Each modality contributes the same number of sampled examples.', opacity * reveal(local, 0.58, 0.74), C.teal);
  if (mode === 'units') return modalityRows(opacity, [5, 10, 24, 16].map((v) => Math.max(1, Math.round(v * p))), 'predicted units') + footer('Each example expands into a different number of training units.', opacity * reveal(local, 0.58, 0.74), C.blue);
  if (mode === 'flops') return bars([['text', 18], ['image', 42], ['video', 94], ['action', 61]], 145, 160, 430, 100, opacity * p, C.rose, '%') + footer('Equal example share can hide unequal compute.', opacity * reveal(local, 0.58, 0.74), C.rose);
  if (mode === 'updates') return bars([['text', 22], ['image', 38], ['video', 82], ['action', 57]], 145, 160, 430, 100, opacity * p, C.green, '') + footer('Gradient norm measures how strongly each modality updates shared parameters.', opacity * reveal(local, 0.58, 0.74), C.green);
  if (mode === 'ledgers') return summary([
    ['1', 'sampled examples'],
    ['2', 'predicted training units'],
    ['3', 'forward and backward FLOPs'],
    ['4', 'update norm at shared parameters'],
    ['5', 'independent decisions after temporal overlap'],
  ], opacity, 'Report all five quantities; sample percentage alone is incomplete.');
  return `${sequenceTokens(208, opacity * p, C.blue, ['text', 'image', 'video', 'action'])}${arrow(480, 270, 480, 329, C.teal, 2, opacity * p)}${text('same sample share ≠ same optimization load', 480, 374, 19, C.ink, 'middle', 620, opacity * p)}`;
}

function timeline(opacity, activeStart, activeEnd, color, progress = 1) {
  let out = line(170, 275, 790, 275, C.faint, 2, opacity);
  for (let i = 0; i < 10; i++) {
    const active = i >= activeStart && i <= activeEnd;
    out += circle(184 + i * 66, 275, active ? 10 : 6, active ? color : C.faint, opacity * (i <= progress * 10 ? 0.9 : 0.25), active ? color : C.faint, 1);
  }
  return out;
}

function drawFeedback(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'failure') return timeline(opacity, 6, 7, C.red, p) + text('camera misses handle', 580, 226, 15, C.red, 'middle', 600, opacity * p) + text('human takes over', 646, 329, 15, C.amber, 'middle', 600, opacity * p) + footer('The rollout shows the failed action, not which alternative would succeed.', opacity * reveal(local, 0.62, 0.78));
  if (mode === 'episode') return timeline(opacity, 0, 9, C.red, p) + footer('Episode outcome copies one failure bit across every action.', opacity * reveal(local, 0.62, 0.78), C.red);
  if (mode === 'apo') return timeline(opacity, 6, 7, C.amber, p) + text('failed action', 580, 226, 14, C.red, 'middle', 600, opacity * p) + text('corrected action', 646, 329, 14, C.green, 'middle', 600, opacity * p) + footer('Action Preference Optimization compares the failed and corrected actions.', opacity * reveal(local, 0.62, 0.78), C.amber);
  if (mode === 'process') return timeline(opacity, 4, 7, C.green, p) + line(514, 221, 514, 329, C.amber, 1.5, opacity * p, '5 5') + text('matched state?', 514, 205, 14, C.amber, 'middle', 600, opacity * p) + footer('Process feedback is valid only across comparable states.', opacity * reveal(local, 0.62, 0.78), C.green);
  return `${timeline(opacity, 6, 7, C.red, 1)}${path('M 170 214 L 170 194 L 790 194 L 790 214', C.red, 1.6, opacity)}${text('TERMINAL FAILURE BIT COVERS THE WHOLE ROLLOUT', 480, 176, 14, C.red, 'middle', 700, opacity)}${rect(560, 245, 160, 60, C.amber, 8, opacity * 0.08, C.amber, 1.5)}${text('defensible failure window', 640, 334, 14, C.amber, 'middle', 650, opacity)}${path('M 646 286 C 682 326 721 348 760 370', C.green, 2, opacity)}${circle(760, 370, 10, C.green, opacity)}${text('observed human correction', 760, 405, 14, C.green, 'middle', 650, opacity)}${footer('Keep credit local to the failure and correction the rollout actually observed.', opacity, C.amber)}`;
}

function drawLearning(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'intro') return `${token('prompt x', 414, 190, opacity, C.blue, 132)}${arrow(480, 248, 480, 324, C.teal, 2, opacity * p)}${text('Which samples and reference signal define the loss?', 480, 373, 19, C.ink, 'middle', 620, opacity * p)}`;
  if (mode === 'ppo') {
    let out = sequenceTokens(176, opacity, C.teal, ['a₁', 'a₂', 'a₃', 'a₄']);
    out += line(240, 351, 720, 351, C.faint, 2, opacity);
    [0.2, -0.1, 0.7, 0.35].forEach((v, i) => { const h = Math.abs(v) * 105 * p; out += rect(295 + i * 118, v >= 0 ? 351 - h : 351, 28, h, v >= 0 ? C.green : C.red, 5, opacity * 0.45); });
    return out + footer('PPO compares sampled actions with a learned value baseline.', opacity * reveal(local, 0.62, 0.78), C.teal);
  }
  if (mode === 'dpo') return `${token('chosen y⁺', 305, 204, opacity, C.green, 154)}${token('rejected y⁻', 501, 204, opacity, C.red, 154)}${text('>', 480, 238, 24, C.ink, 'middle', 700, opacity * p)}${line(348, 342, 612, 342, C.faint, 3, opacity)}${circle(mix(424, 568, p), 342, 10, C.rose, opacity)}${footer('DPO stores the contrast as a fixed preference pair.', opacity * reveal(local, 0.62, 0.78), C.rose)}`;
  if (mode === 'grpo') {
    let out = '';
    [['y₁ · 0', C.red], ['y₂ · 1', C.green], ['y₃ · 1', C.green], ['y₄ · 0', C.red]].forEach(([label, color], i) => { out += token(label, 315 + (i % 2) * 190, 175 + Math.floor(i / 2) * 80, opacity * p, color, 142); });
    out += line(328, 350, 630, 350, C.faint, 2, opacity);
    out += text('group mean', 480, 382, 16, C.amber, 'middle', 620, opacity * p);
    return out + footer('GRPO samples the contrast from one current-policy prompt group.', opacity * reveal(local, 0.62, 0.78), C.amber);
  }
  if (mode === 'gkd') {
    let out = sequenceTokens(174, opacity, C.green, ['s₁', 's₂', 'mistake']);
    out += arrow(480, 234, 480, 300, C.green, 2, opacity * p);
    [0.22, 0.62, 0.35, 0.82, 0.45].forEach((v, i) => { out += rect(365 + i * 48, 402 - v * 90 * p, 26, v * 90 * p, C.green, 4, opacity * 0.45); });
    return out + footer('GKD supplies dense teacher logits on student-generated states.', opacity * reveal(local, 0.62, 0.78), C.green);
  }
  let out = token('same prompt x', 400, 132, opacity, C.blue, 160);
  const cards = [
    [72, 'PPO', 'current rollout', 'value baseline', C.teal],
    [284, 'DPO', 'stored y⁺ / y⁻', 'reference policy', C.rose],
    [496, 'GRPO', 'current group', 'group mean', C.amber],
    [708, 'GKD', 'student prefix', 'teacher logits', C.green],
  ];
  cards.forEach(([x, name, samples, reference, color]) => {
    out += arrow(480, 182, x + 90, 232, color, 1.5, opacity);
    out += rect(x, 244, 180, 148, C.bg, 10, opacity, color, 1.2);
    out += text(name, x + 90, 278, 15, color, 'middle', 700, opacity);
    out += text(samples, x + 90, 322, 14, C.ink, 'middle', 600, opacity);
    out += line(x + 22, 341, x + 158, 341, C.faint, 1, opacity);
    out += text(reference, x + 90, 371, 14, C.muted, 'middle', 560, opacity);
  });
  return out + footer('The prompt is fixed; the sampled data and reference signal change.', opacity, C.amber);
}

function pipelineBlocks(opacity, activeCount = 4) {
  const blocks = [
    ['Hermes Agent', C.blue],
    ['localhost /v1', C.teal],
    ['llama-server', C.green],
    ['GGUF on disk', C.amber],
  ];
  let out = '';
  blocks.forEach(([label, color], i) => {
    const x = 67 + i * 224;
    out += rect(x, 222, 156, 82, C.bg, 13, opacity * (i < activeCount ? 1 : 0.22), color, 1.4);
    out += text(label, x + 78, 270, 15, C.ink, 'middle', 600, opacity * (i < activeCount ? 1 : 0.32));
    if (i < blocks.length - 1) out += arrow(x + 163, 263, x + 210, 263, C.teal, 1.7, opacity * (i < activeCount - 1 ? 1 : 0.25));
  });
  return out;
}

function drawHermes(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'separate') return pipelineBlocks(opacity, 1) + footer('The agent shell and model runtime are separate systems.', opacity * reveal(local, 0.58, 0.74), C.blue);
  if (mode === 'request') return pipelineBlocks(opacity, 3) + circle(mix(230, 711, p), 263, 7, C.teal, opacity) + footer('Hermes sends an OpenAI-compatible request to localhost.', opacity * reveal(local, 0.58, 0.74), C.teal);
  if (mode === 'weights') return pipelineBlocks(opacity, 4) + circle(mix(678, 902, p), 263, 7, C.amber, opacity) + footer('llama-server owns sampling, the KV cache, and GGUF loading.', opacity * reveal(local, 0.58, 0.74), C.amber);
  if (mode === 'return') return pipelineBlocks(opacity, 4) + circle(mix(901, 230, p), 318, 7, C.green, opacity) + text('generated tokens return through the same API boundary', 480, 369, 16, C.green, 'middle', 600, opacity * p);
  return `${pipelineBlocks(opacity, 4)}${text('agent or tool failure', 145, 356, 13, C.blue, 'middle', 650, opacity)}${arrow(145, 339, 145, 306, C.blue, 1.4, opacity)}${text('connection / HTTP failure', 369, 391, 13, C.teal, 'middle', 650, opacity)}${arrow(369, 372, 369, 306, C.teal, 1.4, opacity)}${text('sampling or cache failure', 593, 356, 13, C.green, 'middle', 650, opacity)}${arrow(593, 339, 593, 306, C.green, 1.4, opacity)}${text('load or tokenizer failure', 817, 391, 13, C.amber, 'middle', 650, opacity)}${arrow(817, 372, 817, 306, C.amber, 1.4, opacity)}${footer('Test the layer that owns the failed contract; do not reinstall the whole stack.', opacity, C.amber)}`;
}

const gemmaLong = [
  ['E2B · MLX', 879], ['E4B · MLX', 1682], ['26B A4B · MLX', 2182], ['31B · MLX', 13501],
];
const gemmaRuntime = [
  ['E4B', 1682, 3068], ['26B A4B', 2182, 3227], ['31B', 13501, 24164],
];
const gemmaDecode = [
  ['E2B · MLX', 175.7], ['E4B · MLX', 104.0], ['26B A4B · MLX', 104.4], ['31B · MLX', 23.7],
];
const qwenLong = [
  ['Qwen 3 4B', 1742], ['Qwen 3.5 4B', 2103], ['Qwen 3.5 9B', 2894], ['Qwen 3 14B', 4925],
];
const qwenRuntime = [
  ['Qwen 3.5 9B', 2894, 5158], ['Qwen 3 14B', 4925, 11112],
];
const qwenDecode = [
  ['Qwen 3 4B', 127.8], ['Qwen 3.5 4B', 131.5], ['Qwen 3.5 9B', 92.7], ['Qwen 3 14B', 52.3],
];

function pairedBars(rows, max, opacity) {
  let out = '';
  rows.forEach(([name, mlx, llama], i) => {
    const y = 165 + i * 82;
    out += text(name, 116, y + 15, 15, C.ink, 'start', 560, opacity);
    out += rect(300, y, 380 * mlx / max, 18, C.teal, 5, opacity * 0.42);
    out += text(`MLX ${(mlx / 1000).toFixed(2)} s`, 706, y + 15, 13, C.teal, 'start', 540, opacity);
    out += rect(300, y + 30, 380 * llama / max, 18, C.amber, 5, opacity * 0.42);
    out += text(`llama.cpp ${(llama / 1000).toFixed(2)} s`, 706, y + 45, 13, C.amber, 'start', 540, opacity);
  });
  return out;
}

function drawBenchmark(family, mode, local, opacity) {
  const p = reveal(local);
  const isGemma = family === 'gemma';
  if (mode === 'design') return `${token('512 tokens', 276, 212, opacity, C.teal, 160)}${token('8192 tokens', 524, 212, opacity, C.blue, 160)}${arrow(454, 233, 506, 233, C.muted, 1.5, opacity * p)}${text('same model · same machine · prompt length changes', 480, 334, 18, C.ink, 'middle', 600, opacity * p)}${footer('Measured on one 64 GB M5 Max · April 4, 2026', opacity * reveal(local, 0.58, 0.74))}`;
  if (mode === 'long') {
    const rows = isGemma ? gemmaLong : qwenLong;
    return bars(rows, 118, 155, 440, isGemma ? 14000 : 5200, opacity * p, C.blue, ' ms') + footer('Long prompts separate models through prefill latency.', opacity * reveal(local, 0.60, 0.76), C.blue);
  }
  if (mode === 'runtime') {
    const rows = isGemma ? gemmaRuntime : qwenRuntime;
    return pairedBars(rows, isGemma ? 25000 : 12000, opacity * p) + footer('Runtime choice matters most when prefill is already expensive.', opacity * reveal(local, 0.60, 0.76), C.teal);
  }
  if (mode === 'decode') {
    const rows = isGemma ? gemmaDecode : qwenDecode;
    return bars(rows, 118, 155, 440, isGemma ? 190 : 140, opacity * p, C.green, ' tok/s') + footer('Decode throughput changes less than time to first token.', opacity * reveal(local, 0.60, 0.76), C.green);
  }
  if (mode === 'decision') {
    return isGemma
      ? summary([['E2B', 'sub-second long-prompt TTFT on MLX'], ['E4B', '1.68 s long-prompt TTFT on MLX'], ['26B A4B', '2.18 s long-prompt TTFT on MLX'], ['31B', '13.50 s long-prompt TTFT on MLX']], opacity, 'A model can fit in memory and still have slow first-token latency.')
      : summary([['4B', '1.74–2.10 s long-prompt TTFT on MLX'], ['9B', '2.89 s MLX · 5.16 s llama.cpp'], ['14B', '4.93 s MLX · 11.11 s llama.cpp']], opacity, 'The 4B models have the lowest measured first-token latency.');
  }
  return '';
}

function vehicle(x, y, w, h, color, opacity, label = '') {
  let out = rect(x, y, w, h, C.bg, 6, opacity, color, 1.5);
  out += line(x + w * 0.25, y + 5, x + w * 0.75, y + 5, color, 2, opacity);
  if (label) out += text(label, x + w / 2, y + h + 17, 13, color, 'middle', 650, opacity);
  return out;
}

function cyclist(x, y, opacity, color = C.green, label = '') {
  let out = circle(x - 8, y + 8, 7, C.bg, opacity, color, 1.5);
  out += circle(x + 8, y + 8, 7, C.bg, opacity, color, 1.5);
  out += circle(x, y - 9, 4, color, opacity);
  out += line(x, y - 4, x - 5, y + 6, color, 1.6, opacity);
  out += line(x, y - 4, x + 8, y + 7, color, 1.6, opacity);
  if (label) out += text(label, x, y + 34, 13, color, 'middle', 650, opacity);
  return out;
}

function roadScene(x, y, w, h, progress, opacity, options = {}) {
  const p = clamp(progress);
  const roadX = x + w * 0.18;
  const roadW = w * 0.64;
  const crossY = y + h * 0.27;
  const egoX = x + w * 0.50 - 13;
  const egoY = y + h * 0.78 - 45 * p;
  const leadY = y + h * 0.38 - 22 * p;
  const cyclistX = x + w * 0.90 - 116 * p;
  const cyclistY = crossY + 4;
  const vanX = x + w * 0.72;
  const vanY = crossY - 16;
  let out = rect(x, y, w, h, '#0b0b0b', 12, opacity, C.faint, 1.1);
  out += rect(roadX, y + 1, roadW, h - 2, '#11110f', 0, opacity);
  out += line(x + w * 0.50, y + 8, x + w * 0.50, y + h - 8, C.muted, 1.3, opacity * 0.55, '10 10');
  out += line(roadX, y, roadX, y + h, C.amber, 1.4, opacity * 0.65);
  out += line(roadX + roadW, y, roadX + roadW, y + h, C.amber, 1.4, opacity * 0.65);
  for (let i = 0; i < 7; i++) out += rect(roadX + 5 + i * (roadW - 10) / 7, crossY, (roadW - 24) / 8, 12, C.ink, 1, opacity * 0.38);

  if (options.sensor === 'camera' || options.sensor === 'all') {
    out += path(`M ${egoX + 13} ${egoY} L ${roadX + 12} ${y + 20} L ${roadX + roadW - 12} ${y + 20} Z`, C.blue, 1.2, opacity * 0.52, C.blue);
  }
  if (options.sensor === 'lidar' || options.sensor === 'all') {
    [25, 48, 72].forEach((r) => { out += ring(egoX + 13, egoY + 12, r * (0.55 + 0.25 * p), C.amber, opacity * 0.55, 1); });
  }
  if (options.sensor === 'radar' || options.sensor === 'all') {
    out += path(`M ${egoX + 13} ${egoY} L ${egoX - 52} ${leadY + 10} L ${egoX + 78} ${leadY + 10} Z`, C.rose, 1.2, opacity * 0.42, C.rose);
  }

  if (options.trail) {
    [0, 1, 2].forEach((i) => {
      out += vehicle(egoX, egoY + 18 + i * 21, 26, 38, C.blue, opacity * (0.22 - i * 0.05));
      out += cyclist(cyclistX + 28 + i * 24, cyclistY, opacity * (0.24 - i * 0.05), C.green);
    });
  }
  out += vehicle(egoX, egoY, 26, 38, C.blue, opacity, options.labels ? 'EGO' : '');
  out += vehicle(x + w * 0.50 - 12, leadY, 24, 34, C.rose, opacity, options.labels ? 'LEAD' : '');

  const cyclistHidden = options.hideCyclist || (p > 0.22 && p < 0.52);
  if (!cyclistHidden) out += cyclist(cyclistX, cyclistY, opacity, C.green, options.labels ? 'CYCLIST' : '');
  if (cyclistHidden && options.showPrediction) {
    out += ring(cyclistX, cyclistY, 18 + 12 * p, C.green, opacity, 1.4, '5 4');
    out += text('predicted', cyclistX - 32, cyclistY + 39, 13, C.green, 'end', 650, opacity);
  }
  out += vehicle(vanX, vanY, 34, 64, C.amber, opacity, options.labels ? 'VAN' : '');

  if (options.rain) {
    for (let i = 0; i < 15; i++) {
      const rx = x + 15 + ((i * 41) % (w - 30));
      const ry = y + 12 + ((i * 29) % (h - 24));
      out += line(rx, ry, rx - 7, ry + 16, C.blue, 1.2, opacity * 0.55);
    }
  }
  return out;
}

function forwardDrivingScene(x, y, w, h, progress, opacity, options = {}) {
  const p = clamp(progress);
  const horizon = y + h * 0.25;
  const actorScale = 0.55 + 0.75 * p;
  const actorX = x + w * (0.68 - 0.11 * p);
  const actorY = y + h * (0.48 + 0.23 * p);
  let out = rect(x, y, w, h, '#0b0b0b', 12, opacity, C.faint, 1.1);
  out += path(`M ${x + 18} ${y + h} L ${x + w * 0.43} ${horizon} L ${x + w * 0.57} ${horizon} L ${x + w - 18} ${y + h}`, C.muted, 1.4, opacity * 0.55);
  out += line(x + w * 0.50, horizon, x + w * 0.50, y + h, C.muted, 1.1, opacity * 0.38, '10 10');
  for (let i = 0; i < 6; i++) out += rect(x + 25 + i * (w - 60) / 6, y + h * 0.53, (w - 80) / 8, 9, C.ink, 1, opacity * 0.30);
  out += vehicle(x + w * 0.47, y + h * (0.38 + 0.12 * p), 26 + 12 * p, 28 + 18 * p, C.rose, opacity, options.labels ? 'LEAD' : '');
  out += vehicle(x + w * 0.78, y + h * 0.36, 38, 72, C.amber, opacity, options.labels ? 'VAN' : '');
  if (p < 0.30) out += circle(actorX, actorY, 5 * actorScale, C.green, opacity);
  if (p >= 0.30) out += cyclist(actorX, actorY, opacity, C.green, options.labels ? 'CYCLIST' : '');
  if (options.boxActor) out += rect(actorX - 24 * actorScale, actorY - 27 * actorScale, 48 * actorScale, 58 * actorScale, C.bg, 3, opacity, C.green, 1.4);
  if (options.rain) {
    for (let i = 0; i < 15; i++) out += line(x + 15 + ((i * 43) % (w - 30)), y + 12 + ((i * 31) % (h - 24)), x + 8 + ((i * 43) % (w - 30)), y + 28 + ((i * 31) % (h - 24)), C.blue, 1, opacity * 0.55);
  }
  return out;
}

function sensorEvidenceCards(opacity) {
  const cards = [
    ['CAMERA', 'cyclist + lane', C.blue],
    ['LiDAR', '3D box: 24 m', C.amber],
    ['RADAR', 'closing: 5 m/s', C.rose],
  ];
  return cards.map(([name, value, color], i) => {
    const x = 94 + i * 258;
    return `${rect(x, 182, 214, 96, C.bg, 12, opacity, color, 1.3)}${text(name, x + 18, 215, 14, color, 'start', 700, opacity, 1.2)}${text(value, x + 18, 251, 16, C.ink, 'start', 580, opacity)}${cyclist(x + 186, 228, opacity, C.green)}`;
  }).join('');
}

function drawCameraEncoder(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return `${forwardDrivingScene(64, 180, 250, 190, 0.08, opacity, { boxActor: true })}${forwardDrivingScene(355, 180, 250, 190, 0.36, opacity, { boxActor: true })}${forwardDrivingScene(646, 180, 250, 190, 0.78, opacity, { boxActor: true })}${text('distant · a few pixels', 189, 405, 15, C.red, 'middle', 650, opacity)}${text('approaching', 480, 405, 15, C.amber, 'middle', 650, opacity)}${text('close · clear shape', 771, 405, 15, C.green, 'middle', 650, opacity)}${footer('The same cyclist grows in the image as the ego car closes distance.', opacity, C.green)}`;
  if (mode === 'coarse') return `${forwardDrivingScene(70, 170, 340, 220, 0.31, opacity, { boxActor: true })}${ring(290, 291, 24, C.green, opacity, 1.8)}${text('CYCLIST', 290, 334, 14, C.green, 'middle', 700, opacity)}${arrow(434, 280, 528, 280, C.red, 2, opacity * p)}${spatialGrid(566, 188, 286, 188, 5, 4, opacity, C.red, { col: 3, row: 1, color: C.red, crossed: true })}${text('one stride-16 cell mixes actor + background', 709, 409, 15, C.red, 'middle', 650, opacity)}${footer('The coarse map has no separate cyclist cell for later geometry to recover.', opacity, C.red)}`;
  if (mode === 'pyramid') return `${forwardDrivingScene(60, 172, 310, 210, 0.31, opacity, { boxActor: true })}${ring(260, 288, 23, C.green, opacity, 1.8)}${text('CYCLIST', 260, 329, 13, C.green, 'middle', 700, opacity)}${arrow(392, 278, 470, 278, C.teal, 2, opacity * p)}${text('FINE MAP · actor boundary', 632, 169, 14, C.green, 'middle', 700, opacity)}${spatialGrid(486, 184, 292, 112, 10, 4, opacity, C.green, { col: 7, row: 1, color: C.green })}${text('COARSE MAP · intersection context', 676, 329, 14, C.teal, 'middle', 700, opacity)}${spatialGrid(562, 344, 228, 66, 5, 2, opacity, C.teal)}${footer('The pyramid keeps a cyclist cell and a wider view of the lane and crosswalk.', opacity, C.teal)}`;
  if (mode === 'supervision') return `${forwardDrivingScene(70, 170, 340, 220, 0.31, opacity, { boxActor: true })}${ring(290, 291, 24, C.green, opacity, 1.8)}${arrow(434, 280, 520, 280, C.green, 2, opacity * p)}${rect(548, 176, 310, 92, C.bg, 10, opacity, C.green, 1.2)}${text('IMAGE LOSS', 703, 208, 14, C.green, 'middle', 700, opacity)}${text('cyclist remains separable in the image map', 703, 244, 15, C.ink, 'middle', 560, opacity)}${rect(548, 286, 310, 92, C.bg, 10, opacity, C.teal, 1.2)}${text('BEV LOSS', 703, 318, 14, C.teal, 'middle', 700, opacity)}${text('cyclist lands in the correct metric cell', 703, 354, 15, C.ink, 'middle', 560, opacity)}${footer('Perspective supervision gives the backbone a direct reason to keep the small actor.', opacity, C.green)}`;
  return `${forwardDrivingScene(62, 170, 330, 220, 0.31, opacity, { boxActor: true })}${ring(275, 291, 25, C.green, opacity, 2)}${text('CYCLIST', 275, 335, 14, C.green, 'middle', 750, opacity)}${arrow(415, 280, 490, 280, C.teal, 2, opacity)}${text('COARSE ONLY', 630, 172, 14, C.red, 'middle', 700, opacity)}${spatialGrid(506, 188, 248, 82, 5, 2, opacity, C.red, { col: 3, row: 0, color: C.red, crossed: true })}${text('actor merged', 782, 235, 14, C.red, 'start', 650, opacity)}${text('FINE + COARSE PYRAMID', 660, 316, 14, C.green, 'middle', 700, opacity)}${spatialGrid(506, 332, 248, 82, 10, 3, opacity, C.green, { col: 7, row: 1, color: C.green })}${text('actor retained', 782, 379, 14, C.green, 'start', 650, opacity)}${footer('If the camera encoder erases the cyclist, projection and fusion cannot recreate it.', opacity, C.amber)}`;
}

function drawLidar(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return `${roadScene(210, 140, 540, 290, p, opacity, { sensor: 'lidar', trail: true, labels: true })}${text('sweep start', 232, 164, 14, C.muted, 'start', 600, opacity)}${text('sweep end', 728, 164, 14, C.amber, 'end', 600, opacity)}${footer('The ego car moves during one scan; old and new returns arrive in different poses.', opacity * reveal(local, 0.60, 0.76))}`;
  if (mode === 'compensate') return `${roadScene(70, 170, 350, 220, p, opacity, { sensor: 'lidar', trail: true })}${arrow(443, 280, 523, 280, C.amber, 2, opacity * p)}${roadScene(548, 170, 340, 220, p, opacity, { sensor: 'lidar' })}${text('motion-compensated', 718, 414, 16, C.amber, 'middle', 650, opacity)}${footer('Ego-motion compensation aligns the static curb; the moving cyclist still changes position.', opacity * reveal(local, 0.62, 0.78), C.amber)}`;
  if (mode === 'pillars') return `${roadScene(75, 175, 330, 210, p, opacity, { sensor: 'lidar' })}${arrow(428, 280, 518, 280, C.amber, 2, opacity * p)}${bev(565, 178, 285, 210, opacity, C.amber)}${rect(720, 248, 36, 42, C.bg, 4, opacity, C.green, 1.5)}${text('one x-y cell', 707, 414, 16, C.amber, 'middle', 650, opacity)}${footer('Pillars keep horizontal location but collapse the height structure inside each cell.', opacity * reveal(local, 0.62, 0.78), C.amber)}`;
  if (mode === 'voxels') return `${roadScene(75, 175, 330, 210, p, opacity, { sensor: 'lidar' })}${arrow(428, 280, 518, 280, C.blue, 2, opacity * p)}${grid(565, 178, 8, 6, 35, C.blue, 0.38 * p, opacity)}${text('occupied 3D cells only', 705, 414, 16, C.blue, 'middle', 650, opacity)}${footer('Sparse voxels preserve height and spend compute only where the scan has returns.', opacity * reveal(local, 0.62, 0.78), C.blue)}`;
  if (mode === 'windows') return `${text('LAYER 1 · vertical sets', 290, 161, 14, C.blue, 'middle', 700, opacity)}${spatialGrid(140, 178, 300, 190, 6, 5, opacity, C.blue, { col: 2, row: 2, color: C.green })}${line(290, 178, 290, 368, C.amber, 3, opacity)}${cyclist(265, 273, opacity, C.green)}${text('cyclist at set edge', 290, 401, 14, C.green, 'middle', 650, opacity)}${arrow(464, 273, 536, 273, C.teal, 2, opacity * p)}${text('LAYER 2 · horizontal sets', 690, 161, 14, C.teal, 'middle', 700, opacity)}${spatialGrid(540, 178, 300, 190, 6, 5, opacity, C.teal, { col: 2, row: 2, color: C.green })}${line(540, 292, 840, 292, C.amber, 3, opacity)}${cyclist(665, 273, opacity, C.green)}${arrow(646, 271, 706, 325, C.green, 1.8, opacity)}${text('next layer crosses the old boundary', 690, 401, 14, C.green, 'middle', 650, opacity)}${footer('Alternating set directions let nearby occupied voxels exchange context without global attention.', opacity, C.teal)}`;
  return `${roadScene(62, 166, 330, 230, 0.72, opacity, { sensor: 'lidar', labels: true })}${ring(275, 232, 25, C.green, opacity, 1.8)}${text('same cyclist returns', 227, 427, 14, C.green, 'middle', 650, opacity)}${arrow(416, 281, 486, 281, C.teal, 2, opacity)}${rect(512, 154, 360, 112, C.bg, 10, opacity, C.amber, 1.2)}${text('PILLAR', 536, 187, 14, C.amber, 'start', 700, opacity)}${text('x-y cell retained', 536, 222, 16, C.ink, 'start', 600, opacity)}${text('height pooled', 842, 222, 16, C.red, 'end', 650, opacity)}${rect(512, 286, 360, 112, C.bg, 10, opacity, C.blue, 1.2)}${text('SPARSE VOXELS', 536, 319, 14, C.blue, 'start', 700, opacity)}${text('x-y-z cells retained', 536, 354, 16, C.ink, 'start', 600, opacity)}${text('occupied cells only', 842, 354, 16, C.green, 'end', 650, opacity)}${footer('Keep the cyclist geometry until the downstream task no longer needs its height.', opacity, C.amber)}`;
}

function radarSweeps(opacity, progress) {
  const p = clamp(progress);
  let out = line(164, 288, 796, 288, C.faint, 2, opacity);
  const ranges = [34, 30, 26];
  ranges.forEach((range, i) => {
    const x = 224 + i * 248;
    const active = clamp(p * 3 - i);
    out += circle(x, 288, 42, C.bg, opacity * active, C.rose, 1.4);
    out += vehicle(x - 14, 239, 28, 40, C.rose, opacity * active);
    out += text(`sweep ${i + 1}`, x, 345, 15, C.rose, 'middle', 700, opacity * active);
    out += text(`${range} m`, x, 374, 18, C.ink, 'middle', 650, opacity * active);
    out += text('−5 m/s radial', x, 402, 15, C.muted, 'middle', 560, opacity * active);
  });
  return out;
}

function radarPlacementLane(y, opacity, corrected = false) {
  const egoX = 142;
  const radarX = 548;
  const cameraX = 668;
  let out = line(118, y, 842, y, C.faint, 2, opacity);
  out += vehicle(egoX, y - 20, 28, 40, C.blue, opacity, 'EGO');
  out += vehicle(radarX, y - 21, 30, 42, C.rose, opacity, 'LEAD CAR');
  out += ring(radarX + 15, y, 29, C.rose, opacity, 1.6);
  out += text('radar return · 26 m', radarX + 15, y - 43, 14, C.rose, 'middle', 700, opacity);
  out += path(`M ${cameraX} ${y - 30} L ${cameraX + 48} ${y - 30} L ${cameraX + 48} ${y + 30} L ${cameraX} ${y + 30} Z`, C.blue, 1.8, opacity * (corrected ? 0.38 : 1), 'none', '6 5');
  out += text('camera box', cameraX + 24, y + 51, 13, C.blue, 'middle', 650, opacity);
  if (corrected) {
    out += arrow(cameraX - 4, y, radarX + 42, y, C.green, 2, opacity);
    out += rect(radarX - 7, y - 32, 44, 64, C.bg, 4, opacity, C.green, 2);
    out += text('moved to measured range', 430, y + 54, 14, C.green, 'middle', 650, opacity);
  } else {
    out += text('still misplaced', cameraX + 24, y - 43, 14, C.red, 'middle', 700, opacity);
  }
  return out;
}

function drawRadar(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return radarSweeps(opacity, p) + footer('Range falls from 34 m to 26 m; Doppler measures a 5 m/s closing speed.', opacity * reveal(local, 0.62, 0.78), C.rose);
  if (mode === 'proposal') return `${radarPlacementLane(274, opacity, false)}${text('PROPOSAL FUSION', 480, 158, 15, C.red, 'middle', 750, opacity)}${footer('The radar confirms that a lead car exists, but the camera branch still controls its placement.', opacity, C.rose)}`;
  if (mode === 'depth') return `${radarPlacementLane(274, opacity, true)}${text('DEPTH FUSION', 480, 158, 15, C.green, 'middle', 750, opacity)}${footer('The radar measurement moves the camera feature to the supported 26 m range.', opacity, C.green)}`;
  if (mode === 'bev') return `${roadScene(150, 155, 660, 255, p, opacity, { sensor: 'radar', labels: true })}${text('LEAD CAR · 26 m · −5 m/s', 480, 184, 15, C.rose, 'middle', 700, opacity)}${text('GUARDRAIL RETURN · 0 m/s', 682, 384, 14, C.muted, 'middle', 600, opacity)}${footer('A radar BEV keeps moving and stationary returns separate until other sensors resolve identity.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  return `${text('PROPOSAL FUSION · box stays wrong', 480, 150, 14, C.red, 'middle', 700, opacity)}${radarPlacementLane(238, opacity, false)}${text('DEPTH FUSION · box moves to 26 m', 480, 331, 14, C.green, 'middle', 700, opacity)}${radarPlacementLane(397, opacity, true)}${footer('Fuse radar before box placement if range must change the geometry.', opacity, C.amber)}`;
}

function drawLifting(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return `${forwardDrivingScene(72, 170, 350, 220, p, opacity, { boxActor: true })}${arrow(447, 280, 532, 280, C.amber, 2, opacity * p)}${roadScene(558, 170, 330, 220, p, opacity, {})}${line(590, 355, 818, 214, C.amber, 2, opacity)}${[0,1,2,3].map((i)=>circle(620+i*54,336-i*32,7,C.faint,opacity)).join('')}${footer('The cyclist occupies one image patch, but could lie at several positions along its 3D ray.', opacity * reveal(local, 0.62, 0.78))}`;
  if (mode === 'lss') { let out = `${forwardDrivingScene(72, 170, 350, 220, p, opacity, { boxActor: true })}${arrow(447, 280, 532, 280, C.teal, 2, opacity * p)}${roadScene(558, 170, 330, 220, p, opacity, {})}${line(590, 355, 818, 214, C.faint, 2, opacity)}`; for (let i=0;i<6;i++) out += circle(608+i*42,344-i*25,8,i===3?C.green:C.teal,opacity*(i===3?1:0.35)); return out + text('depth probability along ray', 712, 408, 16, C.teal, 'middle', 650, opacity) + footer('LSS spreads the cyclist feature over depth bins, then accumulates those bins into BEV.', opacity * reveal(local, 0.62, 0.78), C.teal); }
  if (mode === 'detr3d') return `${forwardDrivingScene(72, 170, 350, 220, p, opacity, { boxActor: true })}${arrow(447, 280, 532, 280, C.blue, 2, opacity * p)}${roadScene(558, 170, 330, 220, p, opacity, {})}${circle(705, 252, 19, C.blue, 0.10 * opacity, C.blue, 1.6)}${line(590, 355, 705, 252, C.blue, 2, opacity * p)}${text('candidate actor at 24 m', 715, 414, 16, C.blue, 'middle', 650, opacity)}${footer('An object query starts from a 3D reference, then asks whether the camera supports it.', opacity * reveal(local, 0.62, 0.78), C.blue)}`;
  if (mode === 'bevformer') { let out = `${forwardDrivingScene(72, 170, 350, 220, p, opacity, { boxActor: true })}${arrow(447, 280, 532, 280, C.teal, 2, opacity * p)}${roadScene(558, 170, 330, 220, p, opacity, {})}`; for(let i=0;i<4;i++){out += circle(705,326-i*31,6,C.teal,opacity*p); out += line(590,355,705,326-i*31,C.teal,1,opacity*p);} return out + text('vertical samples in one BEV cell', 715, 414, 16, C.teal, 'middle', 650, opacity) + footer('A BEV cell projects several height references into the camera and gathers matching evidence.', opacity * reveal(local, 0.62, 0.78), C.teal); }
  return `${forwardDrivingScene(64, 174, 330, 214, 0.31, opacity, { boxActor: true })}${ring(277, 291, 24, C.green, opacity, 2)}${text('one cyclist image patch', 229, 421, 14, C.green, 'middle', 650, opacity)}${arrow(418, 278, 512, 278, C.amber, 2, opacity)}${text('BEV CELLS ALONG THE VIEWING RAY', 704, 161, 14, C.amber, 'middle', 700, opacity)}${spatialGrid(548, 178, 312, 210, 6, 4, opacity, C.teal)}${path('M 568 360 L 832 198', C.amber, 2, opacity, 'none', '7 6')}${rect(754, 182, 48, 48, C.red, 3, opacity * 0.22, C.red, 1.5)}${line(761, 189, 795, 223, C.red, 2, opacity)}${line(795, 189, 761, 223, C.red, 2, opacity)}${text('wrong depth', 778, 250, 13, C.red, 'middle', 700, opacity)}${rect(650, 285, 48, 48, C.green, 3, opacity * 0.22, C.green, 1.5)}${cyclist(674, 307, opacity, C.green)}${text('correct cyclist cell', 674, 361, 13, C.green, 'middle', 700, opacity)}${footer('Depth is placement: an error writes real image evidence into the wrong metric location.', opacity, C.amber)}`;
}

function drawFusion(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return `${sensorEvidenceCards(opacity)}${arrow(201, 300, 480, 364, C.blue, 1.8, opacity * p)}${arrow(459, 300, 480, 364, C.amber, 1.8, opacity * p)}${arrow(717, 300, 480, 364, C.rose, 1.8, opacity * p)}${text('same cyclist · complementary measurements', 480, 405, 18, C.ink, 'middle', 620, opacity * p)}${footer('Camera identifies the actor, LiDAR places it, and radar measures closing speed.', opacity * reveal(local, 0.62, 0.78))}`;
  if (mode === 'point') return `${sensorEvidenceCards(opacity)}${arrow(201, 300, 480, 352, C.blue, 1.7, opacity * p)}${arrow(459, 300, 480, 352, C.amber, 1.7, opacity * p)}${arrow(717, 300, 480, 352, C.rose, 1.7, opacity * p)}${circle(480, 372, 24, C.teal, 0.12 * opacity, C.teal, 1.5)}${text('actor point', 480, 420, 16, C.teal, 'middle', 650, opacity)}${footer('Point fusion combines the three measurements at the cyclist, but keeps no camera-only lane field.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  if (mode === 'query') return `${sensorEvidenceCards(opacity)}${arrow(201, 300, 480, 346, C.blue, 1.7, opacity * p)}${arrow(459, 300, 480, 346, C.amber, 1.7, opacity * p)}${arrow(717, 300, 480, 346, C.rose, 1.7, opacity * p)}${rect(353, 357, 254, 68, C.bg, 12, opacity, C.green, 1.4)}${text('CYCLIST QUERY', 480, 386, 14, C.green, 'middle', 700, opacity)}${text('type · 24 m · −5 m/s', 480, 414, 16, C.ink, 'middle', 600, opacity)}${footer('Object-query fusion keeps a compact cyclist state, but not the whole road surface.', opacity * reveal(local, 0.62, 0.78), C.green)}`;
  if (mode === 'bev') return `${sensorEvidenceCards(opacity)}${arrow(201, 300, 480, 326, C.blue, 1.7, opacity * p)}${arrow(459, 300, 480, 326, C.amber, 1.7, opacity * p)}${arrow(717, 300, 480, 326, C.rose, 1.7, opacity * p)}${roadScene(350, 336, 260, 108, p, opacity, {})}${footer('Dense BEV keeps the cyclist, lane boundary, crosswalk, and free space in one spatial field.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  return `${sensorEvidenceCards(opacity)}${arrow(201, 297, 214, 333, C.blue, 1.6, opacity)}${arrow(459, 297, 480, 333, C.amber, 1.6, opacity)}${arrow(717, 297, 746, 333, C.rose, 1.6, opacity)}${rect(92, 334, 244, 96, C.bg, 11, opacity, C.teal, 1.2)}${text('POINT', 214, 362, 13, C.teal, 'middle', 700, opacity)}${circle(214, 393, 9, C.green, opacity)}${text('one measured location', 214, 420, 13, C.muted, 'middle', 560, opacity)}${rect(358, 334, 244, 96, C.bg, 11, opacity, C.green, 1.2)}${text('OBJECT QUERY', 480, 362, 13, C.green, 'middle', 700, opacity)}${text('cyclist · 24 m · −5 m/s', 480, 397, 15, C.ink, 'middle', 620, opacity)}${rect(624, 334, 244, 96, C.bg, 11, opacity, C.blue, 1.2)}${text('DENSE BEV', 746, 362, 13, C.blue, 'middle', 700, opacity)}${roadScene(680, 374, 132, 45, 0.72, opacity, {})}${footer('The fusion unit decides whether prediction receives one point, one actor, or the surrounding road field.', opacity, C.amber)}`;
}

function drawDropout(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'failure') return `${roadScene(70, 170, 350, 220, p, opacity, { sensor: 'all', labels: true })}${arrow(444, 280, 520, 280, C.red, 2, opacity * p)}${roadScene(544, 170, 350, 220, p, opacity, { sensor: 'all', labels: true, rain: true })}${text('clear', 245, 415, 16, C.green, 'middle', 650, opacity)}${text('rain + glare', 719, 415, 16, C.red, 'middle', 650, opacity)}${footer('The physical scene changes smoothly, but camera quality can fall before the sensor is fully absent.', opacity * reveal(local, 0.62, 0.78), C.red)}`;
  if (mode === 'unibev') return `${roadScene(92, 170, 330, 220, p, opacity, { sensor: 'all', rain: true })}${rect(548, 178, 300, 198, C.bg, 12, opacity, C.teal, 1.3)}${text('AVAILABLE INPUTS', 698, 211, 14, C.teal, 'middle', 700, opacity)}${text('camera', 585, 253, 16, C.muted, 'start', 600, opacity)}${text('degraded, still present', 815, 253, 16, C.amber, 'end', 600, opacity)}${text('LiDAR', 585, 294, 16, C.muted, 'start', 600, opacity)}${text('present', 815, 294, 16, C.green, 'end', 600, opacity)}${text('radar', 585, 335, 16, C.muted, 'start', 600, opacity)}${text('present', 815, 335, 16, C.green, 'end', 600, opacity)}${footer('An availability mask can represent a missing sensor, but not how trustworthy a present sensor is.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  if (mode === 'metabev') return `${roadScene(92, 170, 330, 220, p, opacity, { sensor: 'all', rain: true })}${arrow(446, 280, 520, 280, C.green, 2, opacity * p)}${rect(548, 178, 300, 198, C.bg, 12, opacity, C.green, 1.3)}${text('SHARED SCENE STATE', 698, 213, 14, C.green, 'middle', 700, opacity)}${text('cyclist: 24 m', 585, 260, 17, C.ink, 'start', 600, opacity)}${text('lead car: closing', 585, 302, 17, C.ink, 'start', 600, opacity)}${text('crosswalk: occupied', 585, 344, 17, C.ink, 'start', 600, opacity)}${footer('Training across sensor subsets teaches one scene state to survive a missing input stream.', opacity * reveal(local, 0.62, 0.78), C.green)}`;
  if (mode === 'grace') return `${roadScene(92, 170, 330, 220, p, opacity, { sensor: 'all', rain: true })}${rect(548, 178, 300, 198, C.bg, 12, opacity, C.amber, 1.3)}${text('RELIABILITY WEIGHTS', 698, 213, 14, C.amber, 'middle', 700, opacity)}${text('camera', 585, 257, 16, C.blue, 'start', 650, opacity)}${text('0.25', 815, 257, 17, C.ink, 'end', 650, opacity)}${rect(650, 241, 110 * 0.25, 18, C.blue, 4, opacity * 0.45)}${text('LiDAR', 585, 302, 16, C.amber, 'start', 650, opacity)}${text('0.95', 815, 302, 17, C.ink, 'end', 650, opacity)}${rect(650, 286, 110 * 0.95, 18, C.amber, 4, opacity * 0.45)}${text('radar', 585, 347, 16, C.rose, 'start', 650, opacity)}${text('0.85', 815, 347, 17, C.ink, 'end', 650, opacity)}${rect(650, 331, 110 * 0.85, 18, C.rose, 4, opacity * 0.45)}${footer('Reliability gating lowers the rain-damaged camera contribution without pretending it vanished.', opacity * reveal(local, 0.62, 0.78), C.amber)}`;
  return `${roadScene(58, 166, 350, 230, 0.72, opacity, { sensor: 'all', labels: true, rain: true })}${text('same cyclist · rain + glare', 233, 426, 14, C.green, 'middle', 650, opacity)}${arrow(431, 279, 506, 279, C.amber, 2, opacity)}${rect(532, 160, 340, 104, C.bg, 11, opacity, C.teal, 1.2)}${text('AVAILABILITY MASK', 556, 193, 14, C.teal, 'start', 700, opacity)}${text('camera = present', 556, 231, 17, C.ink, 'start', 620, opacity)}${text('cannot express degraded', 846, 231, 14, C.red, 'end', 650, opacity)}${rect(532, 286, 340, 104, C.bg, 11, opacity, C.amber, 1.2)}${text('RELIABILITY WEIGHT', 556, 319, 14, C.amber, 'start', 700, opacity)}${text('camera = 0.25', 556, 357, 17, C.ink, 'start', 620, opacity)}${rect(716, 341, 112, 18, C.faint, 4, opacity)}${rect(716, 341, 28, 18, C.blue, 4, opacity * 0.62)}${footer('A present camera can still be unreliable; continuous gating can down-weight it.', opacity, C.amber)}`;
}

function temporalPanels(opacity, progress) {
  const p = clamp(progress);
  const states = [
    ['t₀ · observed', 0.10, { labels: true }],
    ['t₁ · behind van', 0.38, { hideCyclist: true, showPrediction: true, labels: true }],
    ['t₂ · observed again', 0.82, { labels: true }],
  ];
  return states.map(([label, sceneP, options], i) => {
    const x = 64 + i * 300;
    const active = clamp(p * 3 - i);
    return `${roadScene(x, 180, 260, 190, sceneP, opacity * active, options)}${text(label, x + 130, 405, 16, i === 1 ? C.amber : C.green, 'middle', 650, opacity * active)}`;
  }).join('');
}

function drawTemporal(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return temporalPanels(opacity, p) + footer('The cyclist is seen, disappears behind the van, then reappears farther across the crosswalk.', opacity * reveal(local, 0.62, 0.78));
  if (mode === 'warp') return `${roadScene(70, 175, 350, 215, 0.10, opacity, {})}${arrow(444, 280, 520, 280, C.blue, 2, opacity * p)}${roadScene(544, 175, 350, 215, 0.38, opacity, { hideCyclist: true, showPrediction: true })}${text('ego moved 3 m', 245, 415, 16, C.blue, 'middle', 650, opacity)}${text('past field shifted into t₁', 719, 415, 16, C.teal, 'middle', 650, opacity)}${footer('Dense memory first compensates ego motion, then carries the cyclist region through occlusion.', opacity * reveal(local, 0.62, 0.78), C.blue)}`;
  if (mode === 'instances') return `${roadScene(76, 175, 340, 215, 0.38, opacity, { hideCyclist: true, showPrediction: true })}${arrow(441, 280, 520, 280, C.green, 2, opacity * p)}${rect(548, 176, 310, 215, C.bg, 12, opacity, C.green, 1.3)}${text('CYCLIST TRACK', 703, 210, 14, C.green, 'middle', 700, opacity)}${text('last seen: t₀', 580, 252, 16, C.muted, 'start', 560, opacity)}${text('velocity: 4.2 m/s left', 580, 291, 16, C.ink, 'start', 600, opacity)}${text('predicted position: crosswalk', 580, 330, 16, C.ink, 'start', 600, opacity)}${text('uncertainty: growing', 580, 369, 16, C.amber, 'start', 650, opacity)}${footer('Recurrent instances carry the cyclist state forward and expand uncertainty while no pixels support it.', opacity * reveal(local, 0.62, 0.78), C.green)}`;
  if (mode === 'queue') return `${roadScene(76, 175, 340, 215, 0.38, opacity, { hideCyclist: true, showPrediction: true })}${arrow(441, 280, 520, 280, C.teal, 2, opacity * p)}${rect(548, 176, 310, 215, C.bg, 12, opacity, C.teal, 1.3)}${text('BOUNDED FOREGROUND QUEUE', 703, 210, 14, C.teal, 'middle', 700, opacity)}${text('1  cyclist · occluded', 580, 256, 16, C.green, 'start', 650, opacity)}${text('2  lead car · observed', 580, 298, 16, C.rose, 'start', 650, opacity)}${text('3  delivery van · static', 580, 340, 16, C.amber, 'start', 650, opacity)}${text('new low-score clutter dropped', 580, 376, 15, C.muted, 'start', 560, opacity)}${footer('A bounded queue keeps the important hidden cyclist while replacing lower-value foreground entries.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  if (mode === 'correct') return `${roadScene(160, 155, 640, 260, 0.82, opacity, { labels: true })}${ring(482, 226, 32 * (1 - 0.55 * p), C.green, opacity, 1.4)}${text('prediction corrected by current evidence', 480, 445, 16, C.green, 'middle', 650, opacity)}${footer('When the cyclist reappears, the new observation corrects position and contracts uncertainty.', opacity * reveal(local, 0.62, 0.78), C.green)}`;
  return `${temporalPanels(opacity, 1)}${footer('The track is useful because the t₂ observation corrects the t₁ prediction.', opacity, C.amber)}`;
}

function drawLidarContract(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return `${roadScene(180, 145, 600, 280, p, opacity, { sensor: 'all', labels: true })}${footer('The same cyclist scan can supervise training, feed inference, or belong only to a teacher.', opacity * reveal(local, 0.62, 0.78))}`;
  if (mode === 'labels') return `${forwardDrivingScene(70, 170, 340, 220, p, opacity, { boxActor: true })}${arrow(435, 280, 520, 280, C.amber, 2, opacity * p)}${rect(548, 178, 310, 198, C.bg, 12, opacity, C.amber, 1.3)}${text('TRAINING LABEL', 703, 212, 14, C.amber, 'middle', 700, opacity)}${text('cyclist depth = 24 m', 703, 260, 18, C.ink, 'middle', 650, opacity)}${text('camera predicts depth', 703, 310, 16, C.muted, 'middle', 560, opacity)}${text('LiDAR absent at runtime', 703, 348, 16, C.green, 'middle', 650, opacity)}${footer('LiDAR can label camera depth during training without becoming a deployed sensor dependency.', opacity * reveal(local, 0.62, 0.78), C.amber)}`;
  if (mode === 'runtime') return `${forwardDrivingScene(70, 170, 340, 220, p, opacity, { boxActor: true })}${roadScene(550, 170, 340, 220, p, opacity, { sensor: 'lidar' })}${arrow(410, 280, 534, 280, C.teal, 2, opacity * p)}${text('CAMERA', 240, 415, 15, C.blue, 'middle', 700, opacity)}${text('LiDAR AT INFERENCE', 720, 415, 15, C.amber, 'middle', 700, opacity)}${footer('If live LiDAR points enter the model, every deployed vehicle must supply calibrated LiDAR.', opacity * reveal(local, 0.62, 0.78), C.teal)}`;
  if (mode === 'teacher') return `${roadScene(70, 170, 340, 220, p, opacity, { sensor: 'all' })}${arrow(435, 280, 520, 280, C.green, 2, opacity * p)}${forwardDrivingScene(550, 170, 340, 220, p, opacity, { boxActor: true })}${text('CAMERA + LiDAR TEACHER', 240, 415, 15, C.amber, 'middle', 700, opacity)}${text('CAMERA-ONLY STUDENT', 720, 415, 15, C.green, 'middle', 700, opacity)}${footer('The teacher transfers cyclist geometry during training; the deployed student uses camera only.', opacity * reveal(local, 0.62, 0.78), C.green)}`;
  return `${roadScene(54, 166, 330, 230, 0.72, opacity, { sensor: 'lidar', labels: true })}${text('same cyclist scan', 219, 426, 14, C.green, 'middle', 650, opacity)}${arrow(406, 279, 472, 279, C.teal, 2, opacity)}${rect(500, 142, 380, 82, C.bg, 10, opacity, C.amber, 1.2)}${text('DEPTH LABEL', 522, 174, 14, C.amber, 'start', 700, opacity)}${text('dataset only', 522, 204, 15, C.ink, 'start', 600, opacity)}${text('runtime LiDAR: NO', 852, 204, 15, C.green, 'end', 700, opacity)}${rect(500, 242, 380, 82, C.bg, 10, opacity, C.red, 1.2)}${text('LIVE MODEL INPUT', 522, 274, 14, C.red, 'start', 700, opacity)}${text('deployed graph', 522, 304, 15, C.ink, 'start', 600, opacity)}${text('runtime LiDAR: YES', 852, 304, 15, C.red, 'end', 700, opacity)}${rect(500, 342, 380, 82, C.bg, 10, opacity, C.blue, 1.2)}${text('TEACHER SIGNAL', 522, 374, 14, C.blue, 'start', 700, opacity)}${text('training graph only', 522, 404, 15, C.ink, 'start', 600, opacity)}${text('runtime LiDAR: NO', 852, 404, 15, C.green, 'end', 700, opacity)}${footer('The same scan creates three different deployment contracts.', opacity, C.amber)}`;
}

function buildStory(titleSteps, draw) {
  const durations = titleSteps.map((step) => step.seconds ?? DEFAULT_STEP_SECONDS);
  const boundaries = durations.reduce((ends, duration) => {
    ends.push((ends.at(-1) ?? 0) + duration);
    return ends;
  }, []);
  const totalSeconds = boundaries.at(-1);

  return {
    steps: titleSteps,
    snapshot(index) {
      const step = titleSteps[index];
      if (!step) throw new Error(`No storyboard step ${index}`);
      const body = `${top(index + 1, titleSteps.length, step.title, 1)}${draw(step.mode, 1, 1)}`;
      return {
        svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${esc(step.description ?? step.title)}"><rect width="${WIDTH}" height="${HEIGHT}" fill="${C.bg}"/>${body}</svg>`,
        title: step.title,
        description: step.description ?? step.title,
      };
    },
    frame(frame) {
      const totalFrames = Math.round(totalSeconds * FPS);
      const t = Math.min(frame / FPS, totalSeconds - 1 / FPS);
      const index = Math.max(0, boundaries.findIndex((end) => t < end));
      const step = titleSteps[index];
      const start = index === 0 ? 0 : boundaries[index - 1];
      const duration = durations[index];
      const elapsed = t - start;
      const transitionSeconds = step.transitionSeconds ?? DEFAULT_TRANSITION_SECONDS;
      const buildSeconds = Math.min(step.buildSeconds ?? DEFAULT_BUILD_SECONDS, duration - transitionSeconds);
      const drawProgress = clamp(elapsed / buildSeconds);
      const opacity = ease(elapsed / transitionSeconds);
      const previousIndex = (index - 1 + titleSteps.length) % titleSteps.length;
      const previousStep = titleSteps[previousIndex];
      const previousOpacity = elapsed < transitionSeconds ? 1 - opacity : 0;
      const previousBody = previousOpacity > 0
        ? `${top(previousIndex + 1, titleSteps.length, previousStep.title, previousOpacity)}${draw(previousStep.mode, 1, previousOpacity)}`
        : '';
      const body = `${previousBody}${top(index + 1, titleSteps.length, step.title, opacity)}${draw(step.mode, drawProgress, opacity)}`;
      return {
        svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${esc(step.description ?? step.title)}"><rect width="${WIDTH}" height="${HEIGHT}" fill="${C.bg}"/>${body}</svg>`,
        totalFrames,
      };
    },
  };
}

export const CALM_BLOG_STORIES = {
  'blog-attention-memory.gif': buildStory([
    { title: 'How four attention variants store the same token sequence.', mode: 'intro', seconds: 6.5, buildSeconds: 2.4 },
    { title: 'MHA stores a full token history for every head.', mode: 'mha', seconds: 9 },
    { title: 'GQA shares fewer full histories across query heads.', mode: 'gqa', seconds: 9 },
    { title: 'MLA stores one compressed latent vector per token.', mode: 'mla', seconds: 9 },
    { title: 'DeltaNet updates one fixed-size recurrent state.', mode: 'delta', seconds: 8.5 },
    { title: 'Storage format sets cache size and retrieval limits.', mode: 'summary', seconds: 12 },
  ], drawAttention),
  'blog-vlm-evidence-contract.gif': buildStory([
    { title: 'Different outputs require different visual details.', mode: 'intro', seconds: 6.5, buildSeconds: 2.4 },
    { title: 'CLIP retains features for image-text matching.', mode: 'clip', seconds: 8 },
    { title: 'LLaVA retains features used to generate text.', mode: 'llava', seconds: 8.5 },
    { title: 'Molmo retains phrase-to-location coordinates.', mode: 'molmo', seconds: 9 },
    { title: 'π0 retains time-indexed state for action prediction.', mode: 'pi0', seconds: 9 },
    { title: 'The required output determines which details are retained.', mode: 'summary', seconds: 11.5 },
  ], drawVlm),
  'blog-multimodal-gradient-budget.gif': buildStory([
    { title: 'Equal sample percentages can produce unequal optimization load.', mode: 'intro', seconds: 7, buildSeconds: 2.6 },
    { title: 'First count sampled examples by modality.', mode: 'examples', seconds: 8 },
    { title: 'Examples expand into different numbers of units.', mode: 'units', seconds: 8.5 },
    { title: 'Those units consume different amounts of compute.', mode: 'flops', seconds: 9 },
    { title: 'Gradient norms show unequal updates to shared parameters.', mode: 'updates', seconds: 9 },
    { title: 'Report five ledgers, including independent decisions.', mode: 'ledgers', seconds: 11.5 },
  ], drawBudget),
  'blog-vla-feedback-attribution.gif': buildStory([
    { title: 'One failed rollout does not reveal a successful alternative.', mode: 'failure', seconds: 7, buildSeconds: 2.6 },
    { title: 'Episode outcomes label the whole trajectory.', mode: 'episode', seconds: 8 },
    { title: 'APO compares the failed action with the human correction.', mode: 'apo', seconds: 9 },
    { title: 'Process feedback requires progress or matched states.', mode: 'process', seconds: 9 },
    { title: 'Observed feedback limits which labels the loss can use.', mode: 'summary', seconds: 11 },
  ], drawFeedback),
  'blog-rl-learning-signals.gif': buildStory([
    { title: 'Four methods define their training comparisons differently.', mode: 'intro', seconds: 6.5, buildSeconds: 2.4 },
    { title: 'PPO compares sampled actions with value estimates.', mode: 'ppo', seconds: 8 },
    { title: 'DPO compares a fixed chosen and rejected response.', mode: 'dpo', seconds: 8 },
    { title: 'GRPO compares rewards within one sampled prompt group.', mode: 'grpo', seconds: 9 },
    { title: 'GKD matches teacher probabilities on student states.', mode: 'gkd', seconds: 9 },
    { title: 'Compare sampled data, feedback unit, and reference signal.', mode: 'summary', seconds: 11.5 },
  ], drawLearning),
  'blog-hermes-local-stack.gif': buildStory([
    { title: 'The agent shell and model runtime are separate.', mode: 'separate', seconds: 7, buildSeconds: 2.6 },
    { title: 'Hermes sends one request across a local API.', mode: 'request', seconds: 8 },
    { title: 'llama-server loads and runs the GGUF.', mode: 'weights', seconds: 8.5 },
    { title: 'llama-server returns generated tokens through the local API.', mode: 'return', seconds: 8 },
    { title: 'Map each failure to the agent, API, server, or model file.', mode: 'summary', seconds: 10.5 },
  ], drawHermes),
  'local-gemma-long-prompt-latency.gif': buildStory([
    { title: 'The benchmark changes one variable: prompt length.', mode: 'design', seconds: 7, buildSeconds: 2.6 },
    { title: 'Long prompts separate Gemma 4 models in prefill.', mode: 'long', seconds: 9.5 },
    { title: 'Runtime choice compounds long-prompt latency.', mode: 'runtime', seconds: 9.5 },
    { title: 'Decode speed moves less than first-token latency.', mode: 'decode', seconds: 9 },
    { title: 'Memory fit does not guarantee low first-token latency.', mode: 'decision', seconds: 11 },
  ], (mode, local, opacity) => drawBenchmark('gemma', mode, local, opacity)),
  'local-qwen-long-prompt-latency.gif': buildStory([
    { title: 'The benchmark changes one variable: prompt length.', mode: 'design', seconds: 7, buildSeconds: 2.6 },
    { title: 'Long prompts expose the Qwen prefill cost.', mode: 'long', seconds: 9.5 },
    { title: 'MLX and llama.cpp diverge as models grow.', mode: 'runtime', seconds: 9.5 },
    { title: 'Decode remains steadier than first-token latency.', mode: 'decode', seconds: 9 },
    { title: 'The 4B models have the lowest measured first-token latency.', mode: 'decision', seconds: 11 },
  ], (mode, local, opacity) => drawBenchmark('qwen', mode, local, opacity)),
  'autonomous-perception-camera-encoder.gif': buildStory([
    { title: 'The ego car closes on a cyclist emerging beside a parked van.', mode: 'input', seconds: 9, buildSeconds: 4.2 },
    { title: 'A coarse feature map can erase the distant cyclist.', mode: 'coarse', seconds: 9.5, buildSeconds: 4 },
    { title: 'Fine maps keep the cyclist; coarse maps keep road context.', mode: 'pyramid', seconds: 10, buildSeconds: 4 },
    { title: 'Image and BEV losses protect the same cyclist evidence.', mode: 'supervision', seconds: 10, buildSeconds: 4 },
    { title: 'Design the encoder around the smallest actor that must survive.', mode: 'summary', seconds: 12 },
  ], drawCameraEncoder),
  'autonomous-perception-lidar-encoder.gif': buildStory([
    { title: 'The ego car moves while one LiDAR sweep is being measured.', mode: 'input', seconds: 9.5, buildSeconds: 4.2 },
    { title: 'Compensation aligns the curb, not the independently moving cyclist.', mode: 'compensate', seconds: 10, buildSeconds: 4.2 },
    { title: 'Pillars keep x-y location but collapse height inside each cell.', mode: 'pillars', seconds: 9.5, buildSeconds: 4 },
    { title: 'Sparse voxels retain the cyclist height structure.', mode: 'voxels', seconds: 9.5, buildSeconds: 4 },
    { title: 'Alternating attention sets cross one sparse-window boundary.', mode: 'windows', seconds: 9.5, buildSeconds: 4 },
    { title: 'Preserve the geometry that separates cyclist, van, curb, and road.', mode: 'summary', seconds: 12.5 },
  ], drawLidar),
  'autonomous-perception-radar-encoder.gif': buildStory([
    { title: 'Three radar sweeps show the lead car closing from 34 m to 26 m.', mode: 'input', seconds: 11, buildSeconds: 5.5 },
    { title: 'Late radar fusion can confirm a camera proposal.', mode: 'proposal', seconds: 9.5, buildSeconds: 4 },
    { title: 'Earlier radar fusion can move the proposal to the measured range.', mode: 'depth', seconds: 10, buildSeconds: 4.2 },
    { title: 'Radar BEV retains moving and stationary returns separately.', mode: 'bev', seconds: 10, buildSeconds: 4.2 },
    { title: 'Fuse radar early if its range must change object geometry.', mode: 'summary', seconds: 12 },
  ], drawRadar),
  'autonomous-perception-camera-lifting.gif': buildStory([
    { title: 'The cyclist image patch could lie anywhere along one 3D ray.', mode: 'input', seconds: 9.5, buildSeconds: 4 },
    { title: 'LSS spreads the cyclist feature across predicted depth bins.', mode: 'lss', seconds: 10, buildSeconds: 4.2 },
    { title: 'An object query tests one candidate cyclist position.', mode: 'detr3d', seconds: 9.5, buildSeconds: 4 },
    { title: 'A BEV cell samples several heights against the camera image.', mode: 'bevformer', seconds: 10, buildSeconds: 4.2 },
    { title: 'Wrong depth writes cyclist evidence into the wrong BEV cell.', mode: 'summary', seconds: 12 },
  ], drawLifting),
  'autonomous-perception-fusion-granularity.gif': buildStory([
    { title: 'Three sensors describe the same cyclist differently.', mode: 'input', seconds: 9, buildSeconds: 4 },
    { title: 'Point fusion keeps measurements at the cyclist location.', mode: 'point', seconds: 9.5, buildSeconds: 4 },
    { title: 'An object query keeps one compact cyclist state.', mode: 'query', seconds: 9.5, buildSeconds: 4 },
    { title: 'Dense BEV also keeps lane, crosswalk, and free space.', mode: 'bev', seconds: 10, buildSeconds: 4.2 },
    { title: 'Choose the fusion unit from what prediction must inspect.', mode: 'summary', seconds: 12 },
  ], drawFusion),
  'autonomous-perception-modality-dropout.gif': buildStory([
    { title: 'Rain degrades the camera before the camera disappears.', mode: 'failure', seconds: 10, buildSeconds: 4.5 },
    { title: 'Availability records which sensor streams are present.', mode: 'unibev', seconds: 9.5, buildSeconds: 4 },
    { title: 'Training across sensor sets preserves one cyclist state.', mode: 'metabev', seconds: 10, buildSeconds: 4.2 },
    { title: 'Reliability weights reduce the rain-damaged camera input.', mode: 'grace', seconds: 10, buildSeconds: 4.2 },
    { title: 'Missing input and degraded input are different failures.', mode: 'summary', seconds: 12 },
  ], drawDropout),
  'autonomous-perception-temporal-memory.gif': buildStory([
    { title: 'The cyclist is observed, occluded by the van, then observed again.', mode: 'input', seconds: 11, buildSeconds: 5.5 },
    { title: 'A warped field carries the cyclist region after ego motion.', mode: 'warp', seconds: 10, buildSeconds: 4.2 },
    { title: 'A recurrent track predicts the hidden cyclist and grows uncertainty.', mode: 'instances', seconds: 10.5, buildSeconds: 4.2 },
    { title: 'A bounded queue retains the hidden cyclist over weaker clutter.', mode: 'queue', seconds: 10, buildSeconds: 4.2 },
    { title: 'Fresh pixels correct the cyclist track when it reappears.', mode: 'correct', seconds: 10, buildSeconds: 4.2 },
    { title: 'Observe, predict through occlusion, then correct with evidence.', mode: 'summary', seconds: 12.5 },
  ], drawTemporal),
  'autonomous-perception-lidar-training-contracts.gif': buildStory([
    { title: 'One cyclist scan can give LiDAR three different roles.', mode: 'input', seconds: 9, buildSeconds: 4 },
    { title: 'LiDAR can label cyclist depth during camera training only.', mode: 'labels', seconds: 10, buildSeconds: 4.2 },
    { title: 'Live LiDAR input creates a deployed sensor dependency.', mode: 'runtime', seconds: 10, buildSeconds: 4.2 },
    { title: 'A LiDAR teacher can train a camera-only deployed student.', mode: 'teacher', seconds: 10, buildSeconds: 4.2 },
    { title: 'Locate the LiDAR tensor in dataset, deployed graph, or teacher.', mode: 'summary', seconds: 12 },
  ], drawLidarContract),
};

export async function renderBlogExplainerFrames(names = CALM_BLOG_GIFS, options = {}) {
  const root = options.root ?? process.cwd();
  const outputDir = options.outputDir ?? join(root, 'public/assets/images/blog-explainer-frames');
  const manifest = {};
  mkdirSync(outputDir, { recursive: true });

  for (const name of names) {
    const story = CALM_BLOG_STORIES[name];
    if (!story) throw new Error(`No Blog explainer storyboard for ${name}`);
    const storyDirName = name.replace(/\.gif$/, '');
    const storyDir = join(outputDir, storyDirName);
    rmSync(storyDir, { recursive: true, force: true });
    mkdirSync(storyDir, { recursive: true });
    const frames = [];

    for (let index = 0; index < story.steps.length; index++) {
      const snapshot = story.snapshot(index);
      const filename = `frame-${String(index + 1).padStart(2, '0')}.webp`;
      await sharp(Buffer.from(snapshot.svg))
        .webp({ quality: 92, smartSubsample: true })
        .toFile(join(storyDir, filename));
      frames.push({
        src: `/assets/images/blog-explainer-frames/${storyDirName}/${filename}`,
        title: snapshot.title,
        description: snapshot.description,
      });
    }

    manifest[name] = { frames };
    console.log(`generated ${frames.length} explainer frames for ${name}`);
  }

  writeFileSync(join(outputDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

export async function renderCalmBlogGifs(names, options = {}) {
  const root = options.root ?? process.cwd();
  const outputDir = options.outputDir ?? join(root, 'public/assets/images');
  const scratchRoot = options.scratchRoot ?? join(root, '.tmp-calm-blog-gifs');
  mkdirSync(outputDir, { recursive: true });
  mkdirSync(scratchRoot, { recursive: true });

  for (const name of names) {
    const story = CALM_BLOG_STORIES[name];
    if (!story) throw new Error(`No calm Blog GIF storyboard for ${name}`);
    const { totalFrames } = story.frame(0);
    const frameDir = join(scratchRoot, name.replace(/\.gif$/, ''));
    rmSync(frameDir, { recursive: true, force: true });
    mkdirSync(frameDir, { recursive: true });
    for (let frame = 0; frame < totalFrames; frame++) {
      const { svg } = story.frame(frame);
      await sharp(Buffer.from(svg)).png().toFile(join(frameDir, `frame-${String(frame).padStart(4, '0')}.png`));
    }
    const result = spawnSync('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-framerate', String(FPS),
      '-i', join(frameDir, 'frame-%04d.png'),
      '-filter_complex', '[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
      '-loop', '0',
      join(outputDir, name),
    ], { stdio: 'inherit' });
    if (result.status !== 0) throw new Error(`ffmpeg failed for ${name}`);
    rmSync(frameDir, { recursive: true, force: true });
    console.log(`generated ${join(outputDir, name)}`);
  }

  rmSync(scratchRoot, { recursive: true, force: true });
}
