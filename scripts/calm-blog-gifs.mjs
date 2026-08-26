import { mkdirSync, rmSync } from 'node:fs';
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

function sceneAlpha(elapsed, duration, transitionSeconds) {
  return ease(elapsed / transitionSeconds) * ease((duration - elapsed) / transitionSeconds);
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

function pointCloud(x, y, w, h, opacity = 1, color = C.amber) {
  let out = rect(x, y, w, h, C.bg, 12, opacity, C.faint, 1);
  for (let i = 0; i < 72; i++) {
    const px = x + 16 + ((i * 43) % (w - 32));
    const py = y + 16 + ((i * 29 + Math.floor(i / 7) * 17) % (h - 32));
    out += circle(px, py, i % 11 === 0 ? 3 : 1.8, color, opacity * (i % 5 === 0 ? 0.9 : 0.5));
  }
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
  return summary([
    ['MHA', 'full history per head · largest cache'],
    ['GQA', 'fewer shared histories · fewer K/V maps'],
    ['MLA', 'compressed latent per token · reconstruction bottleneck'],
    ['DeltaNet', 'one recurrent state · bounded, lossy memory'],
  ], opacity, 'Storage format sets cache size and which past details can be retrieved.');
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
  return summary([
    ['CLIP', 'features for image-text matching'],
    ['LLaVA', 'features used to generate text'],
    ['Molmo', 'phrase-to-location coordinates'],
    ['π0', 'time-indexed state for action prediction'],
  ], opacity, 'Different outputs require different visual details.');
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
  ], opacity, 'Report all four quantities; sample percentage alone is incomplete.');
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
  return summary([
    ['Outcome', 'labels the whole trajectory'],
    ['APO', 'uses the local intervention and correction'],
    ['Process', 'needs progress or matched states'],
  ], opacity, 'The loss cannot supply an alternative action that was never observed.');
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
  return summary([
    ['PPO', 'current rollout + learned baseline'],
    ['DPO', 'fixed chosen and rejected pair'],
    ['GRPO', 'current-policy prompt group'],
    ['GKD', 'teacher distribution at student states'],
  ], opacity, 'Compare methods by sampled data, feedback unit, and reference signal.');
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
  return summary([
    ['Agent', 'tools, sessions, and skills'],
    ['API', 'stable localhost contract'],
    ['Server', 'sampling and KV cache'],
    ['GGUF', 'weights and tokenizer'],
  ], opacity, 'A model-load error originates in llama-server or the GGUF file, not Hermes.');
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

function featurePyramid(opacity, p) {
  let out = '';
  [[310, 185, 340, 180], [350, 215, 260, 125], [390, 245, 180, 72]].forEach(([x, y, w, h], i) => {
    out += rect(x, y, w, h, C.bg, 10, opacity * (0.42 + i * 0.18) * p, C.teal, 1.3);
    const cells = 4 + i * 2;
    for (let c = 1; c < cells; c++) out += line(x + w * c / cells, y, x + w * c / cells, y + h, C.faint, 1, opacity * p);
  });
  return out;
}

function drawCameraEncoder(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return cameraView(260, 150, 440, 270, opacity) + footer('The encoder must retain small distant actors and wide scene context.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'coarse') return `${cameraView(110, 175, 300, 210, opacity)}${arrow(432, 280, 542, 280, C.red, 2, opacity * p)}${grid(590, 205, 5, 4, 38, C.red, p, opacity)}${footer('One coarse map is cheap, but small actors can disappear.', opacity * reveal(local, 0.60, 0.76), C.red)}`;
  if (mode === 'pyramid') return featurePyramid(opacity, p) + footer('A feature pyramid carries precise detail and broad context together.', opacity * reveal(local, 0.60, 0.76), C.teal);
  if (mode === 'supervision') return `${cameraView(120, 175, 300, 210, opacity)}${arrow(444, 280, 542, 280, C.green, 2, opacity * p)}${bev(590, 190, 240, 170, opacity * p, C.green)}${text('2D + 3D loss', 710, 390, 16, C.green, 'middle', 620, opacity * p)}${footer('Perspective-view supervision protects features needed after BEV lifting.', opacity * reveal(local, 0.60, 0.76), C.green)}`;
  return summary([['Coarse map', 'lowest cost · weakest small-object detail'], ['Feature pyramid', 'detail plus scene context'], ['Perspective loss', 'supervises image features before BEV projection']], opacity, 'Resolution, pyramid levels, and auxiliary loss set compute and retained detail.');
}

function drawLidar(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return pointCloud(220, 145, 520, 270, opacity, C.amber) + footer('LiDAR begins as sparse points in continuous 3D space.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'pillars') return `${pointCloud(96, 175, 300, 200, opacity, C.amber)}${arrow(420, 275, 538, 275, C.amber, 2, opacity * p)}${bev(585, 185, 245, 180, opacity * p, C.amber)}${footer('PointPillars collapses height early into a dense BEV grid.', opacity * reveal(local, 0.60, 0.76), C.amber)}`;
  if (mode === 'voxels') return `${pointCloud(96, 175, 300, 200, opacity, C.amber)}${arrow(420, 275, 538, 275, C.blue, 2, opacity * p)}${grid(585, 180, 7, 5, 35, C.blue, 0.58 * p, opacity)}${footer('SECOND keeps occupied voxels sparse before producing BEV features.', opacity * reveal(local, 0.60, 0.76), C.blue)}`;
  if (mode === 'dsvt') return `${grid(205, 165, 11, 7, 42, C.blue, 0.58, opacity)}${rect(248, 208, 168, 126, C.bg, 8, opacity * p, C.teal, 2)}${rect(458, 208, 168, 126, C.bg, 8, opacity * p, C.teal, 2)}${footer('DSVT attends inside sparse windows without densifying the full volume.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  if (mode === 'voxelnext') return `${grid(255, 165, 11, 7, 42, C.green, 0.46, opacity)}${circle(413, 250, 17, C.amber, opacity * p, C.amber, 1)}${circle(623, 292, 17, C.amber, opacity * p, C.amber, 1)}${footer('VoxelNeXt keeps detection sparse instead of adding a dense BEV head.', opacity * reveal(local, 0.60, 0.76), C.green)}`;
  return summary([['PointPillars', 'collapse height early'], ['SECOND', 'sparse voxels, then dense BEV'], ['DSVT', 'attention inside sparse windows'], ['VoxelNeXt', 'sparse backbone and sparse head']], opacity, 'Each encoder converts sparse features to dense features at a different stage.');
}

function radarReturns(opacity) {
  let out = bev(248, 150, 464, 270, opacity, C.rose);
  [[320, 210], [391, 328], [527, 230], [638, 350], [588, 188]].forEach(([x, y], i) => { out += circle(x, y, i % 2 ? 5 : 8, C.rose, opacity, C.rose, 1); out += line(x - 20, y, x + 20, y, C.rose, 1, opacity * 0.45); });
  return out;
}

function drawRadar(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return radarReturns(opacity) + footer('Radar is sparse, noisy, and useful for range and radial velocity.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'proposal') return `${cameraView(105, 178, 300, 205, opacity)}${radarReturns(opacity * 0.55)}${arrow(702, 286, 821, 286, C.rose, 2, opacity * p)}${rect(806, 240, 78, 78, C.bg, 8, opacity * p, C.amber, 1.5)}${footer('Proposal-stage fusion lets radar confirm camera candidates.', opacity * reveal(local, 0.60, 0.76), C.rose)}`;
  if (mode === 'depth') return `${cameraView(104, 178, 300, 205, opacity)}${arrow(428, 282, 544, 282, C.rose, 2, opacity * p)}${line(585, 350, 790, 191, C.amber, 3, opacity)}${[0,1,2,3,4].map((i)=>circle(607+i*42,333-i*32,6,i===3?C.rose:C.faint,opacity*(i===3?p:0.7))).join('')}${footer('Depth-stage fusion uses radar to sharpen where image evidence lands.', opacity * reveal(local, 0.60, 0.76), C.amber)}`;
  if (mode === 'bev') return `${radarReturns(opacity)}${arrow(480, 425, 480, 455, C.teal, 2, opacity * p)}${footer('An independent radar BEV preserves radar evidence before fusion.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  return summary([['Proposal', 'confirm camera candidates'], ['Depth', 'change estimated object depth'], ['Independent BEV', 'retain radar range and radial velocity']], opacity, 'Fusion stage determines whether radar can move an estimate or only confirm it.');
}

function cameraRay(opacity) {
  return `${cameraView(90, 175, 280, 205, opacity)}${line(370, 278, 850, 210, C.amber, 2, opacity)}${circle(754, 224, 8, C.amber, opacity)}`;
}

function drawLifting(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return cameraRay(opacity) + footer('A pixel ray does not reveal one unique 3D point.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'lss') { let out = cameraRay(opacity); for (let i=0;i<7;i++) out += circle(472+i*48,264-i*7,7,i===4?C.teal:C.faint,opacity*(i===4?p:0.8)); return out + footer('LSS predicts a depth distribution along each image ray.', opacity * reveal(local, 0.60, 0.76), C.teal); }
  if (mode === 'detr3d') return `${cameraView(90, 175, 280, 205, opacity)}${circle(670, 250, 18, C.blue, 0.08 * opacity, C.blue, 1.5)}${line(370, 278, 670, 250, C.blue, 2, opacity * p)}${text('3D reference point', 670, 302, 16, C.blue, 'middle', 620, opacity)}${footer('DETR3D projects one object query into the camera views.', opacity * reveal(local, 0.60, 0.76), C.blue)}`;
  if (mode === 'bevformer') { let out = `${cameraView(90, 175, 280, 205, opacity)}${bev(590, 175, 250, 205, opacity, C.teal)}`; for(let i=0;i<5;i++){out += circle(680,326-i*35,6,C.teal,opacity*p); out += line(370,278,680,326-i*35,C.teal,1,opacity*p);} return out + footer('BEVFormer samples vertical reference points from each BEV cell.', opacity * reveal(local, 0.60, 0.76), C.teal); }
  return summary([['LSS', 'pixel ray + depth distribution'], ['DETR3D', 'object query + 3D reference point'], ['BEVFormer', 'BEV cell + vertical reference points']], opacity, 'Each method chooses different 3D sample locations for image features.');
}

function actorEvidence(opacity) {
  return `${circle(266, 230, 12, C.blue, opacity, C.blue, 1)}${text('camera', 266, 270, 14, C.blue, 'middle', 600, opacity)}${circle(480, 230, 12, C.amber, opacity, C.amber, 1)}${text('LiDAR', 480, 270, 14, C.amber, 'middle', 600, opacity)}${circle(694, 230, 12, C.rose, opacity, C.rose, 1)}${text('radar', 694, 270, 14, C.rose, 'middle', 600, opacity)}`;
}

function drawFusion(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return actorEvidence(opacity) + text('one actor · camera, LiDAR, and radar features', 480, 352, 19, C.ink, 'middle', 620, opacity * p);
  if (mode === 'point') return `${actorEvidence(opacity)}${[266,480,694].map((x)=>arrow(x,285,480,372,x===266?C.blue:x===480?C.amber:C.rose,1.7,opacity*p)).join('')}${circle(480,385,22,C.teal,0.12*opacity,C.teal,1.5)}${footer('Point fusion aligns evidence at explicit 3D samples.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  if (mode === 'query') return `${actorEvidence(opacity)}${[266,480,694].map((x)=>arrow(x,285,480,356,x===266?C.blue:x===480?C.amber:C.rose,1.7,opacity*p)).join('')}${token('object query', 414, 368, opacity, C.green, 132)}${footer('Object-query fusion asks each modality about one candidate actor.', opacity * reveal(local, 0.60, 0.76), C.green)}`;
  if (mode === 'bev') return `${actorEvidence(opacity)}${[266,480,694].map((x)=>arrow(x,285,480,325,x===266?C.blue:x===480?C.amber:C.rose,1.7,opacity*p)).join('')}${bev(342,335,276,110,opacity*p,C.teal)}${footer('Dense-BEV fusion keeps a shared spatial field for downstream tasks.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  return summary([['Point', 'combine features at one 3D sample'], ['Object query', 'combine features for one candidate actor'], ['Dense BEV', 'combine features in every spatial cell']], opacity, 'Fusion unit determines where sensor disagreements can be detected.');
}

function modalityChips(opacity, levels = [1, 1, 1]) {
  const names = [['camera', C.blue], ['LiDAR', C.amber], ['radar', C.rose]];
  return names.map(([name, color], i) => `${rect(250 + i * 170, 190, 130, 52, C.bg, 11, opacity * (0.18 + 0.82 * levels[i]), color, 1.3)}${text(name, 315 + i * 170, 222, 15, levels[i] > 0.3 ? C.ink : C.muted, 'middle', 600, opacity)}`).join('');
}

function drawDropout(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'failure') return modalityChips(opacity, [1, 0.08, 1]) + line(420, 176, 536, 256, C.red, 4, opacity * p) + line(536, 176, 420, 256, C.red, 4, opacity * p) + footer('A missing sensor has no input; a corrupted sensor supplies incorrect input.', opacity * reveal(local, 0.58, 0.74), C.red);
  if (mode === 'unibev') return `${modalityChips(opacity, [1, 0, 1])}${text('availability mask', 480, 315, 18, C.teal, 'middle', 620, opacity * p)}${footer('UniBEV trains the shared representation under modality dropout.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  if (mode === 'metabev') return `${modalityChips(opacity, [1, 0.2, 1])}${arrow(315,258,480,355,C.blue,1.7,opacity*p)}${arrow(485,258,480,355,C.amber,1.7,opacity*p)}${arrow(655,258,480,355,C.rose,1.7,opacity*p)}${circle(480,375,25,C.green,0.12*opacity,C.green,1.5)}${footer('MetaBEV learns one task representation across available sensor sets.', opacity * reveal(local, 0.60, 0.76), C.green)}`;
  if (mode === 'grace') return `${modalityChips(opacity, [1, 0.35 + 0.65 * p, 0.5])}${text('1.00',315,280,15,C.blue,'middle',650,opacity)}${text((0.35+0.65*p).toFixed(2),485,280,15,C.amber,'middle',650,opacity)}${text('0.50',655,280,15,C.rose,'middle',650,opacity)}${footer('Grace-BEV gates each modality by estimated reliability.', opacity * reveal(local, 0.60, 0.76), C.amber)}`;
  return summary([['Availability', 'which sensors supplied input'], ['Reliability', 'weight assigned to each supplied input'], ['Fused task state', 'representation passed to prediction heads']], opacity, 'Track sensor availability separately from estimated sensor reliability.');
}

function framesRow(opacity, count = 4) {
  let out = '';
  for (let i = 0; i < count; i++) {
    out += bev(112 + i * 188, 175, 154, 170, opacity * (0.45 + 0.55 * i / Math.max(1, count - 1)), C.teal);
    out += text(`t${i ? `-${count - 1 - i}` : `-${count - 1}`}`, 189 + i * 188, 374, 13, C.muted, 'middle', 540, opacity);
  }
  return out;
}

function drawTemporal(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return framesRow(opacity, 4) + footer('Temporal memory must align past features after ego motion and retain occluded actors.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'warp') return `${bev(150,175,260,190,opacity,C.blue)}${arrow(432,270,542,270,C.blue,2,opacity*p)}${bev(570,175,260,190,opacity*p,C.teal)}${path('M 608 320 Q 700 232 796 206',C.amber,3,opacity*p)}${footer('BEVFormer warps the previous dense field into the new ego frame.', opacity * reveal(local, 0.60, 0.76), C.blue)}`;
  if (mode === 'instances') { let out = bev(210,160,540,220,opacity*0.35,C.faint); [[320,225],[480,298],[650,210]].forEach(([x,y],i)=>{out += circle(x,y,16,C.green,0.08*opacity,C.green,1.5); out += arrow(x-35,y+25,x,y,C.green,1.5,opacity*p); out += text(`q${i+1}`,x,y+5,13,C.green,'middle',650,opacity);}); return out + footer('StreamPETR transforms recurrent instances and adds fresh anchors.', opacity * reveal(local, 0.60, 0.76), C.green); }
  if (mode === 'queue') { let out = ''; for(let i=0;i<6;i++){out += token(`q${i+1}`,165+i*108,225,opacity*(i<4?1:0.35),i<4?C.teal:C.faint,82);} out += arrow(180,304,752,304,C.teal,2,opacity*p); return out + footer('Sparse4D keeps a bounded queue of foreground queries.', opacity * reveal(local, 0.60, 0.76), C.teal); }
  return summary([['Warped field', 'dense spatial history'], ['Recurrent instances', 'tracked actor state plus new anchors'], ['Bounded queue', 'fixed number of foreground queries']], opacity, 'Dense maps, actor instances, and bounded queues retain different history.');
}

function drawLidarContract(mode, local, opacity) {
  const p = reveal(local);
  if (mode === 'input') return pointCloud(220,145,520,270,opacity,C.amber) + footer('The same LiDAR scan can play three different training roles.', opacity * reveal(local, 0.58, 0.74));
  if (mode === 'labels') return `${cameraView(110,175,300,205,opacity)}${arrow(432,278,548,278,C.amber,2,opacity*p)}${grid(590,190,7,5,34,C.amber,p,opacity)}${text('depth labels',710,397,16,C.amber,'middle',620,opacity)}${footer('LiDAR can supervise depth while camera remains the runtime input.', opacity * reveal(local, 0.60, 0.76), C.amber)}`;
  if (mode === 'runtime') return `${cameraView(95,175,280,205,opacity)}${pointCloud(595,175,270,205,opacity,C.amber)}${arrow(390,278,558,278,C.teal,2,opacity*p)}${text('both streams at inference',480,413,16,C.teal,'middle',620,opacity)}${footer('Runtime sparse-depth input makes LiDAR part of the deployed contract.', opacity * reveal(local, 0.60, 0.76), C.teal)}`;
  if (mode === 'teacher') return `${pointCloud(105,175,280,205,opacity,C.amber)}${text('teacher',245,412,15,C.amber,'middle',620,opacity)}${arrow(408,278,548,278,C.green,2,opacity*p)}${cameraView(575,175,280,205,opacity)}${text('camera student',715,412,15,C.green,'middle',620,opacity)}${footer('A LiDAR-camera teacher can disappear after distillation.', opacity * reveal(local, 0.60, 0.76), C.green)}`;
  return summary([['Labels', 'training-only geometric target'], ['Runtime input', 'deployed sensor dependency'], ['Teacher', 'training path that can be removed']], opacity, 'Training with LiDAR does not always mean requiring LiDAR at inference.');
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
      const opacity = sceneAlpha(elapsed, duration, transitionSeconds);
      const body = `${top(index + 1, titleSteps.length, step.title, opacity)}${draw(step.mode, drawProgress, opacity)}`;
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
    { title: 'Report examples, units, FLOPs, and gradient norms.', mode: 'ledgers', seconds: 11.5 },
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
    { title: 'Camera resolution determines which scene details are retained.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'One coarse map can erase small actors.', mode: 'coarse', seconds: 8 },
    { title: 'A feature pyramid carries detail and context.', mode: 'pyramid', seconds: 8.5 },
    { title: 'Perspective losses supervise features before BEV projection.', mode: 'supervision', seconds: 9 },
    { title: 'Resolution and pyramid levels trade compute for retained detail.', mode: 'summary', seconds: 11 },
  ], drawCameraEncoder),
  'autonomous-perception-lidar-encoder.gif': buildStory([
    { title: 'LiDAR begins as sparse points in continuous 3D.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'PointPillars collapses height early.', mode: 'pillars', seconds: 8 },
    { title: 'SECOND keeps occupied voxels sparse first.', mode: 'voxels', seconds: 8.5 },
    { title: 'DSVT attends inside sparse windows.', mode: 'dsvt', seconds: 8.5 },
    { title: 'VoxelNeXt keeps the detection head sparse.', mode: 'voxelnext', seconds: 8.5 },
    { title: 'Each LiDAR encoder discards sparsity at a different stage.', mode: 'summary', seconds: 11.5 },
  ], drawLidar),
  'autonomous-perception-radar-encoder.gif': buildStory([
    { title: 'Radar supplies sparse range and radial-velocity measurements.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'Proposal fusion confirms camera candidates.', mode: 'proposal', seconds: 8 },
    { title: 'Depth fusion sharpens geometric placement.', mode: 'depth', seconds: 8.5 },
    { title: 'A separate radar BEV retains range and radial velocity.', mode: 'bev', seconds: 9 },
    { title: 'Fusion stage limits whether radar can move or confirm estimates.', mode: 'summary', seconds: 11 },
  ], drawRadar),
  'autonomous-perception-camera-lifting.gif': buildStory([
    { title: 'A camera pixel defines a ray, not a 3D point.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'LSS distributes evidence along depth bins.', mode: 'lss', seconds: 8.5 },
    { title: 'DETR3D projects an object reference point.', mode: 'detr3d', seconds: 8.5 },
    { title: 'BEVFormer samples vertical references from BEV.', mode: 'bevformer', seconds: 9 },
    { title: 'Each method chooses different 3D sample locations.', mode: 'summary', seconds: 11 },
  ], drawLifting),
  'autonomous-perception-fusion-granularity.gif': buildStory([
    { title: 'Compare how one actor is fused across three sensor streams.', mode: 'input', seconds: 6.5, buildSeconds: 2.4 },
    { title: 'Point fusion aligns evidence at 3D samples.', mode: 'point', seconds: 8 },
    { title: 'Object queries gather candidate-centered evidence.', mode: 'query', seconds: 8.5 },
    { title: 'Dense BEV fusion keeps a shared spatial field.', mode: 'bev', seconds: 8.5 },
    { title: 'Fusion unit determines where sensor disagreement is detected.', mode: 'summary', seconds: 11 },
  ], drawFusion),
  'autonomous-perception-modality-dropout.gif': buildStory([
    { title: 'Missing and corrupted sensors require different handling.', mode: 'failure', seconds: 7, buildSeconds: 2.6 },
    { title: 'UniBEV trains with explicit modality dropout.', mode: 'unibev', seconds: 8 },
    { title: 'MetaBEV learns one state across sensor sets.', mode: 'metabev', seconds: 8.5 },
    { title: 'Grace-BEV gates present sensors by reliability.', mode: 'grace', seconds: 9 },
    { title: 'Track sensor availability separately from sensor reliability.', mode: 'summary', seconds: 11 },
  ], drawDropout),
  'autonomous-perception-temporal-memory.gif': buildStory([
    { title: 'Temporal memory must align motion and retain occluded actors.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'BEVFormer warps a dense field into the new frame.', mode: 'warp', seconds: 9 },
    { title: 'StreamPETR carries recurrent actor instances.', mode: 'instances', seconds: 9 },
    { title: 'Sparse4D keeps a bounded foreground-query queue.', mode: 'queue', seconds: 9 },
    { title: 'Dense maps, actor instances, and queues retain different history.', mode: 'summary', seconds: 11 },
  ], drawTemporal),
  'autonomous-perception-lidar-training-contracts.gif': buildStory([
    { title: 'One LiDAR scan can play three training roles.', mode: 'input', seconds: 7, buildSeconds: 2.6 },
    { title: 'LiDAR may create depth labels used only during training.', mode: 'labels', seconds: 8 },
    { title: 'Runtime sparse depth becomes a sensor dependency.', mode: 'runtime', seconds: 9 },
    { title: 'A LiDAR teacher can supervise a camera-only deployed model.', mode: 'teacher', seconds: 9 },
    { title: 'Training input is not always deployment input.', mode: 'summary', seconds: 11 },
  ], drawLidarContract),
};

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
