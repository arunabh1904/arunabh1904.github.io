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
const SCRATCH = join(ROOT, '.tmp-distributed-training-gifs');

const C = {
  bg: '#071019',
  panel: '#101f2d',
  grid: '#294052',
  ink: '#f2f7f8',
  muted: '#91a8b7',
  cyan: '#59d8ff',
  amber: '#ffc565',
  pink: '#ff75ae',
  green: '#76e0a4',
  violet: '#b49dff',
  red: '#ff7b76',
};

const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
const mix = (a, b, t) => a + (b - a) * t;
const ease = (value) => {
  const t = clamp(value);
  return t * t * (3 - 2 * t);
};
const phase = (t, start, end) => ease((t - start) / (end - start));
const pulse = (t) => 0.5 - 0.5 * Math.cos(t * Math.PI * 2);
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

function glow(x, y, r, color, opacity = 1) {
  return `${circle(x, y, r, color, opacity * 0.13)}${circle(x, y, Math.max(3, r * 0.27), color, opacity)}`;
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

function rankCard(x, y, rank, color, value, opacity = 1) {
  return `${rect(x, y, 138, 78, color, 13, 0.045 + 0.045 * opacity, color, 1)}${label(`RANK ${rank}`, x + 15, y + 24, 10, color, 'start', 700, opacity, 1)}${label(value, x + 69, y + 56, 19, C.ink, 'middle', 650, opacity)}`;
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

function collectiveOrderFrame(frame) {
  const t = cycle(frame);
  const first = phase(t, 0.05, 0.34);
  const wait = phase(t, 0.34, 0.67);
  const reset = 1 - phase(t, 0.86, 0.98);
  const blink = 0.55 + 0.45 * pulse(t * 3);
  let body = header('01', 'PROCESS GROUP · COLLECTIVE ORDER', 'Ranks must enter the same collective sequence');
  body += panel(42, 122, 876, 318, 'ONE TRAINING STEP · DIVERGENT CONTROL FLOW', C.red);

  const rows = [
    { y: 180, rank: 0, first: 'all_reduce(A)', second: 'broadcast(B)', color: C.cyan },
    { y: 265, rank: 1, first: 'broadcast(B)', second: 'all_reduce(A)', color: C.amber },
  ];
  rows.forEach((row) => {
    body += rankCard(72, row.y, row.rank, row.color, `device ${row.rank}`);
    body += arrow(222, row.y + 39, 304, row.y + 39, row.color, 1.5, 0.5 + 0.4 * first);
    body += rect(316, row.y + 7, 204, 64, row.color, 11, 0.04 + 0.08 * first, row.color, 1);
    body += label(row.first, 418, row.y + 45, 14, C.ink, 'middle', 620);
    body += arrow(532, row.y + 39, 606, row.y + 39, C.grid, 1.5, 0.3);
    body += rect(618, row.y + 7, 214, 64, C.grid, 11, 0.035, C.grid, 1);
    body += label(row.second, 725, row.y + 45, 14, C.muted, 'middle', 600, 0.72);
    body += flow(222, row.y + 39, 504, row.y + 39, first, row.color, reset);
  });

  body += line(418, 235, 418, 258, C.red, 2, 0.3 + 0.65 * wait, '5 5');
  body += circle(418, 246, 17, C.red, 0.04 + 0.08 * wait * blink, C.red, 1.2);
  body += label('≠', 418, 253, 20, C.red, 'middle', 700, wait * blink);
  body += rect(316, 367, 516, 42, C.red, 18, 0.04 + 0.08 * wait, C.red, 1);
  body += label(wait > 0.5 ? 'WAITING: peers entered incompatible collectives' : 'compare the next collective on every rank', 574, 394, 12, wait > 0.5 ? C.red : C.muted, 'middle', 620);
  body += label('A hang can be control-flow divergence, not a failed network.', 480, 494, 13, C.green, 'middle', 620);
  return svg(body, 'Rank zero enters all-reduce while rank one enters broadcast. Both ranks wait because collective calls occur in an incompatible order, illustrating a distributed hang caused by divergent control flow.');
}

function ddpFrame(frame) {
  const t = cycle(frame);
  const backward = phase(t, 0.04, 0.30);
  const reduce = phase(t, 0.28, 0.62);
  const update = phase(t, 0.60, 0.86);
  const gradients = ['g₀', 'g₁', 'g₂', 'g₃'];
  const colors = [C.cyan, C.amber, C.pink, C.violet];
  const xs = [56, 278, 500, 722];
  let body = header('02', 'DISTRIBUTED DATA PARALLEL', 'Local gradients become one shared update');
  body += panel(34, 122, 892, 333, 'BACKWARD · BUCKET ALL-REDUCE · OPTIMIZER STEP', C.cyan);

  xs.forEach((x, rank) => {
    const y = 168;
    body += rect(x, y, 182, 124, colors[rank], 14, 0.035 + 0.04 * backward, colors[rank], 1);
    body += label(`RANK ${rank}`, x + 16, y + 25, 10, colors[rank], 'start', 700, 1, 1);
    body += label(`batch ${rank}`, x + 91, y + 55, 12, C.muted, 'middle', 550);
    body += arrow(x + 91, y + 66, x + 91, y + 91, colors[rank], 1.4, 0.35 + 0.55 * backward);
    body += rect(x + 51, y + 84, 80, 26, colors[rank], 6, 0.06 + 0.12 * backward, colors[rank], 0.8);
    body += label(gradients[rank], x + 91, y + 102, 13, C.ink, 'middle', 650);
    body += flow(x + 91, y + 110, 480, 349, reduce, colors[rank], 0.35 + 0.55 * reduce);
  });

  body += rect(387, 326, 186, 54, C.green, 14, 0.04 + 0.11 * reduce, C.green, 1.2);
  body += label('ALL-REDUCE', 480, 348, 10, C.green, 'middle', 700, 1, 1.2);
  body += label('ḡ = (g₀ + g₁ + g₂ + g₃) / 4', 480, 368, 12, C.ink, 'middle', 620);
  xs.forEach((x) => {
    const targetX = x + 91;
    body += flow(480, 380, targetX, 414, update, C.green, 0.35 + 0.55 * update);
    body += rect(x + 36, 402, 110, 32, C.green, 8, 0.035 + 0.12 * update, C.green, 0.8);
    body += label('θ ← θ − ηḡ', targetX, 423, 11, C.ink, 'middle', 620, 0.45 + 0.55 * update);
  });
  body += label('Different batches; identical gradient result; identical replicas.', 480, 500, 13, C.green, 'middle', 620);
  return svg(body, 'Four DistributedDataParallel ranks compute gradients from different batches. An all-reduce averages those gradients and returns the same result to every rank before each replica applies the same optimizer update.');
}

function fsdpFrame(frame) {
  const t = cycle(frame);
  const gather = phase(t, 0.04, 0.34);
  const compute = phase(t, 0.34, 0.62);
  const reshard = phase(t, 0.62, 0.88);
  const colors = [C.cyan, C.amber, C.pink, C.violet];
  const xs = [118, 310, 502, 694];
  let body = header('03', 'FULLY SHARDED DATA PARALLEL', 'Materialize parameters only around computation');
  body += panel(42, 122, 876, 326, 'SHARDED · ALL-GATHER · COMPUTE · RESHARD', C.violet);

  body += label('persistent state on four ranks', 78, 177, 10, C.muted, 'start', 600, 1, 1);
  xs.forEach((x, rank) => {
    body += label(`R${rank}`, x + 47, 207, 10, colors[rank], 'middle', 700);
    body += rect(x, 220, 94, 38, colors[rank], 8, 0.08, colors[rank], 1);
    body += rect(x + rank * 23.5, 220, 23.5, 38, colors[rank], 5, 0.26, colors[rank], 0.6);
    body += flow(x + 47, 258, x + 47, 303, gather, colors[rank], 0.25 + 0.7 * gather);
  });

  body += label('ALL-GATHER MATERIALIZES A FULL PARAMETER ON EACH RANK', 480, 286, 9, C.violet, 'middle', 700, 0.4 + 0.6 * gather, 0.7);
  xs.forEach((x) => {
    body += rect(x, 302, 94, 43, C.violet, 8, 0.025 + 0.05 * gather, C.violet, 0.8);
    colors.forEach((color, shard) => {
      body += rect(x + 7 + shard * 20, 312, 17, 23, color, 4, 0.07 + 0.17 * gather, color, 0.5);
    });
    body += circle(x + 47, 323, 30, C.green, 0.02 + 0.05 * compute * pulse(t * 2), C.green, 0.8);
    body += label('compute', x + 47, 327, 8, C.green, 'middle', 650, compute);
  });

  xs.forEach((x, rank) => {
    body += flow(x + 47, 345, x + 47, 408, reshard, colors[rank], 0.2 + 0.75 * reshard);
    body += rect(x, 395, 94, 38, colors[rank], 7, 0.03 + 0.13 * reshard, colors[rank], 0.8);
    body += rect(x + rank * 23.5, 395, 23.5, 38, colors[rank], 5, 0.05 + 0.2 * reshard, colors[rank], 0.6);
    body += label(`keep shard ${rank}`, x + 47, 421, 9, C.ink, 'middle', 600, 0.35 + 0.65 * reshard);
  });
  body += label('The memory win comes from resharing state between compute regions.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'FSDP begins with a different parameter shard on each rank, all-gathers a full parameter for a compute region, then reshares the parameter so each rank retains only its shard between compute regions.');
}

function parallelismFrame(frame) {
  const t = cycle(frame);
  const move = phase(t, 0.04, 0.72);
  const pulseValue = 0.55 + 0.45 * pulse(t * 1.5);
  const xs = [28, 260, 492, 724];
  const colors = [C.cyan, C.violet, C.amber, C.green];
  const names = ['DATA PARALLEL', 'TENSOR PARALLEL', 'PIPELINE PARALLEL', 'HYBRID MESH'];
  let body = header('04', 'DATA · TENSOR · PIPELINE · DEVICE MESH', 'Parallel dimensions split different objects');
  xs.forEach((x, index) => { body += panel(x, 122, 208, 326, names[index], colors[index]); });

  // Data parallelism: each rank owns a complete model but sees a different batch shard.
  for (let rank = 0; rank < 3; rank++) {
    const y = 170 + rank * 70;
    body += rect(45, y, 52, 42, C.cyan, 7, 0.05 + 0.08 * move, C.cyan, 0.8);
    body += label(`B${rank}`, 71, y + 27, 10, C.ink, 'middle', 650);
    body += arrow(102, y + 21, 126, y + 21, C.cyan, 1.2, 0.45 + 0.45 * move);
    body += rect(132, y, 75, 42, C.cyan, 7, 0.035, C.cyan, 0.8);
    body += label('MODEL', 169, y + 26, 9, C.ink, 'middle', 650);
    body += flow(102, y + 21, 196, y + 21, clamp(move - rank * 0.08), C.cyan, 0.75);
  }
  body += label('split batches', 132, 397, 11, C.ink, 'middle', 650);
  body += label('replicate parameters', 132, 419, 10, C.muted, 'middle', 500);

  // Tensor parallelism: one matrix operation is partitioned across ranks.
  body += label('ONE OPERATION', 364, 174, 10, C.muted, 'middle', 650, 1, 1);
  for (let col = 0; col < 4; col++) {
    const color = col < 2 ? C.violet : C.pink;
    body += rect(292 + col * 36, 196, 31, 112, color, 5, 0.05 + 0.1 * move, color, 0.8);
    for (let row = 1; row < 4; row++) body += line(293 + col * 36, 196 + row * 28, 322 + col * 36, 196 + row * 28, color, 0.7, 0.34);
  }
  body += label('R0', 327, 330, 10, C.violet, 'middle', 700);
  body += label('R1', 399, 330, 10, C.pink, 'middle', 700);
  body += arrow(314, 351, 414, 351, C.green, 1.4, 0.35 + 0.5 * move);
  body += glow(mix(314, 414, move), 351, 10, C.green, 0.72);
  body += label('split matrix work', 364, 397, 11, C.ink, 'middle', 650);
  body += label('communicate partial results', 364, 419, 10, C.muted, 'middle', 500);

  // Pipeline parallelism: consecutive layer ranges live on different stages.
  const stageColors = [C.amber, C.pink, C.violet];
  for (let stage = 0; stage < 3; stage++) {
    const x = 516 + stage * 62;
    body += rect(x, 205, 50, 112, stageColors[stage], 9, 0.04 + 0.08 * move, stageColors[stage], 0.9);
    body += label(`S${stage}`, x + 25, 229, 10, stageColors[stage], 'middle', 700);
    for (let layer = 0; layer < 3; layer++) body += rect(x + 10, 243 + layer * 21, 30, 14, stageColors[stage], 4, 0.08, stageColors[stage], 0.5);
    if (stage < 2) body += arrow(x + 51, 261, x + 61, 261, C.amber, 1.2, 0.55);
  }
  const pipelineX = 541 + move * 124;
  body += glow(pipelineX, 261, 12, C.amber, 0.8);
  body += label('microbatch', pipelineX, 188, 9, C.amber, 'middle', 600);
  body += label('split layer depth', 596, 397, 11, C.ink, 'middle', 650);
  body += label('stream microbatches', 596, 419, 10, C.muted, 'middle', 500);

  // DeviceMesh: rows and columns assign independent parallel dimensions.
  body += label('2 × 2 DEVICE MESH', 828, 174, 10, C.muted, 'middle', 650, 1, 1);
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < 2; col++) {
      const x = 770 + col * 72;
      const y = 207 + row * 72;
      const rank = row * 2 + col;
      body += rect(x, y, 55, 50, col === 0 ? C.violet : C.pink, 9, 0.045 + 0.055 * pulseValue, row === 0 ? C.cyan : C.green, 1.2);
      body += label(`R${rank}`, x + 27.5, y + 30, 10, C.ink, 'middle', 700);
    }
  }
  body += arrow(775, 331, 892, 331, C.violet, 1.4, 0.7);
  body += label('tensor dimension', 833, 351, 9, C.violet, 'middle', 600);
  body += arrow(902, 211, 902, 324, C.cyan, 1.4, 0.7);
  body += label('data', 908, 273, 9, C.cyan, 'start', 600);
  body += label('compose dimensions', 828, 397, 11, C.ink, 'middle', 650);
  body += label('one rank per coordinate', 828, 419, 10, C.muted, 'middle', 500);

  body += label('“Model parallel” is the umbrella: tensor and pipeline parallelism split model computation.', 480, 501, 13, C.green, 'middle', 620);
  return svg(body, 'Data parallelism sends different batches through full model replicas. Tensor parallelism partitions one matrix operation. Pipeline parallelism assigns consecutive layer ranges to stages and streams microbatches. A DeviceMesh composes data and tensor dimensions; model parallelism is the umbrella for splitting model computation.');
}

const animations = [
  ['pytorch-distributed-collective-order.gif', collectiveOrderFrame],
  ['pytorch-distributed-ddp-all-reduce.gif', ddpFrame],
  ['pytorch-distributed-fsdp-lifecycle.gif', fsdpFrame],
  ['pytorch-distributed-parallelism-map.gif', parallelismFrame],
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
