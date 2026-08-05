import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
let sharp;
try {
  sharp = require('sharp');
} catch {
  const fallback = process.env.PERCEPTION_SHARP_PATH;
  if (!fallback) {
    throw new Error('Install dependencies or set PERCEPTION_SHARP_PATH to a Sharp module directory.');
  }
  sharp = require(fallback);
}

const WIDTH = 960;
const HEIGHT = 540;
const FPS = 10;
const FRAMES = 50;
const ROOT = process.cwd();
const OUTPUT = join(ROOT, 'public/assets/images');
const SCRATCH = join(ROOT, '.tmp-perception-gifs');

const C = {
  bg: '#081019',
  bg2: '#0c1622',
  panel: '#101d2a',
  panel2: '#142434',
  grid: '#26394a',
  ink: '#f0f6f8',
  muted: '#8fa6b5',
  dim: '#526979',
  camera: '#58d8ff',
  lidar: '#ffc15a',
  radar: '#ff6fae',
  metric: '#78e0a5',
  task: '#b19aff',
  danger: '#ff7a76',
};

const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const mix = (a, b, t) => a + (b - a) * t;
const ease = (t) => {
  const x = clamp(t);
  return x * x * (3 - 2 * x);
};
const ping = (t) => 0.5 - 0.5 * Math.cos(t * Math.PI * 2);
const cycle = (frame) => frame / FRAMES;
const phase = (t, start, end) => ease((t - start) / (end - start));
const alpha = (value) => clamp(value).toFixed(3);

function esc(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function rect(x, y, w, h, fill, radius = 14, opacity = 1, stroke = 'none', sw = 1) {
  const resolvedFill = fill === C.panel ? 'url(#panel-fill)' : fill;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${resolvedFill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-width="${sw}"/>`;
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

function text(value, x, y, size = 16, fill = C.ink, anchor = 'start', weight = 500, opacity = 1, spacing = 0) {
  return `<text x="${x}" y="${y}" fill="${fill}" fill-opacity="${alpha(opacity)}" font-family="SF Pro Display, SF Pro Text, Arial, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" letter-spacing="${spacing}">${esc(value)}</text>`;
}

function glowCircle(x, y, r, color, opacity = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" fill-opacity="${alpha(opacity * 0.16)}" filter="url(#soft-glow)"/>${circle(x, y, Math.max(2, r * 0.22), color, opacity)}`;
}

function flowDot(x1, y1, x2, y2, p, color, r = 4, opacity = 1) {
  const q = clamp(p);
  return glowCircle(mix(x1, x2, q), mix(y1, y2, q), r * 2.8, color, opacity);
}

function arrow(x1, y1, x2, y2, color, sw = 2, opacity = 1) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const len = 8 + sw;
  const wing = 4 + sw * 0.5;
  const ax = x2 - Math.cos(angle) * len;
  const ay = y2 - Math.sin(angle) * len;
  const p1x = ax + Math.cos(angle + Math.PI / 2) * wing;
  const p1y = ay + Math.sin(angle + Math.PI / 2) * wing;
  const p2x = ax + Math.cos(angle - Math.PI / 2) * wing;
  const p2y = ay + Math.sin(angle - Math.PI / 2) * wing;
  return `${line(x1, y1, x2, y2, color, sw, opacity)}<polygon points="${x2},${y2} ${p1x},${p1y} ${p2x},${p2y}" fill="${color}" fill-opacity="${alpha(opacity)}"/>`;
}

function header(index, kicker, titleValue) {
  return `${text(index, 54, 52, 13, C.metric, 'start', 600, 1, 2.2)}${text(kicker.toUpperCase(), 92, 52, 13, C.muted, 'start', 600, 1, 2.2)}${text(titleValue, 54, 92, 29, C.ink, 'start', 600)}`;
}

function panel(x, y, w, h, label, accent = C.grid) {
  return `<g filter="url(#panel-shadow)">${rect(x, y, w, h, C.panel, 18, 0.94, accent, 1)}</g>${path(`M ${x+18} ${y+1} L ${x+82} ${y+1}`,accent,1.6,0.72)}${circle(x+w-20,y+20,2.5,accent,0.58)}${text(label, x + 18, y + 30, 13, C.muted, 'start', 600, 1, 1.2)}`;
}

function sensorChip(x, y, label, color, active = 1) {
  return `${rect(x, y, 92, 34, color, 17, 0.08 + active * 0.12, color, 1)}${circle(x + 18, y + 17, 4, color, active)}${text(label, x + 31, y + 22, 13, active > 0.25 ? C.ink : C.dim, 'start', 500, 1)}`;
}

function bevGrid(x, y, w, h, opacity = 1, accent = C.grid) {
  let out = rect(x, y, w, h, C.bg2, 12, 0.9 * opacity, accent, 1);
  for (let i = 1; i < 7; i++) out += line(x + (w * i) / 7, y + 1, x + (w * i) / 7, y + h - 1, C.grid, 1, 0.58 * opacity);
  for (let j = 1; j < 4; j++) out += line(x + 1, y + (h * j) / 4, x + w - 1, y + (h * j) / 4, C.grid, 1, 0.58 * opacity);
  return out;
}

function cube(x, y, s, color, opacity = 1) {
  const dx = s * 0.35;
  const dy = s * 0.24;
  return `<path d="M ${x} ${y} L ${x + s} ${y} L ${x + s + dx} ${y - dy} L ${x + dx} ${y - dy} Z" fill="${color}" fill-opacity="${alpha(opacity * 0.18)}" stroke="${color}" stroke-opacity="${alpha(opacity * 0.55)}"/><path d="M ${x + s} ${y} L ${x + s} ${y + s} L ${x + s + dx} ${y + s - dy} L ${x + s + dx} ${y - dy} Z" fill="${color}" fill-opacity="${alpha(opacity * 0.10)}" stroke="${color}" stroke-opacity="${alpha(opacity * 0.4)}"/><rect x="${x}" y="${y}" width="${s}" height="${s}" fill="${color}" fill-opacity="${alpha(opacity * 0.07)}" stroke="${color}" stroke-opacity="${alpha(opacity * 0.55)}"/>`;
}

function vehicle(x, y, color = C.ink, opacity = 1, scale = 1) {
  return `${rect(x - 16 * scale, y - 27 * scale, 32 * scale, 54 * scale, color, 8 * scale, 0.12 * opacity, color, 1)}${rect(x - 10 * scale, y - 15 * scale, 20 * scale, 21 * scale, color, 5 * scale, 0.16 * opacity)}${circle(x - 14 * scale, y - 16 * scale, 3 * scale, color, opacity)}${circle(x + 14 * scale, y - 16 * scale, 3 * scale, color, opacity)}${circle(x - 14 * scale, y + 17 * scale, 3 * scale, color, opacity)}${circle(x + 14 * scale, y + 17 * scale, 3 * scale, color, opacity)}`;
}

function crosshair(x, y, color, opacity = 1, radius = 10) {
  return `${circle(x, y, radius, color, 0.04 * opacity, color, 1.5)}${line(x - radius - 5, y, x + radius + 5, y, color, 1, opacity)}${line(x, y - radius - 5, x, y + radius + 5, color, 1, opacity)}${circle(x, y, 2.5, color, opacity)}`;
}

function featureMap(x, y, w, h, color, cells = 6, opacity = 1) {
  let out = rect(x, y, w, h, C.panel2, 8, 1, color, 0.85 * opacity);
  for (let i = 1; i < cells; i++) out += line(x + (w * i) / cells, y + 2, x + (w * i) / cells, y + h - 2, color, 0.8, 0.16 * opacity);
  for (let i = 1; i < Math.max(2, Math.round(cells * h / w)); i++) out += line(x + 2, y + (h * i) / Math.max(2, Math.round(cells * h / w)), x + w - 2, y + (h * i) / Math.max(2, Math.round(cells * h / w)), color, 0.8, 0.16 * opacity);
  return out;
}

function svg(body, description) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${esc(description)}">
  <defs>
    <radialGradient id="bg-glow" cx="50%" cy="0%" r="92%"><stop offset="0%" stop-color="#173248"/><stop offset="55%" stop-color="${C.bg}"/><stop offset="100%" stop-color="#060b12"/></radialGradient>
    <linearGradient id="panel-fill" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#142737"/><stop offset="48%" stop-color="#0f1d2a"/><stop offset="100%" stop-color="#0a151f"/></linearGradient>
    <pattern id="micro-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 L 0 0 0 32" fill="none" stroke="#8cb4c8" stroke-width="0.5" stroke-opacity="0.10"/></pattern>
    <filter id="soft-glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="7"/></filter>
    <filter id="panel-shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#02070c" flood-opacity="0.34"/></filter>
  </defs>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#bg-glow)"/>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#micro-grid)" opacity="0.18"/>
  <circle cx="870" cy="90" r="190" fill="${C.camera}" fill-opacity="0.022"/>
  <circle cx="90" cy="500" r="210" fill="${C.metric}" fill-opacity="0.018"/>
  <path d="M 0 116 C 250 100 680 132 960 104" fill="none" stroke="#85aec3" stroke-width="1" stroke-opacity="0.055"/>
  ${body}
  </svg>`;
}

function visionFrame(frame) {
  const t = cycle(frame);
  const project = phase(t, 0.08, 0.42);
  const sample = phase(t, 0.38, 0.72);
  const pulse = 0.55 + 0.45 * ping(t * 1.5);
  let b = header('01', 'DETR3D · BEVFORMER', 'A 3D point projects to a different pixel in each camera');

  b += panel(44, 126, 226, 338, '1 · METRIC POINT', C.metric);
  b += bevGrid(68, 174, 178, 220, 1, C.metric);
  b += text('ego frame', 86, 420, 12, C.muted, 'start', 500);
  b += vehicle(157, 363, C.ink, 0.72, 0.48);
  const pointX = 198;
  const pointY = 235;
  b += line(157, 354, pointX, pointY, C.metric, 1.5, 0.38, '5 6');
  b += glowCircle(pointX, pointY, 14 + 4 * pulse, C.metric, 0.92);
  b += text('X = (x, y, z)', pointX, 207, 13, C.metric, 'middle', 600);

  b += panel(292, 126, 356, 338, '2 · CALIBRATED PROJECTION', C.camera);
  const cameras = [
    { y: 174, label: 'front camera', px: 508, py: 226 },
    { y: 304, label: 'left camera', px: 390, py: 356 },
  ];
  cameras.forEach((cam, i) => {
    b += rect(316, cam.y, 308, 108, C.bg2, 12, 0.92, C.grid, 1);
    b += text(cam.label, 330, cam.y + 23, 12, C.muted, 'start', 600);
    b += path(`M 342 ${cam.y + 92} L 442 ${cam.y + 39} L 598 ${cam.y + 92}`, C.grid, 1.5, 0.72);
    b += line(470, cam.y + 34, 470, cam.y + 96, C.grid, 1, 0.28);
    b += line(328, cam.y + 66, 612, cam.y + 66, C.grid, 1, 0.28);
    b += crosshair(cam.px, cam.py, C.camera, 0.22 + 0.78 * project, 9 + 2 * pulse);
    const startY = i === 0 ? 245 : 248;
    b += flowDot(260, startY, cam.px, cam.py, project, C.metric, 3.2, 0.82);
  });
  b += text('uᵢ = π(Kᵢ, Rᵢ, tᵢ, X)', 470, 448, 13, C.ink, 'middle', 500);

  b += panel(670, 126, 246, 338, '3 · SAMPLE FEATURE LEVELS', C.camera);
  const levels = [
    { x: 696, y: 178, w: 188, h: 76, stride: 'stride 4', px: 817, py: 216 },
    { x: 715, y: 273, w: 150, h: 62, stride: 'stride 8', px: 812, py: 304 },
    { x: 735, y: 354, w: 110, h: 48, stride: 'stride 16', px: 806, py: 378 },
  ];
  levels.forEach((q, i) => {
    b += featureMap(q.x, q.y, q.w, q.h, C.camera, 7 - i, 0.65 + 0.35 * sample);
    b += text(q.stride, q.x + 8, q.y + 18, 10, C.muted, 'start', 600);
    b += crosshair(q.px, q.py, C.camera, 0.18 + 0.72 * sample, 6 + i);
    if (i < 2) b += arrow(q.px, q.py + 12, levels[i + 1].px, levels[i + 1].py - 12, C.camera, 1.3, 0.28 + 0.5 * sample);
  });
  b += text('Fᵢˡ(uᵢ / strideₗ)', 793, 429, 13, C.ink, 'middle', 500);
  b += text('FPN is inherited; projection is the 3D step.', 480, 500, 13, C.metric, 'middle', 600);
  return svg(b, 'DETR3D and BEVFormer start from one metric 3D point, project it to a different pixel in each calibrated camera, and sample inherited multiscale image features at stride-adjusted coordinates.');
}

function lidarFrame(frame) {
  const t = cycle(frame);
  const encode = phase(t, 0.08, 0.42);
  const densify = phase(t, 0.38, 0.72);
  const pulse = 0.55 + 0.45 * ping(t * 1.5);
  let b = header('02', 'POINTPILLARS · SECOND · DSVT · VOXELNEXT', 'Where LiDAR models become dense');
  const cards = [
    { x: 36, label: 'POINTPILLARS', color: C.grid },
    { x: 342, label: 'SECOND · DSVT', color: C.lidar },
    { x: 648, label: 'VOXELNEXT', color: C.metric },
  ];
  cards.forEach((c) => b += panel(c.x, 126, 276, 340, c.label, c.color));

  const sparseStack = (ox, oy, color, opacity = 1) => {
    let out = path(`M ${ox} ${oy + 94} L ${ox + 92} ${oy + 54} L ${ox + 176} ${oy + 92}`, C.grid, 1.3, 0.6);
    const pts = [[12,82],[38,72],[66,90],[94,64],[126,84],[156,70],[47,31],[78,19],[111,34],[142,24]];
    pts.forEach(([dx,dy], i) => out += cube(ox + dx, oy + dy, 14, i > 5 ? C.camera : color, opacity * (0.72 + 0.28 * Math.sin(i + t * 6) ** 2)));
    return out;
  };

  // PointPillars: each vertical column is pooled before a dense 2D backbone.
  b += sparseStack(80, 175, C.lidar, 0.9);
  [96,128,160,192,224].forEach((x, i) => {
    const top = 204 + (i % 2) * 18;
    b += rect(x, top, 22, 92 - (top - 204), C.lidar, 5, 0.05 + 0.13 * encode, C.lidar, 1);
    b += flowDot(x + 11, top, x + 11, 299, densify, C.lidar, 2.4, 0.82);
  });
  b += arrow(174, 310, 174, 337, C.lidar, 1.7, 0.8);
  b += bevGrid(86, 342, 176, 72, 0.78 + 0.22 * densify, C.lidar);
  for (let i = 0; i < 18; i++) b += rect(88 + (i % 9) * 19.2, 344 + Math.floor(i / 9) * 34, 18, 32, C.lidar, 2, (0.03 + 0.13 * densify) * (0.5 + (i % 4) / 6));
  b += text('collapse height first', 174, 442, 12, C.ink, 'middle', 600);

  // SECOND and DSVT keep only occupied 3D cells active, then pool height for BEV heads.
  b += sparseStack(386, 175, C.lidar, 0.9);
  const sparseNodes = [[410,274],[452,247],[493,282],[530,232],[566,268],[598,218]];
  sparseNodes.forEach(([x,y], i) => {
    b += cube(x, y, 18, i > 3 ? C.camera : C.lidar, 0.45 + 0.55 * encode);
    if (i > 0) b += line(sparseNodes[i-1][0] + 12, sparseNodes[i-1][1] + 4, x + 8, y + 4, C.lidar, 1.3, 0.12 + 0.5 * pulse * encode);
  });
  b += text('sparse 3D mixing', 480, 322, 11, C.lidar, 'middle', 600);
  b += arrow(480, 330, 480, 349, C.lidar, 1.7, 0.8);
  b += bevGrid(392, 354, 176, 60, 0.75 + 0.25 * densify, C.lidar);
  b += rect(431, 371, 38, 26, C.lidar, 4, 0.08 + 0.20 * densify);
  b += rect(490, 362, 28, 42, C.camera, 4, 0.07 + 0.18 * densify);
  b += text('densify at the BEV head', 480, 442, 12, C.ink, 'middle', 600);

  // VoxelNeXt keeps sparse voxels through the prediction head.
  b += sparseStack(692, 175, C.metric, 0.9);
  const candidates = [[719,274,C.lidar],[762,250,C.lidar],[806,285,C.lidar],[838,230,C.camera]];
  candidates.forEach(([x,y,c], i) => {
    b += cube(x, y, 20, c, 0.42 + 0.58 * encode);
    if (i > 0) b += line(candidates[i-1][0] + 13, candidates[i-1][1] + 5, x + 9, y + 5, C.metric, 1.3, 0.12 + 0.48 * pulse * encode);
  });
  b += arrow(785, 316, 785, 353, C.metric, 1.7, 0.82);
  b += rect(714, 357, 142, 58, C.bg2, 10, 0.9, C.metric, 1);
  b += crosshair(760, 385, C.metric, 0.25 + 0.75 * densify, 12);
  b += crosshair(820, 379, C.camera, 0.20 + 0.72 * densify, 9);
  b += text('boxes from active voxels', 785, 442, 12, C.ink, 'middle', 600);

  b += text('The lineage moves densification later—and VoxelNeXt removes the dense head.', 480, 501, 13, C.metric, 'middle', 600);
  return svg(b, 'PointPillars collapses height before a dense two-dimensional backbone, SECOND and DSVT preserve sparse three-dimensional cells until a later BEV head, and VoxelNeXt predicts boxes directly from active voxels.');
}

function radarFrame(frame) {
  const t = cycle(frame);
  const travel = phase(t, 0.08, 0.58);
  const settle = phase(t, 0.52, 0.80);
  const pulse = 0.55 + 0.45 * ping(t * 1.6);
  let b = header('03', 'CRAFT · CRN · RCBEVDET', 'Where radar enters camera perception');
  const cards = [
    {x:34,label:'CRAFT · PROPOSAL STAGE',accent:C.radar},
    {x:342,label:'CRN · DEPTH STAGE',accent:C.camera},
    {x:650,label:'RCBEVDET · BEV STAGE',accent:C.metric},
  ];
  cards.forEach((c)=>b += panel(c.x,126,276,342,c.label,c.accent));

  // One actor and its radar returns are represented at three different intervention points.
  let x=34;
  b += rect(x+24,170,228,72,C.bg2,10,1,C.camera,1);
  b += text('camera box: weak range',x+39,191,11,C.camera,'start',600);
  b += rect(x+105,202,66,27,C.camera,5,0.08+0.10*pulse,C.camera,1);
  b += path(`M ${x+132} 229 L ${x+62} 374 M ${x+158} 229 L ${x+224} 374`,C.camera,1.2,0.25,'none','4 5');
  const craftReturns=[[x+79,352],[x+116,316],[x+162,303],[x+207,342]];
  craftReturns.forEach(([rx,ry],i)=>{
    const chosen=i===1||i===2;
    b += circle(rx,ry,chosen?7:5,C.radar,chosen?0.88:0.24,C.radar,1);
    b += arrow(rx,ry,rx+13+i*2,ry-3,C.metric,1.3,chosen?0.75:0.22);
    if(chosen){
      b += path(`M ${rx} ${ry} Q ${x+138} 266 ${x+138} 226`,C.radar,1.4,0.18+0.50*travel,'none','4 5');
      b += flowDot(rx,ry,x+138,226,travel,C.radar,2.5,0.88);
    }
  });
  b += text('associate a soft polar set',x+138,406,11,C.radar,'middle',600);
  b += text('around each camera proposal',x+138,427,11,C.muted,'middle',500);
  b += text('proposal chooses returns',x+138,450,11,C.ink,'middle',600);

  // CRN changes the depth distribution before lifting, then aligns camera and radar BEV.
  x=342;
  b += rect(x+24,170,228,74,C.bg2,10,1,C.camera,1);
  const ray = [[x+60,231],[x+98,218],[x+136,205],[x+174,192],[x+218,178]];
  ray.forEach(([dx,dy],i)=>{
    const hit=i===3;
    const before=[0.12,0.20,0.32,0.25,0.11][i];
    const after=hit?0.92:before*(1-settle*0.62);
    b += circle(dx,dy,4+8*after,C.camera,0.12+0.55*after,C.radar,hit?1.2:0);
  });
  b += glowCircle(x+174,192,10,C.radar,0.25+0.60*settle);
  b += text('radar sharpens p(depth | pixel)',x+138,268,11,C.radar,'middle',600);
  b += arrow(x+138,278,x+138,306,C.radar,1.7,0.8);
  b += bevGrid(x+24,314,228,78,1,C.camera);
  const roughX=x+111, alignedX=x+158, targetY=352;
  b += rect(roughX-14,targetY-9,28,18,C.camera,5,0.09,C.camera,1);
  b += line(roughX,targetY,alignedX,targetY-5,C.metric,1.6,0.28+0.55*settle,'4 4');
  b += rect(alignedX-14,targetY-14,28,18,C.radar,5,0.08+0.16*settle,C.radar,1);
  b += text('then align radar and camera BEV',x+138,421,11,C.muted,'middle',500);
  b += text('radar changes the lift',x+138,450,11,C.ink,'middle',600);

  // RCBEVDet learns a radar-specific BEV before cross-modal alignment.
  x=650;
  b += rect(x+24,174,104,62,C.radar,10,0.06,C.radar,1);
  b += text('point stream',x+76,197,11,C.ink,'middle',600);
  b += text('RCS scatter',x+76,220,10,C.radar,'middle',500);
  b += rect(x+148,174,104,62,C.metric,10,0.06,C.metric,1);
  b += text('transformer',x+200,197,11,C.ink,'middle',600);
  b += text('global context',x+200,220,10,C.metric,'middle',500);
  [0,1,2].forEach((i)=>{
    b += circle(x+43+i*31,256,3+i,C.radar,0.72);
    b += circle(x+169+i*31,256,4,C.metric,0.62);
    if(i<2)b += line(x+173+i*31,256,x+196+i*31,256,C.metric,1,0.40);
  });
  b += flowDot(x+76,262,x+122,307,travel,C.radar,2.5,0.86);
  b += flowDot(x+200,262,x+154,307,travel,C.metric,2.5,0.86);
  b += circle(x+138,312,22,C.task,0.07+0.07*pulse,C.task,1);
  b += text('⊕',x+138,319,19,C.ink,'middle',500,0.9);
  b += arrow(x+138,337,x+138,350,C.task,1.7,0.8);
  b += bevGrid(x+24,354,228,54,1,C.task);
  b += rect(x+116,370,40,21,C.radar,4,0.08+0.14*settle,C.radar,1);
  b += text('build radar BEV, then align',x+138,428,11,C.muted,'middle',500);
  b += text('radar owns a BEV encoder',x+138,450,11,C.ink,'middle',600);
  return svg(b, 'CRAFT associates radar returns around each camera proposal, CRN changes the camera depth distribution before lifting, and RCBEVDet builds a radar-specific BEV before camera-radar alignment.');
}

function liftingFrame(frame) {
  const t = cycle(frame);
  const q = phase(t, 0.06, 0.54);
  const settle = phase(t, 0.50, 0.78);
  const pulse = 0.55 + 0.45 * ping(t * 1.4);
  let b = header('04', 'LSS · DETR3D · BEVFORMER', 'Three camera-to-3D mechanisms');
  const cards = [
    {x:34,label:'LSS · BEVDEPTH',accent:C.camera},
    {x:342,label:'DETR3D',accent:C.metric},
    {x:650,label:'BEVFORMER',accent:C.task},
  ];
  cards.forEach((c) => {
    b += panel(c.x,126,276,340,c.label,c.accent);
  });

  // Lift-Splat-Shoot: an image pixel chooses a probability distribution along one calibrated ray.
  const ax=34;
  b += rect(ax+22,168,232,92,C.bg2,10,1,C.grid,1);
  b += path(`M ${ax+43} 248 L ${ax+110} 188 L ${ax+232} 188 L ${ax+246} 248`,C.grid,1.3,0.72);
  b += crosshair(ax+142,211,C.camera,0.88,8+2*pulse);
  b += text('one image feature',ax+142,248,10,C.camera,'middle',600);
  b += arrow(ax+142,266,ax+142,284,C.camera,1.5,0.78);
  b += bevGrid(ax+22,302,232,108,1,C.camera);
  const origin=[ax+51,389];
  const depths=[[ax+82,374],[ax+112,359],[ax+143,344],[ax+174,329],[ax+207,313]];
  b += line(origin[0],origin[1],depths[4][0]+10,depths[4][1]-5,C.camera,2,0.34);
  depths.forEach(([x,y],i)=>{
    const w=[0.12,0.27,0.83,0.42,0.16][i];
    const px=mix(origin[0],x,q), py=mix(origin[1],y,q);
    b += circle(px,py,4+7*w,C.camera,0.12+0.65*w,C.camera,0.6);
  });
  b += text('predict p(depth) on this ray',ax+138,437,11,C.camera,'middle',600);

  // DETR3D: each object query owns a 3D reference point and asks cameras for evidence there.
  const bx=342;
  b += rect(bx+22,168,108,102,C.bg2,10,1,C.grid,1);
  b += rect(bx+146,168,108,102,C.bg2,10,1,C.grid,1);
  b += text('front',bx+34,188,10,C.muted,'start',600);
  b += text('left',bx+158,188,10,C.muted,'start',600);
  b += path(`M ${bx+30} 258 L ${bx+74} 200 L ${bx+122} 258`,C.grid,1.2,0.62);
  b += path(`M ${bx+154} 258 L ${bx+205} 200 L ${bx+246} 258`,C.grid,1.2,0.62);
  const p1=[bx+87,221], p2=[bx+198,239];
  b += crosshair(p1[0],p1[1],C.metric,0.22+0.78*q,7);
  b += crosshair(p2[0],p2[1],C.metric,0.22+0.78*q,7);
  b += bevGrid(bx+22,304,232,106,1,C.metric);
  const query=[bx+141,352];
  b += cube(query[0]-12,query[1]-12,25,C.metric,0.55+0.35*pulse);
  b += text('3D reference point',query[0],397,10,C.metric,'middle',600);
  b += path(`M ${query[0]} ${query[1]} Q ${bx+105} 292 ${p1[0]} ${p1[1]}`,C.metric,1.5,0.18+0.52*q,'none','5 5');
  b += path(`M ${query[0]} ${query[1]} Q ${bx+179} 292 ${p2[0]} ${p2[1]}`,C.metric,1.5,0.18+0.52*q,'none','5 5');
  b += flowDot(query[0],query[1],p1[0],p1[1],q,C.metric,2.6,0.86);
  b += flowDot(query[0],query[1],p2[0],p2[1],q,C.metric,2.6,0.86);
  b += text('query chooses (x, y, z)',bx+138,437,11,C.metric,'middle',600);

  // BEVFormer: each BEV cell owns x,y and samples several z references along a vertical pillar.
  const cx=650;
  b += rect(cx+22,168,232,92,C.bg2,10,1,C.grid,1);
  b += path(`M ${cx+34} 248 L ${cx+90} 194 L ${cx+242} 194`,C.grid,1.2,0.64);
  const imageSamples=[[cx+86,230],[cx+132,216],[cx+183,202]];
  imageSamples.forEach(([x,y],i)=>b += crosshair(x,y,C.task,0.18+0.72*q,5+i));
  b += bevGrid(cx+22,304,232,106,1,C.task);
  const cell=[cx+146,365];
  b += rect(cell[0]-14,cell[1]-11,28,22,C.task,4,0.10+0.15*settle,C.task,1);
  const pillarX=cx+102;
  b += line(pillarX,379,pillarX,292,C.task,2,0.72);
  [306,330,354,378].forEach((y)=>b += glowCircle(pillarX,y,7,C.task,0.34+0.52*pulse));
  imageSamples.forEach(([x,y],i)=>{
    const py=306+i*24;
    b += path(`M ${pillarX} ${py} Q ${cx+138} 276 ${x} ${y}`,C.task,1.3,0.15+0.48*q,'none','4 5');
    b += flowDot(pillarX,py,x,y,q,C.task,2.5,0.82);
  });
  b += text('BEV cell fixes (x, y); sample z',cx+138,437,11,C.task,'middle',600);
  b += text('image chooses depth',172,500,12,C.camera,'middle',600);
  b += text('object query chooses x, y, z',480,500,12,C.metric,'middle',600);
  b += text('BEV cell chooses x, y and samples z',788,500,12,C.task,'middle',600);
  return svg(b, 'Lift-Splat-Shoot predicts depth along each calibrated image ray, DETR3D projects an object query reference point into the cameras, and BEVFormer lifts each BEV cell into a vertical pillar of reference points.');
}

function fusionFrame(frame) {
  const t = cycle(frame);
  const flow = phase(t,0.05,0.60);
  const settle = phase(t,0.54,0.82);
  const pulse = 0.55+0.45*ping(t*1.4);
  let b=header('05','POINTPAINTING · FUTR3D · BEVFUSION','What survives point, query, and BEV fusion');
  const cards=[
    {x:34,label:'POINTPAINTING · POINTS',accent:C.lidar},
    {x:342,label:'FUTR3D · QUERIES',accent:C.metric},
    {x:650,label:'BEVFUSION · BEV',accent:C.task},
  ];
  cards.forEach((c)=>b+=panel(c.x,126,276,340,c.label,c.accent));

  const sharedScene=(x,y,accent)=>{
    let out=bevGrid(x,y,232,142,1,accent);
    out+=path(`M ${x+22} ${y+128} Q ${x+92} ${y+70} ${x+210} ${y+22}`,C.camera,5,0.23);
    out+=path(`M ${x+34} ${y+132} Q ${x+102} ${y+78} ${x+216} ${y+30}`,C.camera,1.5,0.72,'none','7 6');
    out+=vehicle(x+148,y+70,C.ink,0.72,0.45);
    [[126,49],[140,57],[156,60],[166,77],[150,88],[132,83],[64,111],[194,40]].forEach(([dx,dy],i)=>out+=circle(x+dx,y+dy,3,C.lidar,i<6?0.88:0.42));
    return out;
  };

  // PointPainting transfers image semantics only at LiDAR hit locations.
  let x=34;
  b+=sharedScene(x+22,170,C.lidar);
  b+=text('camera semantics + LiDAR hits',x+138,330,10,C.muted,'middle',500);
  b+=arrow(x+138,339,x+138,356,C.lidar,1.6,0.8);
  const painted=[[x+72,393],[x+100,381],[x+131,398],[x+159,378],[x+194,391]];
  painted.forEach(([px,py],i)=>{
    b+=flowDot(x+138,304,px,py,flow,C.camera,2.2,0.72);
    b+=circle(px,py,5,i<4?C.lidar:C.camera,0.55+0.28*settle,i<4?C.camera:C.lidar,1);
  });
  b+=path(`M ${x+54} 424 Q ${x+138} 384 ${x+222} 365`,C.camera,2,0.08*(1-settle),'none','6 6');
  b+=text('only semantics at hit points survive',x+138,449,11,C.lidar,'middle',600);

  // FUTR3D samples every sensor around an object hypothesis; the carrier is an actor query.
  x=342;
  b+=sharedScene(x+22,170,C.metric);
  const qx=x+169,qy=240;
  b+=circle(qx,qy,24,C.metric,0.06+0.06*pulse,C.metric,1.2);
  b+=cube(qx-12,qy-10,24,C.metric,0.58);
  [[x+55,193,C.camera],[x+74,286,C.lidar],[x+229,204,C.radar]].forEach(([sx,sy,c],i)=>{
    b+=path(`M ${qx} ${qy} Q ${x+138} ${210+i*35} ${sx} ${sy}`,c,1.4,0.18+0.50*flow,'none','4 5');
    b+=flowDot(qx,qy,sx,sy,flow,c,2.4,0.82);
  });
  b+=text('one 3D object hypothesis',x+138,330,10,C.muted,'middle',500);
  b+=arrow(x+138,339,x+138,358,C.metric,1.6,0.8);
  b+=circle(x+138,395,26,C.metric,0.08+0.08*settle,C.metric,1.2);
  b+=vehicle(x+138,395,C.metric,0.82,0.42);
  b+=text('actor evidence survives; fields do not',x+138,449,11,C.metric,'middle',600);

  // BEVFusion aligns two dense metric maps, retaining scene semantics and object geometry.
  x=650;
  b+=sharedScene(x+22,170,C.task);
  b+=text('camera BEV',x+74,329,10,C.camera,'middle',600);
  b+=text('+',x+138,329,13,C.ink,'middle',600);
  b+=text('LiDAR BEV',x+202,329,10,C.lidar,'middle',600);
  b+=arrow(x+138,338,x+138,354,C.task,1.6,0.8);
  b+=bevGrid(x+36,362,204,66,1,C.task);
  b+=path(`M ${x+48} 420 Q ${x+128} 386 ${x+224} 369`,C.camera,4,0.14+0.24*settle);
  b+=rect(x+142,378,38,25,C.lidar,5,0.08+0.18*settle,C.lidar,1);
  b+=vehicle(x+161,390,C.ink,0.72,0.34);
  b+=circle(x+161,390,18,C.task,0.03+0.04*pulse);
  b+=text('object geometry and scene fields survive',x+138,449,11,C.task,'middle',600);
  b+=text('The carrier decides which evidence can reach the heads.',480,500,13,C.metric,'middle',600);
  return svg(b,'PointPainting can transfer camera semantics only where LiDAR has a return, FUTR3D gathers sensor evidence around sparse object queries, and BEVFusion preserves dense camera semantics and LiDAR geometry on an aligned metric grid.');
}

function dropoutFrame(frame) {
  const t=cycle(frame);
  const slot=Math.floor(t*3)%3;
  const local=(t*3)%1;
  const states=[
    {label:'BOTH HEALTHY',cam:1,lidar:1,camHealth:0.90},
    {label:'LIDAR MISSING',cam:1,lidar:0,camHealth:0.90},
    {label:'CAMERA DEGRADED',cam:1,lidar:1,camHealth:0.20},
  ];
  const state=states[slot];
  const pulse=0.55+0.45*ping(local);
  let b=header('06','UNIBEV · METABEV · GRACE-BEV','Missing and degraded sensors require different signals');
  const cards=[
    {x:34,label:'UNIBEV · AVAILABILITY',accent:C.metric},
    {x:342,label:'METABEV · AVAILABILITY',accent:C.task},
    {x:650,label:'GRACE-BEV · RELIABILITY',accent:C.radar},
  ];
  cards.forEach((c)=>b += panel(c.x,126,276,342,c.label,c.accent));

  const scenario=(x)=>{
    let out=text(state.label,x+138,171,11,state.label.includes('DEGRADED')?C.danger:C.ink,'middle',700,1,0.8);
    out+=sensorChip(x+20,184,'camera',C.camera,state.cam);
    out+=sensorChip(x+164,184,'LiDAR',C.lidar,state.lidar);
    if(state.camHealth<0.5) out+=line(x+24,188,x+108,214,C.danger,2,0.82);
    return out;
  };

  // UniBEV samples missing-modality packets and normalizes only the streams marked present.
  let x=34;
  b += scenario(x);
  [[1,1],[1,0],[0,1]].forEach((set,i)=>{
    const xx=x+44+i*65;
    const current=set[0]===state.cam&&set[1]===state.lidar;
    b += rect(xx,238,48,24,C.panel2,12,1,current?C.metric:C.grid,current?1.5:1);
    b += circle(xx+17,250,3,C.camera,set[0]);
    b += circle(xx+31,250,3,C.lidar,set[1]);
  });
  const uniSum=Math.max(state.cam+state.lidar,0.01);
  b += line(x+66,218,x+120,302,C.camera,2,0.20+state.cam*0.66);
  b += line(x+210,218,x+156,302,C.lidar,2,0.20+state.lidar*0.66);
  b += circle(x+138,320,30,C.metric,0.08,C.metric,1.2);
  b += text('Σ / |M|',x+138,325,14,C.ink,'middle',500,0.92);
  b += text(`${state.cam ? (state.cam/uniSum).toFixed(1) : '0'}  ·  ${state.lidar ? (state.lidar/uniSum).toFixed(1) : '0'}`,x+138,361,11,C.muted,'middle',500,0.8,0.4);
  b += arrow(x+138,353,x+138,380,C.metric,1.8,0.75);
  b += bevGrid(x+62,386,152,42,1,C.metric);
  b += text('drop streams; normalize survivors',x+138,450,11,C.metric,'middle',600);

  // MetaBEV lets BEV queries retrieve from the encoders marked available.
  x=342;
  b += scenario(x);
  b += text('queries see the availability mask',x+138,254,10,C.muted,'middle',500);
  b += bevGrid(x+48,304,180,92,1,C.task);
  const queries=[[x+82,334],[x+138,354],[x+194,326]];
  queries.forEach(([qx,qy],i)=>{
    b += circle(qx,qy,6,C.task,0.72,C.task,1);
    if(state.cam){
      const sx=x+66,sy=218;
      b += path(`M ${qx} ${qy} Q ${x+100+i*12} 260 ${sx} ${sy}`,C.camera,1.3,0.18+0.50*state.cam,'none','4 5');
      if(i===0)b += flowDot(qx,qy,sx,sy,phase(local,0.05,0.7),C.camera,2.2,0.8);
    }
    if(state.lidar){
      const sx=x+210,sy=218;
      b += path(`M ${qx} ${qy} Q ${x+176-i*10} 260 ${sx} ${sy}`,C.lidar,1.3,0.18+0.50*state.lidar,'none','4 5');
      if(i===2)b += flowDot(qx,qy,sx,sy,phase(local,0.05,0.7),C.lidar,2.2,0.8);
    }
  });
  b += text('attend only to available encoders',x+138,450,11,C.task,'middle',600);

  // Grace-BEV adds a reliability signal, so present does not have to mean trusted.
  x=650;
  b += scenario(x);
  const camTrust=state.cam ? state.camHealth + 0.03*pulse : 0;
  const lidarTrust=state.lidar ? 0.90 : 0;
  b += text('estimated trust',x+138,252,10,C.muted,'middle',500);
  b += text(`gcam ${camTrust.toFixed(2)}`,x+28,276,11,C.camera,'start',500);
  b += rect(x+28,286,88,8,C.panel2,4,1,C.grid,0);
  b += rect(x+28,286,88*camTrust,8,C.camera,4,0.72,C.camera,0);
  b += text(`gLiDAR ${lidarTrust.toFixed(2)}`,x+160,276,11,C.lidar,'start',500);
  b += rect(x+160,286,88,8,C.panel2,4,1,C.grid,0);
  b += rect(x+160,286,88*lidarTrust,8,C.lidar,4,0.72,C.lidar,0);
  b += line(x+66,218,x+121,326,C.camera,1+3*camTrust,0.20+0.68*camTrust);
  b += line(x+210,218,x+155,326,C.lidar,1+3*lidarTrust,0.20+0.68*lidarTrust);
  b += circle(x+138,334,30,C.radar,0.08+0.04*pulse,C.radar,1.2);
  b += text('gated',x+138,339,12,C.ink,'middle',500,0.9);
  b += arrow(x+138,366,x+138,384,C.radar,1.8,0.75);
  b += bevGrid(x+62,388,152,40,1,C.radar);
  b += text('estimate health; gate each stream',x+138,450,11,C.radar,'middle',600);
  b += text('Availability says whether data arrived; reliability says whether to trust it.',480,500,13,C.metric,'middle',600);
  return svg(b,'UniBEV and MetaBEV condition fusion on which sensors are present, while Grace-BEV estimates reliability so a present but degraded sensor can be downweighted.');
}

function multitaskFrame(frame){
  const t=cycle(frame);
  const draw=phase(t,0.06,0.58);
  const settle=phase(t,0.52,0.84);
  const pulse=0.55+0.45*ping(t*1.5);
  let b=header('07','KENDALL ET AL. · GRADNORM · PCGRAD','Loss scale, training rate, and gradient conflict');
  const cards=[
    {x:34,label:'KENDALL · LOSS SCALE',accent:C.camera},
    {x:342,label:'GRADNORM · TRAINING RATE',accent:C.metric},
    {x:650,label:'PCGRAD · DIRECTION',accent:C.task},
  ];
  cards.forEach((c)=>b+=panel(c.x,126,276,342,c.label,c.accent));

  const rawVectors=(x)=>{
    const o={x:x+138,y:325};
    const a={x:x+66,y:222};
    const c={x:x+230,y:372};
    let out=line(x+36,o.y,x+244,o.y,C.grid,1,0.58);
    out+=line(o.x,184,o.x,408,C.grid,1,0.58);
    out+=line(o.x,o.y,a.x,a.y,C.camera,1.4,0.40,'5 5');
    out+=line(o.x,o.y,c.x,c.y,C.task,1.4,0.40,'5 5');
    out+=text('raw g₁',a.x-4,a.y-10,10,C.camera,'middle',600,0.72);
    out+=text('raw g₂',c.x+2,c.y+17,10,C.task,'middle',600,0.72);
    return {out,o,a,c};
  };

  // Kendall changes the contribution of each loss through learned uncertainty scales.
  let x=34;
  let v=rawVectors(x); b+=v.out;
  const k1={x:mix(v.o.x,x+88,draw),y:mix(v.o.y,253,draw)};
  const k2={x:mix(v.o.x,x+206,draw),y:mix(v.o.y,360,draw)};
  b+=arrow(v.o.x,v.o.y,k1.x,k1.y,C.camera,2.8,0.88);
  b+=arrow(v.o.x,v.o.y,k2.x,k2.y,C.task,2.8,0.88);
  b+=text('1 / 2σ₁²',x+62,188,11,C.camera,'start',600);
  b+=text('1 / 2σ₂²',x+214,405,11,C.task,'middle',600);
  b+=text('learn uncertainty; rescale loss',x+138,443,11,C.camera,'middle',600);

  // GradNorm changes task weights until gradient norms reflect relative training rates.
  x=342;
  v=rawVectors(x); b+=v.out;
  const g1={x:mix(v.o.x,x+92,settle),y:mix(v.o.y,259,settle)};
  const g2={x:mix(v.o.x,x+200,settle),y:mix(v.o.y,357,settle)};
  b+=arrow(v.o.x,v.o.y,g1.x,g1.y,C.camera,2.8,0.88);
  b+=arrow(v.o.x,v.o.y,g2.x,g2.y,C.task,2.8,0.88);
  b+=line(x+88,244,x+88,272,C.ink,1,0.36,'2 3');
  b+=line(x+204,343,x+204,371,C.ink,1,0.36,'2 3');
  b+=text('target norms from training rate',x+138,192,11,C.metric,'middle',600);
  b+=text('change weights; match gradient norms',x+138,443,11,C.metric,'middle',600);

  // PCGrad changes direction when task gradients have a negative dot product.
  x=650;
  v=rawVectors(x); b+=v.out;
  b+=arrow(v.o.x,v.o.y,v.a.x,v.a.y,C.camera,2.6,0.82);
  b+=arrow(v.o.x,v.o.y,v.c.x,v.c.y,C.task,2.6,0.82);
  const projected={x:x+185,y:233};
  b+=line(v.a.x,v.a.y,projected.x,projected.y,C.danger,1.4,0.18+0.55*draw,'4 5');
  b+=arrow(v.o.x,v.o.y,mix(v.o.x,projected.x,draw),mix(v.o.y,projected.y,draw),C.metric,3,0.24+0.70*draw);
  b+=text('g₁ · g₂ < 0',x+138,192,11,C.danger,'middle',700);
  b+=text('projected g₁',projected.x,projected.y-12,10,C.metric,'middle',600);
  b+=circle(projected.x,projected.y,7+3*pulse,C.metric,0.03+0.04*pulse);
  b+=text('remove the conflicting component',x+138,443,11,C.task,'middle',600);
  b+=text('Kendall and GradNorm change magnitude; PCGrad changes direction.',480,500,13,C.metric,'middle',600);
  return svg(b,'The same two task gradients are shown in every panel: Kendall uncertainty weighting and GradNorm change their magnitudes, while PCGrad projects a conflicting component and changes gradient direction.');
}

function lidarContractFrame(frame){
  const t=cycle(frame);
  const train=phase(t,0.02,0.44);
  const drive=phase(t,0.48,0.86);
  let b=header('08','BEVDEPTH · SPARSE-TO-DENSE · CRKD','LiDAR can be a label, runtime input, or teacher');
  const cols=[
    {x:34,label:'BEVDEPTH · DEPTH LABELS',accent:C.camera},
    {x:342,label:'SPARSE-TO-DENSE · INPUT',accent:C.lidar},
    {x:650,label:'CRKD · TEACHER',accent:C.metric},
  ];
  cols.forEach((c)=>{
    b+=panel(c.x,126,276,342,c.label,c.accent);
    b+=text('TRAIN',c.x+24,174,11,C.muted,'start',600,1,1.2);
    b+=line(c.x+24,290,c.x+252,290,C.grid,1.2,0.8,'5 6');
    b+=text('DRIVE',c.x+24,319,11,C.muted,'start',600,1,1.2);
  });
  // Depth labels: LiDAR supervises depth, but only camera remains onboard.
  let x=34;
  b+=sensorChip(x+24,192,'camera',C.camera,1);
  b+=sensorChip(x+150,192,'LiDAR',C.lidar,1);
  b+=line(x+70,226,x+138,260,C.camera,2,0.8);
  b+=line(x+196,226,x+138,260,C.lidar,2,0.8,'5 5');
  b+=flowDot(x+196,226,x+138,260,train,C.lidar,3,0.9);
  b+=circle(x+138,262,22,C.camera,0.10,C.camera,1);
  b+=text('depth',x+138,267,11,C.ink,'middle',500,0.9);
  b+=sensorChip(x+24,338,'camera',C.camera,1);
  b+=arrow(x+118,355,x+190,355,C.camera,2,0.82);
  b+=bevGrid(x+194,334,58,48,1,C.camera);
  b+=flowDot(x+118,355,x+194,355,drive,C.camera,3,0.9);
  b+=text('projected LiDAR → depth labels',x+138,444,10,C.camera,'middle',500,0.9,0.05);
  b+=text('camera-only drive',x+138,460,10,C.muted,'middle',500,0.82,0.05);

  // Depth completion: LiDAR is an input in both graphs.
  x=342;
  b+=sensorChip(x+24,192,'camera',C.camera,1);
  b+=sensorChip(x+150,192,'LiDAR',C.lidar,1);
  b+=arrow(x+70,226,x+138,260,C.camera,1.8,0.75);
  b+=arrow(x+196,226,x+138,260,C.lidar,1.8,0.85);
  b+=circle(x+138,262,22,C.lidar,0.10,C.lidar,1);
  b+=text('dense z',x+138,267,11,C.ink,'middle',500,0.9);
  b+=sensorChip(x+24,338,'camera',C.camera,1);
  b+=sensorChip(x+150,338,'LiDAR',C.lidar,1);
  b+=line(x+70,372,x+138,414,C.camera,2,0.75);
  b+=line(x+196,372,x+138,414,C.lidar,2,0.85);
  b+=flowDot(x+196,372,x+138,414,drive,C.lidar,3,0.9);
  b+=circle(x+138,416,18,C.lidar,0.12,C.lidar,1);
  b+=text('sparse runtime depth → dense depth',x+138,444,10,C.lidar,'middle',500,0.9,0.02);
  b+=text('LiDAR stays in the graph',x+138,460,10,C.muted,'middle',500,0.82,0.05);

  // Distillation: privileged teacher transfers to a cheaper student.
  x=650;
  b+=rect(x+24,190,96,62,C.lidar,12,0.06,C.lidar,1);
  b+=text('teacher',x+72,218,12,C.ink,'middle',500,0.9);
  b+=circle(x+53,236,4,C.camera,0.9); b+=circle(x+70,236,4,C.lidar,0.9);
  b+=rect(x+156,190,96,62,C.metric,12,0.06,C.metric,1);
  b+=text('student',x+204,218,12,C.ink,'middle',500,0.9);
  b+=circle(x+184,236,4,C.camera,0.9); b+=circle(x+202,236,4,C.radar,0.9);
  b+=arrow(x+122,222,x+154,222,C.metric,2,0.85);
  b+=flowDot(x+122,222,x+154,222,train,C.metric,3,0.9);
  b+=sensorChip(x+24,338,'camera',C.camera,1);
  b+=sensorChip(x+150,338,'radar',C.radar,1);
  b+=line(x+70,372,x+138,414,C.camera,2,0.75);
  b+=line(x+196,372,x+138,414,C.radar,2,0.75);
  b+=flowDot(x+196,372,x+138,414,drive,C.radar,3,0.9);
  b+=circle(x+138,416,18,C.metric,0.12,C.metric,1);
  b+=text('camera+LiDAR teacher',x+138,444,10,C.metric,'middle',500,0.9,0.05);
  b+=text('→ camera+radar student',x+138,460,10,C.muted,'middle',500,0.82,0.05);
  return svg(b,'LiDAR can provide training-only depth labels, remain a runtime input for depth completion, or supervise a cheaper student through distillation; the deployment contracts are different.');
}

function temporalFrame(frame){
  const t=cycle(frame);
  const carry=phase(t,0.04,0.58);
  const refresh=phase(t,0.55,0.80);
  const pulse=0.55+0.45*ping(t*1.5);
  let b=header('09','BEVDET4D · SPARSE4D V2 · STREAMPETR','What crosses the frame boundary');
  const cols=[
    {x:34,label:'BEVDET4D · DENSE FIELD',accent:C.camera},
    {x:342,label:'SPARSE4D V2 · INSTANCES',accent:C.metric},
    {x:650,label:'STREAMPETR · QUERY QUEUE',accent:C.task},
  ];
  cols.forEach((c)=>{
    b+=panel(c.x,126,276,342,c.label,c.accent);
    b+=text('t − 1',c.x+54,178,11,C.muted,'middle',500,0.8,0.5);
    b+=text('t',c.x+222,178,11,C.muted,'middle',500,0.8,0.5);
    b+=arrow(c.x+104,293,c.x+170,293,c.accent,2,0.68);
  });

  // Dense recurrence carries the complete BEV state after ego-motion alignment.
  let x=34;
  b+=bevGrid(x+18,194,102,194,1,C.camera);
  b+=bevGrid(x+156,194,102,194,1,C.camera);
  b+=path(`M ${x+37} 372 Q ${x+70} 270 ${x+103} 210`,C.metric,5,0.20);
  b+=path(`M ${x+175} 372 Q ${x+208} 270 ${x+241} 210`,C.metric,5,0.20+0.18*refresh);
  [[x+50,230,x+188,230],[x+78,270,x+216,270],[x+54,345,x+192,345]].forEach(([x1,y1,x2,y2])=>b+=flowDot(x1,y1,x2,y2,carry,C.camera,2.2,0.46));
  const denseOld={x:x+74,y:310};
  const denseWarp={x:x+203,y:301};
  const denseTrue={x:x+225,y:276};
  b+=rect(denseOld.x-13,denseOld.y-8,26,16,C.camera,5,0.16,C.camera,1);
  b+=path(`M ${denseOld.x} ${denseOld.y} Q ${x+140} 265 ${denseWarp.x} ${denseWarp.y}`,C.camera,1.6,0.30+0.40*carry,'none','5 5');
  b+=rect(mix(denseOld.x,denseWarp.x,carry)-13,mix(denseOld.y,denseWarp.y,carry)-8,26,16,C.camera,5,0.10,C.camera,1);
  b+=rect(denseTrue.x-13,denseTrue.y-8,26,16,C.metric,5,0.08+0.14*refresh,C.metric,1);
  b+=line(denseWarp.x,denseWarp.y,denseTrue.x,denseTrue.y,C.danger,1.5,0.4+0.25*pulse,'4 4');
  b+=circle(x+184,358,5+4*refresh,C.task,0.08+0.22*refresh,C.task,1);
  b+=text('warp every BEV cell',x+138,426,11,C.camera,'middle',600);
  b+=text('scene context survives',x+138,449,10,C.muted,'middle',500);

  // Sparse4D v2 transforms existing instances and adds fresh anchors for births.
  x=342;
  b+=rect(x+18,194,102,194,C.bg2,12,0.9,C.grid,1);
  b+=rect(x+156,194,102,194,C.bg2,12,0.9,C.grid,1);
  const oldActors=[[x+62,250],[x+88,330]];
  const newActors=[[x+188,236],[x+218,306]];
  oldActors.forEach(([ox,oy],i)=>{
    b+=circle(ox,oy,12,C.metric,0.12,C.metric,1);
    b+=rect(ox-10,oy-6,20,12,C.metric,4,0.10,C.metric,1);
    const [nx,ny]=newActors[i];
    b+=path(`M ${ox} ${oy} Q ${x+138} ${oy-20} ${nx} ${ny}`,C.metric,1.8,0.28+0.45*carry,'none','5 5');
    b+=circle(mix(ox,nx,carry),mix(oy,ny,carry),8,C.metric,0.16,C.metric,1);
  });
  b+=text('ego transform',x+138,269,10,C.metric,'middle',600);
  const birthX=x+188,birthY=358;
  b+=circle(birthX,birthY,6+5*refresh,C.task,0.08+0.22*refresh,C.task,1);
  b+=path(`M ${x+238} 382 Q ${x+220} 366 ${birthX} ${birthY}`,C.task,1.6,0.22+0.50*refresh,'none','4 5');
  b+=text('fresh anchor',x+224,378,9,C.task,'middle',600);
  b+=text('transform prior instances',x+138,426,11,C.metric,'middle',600);
  b+=text('fresh anchors discover births',x+138,449,10,C.muted,'middle',500);

  // StreamPETR carries a bounded FIFO of high-confidence foreground queries; background is discarded.
  x=650;
  const queueY=[208,250,292,334];
  queueY.forEach((y,i)=>{
    const fg=i<2;
    b+=rect(x+26,y,86,28,fg?C.task:C.grid,8,fg?0.09:0.04,fg?C.task:C.grid,1);
    b+=circle(x+43,y+14,4,fg?C.task:C.dim,fg?0.88:0.42);
    b+=text(fg?`foreground q${i+1}`:`background q${i+1}`,x+53,y+18,9,fg?C.ink:C.dim,'start',500);
  });
  b+=text('top-K foreground',x+69,385,9,C.task,'middle',600);
  b+=arrow(x+116,278,x+160,278,C.task,2,0.75);
  [222,274].forEach((y,i)=>{
    b+=rect(x+166,y,82,30,C.task,8,0.07+0.08*carry,C.task,1);
    b+=text(`carried q${i+1}`,x+207,y+20,9,C.ink,'middle',500);
    b+=flowDot(x+112,queueY[i]+14,x+166,y+15,carry,C.task,2.2,0.82);
  });
  b+=line(x+118,queueY[2]+14,x+152,queueY[2]+14,C.danger,1.5,0.5,'4 4');
  b+=text('discard',x+138,queueY[2]+8,9,C.danger,'middle',600);
  b+=circle(x+205,350,6+5*refresh,C.metric,0.08+0.22*refresh,C.metric,1);
  b+=text('fresh query',x+205,376,9,C.metric,'middle',600);
  b+=text('enqueue top foreground queries',x+138,426,11,C.task,'middle',600);
  b+=text('background memory is bounded',x+138,449,10,C.muted,'middle',500);
  b+=text('Dense memory retains fields; sparse memory must manage births and aging.',480,500,13,C.metric,'middle',600);
  return svg(b,'BEVDet4D carries a dense bird eye view field, Sparse4D version 2 transforms prior object instances and adds fresh anchors, and StreamPETR retains a bounded queue of foreground queries while adding fresh queries for new objects.');
}

const animations = [
  ['autonomous-perception-vision-encoder.gif', visionFrame],
  ['autonomous-perception-lidar-encoder.gif', lidarFrame],
  ['autonomous-perception-radar-encoder.gif', radarFrame],
  ['autonomous-perception-camera-lifting.gif', liftingFrame],
  ['autonomous-perception-fusion-granularity.gif', fusionFrame],
  ['autonomous-perception-modality-dropout.gif', dropoutFrame],
  ['autonomous-perception-multitask-gradients.gif', multitaskFrame],
  ['autonomous-perception-lidar-training-contracts.gif', lidarContractFrame],
  ['autonomous-perception-temporal-memory.gif', temporalFrame],
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
  const filter = '[0:v]split[a][b];[a]palettegen=max_colors=80:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle';
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
