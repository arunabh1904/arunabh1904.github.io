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
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" fill-opacity="${alpha(opacity)}" stroke="${stroke}" stroke-width="${sw}"/>`;
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
  return `${rect(x, y, w, h, C.panel, 18, 0.9, accent, 1)}${text(label, x + 18, y + 30, 13, C.muted, 'start', 600, 1, 1.2)}`;
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

function sceneRoad(x, y, w, h) {
  const horizon = y + h * 0.34;
  let out = rect(x, y, w, h, C.bg2, 14, 1, C.grid, 1);
  out += `<path d="M ${x + w * 0.34} ${y + h} L ${x + w * 0.46} ${horizon} L ${x + w * 0.54} ${horizon} L ${x + w * 0.70} ${y + h} Z" fill="${C.panel2}"/>`;
  out += line(x + w * 0.5, horizon + 12, x + w * 0.52, y + h, C.muted, 2, 0.32, '10 10');
  out += line(x + w * 0.42, horizon, x + w * 0.17, y + h, C.grid, 2, 0.8);
  out += line(x + w * 0.58, horizon, x + w * 0.86, y + h, C.grid, 2, 0.8);
  out += circle(x + w * 0.18, horizon - 4, 28, C.camera, 0.05);
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

function svg(body, description) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="${esc(description)}">
  <defs>
    <radialGradient id="bg-glow" cx="50%" cy="0%" r="92%"><stop offset="0%" stop-color="#173248"/><stop offset="55%" stop-color="${C.bg}"/><stop offset="100%" stop-color="#060b12"/></radialGradient>
    <filter id="soft-glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="7"/></filter>
  </defs>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="url(#bg-glow)"/>
  <circle cx="870" cy="90" r="190" fill="${C.camera}" fill-opacity="0.022"/>
  <circle cx="90" cy="500" r="210" fill="${C.metric}" fill-opacity="0.018"/>
  ${body}
  </svg>`;
}

function visionFrame(frame) {
  const t = cycle(frame);
  const travel = phase(t, 0.08, 0.56);
  const settle = phase(t, 0.55, 0.78);
  const pulse = 0.65 + 0.35 * ping(t * 2);
  let b = header('01', 'vision encoder', 'Keep the small signal');
  b += sceneRoad(52, 126, 360, 340);
  b += text('camera', 72, 154, 13, C.muted, 'start', 600, 1, 1);
  const actorX = 52 + 360 * 0.555;
  const actorY = 126 + 340 * 0.405;
  b += rect(actorX - 5, actorY - 8, 10, 16, C.camera, 3, 0.28 + pulse * 0.35, C.camera, 1);
  b += circle(actorX, actorY, 18 + 3 * pulse, C.camera, 0.05 + 0.05 * pulse);
  for (let i = 1; i < 8; i++) b += line(52 + (360 * i) / 8, 126, 52 + (360 * i) / 8, 466, C.camera, 1, 0.04);
  for (let j = 1; j < 6; j++) b += line(52, 126 + (340 * j) / 6, 412, 126 + (340 * j) / 6, C.camera, 1, 0.04);

  b += panel(470, 126, 438, 142, 'COARSE ONLY', C.grid);
  b += panel(470, 286, 438, 180, 'MULTI-SCALE', C.camera);
  const coarseXs = [520, 628, 738, 842];
  coarseXs.forEach((x, i) => {
    const s = 74 - i * 12;
    b += rect(x, 177 + i * 3, s, s * 0.65, C.panel2, 8, 1, C.grid, 1);
    for (let k = 1; k < 4; k++) b += line(x + (s * k) / 4, 180 + i * 3, x + (s * k) / 4, 174 + i * 3 + s * 0.65, C.grid, 1, 0.55);
    if (i < 3) b += arrow(x + s + 7, 198, coarseXs[i + 1] - 8, 198, C.dim, 1.5, 0.7);
  });
  b += flowDot(actorX, actorY, 535, 200, travel, C.camera, 4, 1 - settle * 0.85);
  b += flowDot(535, 200, 846, 199, clamp((travel - 0.34) / 0.66), C.camera, 4, 0.85 - settle * 0.8);
  b += circle(862, 202, 3 + 7 * (1 - settle), C.camera, 0.22 * (1 - settle));

  const levels = [
    { x: 520, y: 342, w: 118, h: 78, o: 1 },
    { x: 646, y: 354, w: 96, h: 64, o: 0.82 },
    { x: 752, y: 368, w: 72, h: 48, o: 0.62 },
  ];
  levels.forEach((q, i) => {
    b += rect(q.x, q.y, q.w, q.h, C.panel2, 9, 1, C.camera, 0.9);
    for (let k = 1; k < 5; k++) b += line(q.x + (q.w * k) / 5, q.y + 2, q.x + (q.w * k) / 5, q.y + q.h - 2, C.camera, 1, 0.16);
    for (let k = 1; k < 3; k++) b += line(q.x + 2, q.y + (q.h * k) / 3, q.x + q.w - 2, q.y + (q.h * k) / 3, C.camera, 1, 0.16);
    if (i < 2) b += arrow(q.x + q.w + 6, q.y + q.h * 0.5, levels[i + 1].x - 6, levels[i + 1].y + levels[i + 1].h * 0.5, C.camera, 1.5, 0.72);
  });
  const fx = mix(actorX, 579, travel);
  const fy = mix(actorY, 380, travel);
  b += glowCircle(fx, fy, 12, C.camera, 0.95);
  b += arrow(824, 392, 870, 392, C.metric, 2, 0.85);
  b += rect(858, 370, 30, 42, C.metric, 6, 0.16 + 0.18 * settle, C.metric, 1);
  b += text('retained', 873, 438, 12, C.metric, 'middle', 500, 0.9, 0.6);
  return svg(b, 'A tiny distant actor disappears in a coarse-only vision encoder but remains visible in a multiscale feature pyramid.');
}

function lidarFrame(frame) {
  const t = cycle(frame);
  const build = phase(t, 0.04, 0.40);
  const move = phase(t, 0.34, 0.70);
  const pulse = 0.65 + 0.35 * ping(t * 1.5);
  let b = header('02', 'lidar encoder', 'Choose when height disappears');
  b += panel(52, 126, 310, 342, 'SPARSE RETURNS', C.lidar);
  const lower = [[90,410],[118,396],[146,419],[176,389],[204,414],[237,393],[270,422],[302,397],[326,414],[116,344],[154,358],[215,349],[286,360]];
  const upper = [[126,274],[158,258],[190,282],[222,263],[254,278],[286,257],[315,276]];
  [...lower, ...upper].forEach(([x,y], i) => {
    const o = 0.55 + 0.45 * Math.sin((i * 1.7 + t * Math.PI * 2)) ** 2;
    b += glowCircle(x, y, 6, i < lower.length ? C.lidar : C.camera, o);
  });
  b += path('M 78 435 L 168 348 L 337 378', C.grid, 2, 0.8);
  b += path('M 105 313 L 173 248 L 325 258', C.grid, 2, 0.6);
  b += vehicle(203, 431, C.ink, 0.75, 0.55);

  b += panel(400, 126, 508, 150, 'PILLARS  →  EARLY BEV', C.grid);
  const pillarXs = [462, 510, 558, 606, 654, 702];
  pillarXs.forEach((x, i) => {
    const hi = 25 + (i % 3) * 13;
    const yTop = mix(170 - hi, 204, move);
    const yBot = 219;
    b += rect(x, yTop, 28, Math.max(8, yBot - yTop), C.lidar, 5, 0.08 + 0.14 * build, C.lidar, 1);
    if (i === 2 || i === 3) b += circle(x + 14, mix(170, 210, move), 5, C.camera, 0.75 * (1 - move));
  });
  b += arrow(760, 204, 816, 204, C.muted, 2, 0.75);
  b += bevGrid(824, 168, 60, 72, 1, C.grid);
  b += circle(848, 203, 8 + 2 * pulse, C.lidar, 0.25);
  b += circle(848, 203, 4, C.camera, 0.55 * (1 - move));
  b += text('fast · planar', 856, 258, 12, C.muted, 'end', 500, 0.9, 0.4);

  b += panel(400, 294, 508, 174, 'SPARSE 3D  →  LATE BEV', C.lidar);
  const vox = [[462,405,C.lidar],[510,385,C.lidar],[558,412,C.lidar],[606,392,C.lidar],[520,340,C.camera],[574,326,C.camera],[632,346,C.camera],[686,401,C.lidar]];
  vox.forEach(([x,y,c], i) => {
    b += cube(x, y, 24, c, build);
    if (i > 0 && i < 7 && (i % 2 === 0 || i === 5)) b += line(vox[i-1][0]+18, vox[i-1][1]+8, x+12, y+8, c, 1.5, 0.16 + 0.42 * pulse * build);
  });
  b += arrow(746, 383, 810, 383, C.lidar, 2, 0.8);
  b += bevGrid(824, 343, 60, 72, 1, C.lidar);
  b += circle(844, 395, 6, C.lidar, 0.65);
  b += circle(858, 365, 6, C.camera, 0.65);
  b += text('height survives', 856, 448, 12, C.lidar, 'end', 500, 0.9, 0.4);
  return svg(b, 'LiDAR points are compared through early pillar collapse and sparse 3D encoding that preserves overpass height before bird eye view compression.');
}

function radarFrame(frame) {
  const t = cycle(frame);
  const travel = phase(t, 0.08, 0.52);
  const erase = phase(t, 0.48, 0.72);
  const pulse = 0.55 + 0.45 * ping(t * 2);
  let b = header('03', 'radar encoder', 'Keep the measurement, not only the dot');
  b += panel(52, 126, 310, 342, 'POLAR RETURNS', C.radar);
  const ox = 203, oy = 430;
  [58,105,152,199].forEach((r) => b += path(`M ${ox-r} ${oy} A ${r} ${r} 0 0 1 ${ox+r} ${oy}`, C.grid, 1.2, 0.58));
  [-60,-30,0,30,60].forEach((a) => {
    const rad = (a-90) * Math.PI / 180;
    b += line(ox, oy, ox + 210*Math.cos(rad), oy + 210*Math.sin(rad), C.grid, 1.2, 0.48);
  });
  const returns = [
    {x:128,y:318,v:22,rcs:6},{x:170,y:260,v:36,rcs:9},{x:237,y:292,v:18,rcs:5},{x:282,y:232,v:44,rcs:11},{x:305,y:345,v:14,rcs:4}
  ];
  returns.forEach((p,i) => {
    b += circle(p.x,p.y,7 + p.rcs*0.18,C.radar,0.25,C.radar,1);
    b += arrow(p.x,p.y,p.x + p.v*0.55,p.y-(i%2?7:2),C.metric,2,0.78);
    b += circle(p.x,p.y,15 + (i%3)*4,C.radar,0.04 + 0.035*pulse,'none');
  });
  b += vehicle(ox, 442, C.ink, 0.8, 0.48);

  b += panel(400, 126, 508, 150, 'RASTERIZE EARLY', C.grid);
  b += bevGrid(438, 168, 180, 78, 1, C.grid);
  returns.slice(0,4).forEach((p,i) => {
    const tx = 462 + i*39, ty = 196 + (i%2)*20;
    b += flowDot(p.x,p.y,tx,ty,travel,C.radar,3,0.8);
    b += circle(tx,ty,9,C.radar,0.20 + 0.08*(1-erase));
  });
  b += arrow(648,207,716,207,C.dim,2,0.7);
  b += rect(736,173,140,68,C.panel2,12,1,C.grid,1);
  b += circle(770,207,13,C.radar,0.14,C.radar,1);
  b += circle(810,207,13,C.radar,0.14,C.radar,1);
  b += circle(850,207,13,C.radar,0.14,C.radar,1);
  b += text('where', 806, 258, 12, C.muted, 'middle', 500, 0.9, 0.5);

  b += panel(400, 294, 508, 174, 'ATTRIBUTE-AWARE TOKENS', C.radar);
  const tokenXs = [454,526,598,670];
  tokenXs.forEach((x,i) => {
    const y=350+(i%2)*40;
    b += rect(x,y,54,34,C.radar,10,0.08+0.06*pulse,C.radar,1);
    b += circle(x+14,y+17,4+i,C.radar,0.9);
    b += arrow(x+27,y+22,x+42+i*2,y+12,C.metric,1.6,0.8);
    b += flowDot(returns[i].x,returns[i].y,x+14,y+17,travel,C.radar,3,0.9);
  });
  b += arrow(744,370,802,370,C.radar,2,0.8);
  b += rect(818,332,66,78,C.panel2,12,1,C.radar,1);
  b += circle(850,366,14+3*pulse,C.radar,0.13,C.radar,1);
  b += arrow(838,390,866,346,C.metric,2.5,0.9);
  b += text('where + motion', 856, 448, 12, C.radar, 'end', 500, 0.9, 0.4);
  return svg(b, 'Radar returns lose Doppler and confidence when rasterized too early, while attribute-aware tokens preserve range, motion, strength, age, and uncertainty.');
}

function liftingFrame(frame) {
  const t = cycle(frame);
  const q = phase(t, 0.06, 0.56);
  const settle = phase(t, 0.52, 0.74);
  const pulse = 0.55 + 0.45 * ping(t * 1.4);
  let b = header('04', 'camera to 3D', 'Three ways to ask “where?”');
  const cards = [
    {x:34,label:'LIFT + SPLAT',cost:'rays × depth',accent:C.camera},
    {x:342,label:'OBJECT QUERIES',cost:'queries × samples',accent:C.metric},
    {x:650,label:'BEV QUERIES',cost:'cells × samples',accent:C.task},
  ];
  cards.forEach((c) => {
    b += panel(c.x,126,276,346,c.label,c.accent);
    b += text(c.cost,c.x+138,450,12,C.muted,'middle',500,0.9,0.6);
    b += rect(c.x+24,166,228,92,C.bg2,10,1,C.grid,1);
    b += path(`M ${c.x+44} 246 L ${c.x+112} 190 L ${c.x+226} 190 L ${c.x+244} 246`,C.grid,1.5,0.8);
    b += bevGrid(c.x+24,318,228,100,1,c.accent);
    b += arrow(c.x+138,272,c.x+138,305,c.accent,2,0.75);
  });
  // Lift and splat: distribute a pixel along a ray, then pool its probability mass.
  const ax=34;
  b += glowCircle(ax+128,215,10,C.camera,0.9);
  const depths=[[ax+78,235],[ax+110,220],[ax+145,206],[ax+182,193],[ax+218,185]];
  depths.forEach(([x,y],i)=>{
    const w=[0.18,0.38,1,0.52,0.2][i];
    const px=mix(ax+128,x,q), py=mix(215,y,q);
    b += circle(px,py,4+5*w,C.camera,0.22+0.55*w);
    const bx=ax+52+i*39, by=342+(i%2)*22;
    b += flowDot(x,y,bx,by,clamp((q-0.4)/0.6),C.camera,3,0.45+0.4*w);
    b += rect(bx-10,by-8,20,16,C.camera,4,0.04+0.16*w*settle,C.camera,0.8);
  });
  b += text('probability mass',ax+138,296,11,C.camera,'middle',500,0.82,0.4);

  // Object queries: only a few hypotheses project into the image.
  const bx=342;
  const queries=[[bx+72,380],[bx+142,350],[bx+208,390]];
  queries.forEach(([x,y],i)=>{
    const sx=[bx+82,bx+146,bx+212][i], sy=[235,208,226][i];
    b += circle(x,y,9,C.metric,0.14,C.metric,1);
    b += path(`M ${x} ${y} Q ${x} 292 ${sx} ${sy}`,C.metric,1.6,0.22+0.52*q,'none','5 5');
    b += flowDot(x,y,sx,sy,q,C.metric,3,0.9);
    b += circle(sx,sy,5+3*pulse,C.metric,0.18+0.25*q);
  });
  b += circle(bx+244,236,5,C.danger,0.7);
  b += text('support only',bx+138,296,11,C.metric,'middle',500,0.82,0.4);

  // BEV queries: a dense metric state, sparse retrieval from images.
  const cx=650;
  for(let row=0;row<3;row++) for(let col=0;col<6;col++){
    const x=cx+48+col*33, y=345+row*28;
    const active=(col+row)%4===0;
    b += circle(x,y,active?5:3,C.task,active?0.75:0.28);
    if(active){
      const sx=cx+58+col*27, sy=232-row*12;
      b += path(`M ${x} ${y} Q ${cx+138} 292 ${sx} ${sy}`,C.task,1.4,0.18+0.46*q,'none','4 5');
      b += flowDot(x,y,sx,sy,q,C.task,3,0.85);
    }
  }
  b += text('dense state · sparse read',cx+138,296,11,C.task,'middle',500,0.82,0.35);
  return svg(b, 'Dense lift and splat assigns image evidence across depth bins, object queries retrieve only around object hypotheses, and BEV queries maintain a dense metric state with sparse image sampling.');
}

function fusionFrame(frame) {
  const t = cycle(frame);
  const flow = phase(t,0.05,0.64);
  const pulse = 0.55+0.45*ping(t*1.4);
  let b=header('05','fusion','Fuse where the output lives');
  const rows=[
    {y:142,label:'POINT',accent:C.lidar,out:'point labels'},
    {y:260,label:'QUERY',accent:C.metric,out:'actors'},
    {y:378,label:'BEV',accent:C.task,out:'scene fields'},
  ];
  rows.forEach((r,ri)=>{
    b += rect(52,r.y,856,94,C.panel,17,0.88,r.accent,1);
    b += text(r.label,76,r.y+30,12,C.muted,'start',600,1,1.3);
    b += sensorChip(150,r.y+30,'camera',C.camera,1);
    b += sensorChip(254,r.y+30,'LiDAR',C.lidar,1);
    b += sensorChip(358,r.y+30,'radar',C.radar,ri===0?0.35:1);
    const mx=574,my=r.y+48;
    b += circle(mx,my,24,r.accent,0.10+0.06*pulse,r.accent,1.2);
    [[196,C.camera],[300,C.lidar],[404,C.radar]].forEach(([sx,c],i)=>{
      const o=(ri===0&&i===2)?0.2:0.7;
      b += line(sx+46,r.y+47,mx-26,my,c,1.6,o);
      b += flowDot(sx+46,r.y+47,mx-22,my,clamp(flow-i*0.05),c,3,o);
    });
    b += arrow(mx+26,my,696,my,r.accent,2,0.8);
    if(ri===0){
      [0,1,2,3,4].forEach(i=>b+=circle(720+i*26,my+(i%2?8:-6),6,i%2?C.camera:C.lidar,0.75));
    }else if(ri===1){
      b += circle(756,my,15,C.metric,0.12,C.metric,1);
      b += rect(739,my-12,34,24,C.metric,7,0.08,C.metric,1);
    }else{
      b += bevGrid(706,r.y+18,112,60,1,C.task);
      b += rect(742,r.y+33,34,22,C.task,5,0.16+0.06*pulse,C.task,1);
    }
    b += text(r.out,874,r.y+54,12,r.accent,'end',500,0.9,0.4);
  });
  return svg(b,'Point fusion paints point-aligned outputs, query fusion gathers sensor evidence around actor hypotheses, and bird eye view fusion supports dense scene outputs.');
}

function dropoutFrame(frame) {
  const t=cycle(frame);
  const slot=Math.floor(t*5)%5;
  const local=(t*5)%1;
  const activeSets=[[1,1,1],[1,0,1],[0,1,1],[1,1,0],[0.35,1,1]];
  const active=activeSets[slot];
  const transition=0.5-0.5*Math.cos(local*Math.PI*2);
  let b=header('06','training','Train the deployment modes');
  const cards=[
    {x:52,label:'ALWAYS COMPLETE',good:false,accent:C.danger},
    {x:500,label:'MODALITY DROPOUT',good:true,accent:C.metric},
  ];
  cards.forEach((c,ci)=>{
    b += panel(c.x,126,408,342,c.label,c.accent);
    const sensorY=174;
    b += sensorChip(c.x+28,sensorY,'camera',C.camera,ci===0?1:active[0]);
    b += sensorChip(c.x+158,sensorY,'LiDAR',C.lidar,ci===0?1:active[1]);
    b += sensorChip(c.x+288,sensorY,'radar',C.radar,ci===0?1:active[2]);
    if(ci===1){
      [0,1,2,3,4].forEach(i=>{
        const xx=c.x+58+i*66;
        const set=activeSets[i];
        b += rect(xx,242,52,24,C.panel2,12,1,i===slot?C.metric:C.grid,i===slot?1.5:1);
        [0,1,2].forEach(j=>b+=circle(xx+13+j*13,254,3,[C.camera,C.lidar,C.radar][j],set[j]));
      });
    }else{
      [0,1,2,3,4].forEach(i=>{
        const xx=c.x+58+i*66;
        b += rect(xx,242,52,24,C.panel2,12,1,C.grid,1);
        [0,1,2].forEach(j=>b+=circle(xx+13+j*13,254,3,[C.camera,C.lidar,C.radar][j],1));
      });
    }
    const fx=c.x+204,fy=327;
    b += circle(fx,fy,42,c.accent,0.08,c.accent,1.2);
    b += text('Σ',fx,fy+9,28,C.ink,'middle',500,0.9);
    [0,1,2].forEach(j=>{
      const sx=c.x+74+j*130;
      const strength=ci===0?active[j]:active[j]/Math.max(0.01,active.reduce((a,v)=>a+v,0));
      b += line(sx+46,208,fx,fy-42,[C.camera,C.lidar,C.radar][j],2,ci===0?active[j]:0.25+strength*1.7);
      if(active[j] < 0.1) b += line(sx+20,184,sx+72,218,C.danger,2.5,0.9);
      if(active[j] < 0.1) b += line(sx+72,184,sx+20,218,C.danger,2.5,0.9);
    });
    const stable=ci===1?0.92:active.reduce((a,v)=>a+v,0)/3;
    const wobble=ci===1?0:Math.sin(local*Math.PI*2)*10*(1-stable);
    b += arrow(fx,fy+46,fx,390,c.accent,2,0.8);
    b += bevGrid(c.x+118+wobble,390,172,56,1,c.accent);
    b += rect(c.x+170+wobble,405,54,25,c.accent,6,0.06+0.15*stable,c.accent,1);
    b += circle(c.x+204,418,28+8*(1-stable)*transition,c.accent,0.04+0.04*stable);
  });
  return svg(b,'A model trained only with complete sensors becomes unstable when a stream disappears, while modality dropout cycles supported sensor combinations and renormalizes fusion over the streams that remain.');
}

function multitaskFrame(frame){
  const t=cycle(frame);
  const pulse=ping(t*1.7);
  const draw=phase(t,0.04,0.55);
  let b=header('07','multi-task learning','Share features. Control the gradients.');
  const cards=[
    {x:52,label:'ONE UNMEASURED SUM',accent:C.danger,controlled:false},
    {x:500,label:'MEASURE · BALANCE · SPLIT',accent:C.task,controlled:true},
  ];
  cards.forEach((c)=>{
    b+=panel(c.x,126,408,342,c.label,c.accent);
    const trunkX=c.x+156+(c.controlled?0:Math.sin(t*Math.PI*4)*6);
    b+=rect(trunkX,266,96,78,C.panel2,14,1,c.accent,1.2);
    b+=text('shared',trunkX+48,298,13,C.ink,'middle',500,0.9);
    b+=text('geometry',trunkX+48,318,13,C.muted,'middle',500,0.9);
    const heads=[
      {x:c.x+52,y:182,label:'boxes',color:C.camera},
      {x:c.x+156,y:182,label:'lanes',color:C.metric},
      {x:c.x+260,y:182,label:'occupancy',color:C.task},
    ];
    heads.forEach((h,i)=>{
      b+=rect(h.x,h.y,96,46,h.color,12,0.06,h.color,1);
      b+=text(h.label,h.x+48,h.y+29,12,C.ink,'middle',500,0.9);
      const tx=trunkX+22+i*26,ty=266;
      b+=line(h.x+48,h.y+46,tx,ty,h.color,c.controlled?1.6:2.8-i*0.3,c.controlled?0.65:0.88);
      b+=flowDot(h.x+48,h.y+46,tx,ty,draw,h.color,3,0.85);
      if(c.controlled){
        b+=rect(mix(h.x+48,tx,0.60)-12,mix(h.y+46,ty,0.60)-7,24,14,C.panel2,7,1,h.color,1);
      }
    });
    if(!c.controlled){
      b+=arrow(trunkX+48,305,trunkX+14,365,C.camera,2.5,0.45+0.4*pulse);
      b+=arrow(trunkX+48,305,trunkX+90,371,C.task,3.6,0.55+0.35*(1-pulse));
      b+=arrow(trunkX+48,305,trunkX+50,382,C.metric,1.4,0.35);
      b+=text('conflict',trunkX+48,414,12,C.danger,'middle',500,0.9,0.5);
    }else{
      b+=arrow(trunkX+48,344,trunkX+48,390,C.task,2.4,0.86);
      b+=rect(c.x+126,390,156,42,C.task,12,0.07,C.task,1);
      b+=text('stable shared update',c.x+204,416,12,C.ink,'middle',500,0.9);
      b+=circle(trunkX+48,305,46+3*pulse,C.task,0.025+0.025*pulse);
    }
  });
  return svg(b,'Unmeasured task losses pull a shared perception trunk with conflicting magnitudes, while measured gradients, balanced weights, and selective splits stabilize the shared update.');
}

function lidarContractFrame(frame){
  const t=cycle(frame);
  const train=phase(t,0.02,0.44);
  const drive=phase(t,0.48,0.86);
  let b=header('08','privileged sensing','Does LiDAR leave the graph?');
  const cols=[
    {x:34,label:'DEPTH LABELS',accent:C.camera},
    {x:342,label:'DEPTH COMPLETION',accent:C.lidar},
    {x:650,label:'DISTILLATION',accent:C.metric},
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
  return svg(b,'LiDAR can provide training-only depth labels, remain a runtime input for depth completion, or supervise a cheaper student through distillation; the deployment contracts are different.');
}

function temporalFrame(frame){
  const t=cycle(frame);
  const carry=phase(t,0.04,0.58);
  const refresh=phase(t,0.55,0.80);
  const pulse=0.55+0.45*ping(t*1.5);
  let b=header('09','temporal state','Choose what crosses the frame boundary');
  const cols=[
    {x:34,label:'DENSE SCENE',cost:'grid × time',accent:C.camera},
    {x:342,label:'SPARSE OBJECTS',cost:'queries × state',accent:C.metric},
    {x:650,label:'HYBRID',cost:'short field + long actors',accent:C.task},
  ];
  cols.forEach((c)=>{
    b+=panel(c.x,126,276,342,c.label,c.accent);
    b+=text(c.cost,c.x+138,449,12,C.muted,'middle',500,0.9,0.45);
    b+=text('t − 1',c.x+54,178,11,C.muted,'middle',500,0.8,0.5);
    b+=text('t',c.x+222,178,11,C.muted,'middle',500,0.8,0.5);
    b+=arrow(c.x+104,293,c.x+170,293,c.accent,2,0.68);
  });

  // Dense BEV: the full field is warped forward; static structure aligns, actors still need motion correction.
  let x=34;
  b+=bevGrid(x+24,194,108,194,1,C.camera);
  b+=bevGrid(x+144,194,108,194,1,C.camera);
  b+=path(`M ${x+54} 368 Q ${x+78} 280 ${x+104} 212`,C.metric,5,0.20);
  b+=path(`M ${x+174} 368 Q ${x+198} 280 ${x+224} 212`,C.metric,5,0.20+0.18*refresh);
  const denseOld={x:x+76,y:300};
  const denseWarp={x:x+196,y:292};
  const denseTrue={x:x+218,y:270};
  b+=rect(denseOld.x-13,denseOld.y-8,26,16,C.camera,5,0.16,C.camera,1);
  b+=path(`M ${denseOld.x} ${denseOld.y} Q ${x+140} 265 ${denseWarp.x} ${denseWarp.y}`,C.camera,1.6,0.30+0.40*carry,'none','5 5');
  b+=rect(mix(denseOld.x,denseWarp.x,carry)-13,mix(denseOld.y,denseWarp.y,carry)-8,26,16,C.camera,5,0.10,C.camera,1);
  b+=rect(denseTrue.x-13,denseTrue.y-8,26,16,C.metric,5,0.08+0.14*refresh,C.metric,1);
  b+=line(denseWarp.x,denseWarp.y,denseTrue.x,denseTrue.y,C.danger,1.5,0.4+0.25*pulse,'4 4');
  b+=text('warp full field',x+138,416,11,C.camera,'middle',500,0.84,0.35);

  // Sparse recurrence: carry selected actor state, reserve a new query for a birth.
  x=342;
  const oldActors=[[x+62,250],[x+92,330]];
  const newActors=[[x+184,238],[x+216,312]];
  oldActors.forEach(([ox,oy],i)=>{
    b+=circle(ox,oy,12,C.metric,0.12,C.metric,1);
    b+=rect(ox-10,oy-6,20,12,C.metric,4,0.10,C.metric,1);
    const [nx,ny]=newActors[i];
    b+=path(`M ${ox} ${oy} Q ${x+138} ${oy-16} ${nx} ${ny}`,C.metric,1.8,0.28+0.45*carry,'none','5 5');
    b+=circle(mix(ox,nx,carry),mix(oy,ny,carry),8,C.metric,0.16,C.metric,1);
  });
  const birthX=x+194,birthY=366;
  b+=circle(birthX,birthY,6+5*refresh,C.task,0.08+0.22*refresh,C.task,1);
  b+=path(`M ${x+168} 388 Q ${x+180} 372 ${birthX} ${birthY}`,C.task,1.6,0.22+0.50*refresh,'none','4 5');
  b+=text('carry actors · birth queries',x+138,416,11,C.metric,'middle',500,0.84,0.35);

  // Hybrid: short dense context seeds and supports longer sparse tracks.
  x=650;
  b+=bevGrid(x+24,194,108,194,0.72,C.task);
  b+=bevGrid(x+144,194,108,194,0.72,C.task);
  b+=path(`M ${x+44} 370 Q ${x+72} 270 ${x+112} 214`,C.metric,4,0.15);
  b+=path(`M ${x+164} 370 Q ${x+192} 270 ${x+232} 214`,C.metric,4,0.15+0.12*refresh);
  const hybridOld=[[x+62,260],[x+92,324]];
  const hybridNew=[[x+182,246],[x+220,302]];
  hybridOld.forEach(([ox,oy],i)=>{
    const [nx,ny]=hybridNew[i];
    b+=circle(ox,oy,9,C.task,0.14,C.task,1);
    b+=path(`M ${ox} ${oy} Q ${x+138} ${oy-12} ${nx} ${ny}`,C.task,1.7,0.30+0.48*carry,'none','5 5');
    b+=circle(mix(ox,nx,carry),mix(oy,ny,carry),7,C.task,0.18,C.task,1);
  });
  const hybridBirthX=x+186,hybridBirthY=352;
  b+=circle(hybridBirthX,hybridBirthY,5+4*refresh,C.metric,0.08+0.22*refresh,C.metric,1);
  b+=line(x+156,hybridBirthY+20,hybridBirthX,hybridBirthY,C.metric,1.5,0.25+0.45*refresh,'4 4');
  b+=text('discover dense · remember sparse',x+138,416,11,C.task,'middle',500,0.84,0.25);
  return svg(b,'Dense temporal memory warps a complete bird eye view field, sparse recurrence carries selected object queries and creates new birth queries, and a hybrid keeps short dense context with longer sparse actor state.');
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
