/**
 * simdpipe benchmark harness (Node/V8).
 *
 * Compares the WASM+SIMD core against an idiomatic scalar-JS rasterizer doing
 * the SAME work, across workload families. V8 is the reference engine.
 *
 * Metric: median frame time (ms) over N timed frames after warmup, plus derived
 * Mpix/s (shaded fragments) and Mtri/s. We report the SIMD vs JS speedup — the
 * honest "does WASM+SIMD actually win, and by how much" number.
 *
 * Usage: node bench/run.mjs [--json out.json] [--size WxH] [--frames N]
 */

import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';
import { createBaseline } from './baseline-js.mjs';
import { writeFileSync } from 'node:fs';

// ---- args ----
const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SIZE = getArg('--size', '512x512');
const [W, H] = SIZE.split('x').map(Number);
const FRAMES = parseInt(getArg('--frames', '60'), 10);
const WARMUP = parseInt(getArg('--warmup', '20'), 10);
const JSON_OUT = getArg('--json', null);

const now = () => Number(process.hrtime.bigint()) / 1e6; // ms

function median(a) { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; }

// ---- scene generators (flat vertex buffers) ----
function vtx(buf, i, x, y, z, r, g, b, a = 1) {
  const o = i * VERTEX_STRIDE;
  buf[o] = x; buf[o + 1] = y; buf[o + 2] = z; buf[o + 3] = 1.0;
  buf[o + 4] = r; buf[o + 5] = g; buf[o + 6] = b; buf[o + 7] = a;
  buf[o + 8] = 0; buf[o + 9] = 0;
}

// deterministic PRNG so both renderers get identical geometry
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Big overlapping triangles → fill-rate / overdraw bound. */
function sceneFill(ntris) {
  const rng = mulberry32(1);
  const buf = new Float32Array(ntris * 3 * VERTEX_STRIDE);
  for (let t = 0; t < ntris; t++) {
    const z = rng(); // random depth for overdraw via depth test
    // three corners spread across most of the screen
    vtx(buf, t * 3 + 0, rng() * W * 0.4, rng() * H, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 1, W * 0.3 + rng() * W * 0.7, rng() * H * 0.4, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 2, rng() * W, H * 0.4 + rng() * H * 0.6, z, rng(), rng(), rng());
  }
  return buf;
}

/** Many tiny triangles → setup-bound (the regime software can win). */
function sceneSmallTris(ntris, px = 8) {
  const rng = mulberry32(2);
  const buf = new Float32Array(ntris * 3 * VERTEX_STRIDE);
  for (let t = 0; t < ntris; t++) {
    const cx = rng() * (W - px), cy = rng() * (H - px), z = rng();
    vtx(buf, t * 3 + 0, cx, cy, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 1, cx + px, cy, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 2, cx, cy + px, z, rng(), rng(), rng());
  }
  return buf;
}

/** Mid-size triangles, moderate count → balanced. */
function sceneBalanced(ntris) {
  const rng = mulberry32(3);
  const buf = new Float32Array(ntris * 3 * VERTEX_STRIDE);
  const s = Math.max(16, Math.min(W, H) / 8);
  for (let t = 0; t < ntris; t++) {
    const cx = rng() * (W - s), cy = rng() * (H - s), z = rng();
    vtx(buf, t * 3 + 0, cx, cy, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 1, cx + s, cy + rng() * s, z, rng(), rng(), rng());
    vtx(buf, t * 3 + 2, cx + rng() * s, cy + s, z, rng(), rng(), rng());
  }
  return buf;
}

// ---- timing ----
function timeFrames(drawFn, frames, warmup) {
  for (let i = 0; i < warmup; i++) drawFn();
  const times = [];
  for (let i = 0; i < frames; i++) {
    const t0 = now();
    drawFn();
    times.push(now() - t0);
  }
  return median(times);
}

// ---- run a family on both renderers ----
async function runFamily(name, buf, ntris, flags) {
  const sp = await createRenderer({ width: W, height: H });
  sp.setFlags(flags);
  const js = createBaseline({ width: W, height: H });
  js.setDepthTest(!!(flags & FLAGS.DEPTH_TEST));

  // simdpipe: pre-upload geometry once to linear memory (measure raster, not copy)
  const floats = ntris * 3 * VERTEX_STRIDE;
  const ptr = sp.alloc(floats * 4);
  sp._module.HEAPF32.set(buf.subarray(0, floats), ptr >> 2);
  const spDraw = () => {
    sp.clear(0xff101018, 1.0);
    sp._module._sp_draw_triangles_flat(ptr, ntris);
  };
  const jsDraw = () => {
    js.clear(0xff101018, 1.0);
    js.drawTriangles(buf, ntris);
  };

  const spMs = timeFrames(spDraw, FRAMES, WARMUP);
  const jsMs = timeFrames(jsDraw, FRAMES, WARMUP);

  // shaded-fragment throughput from one fresh frame's stats
  sp.resetStats(); spDraw(); const spStats = sp.stats();
  js.resetStats(); jsDraw(); const jsStats = js.stats();

  const spMpix = spStats.fragShaded / 1e6 / (spMs / 1000);
  const jsMpix = jsStats.fragShaded / 1e6 / (jsMs / 1000);
  const speedup = jsMs / spMs;

  sp.free(ptr);

  return {
    name, ntris,
    simd_ms: +spMs.toFixed(3), js_ms: +jsMs.toFixed(3),
    speedup: +speedup.toFixed(2),
    simd_Mpix_s: +spMpix.toFixed(1), js_Mpix_s: +jsMpix.toFixed(1),
    shaded: spStats.fragShaded, tested: spStats.fragTested,
  };
}

// ---- main ----
const results = [];
const DT = FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT; // vertex-color path (no texture)

console.log(`simdpipe bench — ${W}x${H}, ${FRAMES} frames (warmup ${WARMUP}), V8 ${process.version}\n`);

results.push(await runFamily('fill-rate (200 big tris, overdraw)', sceneFill(200), 200, DT));
results.push(await runFamily('balanced (2k mid tris)', sceneBalanced(2000), 2000, DT));
results.push(await runFamily('small-tris (20k @ 8px)', sceneSmallTris(20000, 8), 20000, DT));
results.push(await runFamily('small-tris (50k @ 4px)', sceneSmallTris(50000, 4), 50000, DT));

// table
const pad = (s, n) => String(s).padEnd(n);
const padr = (s, n) => String(s).padStart(n);
console.log(pad('family', 36), padr('simd ms', 9), padr('js ms', 9), padr('speedup', 9), padr('simd Mpx/s', 12), padr('js Mpx/s', 10));
console.log('-'.repeat(36 + 9 + 9 + 9 + 12 + 10 + 5));
for (const r of results) {
  console.log(
    pad(r.name, 36),
    padr(r.simd_ms, 9), padr(r.js_ms, 9),
    padr(r.speedup + 'x', 9),
    padr(r.simd_Mpix_s, 12), padr(r.js_Mpix_s, 10),
  );
}

const geo = results.reduce((a, r) => a * r.speedup, 1) ** (1 / results.length);
console.log('\ngeomean SIMD-vs-JS speedup:', geo.toFixed(2) + 'x');

if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify({
    engine: process.version, size: { W, H }, frames: FRAMES, results,
    geomean_speedup: +geo.toFixed(2),
  }, null, 2));
  console.log('wrote', JSON_OUT);
}
