/**
 * Threaded scaling benchmark — does WASM threads multiply throughput?
 * Loads the pthreads build and rasterizes the same scene across N bands.
 */
import createModule from '../dist/simdpipe-threads.mjs';
import { VERTEX_STRIDE } from '../lib/index.mjs';

const W = 1024, H = 1024;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const M = await createModule();
if (!M._sp_init(W, H)) throw new Error('init failed');
// vertex-color, depth on, persp on
M._sp_set_flags((1 << 0) | (1 << 2));

// deterministic scene: many overlapping mid/large triangles (fill+setup mix)
function mulberry32(seed) {
  return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const NT = 4000;
const rng = mulberry32(7);
const buf = new Float32Array(NT * 3 * VERTEX_STRIDE);
const vtx = (i, x, y, z, r, g, b) => { const o = i * VERTEX_STRIDE; buf[o] = x; buf[o + 1] = y; buf[o + 2] = z; buf[o + 3] = 1; buf[o + 4] = r; buf[o + 5] = g; buf[o + 6] = b; buf[o + 7] = 1; };
const s = 160;
for (let t = 0; t < NT; t++) {
  const cx = rng() * W, cy = rng() * H, z = rng();
  vtx(t * 3 + 0, cx, cy, z, rng(), rng(), rng());
  vtx(t * 3 + 1, cx + (rng() - 0.3) * s, cy + rng() * s, z, rng(), rng(), rng());
  vtx(t * 3 + 2, cx + rng() * s, cy + (rng() - 0.3) * s, z, rng(), rng(), rng());
}
const ptr = M._sp_alloc(buf.length * 4);
M.HEAPF32.set(buf, ptr >> 2);

function frame(nthreads) {
  M._sp_clear(0xff101018, 1.0);          // clear once, serially (avoids band race)
  if (nthreads <= 1) M._sp_draw_triangles_flat(ptr, NT);
  else M._sp_draw_triangles_threaded(ptr, NT, nthreads);
}

function bench(nthreads) {
  for (let i = 0; i < 10; i++) frame(nthreads);   // warmup (spins up pool)
  const times = [];
  for (let i = 0; i < 40; i++) { const t0 = now(); frame(nthreads); times.push(now() - t0); }
  return median(times);
}

console.log(`simdpipe threaded scaling — ${W}x${H}, ${NT} tris, V8 ${process.version}, ${(await import('node:os')).cpus().length} cores\n`);
const base = bench(1);
console.log('threads   ms     speedup');
console.log('-'.repeat(30));
for (const n of [1, 2, 4, 8, 16, 24]) {
  const ms = bench(n);
  console.log(String(n).padEnd(9), ms.toFixed(2).padStart(6), '   ' + (base / ms).toFixed(2) + 'x');
}
