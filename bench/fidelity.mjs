/**
 * Fidelity-ladder benchmark — the core simdpipe thesis: trade fidelity for speed.
 *
 * Same scene (overlapping textured triangles), rendered at descending fidelity
 * tiers. Shows the fps you buy by turning expensive work off. This is the
 * "we lose fidelity but we WILL be fast" claim, quantified.
 *
 * Knobs measured (those wired in Phase 0):
 *   - perspective-correct interpolation  (PERSP_CORRECT)  → affine
 *   - depth test/write                    (DEPTH_TEST)     → off
 *   - texture sample                      (TEXTURE)        → flat vertex color
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';

const W = 512, H = 512;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const r = await createRenderer({ width: W, height: H });

// bind a texture
const TW = 64, TH = 64;
const tex = new Uint32Array(TW * TH);
for (let i = 0; i < tex.length; i++) { const x = i % TW, y = (i / TW) | 0; const on = ((x >> 3) + (y >> 3)) & 1; tex[i] = on ? 0xffe0a040 : 0xff4060e0; }
r.bindTexture(new Uint8Array(tex.buffer), TW, TH);

// scene: overlapping mid triangles with uv + depth
function mulberry32(s) { return function () { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
// shade-bound scene: few BIG overlapping triangles (per-pixel cost dominates,
// which is where filtering/texture/perspective knobs actually matter)
const NT = 80, rng = mulberry32(11);
const buf = new Float32Array(NT * 3 * VERTEX_STRIDE);
const vtx = (i, x, y, z, R, G, B, u, v) => { const o = i * VERTEX_STRIDE; buf[o] = x; buf[o + 1] = y; buf[o + 2] = z; buf[o + 3] = 1; buf[o + 4] = R; buf[o + 5] = G; buf[o + 6] = B; buf[o + 7] = 1; buf[o + 8] = u; buf[o + 9] = v; };
for (let t = 0; t < NT; t++) {
  const z = rng();
  vtx(t * 3 + 0, rng() * W * 0.5, rng() * H, z, rng(), rng(), rng(), 0, 0);
  vtx(t * 3 + 1, W * 0.4 + rng() * W * 0.6, rng() * H * 0.5, z, rng(), rng(), rng(), 4, 0);
  vtx(t * 3 + 2, rng() * W, H * 0.4 + rng() * H * 0.6, z, rng(), rng(), rng(), 0, 4);
}
const ptr = r.alloc(buf.length * 4);
r._module.HEAPF32.set(buf, ptr >> 2);

function bench(flags) {
  r.setFlags(flags);
  const frame = () => { r.clear(0xff101018, 1.0); r._module._sp_draw_triangles_flat(ptr, NT); };
  for (let i = 0; i < 15; i++) frame();
  const times = []; for (let i = 0; i < 50; i++) { const t0 = now(); frame(); times.push(now() - t0); }
  return median(times);
}

const tiers = [
  ['full: bilinear texture + persp + depth',  FLAGS.TEXTURE | FLAGS.BILINEAR | FLAGS.PERSP_CORRECT | FLAGS.DEPTH_TEST],
  ['bilinear → nearest (1 tap vs 4)',         FLAGS.TEXTURE | FLAGS.PERSP_CORRECT | FLAGS.DEPTH_TEST],
  ['nearest + drop perspective → affine',     FLAGS.TEXTURE | FLAGS.DEPTH_TEST],
  ['drop texture → flat vertex color',        FLAGS.PERSP_CORRECT | FLAGS.DEPTH_TEST],
  ['cheapest: affine vertex color',           FLAGS.DEPTH_TEST],
];

console.log(`simdpipe fidelity ladder — ${W}x${H}, ${NT} textured tris, V8 ${process.version}\n`);
const base = bench(tiers[0][1]);
console.log('tier'.padEnd(46), 'ms'.padStart(8), 'fps'.padStart(7), 'vs full'.padStart(9));
console.log('-'.repeat(46 + 8 + 7 + 9 + 3));
for (const [name, flags] of tiers) {
  const ms = bench(flags);
  console.log(name.padEnd(46), ms.toFixed(3).padStart(8), (1000 / ms).toFixed(0).padStart(7), (base / ms).toFixed(2).padStart(8) + 'x');
}
console.log('\n→ each fidelity drop buys speed; pick your point on the curve.');
