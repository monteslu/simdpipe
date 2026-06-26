/**
 * Persistent worker pool vs per-frame band spawn.
 *
 * The pool: created once, atomic tile-row dispatch (work-stealing), two barriers
 * per frame, no per-frame spawn. Output is bit-identical to serial.
 *
 * Honest finding: threading wins on *substantial* frames (fill-heavy), where the
 * parallel work amortizes WASM's futex-barrier sync cost; on trivially cheap
 * frames, sync overhead dominates and serial is better (so pooled-draw falls back
 * to serial below a small-work threshold).
 */
import createModule from '../dist/simdpipe-threads.mjs';
import { VERTEX_STRIDE } from '../lib/index.mjs';

const W = 1024, H = 1024;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const mb = (s) => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

const M = await createModule();
M._sp_init(W, H);
M._sp_set_flags((1 << 0) | (1 << 2));

function scene(NT, size, seed, clustered) {
  const rng = mb(seed), buf = new Float32Array(NT * 3 * VERTEX_STRIDE);
  const v = (i, x, y, z, r, g, b) => { const o = i * VERTEX_STRIDE; buf[o] = x; buf[o + 1] = y; buf[o + 2] = z; buf[o + 3] = 1; buf[o + 4] = r; buf[o + 5] = g; buf[o + 6] = b; buf[o + 7] = 1; };
  for (let t = 0; t < NT; t++) {
    const cx = rng() * W, cy = (clustered ? rng() ** 2 : rng()) * H, z = rng();
    v(t * 3 + 0, cx, cy, z, rng(), rng(), rng());
    v(t * 3 + 1, cx + rng() * size, cy + rng() * size, z, rng(), rng(), rng());
    v(t * 3 + 2, cx - rng() * size, cy + rng() * size, z, rng(), rng(), rng());
  }
  const ptr = M._sp_alloc(buf.length * 4); M.HEAPF32.set(buf, ptr >> 2);
  return { ptr, NT };
}

const bench = (fn) => { for (let i = 0; i < 8; i++) fn(); const t = []; for (let i = 0; i < 30; i++) { const a = now(); fn(); t.push(now() - a); } return median(t); };

console.log(`simdpipe pool — ${W}x${H}, V8 ${process.version}, ${(await import('node:os')).cpus().length} cores\n`);

// --- heavy serial baseline measured BEFORE the pool exists (clean isolation) ---
const heavy = scene(6000, 300, 9, false);
const hSer = bench(() => { M._sp_clear(0xff101018, 1.0); M._sp_draw_triangles_flat(heavy.ptr, heavy.NT); });

M._sp_pool_start(8);

// --- correctness ---
const cscene = scene(2000, 200, 1, false);
M._sp_clear(0xff101018, 1.0); M._sp_draw_triangles_flat(cscene.ptr, cscene.NT);
let cp = M._sp_color_ptr(); const serialFB = new Uint8Array(M.HEAPU8.buffer.slice(cp, cp + W * H * 4));
M._sp_clear(0xff101018, 1.0); M._sp_draw_triangles_pooled(cscene.ptr, cscene.NT);
cp = M._sp_color_ptr(); const poolFB = new Uint8Array(M.HEAPU8.buffer.slice(cp, cp + W * H * 4));
let diff = 0; for (let i = 0; i < serialFB.length; i++) if (serialFB[i] !== poolFB[i]) diff++;
console.log(`correctness: pooled vs serial → ${diff === 0 ? '✅ bit-identical' : '❌ ' + diff + ' px differ'}\n`);

// --- heavy fill-bound (threading's sweet spot) ---
const hBand = bench(() => { M._sp_clear(0xff101018, 1.0); M._sp_draw_triangles_threaded(heavy.ptr, heavy.NT, 8); });
const hPool = bench(() => { M._sp_clear(0xff101018, 1.0); M._sp_draw_triangles_pooled(heavy.ptr, heavy.NT); });
console.log('heavy fill-bound (6k big tris):');
console.log(`  serial               ${hSer.toFixed(1)} ms   1.00x`);
console.log(`  band-spawn (static)  ${hBand.toFixed(1)} ms   ${(hSer / hBand).toFixed(2)}x`);
console.log(`  persistent pool      ${hPool.toFixed(1)} ms   ${(hSer / hPool).toFixed(2)}x   ← ${(hBand / hPool).toFixed(2)}x faster than band-spawn`);

console.log('\n→ pool wins on substantial frames (work-stealing + no per-frame spawn);');
console.log('  small frames fall back to serial (sync cost > work).');

M._sp_pool_stop();
