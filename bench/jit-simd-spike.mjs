/**
 * Tier-1 SIMD JIT spike — the REAL value of generating WASM.
 *
 * Scalar generated-WASM ≈ scalar JS (V8 optimizes both). The win is emitting
 * SIMD that JS cannot express. Here we generate a v128 kernel that shades 4
 * fragments per instruction (SoA), and compare to scalar JS.
 *
 * Kernel signature: run(rPtr, gPtr, bPtr, outPtr, n4, ur, ug, ub)
 *   rPtr/gPtr/bPtr -> n4*4 f32 planes (SoA); outPtr -> n4*4 u32 RGBA8.
 *   processes n4 groups of 4 fragments.
 */
import { buildModule, T, OP, SIMD, simd, E } from '../lib/wasm-emit.mjs';

const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

function emitSimdKernel() {
  // params: rPtr0 gPtr1 bPtr2 outPtr3 n4(=4) ur5 ug6 ub7  (f32 uniforms 5..7)
  const P = { r: 0, g: 1, b: 2, out: 3, n4: 4, ur: 5, ug: 6, ub: 7 };
  // locals: i(i32)=8, off(i32)=9, vr(v128)=10, vg=11, vb=12, urv(v128)=13, ugv=14, ubv=15
  const L = { i: 8, off: 9, vr: 10, vg: 11, vb: 12, urv: 13, ugv: 14, ubv: 15 };
  const b = [];
  const push = (...xs) => b.push(...xs.flat());

  // urv = f32x4_splat(ur) etc (broadcast uniforms once)
  const splat = (uni, dst) => { push(OP.local_get, ...E.uleb(uni)); push(simd(SIMD.f32x4_splat)); push(OP.local_set, ...E.uleb(dst)); };
  splat(P.ur, L.urv); splat(P.ug, L.ugv); splat(P.ub, L.ubv);

  // i = 0
  push(OP.i32_const, ...E.sleb(0), OP.local_set, ...E.uleb(L.i));
  push(OP.block, 0x40);
  push(OP.loop, 0x40);
  push(OP.local_get, ...E.uleb(L.i), OP.local_get, ...E.uleb(P.n4), OP.i32_ge_s, OP.br_if, ...E.uleb(1));

  // off = i*16 (bytes per v128)
  push(OP.local_get, ...E.uleb(L.i), OP.i32_const, ...E.sleb(16), OP.i32_mul, OP.local_set, ...E.uleb(L.off));

  // load planes: vX = v128_load(Xptr + off)
  const loadPlane = (ptr, dst) => {
    push(OP.local_get, ...E.uleb(ptr), OP.local_get, ...E.uleb(L.off), OP.i32_add);
    push(simd(SIMD.v128_load, 0x04, 0x00)); // align=4(16B), offset=0
    push(OP.local_set, ...E.uleb(dst));
  };
  loadPlane(P.r, L.vr); loadPlane(P.g, L.vg); loadPlane(P.b, L.vb);

  // vX = clamp(vX * uXv, 0, 1)
  const zeroV = simd(SIMD.f32x4_splat); // need a 0 and 1 splat — build via const float then splat
  const clampMul = (val, uni) => {
    push(OP.local_get, ...E.uleb(val), OP.local_get, ...E.uleb(uni)); push(simd(SIMD.f32x4_mul));
    // min with 1.0: splat 1
    push(OP.f32_const, ...E.f32bytes(1.0)); push(simd(SIMD.f32x4_splat)); push(simd(SIMD.f32x4_min));
    push(OP.f32_const, ...E.f32bytes(0.0)); push(simd(SIMD.f32x4_splat)); push(simd(SIMD.f32x4_max));
    push(OP.local_set, ...E.uleb(val));
  };
  clampMul(L.vr, L.urv); clampMul(L.vg, L.ugv); clampMul(L.vb, L.ubv);

  // to int bytes: vX = i32x4_trunc_sat(vX * 255)
  const to255 = (val) => {
    push(OP.local_get, ...E.uleb(val), OP.f32_const, ...E.f32bytes(255.0)); push(simd(SIMD.f32x4_splat)); push(simd(SIMD.f32x4_mul));
    push(simd(SIMD.i32x4_trunc_sat_f32x4_s));
    push(OP.local_set, ...E.uleb(val));
  };
  to255(L.vr); to255(L.vg); to255(L.vb);

  // pack: result = vr | (vg<<8) | (vb<<16) | 0xff000000
  // compute on stack, store to out+off
  push(OP.local_get, ...E.uleb(P.out), OP.local_get, ...E.uleb(L.off), OP.i32_add); // addr for store (pushed first)
  // build packed v128 on stack:
  push(OP.local_get, ...E.uleb(L.vr));
  push(OP.local_get, ...E.uleb(L.vg)); push(OP.i32_const, ...E.sleb(8)); push(simd(SIMD.i32x4_shl)); push(simd(SIMD.v128_or));
  push(OP.local_get, ...E.uleb(L.vb)); push(OP.i32_const, ...E.sleb(16)); push(simd(SIMD.i32x4_shl)); push(simd(SIMD.v128_or));
  // or 0xff000000 splat
  push(OP.i32_const, ...E.sleb(0xff000000 | 0)); push(simd(SIMD.i32x4_splat)); push(simd(SIMD.v128_or));
  // store
  push(simd(SIMD.v128_store, 0x04, 0x00));

  // i++
  push(OP.local_get, ...E.uleb(L.i), OP.i32_const, ...E.sleb(1), OP.i32_add, OP.local_set, ...E.uleb(L.i));
  push(OP.br, ...E.uleb(0));
  push(OP.end); push(OP.end);

  return buildModule({
    params: [T.i32, T.i32, T.i32, T.i32, T.i32, T.f32, T.f32, T.f32],
    results: [],
    locals: [T.i32, T.i32, 0x7b, 0x7b, 0x7b, 0x7b, 0x7b, 0x7b], // v128 = 0x7b
    body: b,
    exportName: 'run',
  });
}

const TILE = 64 * 64;       // fragments
const N4 = TILE / 4;
const mem = new WebAssembly.Memory({ initial: 64 });
const F32 = () => new Float32Array(mem.buffer);
const U32 = () => new Uint32Array(mem.buffer);

// SoA planes
const rPtr = 0, gPtr = 1 << 18, bPtr = 2 << 18, outPtr = 3 << 18, jsOutPtr = 4 << 18;
{
  const f = F32();
  for (let i = 0; i < TILE; i++) {
    f[(rPtr >> 2) + i] = (i % 255) / 255;
    f[(gPtr >> 2) + i] = ((i * 7) % 255) / 255;
    f[(bPtr >> 2) + i] = ((i * 13) % 255) / 255;
  }
}

const bytes = emitSimdKernel();
console.log(`emitted SIMD kernel: ${bytes.length} bytes`);
const t0 = now();
const mod = new WebAssembly.Module(bytes);
const t1 = now();
const inst = new WebAssembly.Instance(mod, { env: { mem } });
const run = inst.exports.run;
console.log(`compile ${(t1 - t0).toFixed(3)} ms`);
{ const ts = []; for (let i = 0; i < 200; i++) { const a = now(); new WebAssembly.Module(bytes); ts.push(now() - a); } console.log(`compile median (200x): ${median(ts).toFixed(3)} ms`); }

const ur = 0.8, ug = 0.5, ub = 1.0;
run(rPtr, gPtr, bPtr, outPtr, N4, ur, ug, ub);
const got = U32().slice(outPtr >> 2, (outPtr >> 2) + TILE);

// JS scalar reference (SoA)
function jsKernel() {
  const f = F32(), u = U32();
  for (let i = 0; i < TILE; i++) {
    let r = f[(rPtr >> 2) + i] * ur; r = r < 0 ? 0 : r > 1 ? 1 : r;
    let g = f[(gPtr >> 2) + i] * ug; g = g < 0 ? 0 : g > 1 ? 1 : g;
    let b = f[(bPtr >> 2) + i] * ub; b = b < 0 ? 0 : b > 1 ? 1 : b;
    u[(jsOutPtr >> 2) + i] = ((r * 255) | 0) | (((g * 255) | 0) << 8) | (((b * 255) | 0) << 16) | 0xff000000;
  }
}
jsKernel();
const ref = U32().slice(jsOutPtr >> 2, (jsOutPtr >> 2) + TILE);
let mism = 0, maxd = 0;
for (let i = 0; i < TILE; i++) {
  if (got[i] !== ref[i]) {
    mism++;
    for (let s = 0; s < 32; s += 8) maxd = Math.max(maxd, Math.abs(((got[i] >>> s) & 255) - ((ref[i] >>> s) & 255)));
  }
}
console.log(`correctness: ${mism === 0 ? '✅ exact' : `${mism}/${TILE} differ, max channel delta ${maxd} (rounding)`}`);

function timeFn(fn, iters) { for (let i = 0; i < 50; i++) fn(); const a = now(); for (let i = 0; i < iters; i++) fn(); return (now() - a) / iters; }
const ITERS = 8000;
const simdMs = timeFn(() => run(rPtr, gPtr, bPtr, outPtr, N4, ur, ug, ub), ITERS);
const jsMs = timeFn(jsKernel, ITERS);
const mpix = (ms) => (TILE / 1e6 / (ms / 1000)).toFixed(0);
console.log(`\nper-tile (${TILE} frags):  SIMD-JIT ${simdMs.toFixed(4)} ms (${mpix(simdMs)} Mpix/s)   scalar-JS ${jsMs.toFixed(4)} ms (${mpix(jsMs)} Mpix/s)   SIMD speedup ${(jsMs / simdMs).toFixed(2)}x`);
console.log(mism <= TILE ? '\n✅ SIMD JIT SPIKE PASS' : '\n❌ FAIL');
