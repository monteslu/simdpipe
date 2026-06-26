/**
 * Tier-1 JIT spike — THE crux of the plan.
 *
 * Proves: generate WASM bytes at runtime → host engine (V8) compiles them to
 * native → call into them over a whole tile of fragments via shared memory.
 *
 * The kernel we emit (a stand-in for a compiled fragment shader):
 *   for i in 0..N:  out[i] = clamp(in_r[i] * ur, 0,1) ... packed RGBA8
 * i.e. a per-fragment multiply-by-uniform + pack — the shape of a real shader
 * inner loop. We compare:
 *   (a) generated-WASM kernel  (the JIT)
 *   (b) equivalent JS loop     (what an interpreter/scalar path would do)
 * and report compile latency + steady-state throughput.
 */
import { buildModule, T, OP, E } from '../lib/wasm-emit.mjs';

const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

// ---------- emit a fragment kernel as WASM ----------
// signature: run(inPtr:i32, outPtr:i32, n:i32, ur:f32, ug:f32, ub:f32) -> void
// memory layout: inPtr -> n*3 f32 (r,g,b interleaved); outPtr -> n u32 RGBA8.
function emitKernel() {
  // locals beyond params: i (i32)=local6, base(i32)=7, r(f32)=8,g=9,b=10, tmp(i32)=11
  const P = { inPtr: 0, outPtr: 1, n: 2, ur: 3, ug: 4, ub: 5 };
  const L = { i: 6, base: 7, r: 8, g: 9, b: 10, px: 11 };
  const b = [];
  const push = (...xs) => b.push(...xs.flat());

  // i = 0
  push(OP.i32_const, ...E.sleb(0), OP.local_set, ...E.uleb(L.i));
  // block { loop { if i>=n break; ... ; i++; continue } }
  push(OP.block, 0x40);            //   void block
  push(OP.loop, 0x40);             //     void loop
  // if (i >= n) br 1 (out of block)
  push(OP.local_get, ...E.uleb(L.i), OP.local_get, ...E.uleb(P.n), OP.i32_ge_s, OP.br_if, ...E.uleb(1));

  // base = i*3 ; load r,g,b from inPtr + base*4
  push(OP.local_get, ...E.uleb(L.i), OP.i32_const, ...E.sleb(3), OP.i32_mul, OP.local_set, ...E.uleb(L.base));
  const loadF = (slot, dst) => {
    // addr = inPtr + (base+slot)*4
    push(OP.local_get, ...E.uleb(P.inPtr));
    push(OP.local_get, ...E.uleb(L.base), OP.i32_const, ...E.sleb(slot), OP.i32_add, OP.i32_const, ...E.sleb(4), OP.i32_mul, OP.i32_add);
    push(OP.f32_load, 0x02, 0x00);   // align=2 (4 bytes), offset=0
    push(OP.local_set, ...E.uleb(dst));
  };
  loadF(0, L.r); loadF(1, L.g); loadF(2, L.b);

  // r = clamp(r*ur,0,1), etc.
  const clampMul = (val, uni) => {
    push(OP.local_get, ...E.uleb(val), OP.local_get, ...E.uleb(uni), OP.f32_mul);
    push(OP.f32_const, ...E.f32bytes(1.0), OP.f32_min);
    push(OP.f32_const, ...E.f32bytes(0.0), OP.f32_max);
    push(OP.local_set, ...E.uleb(val));
  };
  clampMul(L.r, P.ur); clampMul(L.g, P.ug); clampMul(L.b, P.ub);

  // px = (i32)(r*255) | (i32)(g*255)<<8 | (i32)(b*255)<<16 | 0xff000000
  const toByte = (val, shift) => {
    push(OP.local_get, ...E.uleb(val), OP.f32_const, ...E.f32bytes(255.0), OP.f32_mul, OP.i32_trunc_f32_s);
    if (shift) { push(OP.i32_const, ...E.sleb(shift), OP.i32_shl); }
  };
  toByte(L.r, 0);
  toByte(L.g, 8); push(OP.i32_or);
  toByte(L.b, 16); push(OP.i32_or);
  push(OP.i32_const, ...E.sleb(0xff000000 | 0), OP.i32_or);
  push(OP.local_set, ...E.uleb(L.px));

  // store px at outPtr + i*4
  push(OP.local_get, ...E.uleb(P.outPtr));
  push(OP.local_get, ...E.uleb(L.i), OP.i32_const, ...E.sleb(4), OP.i32_mul, OP.i32_add);
  push(OP.local_get, ...E.uleb(L.px));
  push(OP.i32_store, 0x02, 0x00);

  // i++ ; continue loop (br 0)
  push(OP.local_get, ...E.uleb(L.i), OP.i32_const, ...E.sleb(1), OP.i32_add, OP.local_set, ...E.uleb(L.i));
  push(OP.br, ...E.uleb(0));
  push(OP.end);   // loop
  push(OP.end);   // block

  return buildModule({
    params: [T.i32, T.i32, T.i32, T.f32, T.f32, T.f32],
    results: [],
    locals: [T.i32, T.i32, T.f32, T.f32, T.f32, T.i32],
    body: b,
    exportName: 'run',
  });
}

const N = 64 * 64;       // a 64x64 tile of fragments
const PAGES = 64;        // 4 MB
const mem = new WebAssembly.Memory({ initial: PAGES });
const HEAP = () => new Uint8Array(mem.buffer);
const F32 = () => new Float32Array(mem.buffer);
const U32 = () => new Uint32Array(mem.buffer);

// layout: in at 0, out at 1MB
const inPtr = 0;
const outPtr = 1 << 20;

// fill input with deterministic r,g,b
{
  const f = F32();
  for (let i = 0; i < N; i++) {
    f[(inPtr >> 2) + i * 3 + 0] = (i % 255) / 255;
    f[(inPtr >> 2) + i * 3 + 1] = ((i * 7) % 255) / 255;
    f[(inPtr >> 2) + i * 3 + 2] = ((i * 13) % 255) / 255;
  }
}

// ---- compile (the JIT step) ----
const bytes = emitKernel();
console.log(`emitted kernel: ${bytes.length} bytes`);

const tCompile0 = now();
const mod = new WebAssembly.Module(bytes);          // sync compile (Node: fine on main; on a worker for big)
const tCompile1 = now();
const inst = new WebAssembly.Instance(mod, { env: { mem } });
const tInst1 = now();
const run = inst.exports.run;

console.log(`compile: ${(tCompile1 - tCompile0).toFixed(3)} ms   instantiate: ${(tInst1 - tCompile1).toFixed(3)} ms`);

// recompile-latency distribution (fresh module each time)
{
  const times = [];
  for (let i = 0; i < 200; i++) { const t0 = now(); new WebAssembly.Module(bytes); times.push(now() - t0); }
  console.log(`compile latency (200x, fresh module): median ${median(times).toFixed(3)} ms`);
}

// ---- correctness: compare to JS reference ----
const ur = 0.8, ug = 0.5, ub = 1.0;
run(inPtr, outPtr, N, ur, ug, ub);
const got = U32().slice(outPtr >> 2, (outPtr >> 2) + N);

function jsKernel(inP, outP, n, ur, ug, ub) {
  const f = F32(), u = U32();
  for (let i = 0; i < n; i++) {
    let r = f[(inP >> 2) + i * 3] * ur; r = r < 0 ? 0 : r > 1 ? 1 : r;
    let g = f[(inP >> 2) + i * 3 + 1] * ug; g = g < 0 ? 0 : g > 1 ? 1 : g;
    let b = f[(inP >> 2) + i * 3 + 2] * ub; b = b < 0 ? 0 : b > 1 ? 1 : b;
    u[(outP >> 2) + i] = ((r * 255) | 0) | (((g * 255) | 0) << 8) | (((b * 255) | 0) << 16) | 0xff000000;
  }
}
const jsOutPtr = 2 << 20;
jsKernel(inPtr, jsOutPtr, N, ur, ug, ub);
const ref = U32().slice(jsOutPtr >> 2, (jsOutPtr >> 2) + N);
let mism = 0; for (let i = 0; i < N; i++) if (got[i] !== ref[i]) mism++;
console.log(`correctness: ${mism === 0 ? '✅ JIT output matches JS reference' : `❌ ${mism}/${N} mismatch`}`);

// ---- throughput: JIT kernel vs JS kernel ----
function timeFn(fn, iters) {
  for (let i = 0; i < 50; i++) fn();
  const t0 = now(); for (let i = 0; i < iters; i++) fn(); return (now() - t0) / iters;
}
const ITERS = 5000;
const jitMs = timeFn(() => run(inPtr, outPtr, N, ur, ug, ub), ITERS);
const jsMs = timeFn(() => jsKernel(inPtr, jsOutPtr, N, ur, ug, ub), ITERS);
const mpix = (ms) => (N / 1e6 / (ms / 1000)).toFixed(0);
console.log(`\nper-tile (${N} frags):  JIT ${jitMs.toFixed(4)} ms (${mpix(jitMs)} Mpix/s)   JS ${jsMs.toFixed(4)} ms (${mpix(jsMs)} Mpix/s)   JIT speedup ${(jsMs / jitMs).toFixed(2)}x`);

if (mism !== 0) process.exit(1);
console.log('\n✅ JIT SPIKE PASS — generated WASM compiled by V8 to native, correct, and fast.');
