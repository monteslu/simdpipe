/**
 * simdpipe competitor harness — same scenes as gl-harness.mjs, rendered through
 * simdpipe (WASM + 128-bit SIMD), single-threaded. Dumps PNG screenshots +
 * framebuffer fingerprints so output is verifiable against the GL renderers.
 *
 * Vertex-color workloads use the fixed-function flat path (drawTriangles). The
 * shade-bound workload uses the Tier-1 JIT shader (the same sin/mix math the GL
 * heavy fragment shader runs).
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../../lib/index.mjs';
import { makeScene, WORKLOADS, STRIDE } from './scene.mjs';
import { writePNG, fbStats } from './png.mjs';

const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const has = (k) => argv.includes(k);
const SIZE = getArg('--size', '512x512');
const [W, H] = SIZE.split('x').map(Number);
const FRAMES = parseInt(getArg('--frames', '60'), 10);
const WARMUP = parseInt(getArg('--warmup', '20'), 10);
const DUMP = getArg('--dump', null);
const TAG = getArg('--tag', 'simdpipe');
const THREADS = parseInt(getArg('--threads', '0'), 10); // >0 → use the persistent pool
const LOWFI = has('--lowfi'); // affine + no-depth (the "do less work" fidelity lever)

const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

// 8-float scene → simdpipe's 10-float VERTEX_STRIDE buffer (x y z invw r g b a u v)
function toSPBuffer(scene, ntris) {
  const n = ntris * 3;
  const out = new Float32Array(n * VERTEX_STRIDE);
  for (let i = 0; i < n; i++) {
    const s = i * STRIDE, o = i * VERTEX_STRIDE;
    out[o] = scene[s]; out[o + 1] = scene[s + 1]; out[o + 2] = scene[s + 2]; out[o + 3] = 1.0;
    out[o + 4] = scene[s + 3]; out[o + 5] = scene[s + 4]; out[o + 6] = scene[s + 5]; out[o + 7] = 1.0;
    out[o + 8] = scene[s + 6]; out[o + 9] = scene[s + 7];
  }
  return out;
}

function timeFrames(drawFn) {
  for (let i = 0; i < WARMUP; i++) drawFn();
  const t = [];
  for (let i = 0; i < FRAMES; i++) { const a = now(); drawFn(); t.push(now() - a); }
  return median(t);
}

const results = [];

// Flags: full fidelity = depth + perspective-correct. Low-fi = neither (affine,
// no z-buffer) — the "do less work" lever. Both still SIMD-rasterize every pixel.
const FULL = FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT;
const FLAGS_USE = LOWFI ? 0 : FULL;

// ---- vertex-color workloads via the fixed-function flat path ----
// Single-thread uses the normal renderer; threaded loads the pthreads module and
// drives the persistent work-stealing pool (the multiplier on top of SIMD).
let sp, M, pooled = false;
if (THREADS > 0) {
  const createThreads = (await import('../../dist/simdpipe-threads.mjs')).default;
  M = await createThreads();
  M._sp_init(W, H);
  M._sp_set_flags(FLAGS_USE);
  M._sp_pool_start(THREADS);
  pooled = true;
  sp = { // minimal shim matching the bits we use
    alloc: (n) => M._sp_alloc(n), free: (p) => M._sp_free(p),
    clear: (c, d) => M._sp_clear(c >>> 0, d),
    getFramebuffer: () => new Uint8Array(M.HEAPU8.buffer, M._sp_color_ptr(), W * H * 4),
    _module: M,
  };
} else {
  sp = await createRenderer({ width: W, height: H });
  sp.setFlags(FLAGS_USE);
  M = sp._module;
}

for (const wl of WORKLOADS) {
  const scene = makeScene(wl.kind, wl.ntris, W, H, { px: wl.px });
  const spbuf = toSPBuffer(scene, wl.ntris);
  const floats = wl.ntris * 3 * VERTEX_STRIDE;
  const ptr = sp.alloc(floats * 4);
  M.HEAPF32.set(spbuf.subarray(0, floats), ptr >> 2);
  const drawFn = pooled ? M._sp_draw_triangles_pooled : M._sp_draw_triangles_flat;
  const draw = () => { sp.clear(0xff180f10, 1.0); drawFn(ptr, wl.ntris); };
  const ms = timeFrames(draw);
  draw();
  const snap = fbStats(sp.getFramebuffer(), W, H);
  if (DUMP) writePNG(`${DUMP}-${TAG}-${wl.kind}.png`, sp.getFramebuffer(), W, H, false);
  sp.free(ptr);
  results.push({ name: wl.name, shader: 'color', ms: +ms.toFixed(3), ntris: wl.ntris, ...snap });
}

// ---- shade-bound: heavy JIT fragment shader over the balanced scene ----
// Driven through drawProgram directly (no GL vertex stage) so the geometry is the
// SAME screen-space scene as the vertex-color path and the GL harness — only the
// fragment shader differs (the same sin/mix math the GL heavy shader runs).
// (Single-thread only — the JIT shade pass over the G-buffer isn't pooled here.)
if (!pooled) {
  const SRC = `
uniform float uT;
void main(){
  float wr = 0.5 + 0.5*sin(color.x*12.0 + uT);
  float wg = 0.5 + 0.5*sin(color.y*12.0 + uT*1.3);
  float wb = 0.5 + 0.5*sin(color.z*12.0 + uT*0.7);
  vec3 c = mix(color.rgb, vec3(wr, wg, wb), 0.5);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;
  const prog = sp.createJITProgram(SRC);
  const scene = makeScene('balanced', 2000, W, H);
  const spbuf = toSPBuffer(scene, 2000);
  const draw = () => { sp.clear(0xff180f10, 1.0); sp.drawProgram(spbuf, 2000, prog, { uT: 0.7 }); };
  const ms = timeFrames(draw);
  draw();
  const snap = fbStats(sp.getFramebuffer(), W, H);
  if (DUMP) writePNG(`${DUMP}-${TAG}-heavy.png`, sp.getFramebuffer(), W, H, false);
  results.push({ name: 'shade-bound (heavy frag, 2k tris)', shader: 'heavy', ms: +ms.toFixed(3), ntris: 2000, jit: prog.jit, ...snap });
}

if (pooled) M._sp_pool_stop();
const label = `simdpipe (WASM+SIMD, ${THREADS > 0 ? THREADS + ' threads' : '1 thread'}${LOWFI ? ', low-fi' : ''})`;
console.log(JSON.stringify({ renderer: label, size: { W, H }, frames: FRAMES, results }));
