/**
 * Fast iteration harness for the balanced + shade single-thread paths only.
 * Times the vertex-color flat path on several triangle counts and prints a
 * coverage/meanRGB fingerprint so a C change can be checked for byte-identity
 * (fingerprint must NOT move) and speed in ~2s, not the 600s full compete.
 *
 * Usage: node bench/compete/micro-balanced.mjs [--size 512x512] [--frames 80]
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../../lib/index.mjs';
import { makeScene, STRIDE } from './scene.mjs';
import { fbStats } from './png.mjs';

const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const [W, H] = getArg('--size', '512x512').split('x').map(Number);
const FRAMES = parseInt(getArg('--frames', '100'), 10);
// WASM tier-up needs ~100 iterations to fully optimize; under-warming reports
// 2-3x inflated, non-monotonic noise. Default high so a bare run is never wrong.
const WARMUP = parseInt(getArg('--warmup', '120'), 10);

const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

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

const sp = await createRenderer({ width: W, height: H });
sp.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);
const M = sp._module;

const CASES = [
  ['balanced 2k', 'balanced', 2000],
  ['balanced 4k', 'balanced', 4000],
  ['balanced 8k', 'balanced', 8000],
  ['dense 16k', 'balanced', 16000],
];

for (const [label, kind, ntris] of CASES) {
  const scene = makeScene(kind, ntris, W, H);
  const spbuf = toSPBuffer(scene, ntris);
  const floats = ntris * 3 * VERTEX_STRIDE;
  const ptr = sp.alloc(floats * 4);
  M.HEAPF32.set(spbuf.subarray(0, floats), ptr >> 2);
  const draw = () => { sp.clear(0xff180f10, 1.0); M._sp_draw_triangles_flat(ptr, ntris); };
  for (let i = 0; i < WARMUP; i++) draw();
  const t = [];
  for (let i = 0; i < FRAMES; i++) { const a = now(); draw(); t.push(now() - a); }
  draw();
  const s = fbStats(sp.getFramebuffer(), W, H);
  sp.free(ptr);
  console.log(`${label.padEnd(14)} ${median(t).toFixed(3).padStart(8)} ms   cov=${s.coverage.toFixed(2)}%  rgb=[${s.meanRGB.join(',')}]  hash=${s.hash}`);
}
