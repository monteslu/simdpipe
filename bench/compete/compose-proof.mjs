/**
 * Generate one labeled side-by-side strip proving every renderer draws the SAME
 * scene. Renders the 'fill' workload at NxN on simdpipe, llvmpipe, GPU, and
 * native-C, then tiles their framebuffers horizontally into proof.png.
 *
 * Run: node bench/compete/compose-proof.mjs [N]
 * (GL renderers are spawned as subprocesses dumping raw RGBA, since they need
 * their own EGL env; simdpipe + native are done here.)
 */
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import os from 'node:os';
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../../lib/index.mjs';
import { makeScene, STRIDE } from './scene.mjs';
import { writePNG } from './png.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const N = parseInt(process.argv[2] || '256', 10);
const tmp = os.tmpdir();

function toSP(scene, nt) {
  const n = nt * 3, o = new Float32Array(n * VERTEX_STRIDE);
  for (let i = 0; i < n; i++) { const s = i * STRIDE, d = i * VERTEX_STRIDE; o[d] = scene[s]; o[d + 1] = scene[s + 1]; o[d + 2] = scene[s + 2]; o[d + 3] = 1; o[d + 4] = scene[s + 3]; o[d + 5] = scene[s + 4]; o[d + 6] = scene[s + 5]; o[d + 7] = 1; }
  return o;
}

const scene = makeScene('fill', 200, N, N);

// simdpipe
const sp = await createRenderer({ width: N, height: N });
sp.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);
const spbuf = toSP(scene, 200);
const ptr = sp.alloc(spbuf.length * 4); sp._module.HEAPF32.set(spbuf, ptr >> 2);
sp.clear(0xff180f10, 1.0); sp._module._sp_draw_triangles_flat(ptr, 200);
const spFB = sp.getFramebuffer().slice();

// native C (writes raw rgba)
let natFB = null;
const bin = join(__dir, 'native-raster');
if (existsSync(bin)) {
  const sb = join(tmp, `proof-${N}.bin`);
  execFileSync(process.execPath, [join(__dir, 'export-scene.mjs'), 'fill', '200', String(N), String(N), sb]);
  const rg = join(tmp, `proof-${N}-native.rgba`);
  execFileSync(bin, [sb, '1', '0', rg]);
  natFB = new Uint8Array(readFileSync(rg));
}

// GL renderers: spawn the dumper which writes raw rgba (flip handled here)
function glRaw(env, tag) {
  const raw = join(tmp, `proof-${N}-${tag}.rgba`);
  execFileSync(process.execPath, [join(__dir, 'gl-dump-raw.mjs'), String(N), raw], { env: { ...process.env, ...env } });
  return new Uint8Array(readFileSync(raw));
}
let llvmFB = null, gpuFB = null;
try { llvmFB = glRaw({ LIBGL_ALWAYS_SOFTWARE: '1', LP_NUM_THREADS: '0' }, 'llvmpipe'); } catch (e) { console.error('llvmpipe dump failed', e.message); }
try { gpuFB = glRaw({}, 'gpu'); } catch (e) { console.error('gpu dump failed', e.message); }

// tile horizontally
const panels = [['simdpipe', spFB], ['llvmpipe', llvmFB], ['GPU', gpuFB], ['native-C', natFB]].filter(p => p[1]);
const cols = panels.length, GAP = 4;
const outW = N * cols + GAP * (cols - 1), outH = N;
const out = new Uint8Array(outW * outH * 4).fill(0);
for (let i = 3; i < out.length; i += 4) out[i] = 255; // opaque
for (let p = 0; p < cols; p++) {
  const fb = panels[p][1], x0 = p * (N + GAP);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const s = (y * N + x) * 4, d = (y * outW + (x0 + x)) * 4;
    out[d] = fb[s]; out[d + 1] = fb[s + 1]; out[d + 2] = fb[s + 2]; out[d + 3] = 255;
  }
}
const outPath = join(__dir, 'proof.png');
writePNG(outPath, out, outW, outH, false);
console.log(`wrote ${outPath} — ${panels.map(p => p[0]).join(' | ')} (each ${N}x${N})`);
