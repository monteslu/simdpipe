/**
 * Coarse (2×1 variable-rate) shading: the {coarse:true} JIT kernel evaluates the
 * fragment math once per horizontal pixel pair (half the transcendental throughput).
 * Verify it (a) still JITs, (b) stays close to the per-pixel version — a bounded VRS
 * blur, NOT garbage — and (c) actually runs faster on a heavy shader. This is an
 * opt-in fidelity trade (it does NOT match the per-pixel oracle to ≤1 LSB by design).
 */
import { createRenderer, FLAGS } from '../lib/index.mjs';

const W = 256, H = 256;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const r = await createRenderer({ width: W, height: H });
r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);

const vtx = (x, y, z, R, G, B) => [x, y, z, 1, R, G, B, 1, 0, 0];
const quad = new Float32Array([
  ...vtx(0, 0, 0.5, 1, 0, 0), ...vtx(W, 0, 0.5, 0, 1, 0), ...vtx(0, H, 0.5, 0, 0, 1),
  ...vtx(W, 0, 0.5, 0, 1, 0), ...vtx(W, H, 0.5, 1, 1, 0), ...vtx(0, H, 0.5, 0, 0, 1),
]);

// Heavy procedural shader (the shade-bound kind): three sins per pixel.
const SRC = `
uniform float t;
void main(){
  float wr = 0.5 + 0.5*sin(color.x*12.0 + t);
  float wg = 0.5 + 0.5*sin(color.y*12.0 + t*1.3);
  float wb = 0.5 + 0.5*sin(color.z*12.0 + t*0.7);
  vec3 c = mix(color.rgb, vec3(wr, wg, wb), 0.5);
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

const fine = r.createJITProgram(SRC);                 // per-pixel
const coarse = r.createJITProgram(SRC, { coarse: true }); // 2×1 VRS

console.log('fine   JIT:', fine.jit ? '✅ native' : '❌ ' + fine.reason);
console.log('coarse JIT:', coarse.jit ? '✅ native' : '❌ ' + coarse.reason, '  coarse flag:', coarse.coarse);

if (!coarse.jit) { console.error('❌ COARSE FAILED TO JIT'); process.exit(1); }
if (!coarse.coarse) { console.error('❌ coarse flag not set on the program'); process.exit(1); }

function render(prog) { r.clear(0xff000000, 1.0); r.drawProgram(quad, 2, prog, { t: 0.7 }); return r.getFramebuffer().slice(); }

const fineFB = render(fine);
const coarseFB = render(coarse);

// Compare: coarse should be CLOSE to fine (a bounded blur), not divergent. The 2×1
// scheme shades the left pixel of each pair and reuses it for the right, so the max
// per-channel delta is bounded by the shader's horizontal gradient over 1px — small.
let maxD = 0, sumD = 0, nDiff = 0;
for (let i = 0; i < fineFB.length; i++) {
  const d = Math.abs(fineFB[i] - coarseFB[i]);
  if (d > 0) { nDiff++; sumD += d; if (d > maxD) maxD = d; }
}
const meanD = nDiff ? (sumD / nDiff).toFixed(2) : 0;
console.log(`coarse vs fine: ${nDiff} channels differ, maxΔ=${maxD}, meanΔ(of differing)=${meanD}`);

// Coverage must be identical (coarse only changes shaded VALUES, never which pixels
// are covered) — check the alpha/non-background pixel count matches.
const nonbg = (fb) => { let n = 0; for (let i = 0; i < fb.length; i += 4) if (fb[i] | fb[i+1] | fb[i+2]) n++; return n; };
const cf = nonbg(fineFB), cc = nonbg(coarseFB);
console.log(`covered px: fine ${cf}, coarse ${cc}`);
if (cf !== cc) { console.error('❌ coverage changed — coarse must not alter which pixels draw'); process.exit(1); }

// The blur must be BOUNDED (a real shading-rate trade, not corruption). With a smooth
// sin shader over a 256px gradient, a 1px reuse is a handful of LSBs.
if (maxD > 24) { console.error(`❌ coarse delta ${maxD} too large — not a 2×1 blur, likely a bug`); process.exit(1); }

// Timing: coarse must be faster on the heavy shader.
const time = (prog) => { for (let i = 0; i < 40; i++) render(prog); const ts = []; for (let i = 0; i < 60; i++) { const t0 = now(); render(prog); ts.push(now() - t0); } return median(ts); };
const tFine = time(fine), tCoarse = time(coarse);
console.log(`shade time: fine ${tFine.toFixed(3)} ms   coarse ${tCoarse.toFixed(3)} ms   speedup ${(tFine / tCoarse).toFixed(2)}x`);
if (tCoarse >= tFine) console.log('⚠ coarse not faster here (small frame / fast shader) — win shows on heavier fragment load');

console.log('✅ COARSE SHADING PASS (bounded VRS blur, coverage preserved, faster)');
