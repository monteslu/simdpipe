/**
 * Tier-1 end-to-end: a JIT'd GLSL shader rendering real geometry, validated
 * against the JS backend (the correctness oracle) and timed against it.
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';
import { compileJS } from '../lib/shader-js.mjs';

const W = 128, H = 128;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const r = await createRenderer({ width: W, height: H });
r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);

const vtx = (x, y, z, R, G, B, u, v) => [x, y, z, 1, R, G, B, 1, u, v];
const quad = new Float32Array([
  ...vtx(0, 0, 0.5, 1, 0, 0, 0, 0), ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0), ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
  ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0), ...vtx(W, H, 0.5, 1, 1, 0, 1, 1), ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
]);

// A procedural shader the JIT can fully emit (no texture):
const SRC = `
uniform float t;
void main(){
  float wr = 0.5 + 0.5*sin(uv.x*12.0 + t);
  float wg = 0.5 + 0.5*sin(uv.y*12.0 + t*1.3);
  vec3 base = mix(color.rgb, vec3(wr, wg, 1.0), 0.5);
  gl_FragColor = vec4(clamp(base, 0.0, 1.0), 1.0);
}`;

const jitProg = r.createJITProgram(SRC);
console.log('JIT path:', jitProg.jit ? '✅ native SIMD kernel' : '❌ fell back to JS — ' + jitProg.reason);

const jsProg = compileJS(SRC);

// render both, compare
function render(prog) { r.clear(0xff000000, 1.0); r.drawProgram(quad, 2, prog, { t: 0.7 }); return r.getFramebuffer().slice(); }
const jitFB = render(jitProg);
const jsFB = render(jsProg);

let diff = 0, maxd = 0;
for (let i = 0; i < W * H * 4; i++) { const d = Math.abs(jitFB[i] - jsFB[i]); if (d) { diff++; maxd = Math.max(maxd, d); } }
// allow small rounding diff (JIT uses a sin approximation + SIMD rounding)
const pctDiff = (100 * diff / (W * H * 4)).toFixed(1);
console.log(`JIT vs JS: ${diff} channels differ (${pctDiff}%), max delta ${maxd} (sin-approx + rounding)`);
const px = (fb, x, y) => { const i = (y * W + x) * 4; return [fb[i], fb[i + 1], fb[i + 2]]; };
console.log('JIT center', px(jitFB, 64, 64), ' JS center', px(jsFB, 64, 64));

// timing
const timeit = (prog) => { for (let i = 0; i < 20; i++) render(prog); const t = []; for (let i = 0; i < 60; i++) { const a = now(); render(prog); t.push(now() - a); } return median(t); };
const jitMs = timeit(jitProg), jsMs = timeit(jsProg);
console.log(`\nshade time: JIT ${jitMs.toFixed(3)} ms   JS ${jsMs.toFixed(3)} ms   JIT speedup ${(jsMs / jitMs).toFixed(2)}x`);

// pass if JIT actually JIT'd, output is close, and there's color variation
let varied = false; const c = px(jitFB, 64, 64), e = px(jitFB, 10, 10);
if (c[0] !== e[0] || c[1] !== e[1] || c[2] !== e[2]) varied = true;
const ok = jitProg.jit && maxd <= 12 && varied;
console.log(ok ? '\n✅ JIT SHADER END-TO-END PASS' : '\n❌ FAIL');
process.exit(ok ? 0 : 1);
