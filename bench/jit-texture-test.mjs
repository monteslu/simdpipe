/**
 * Tier-1 texture sampling in the JIT — proves texture() compiles to a native
 * SIMD kernel (gathering texels straight from linear memory) instead of falling
 * back to the JS backend. Validated against the JS oracle pixel-for-pixel.
 *
 * Per the project thesis: nothing that is pure calculation should EVER fall back
 * to JS. texture() nearest sampling is address arithmetic + 4 scalar loads +
 * lane reassembly — fully expressible in WASM, no gather opcode needed.
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';
import { compileJS } from '../lib/shader-js.mjs';

const W = 128, H = 128;
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

const r = await createRenderer({ width: W, height: H });
r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);

// Build a 64x64 RGBA8 procedural texture (a checkerboard + gradient so every
// channel varies and nearest sampling is visible).
const TW = 64, TH = 64;
const tex = new Uint8Array(TW * TH * 4);
for (let y = 0; y < TH; y++) {
  for (let x = 0; x < TW; x++) {
    const i = (y * TW + x) * 4;
    const checker = ((x >> 3) ^ (y >> 3)) & 1;
    tex[i + 0] = checker ? (x * 4) & 255 : 32;            // R
    tex[i + 1] = checker ? 32 : (y * 4) & 255;            // G
    tex[i + 2] = ((x + y) * 2) & 255;                     // B
    tex[i + 3] = 255;                                      // A
  }
}
r.bindTexture(tex, TW, TH);

const vtx = (x, y, z, R, G, B, u, v) => [x, y, z, 1, R, G, B, 1, u, v];
const quad = new Float32Array([
  ...vtx(0, 0, 0.5, 1, 0, 0, 0, 0), ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0), ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
  ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0), ...vtx(W, H, 0.5, 1, 1, 0, 1, 1), ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
]);

// A shader that SAMPLES THE TEXTURE — the case that used to fall back to JS.
const SRC = `
uniform float t;
void main(){
  vec4 tc = texture(uv);
  // modulate by vertex color so it's not a pure passthrough
  gl_FragColor = vec4(tc.rgb * (0.5 + 0.5*color.rgb), 1.0);
}`;

const jitProg = r.createJITProgram(SRC);
console.log('JIT path:', jitProg.jit ? '✅ native SIMD kernel' : '❌ fell back to JS — ' + jitProg.reason);
console.log('needsTexture:', jitProg.needsTexture);

const jsProg = compileJS(SRC);

function render(prog) { r.clear(0xff000000, 1.0); r.drawProgram(quad, 2, prog, { t: 0.0 }); return r.getFramebuffer().slice(); }
const jitFB = render(jitProg);
const jsFB = render(jsProg);

let diff = 0, maxd = 0;
for (let i = 0; i < W * H * 4; i++) { const d = Math.abs(jitFB[i] - jsFB[i]); if (d) { diff++; maxd = Math.max(maxd, d); } }
const pctDiff = (100 * diff / (W * H * 4)).toFixed(1);
console.log(`JIT vs JS: ${diff} channels differ (${pctDiff}%), max delta ${maxd}`);
const px = (fb, x, y) => { const i = (y * W + x) * 4; return [fb[i], fb[i + 1], fb[i + 2], fb[i + 3]]; };
console.log('JIT @(20,20)', px(jitFB, 20, 20), ' JS @(20,20)', px(jsFB, 20, 20));
console.log('JIT @(96,40)', px(jitFB, 96, 40), ' JS @(96,40)', px(jsFB, 96, 40));

// timing
const timeit = (prog) => { for (let i = 0; i < 20; i++) render(prog); const t = []; for (let i = 0; i < 60; i++) { const a = now(); render(prog); t.push(now() - a); } return median(t); };
const jitMs = timeit(jitProg), jsMs = timeit(jsProg);
console.log(`\nshade time: JIT ${jitMs.toFixed(3)} ms   JS ${jsMs.toFixed(3)} ms   JIT speedup ${(jsMs / jitMs).toFixed(2)}x`);

// pass: JIT actually JIT'd (no JS fallback), output matches the oracle, and the
// sampled image actually varies across the surface.
let varied = false; const c = px(jitFB, 20, 20), e = px(jitFB, 96, 40);
if (c[0] !== e[0] || c[1] !== e[1] || c[2] !== e[2]) varied = true;
const okNearest = jitProg.jit && maxd <= 1 && varied;
console.log(okNearest ? '\n✅ JIT TEXTURE NEAREST PASS (no JS fallback, matches oracle)' : '\n❌ NEAREST FAIL');

// ---- bilinear: same shader, 4-tap filter, JIT vs JS oracle ----
console.log('\n--- bilinear (4-tap) ---');
const jitBil = r.createJITProgram(SRC, { bilinear: true });
const jsBil = compileJS(SRC, { bilinear: true });
console.log('JIT path:', jitBil.jit ? '✅ native SIMD kernel' : '❌ fell back to JS — ' + jitBil.reason);
const jitB = render(jitBil), jsB = render(jsBil);
let diffB = 0, maxdB = 0;
for (let i = 0; i < W * H * 4; i++) { const d = Math.abs(jitB[i] - jsB[i]); if (d) { diffB++; maxdB = Math.max(maxdB, d); } }
console.log(`bilinear JIT vs JS: ${diffB} channels differ (${(100 * diffB / (W * H * 4)).toFixed(1)}%), max delta ${maxdB}`);
console.log('JIT @(64,64)', px(jitB, 64, 64), ' JS @(64,64)', px(jsB, 64, 64));
const jitBms = timeit(jitBil);
console.log(`bilinear shade time: JIT ${jitBms.toFixed(3)} ms (nearest was ${jitMs.toFixed(3)} ms)`);
// bilinear allows a slightly looser delta (float blend rounding across 4 taps)
const okBil = jitBil.jit && maxdB <= 2;
console.log(okBil ? '✅ JIT TEXTURE BILINEAR PASS (no JS fallback, matches oracle)' : '❌ BILINEAR FAIL');

const ok = okNearest && okBil;
console.log(ok ? '\n✅ JIT TEXTURE SAMPLING PASS' : '\n❌ FAIL');
process.exit(ok ? 0 : 1);
