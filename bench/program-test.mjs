/** Programmable shader path test — proves simdpipe runs custom fragment shaders. */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';
import { shaders } from '../lib/program.mjs';

const W = 64, H = 64;
const r = await createRenderer({ width: W, height: H });
r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);

const vtx = (x, y, z, R, G, B, u, v) => [x, y, z, 1, R, G, B, 1, u, v];
// full-screen quad as two triangles, with uv 0..1 and a color gradient
const quad = new Float32Array([
  ...vtx(0, 0, 0.5, 1, 0, 0, 0, 0),
  ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0),
  ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
  ...vtx(W, 0, 0.5, 0, 1, 0, 1, 0),
  ...vtx(W, H, 0.5, 1, 1, 0, 1, 1),
  ...vtx(0, H, 0.5, 0, 0, 1, 0, 1),
]);

const px = (fb, x, y) => { const i = (y * W + x) * 4; return [fb[i], fb[i + 1], fb[i + 2], fb[i + 3]]; };
let pass = true;

// --- vertexColor shader ---
r.clear(0xff000000, 1.0);
r.drawProgram(quad, 2, shaders.vertexColor());
let fb = r.getFramebuffer();
const c00 = px(fb, 1, 1), c63 = px(fb, 62, 1);
console.log('vertexColor  topleft', c00, ' topright', c63);
// top-left ~ red, top-right ~ green
if (!(c00[0] > 150 && c63[1] > 150)) { console.error('FAIL vertexColor gradient'); pass = false; }

// --- procedural shader (animated) ---
r.clear(0xff000000, 1.0);
r.drawProgram(quad, 2, shaders.procedural(), { t: 1.0 });
fb = r.getFramebuffer();
const proc = px(fb, 32, 32);
console.log('procedural   center', proc);
if (proc[3] !== 255) { console.error('FAIL procedural alpha'); pass = false; }
// it should differ from the vertexColor result somewhere → programmable
const center_vc = (() => { r.clear(0xff000000, 1); r.drawProgram(quad, 2, shaders.vertexColor()); return px(r.getFramebuffer(), 32, 32); })();
if (proc[0] === center_vc[0] && proc[1] === center_vc[1] && proc[2] === center_vc[2]) { console.error('FAIL: procedural == vertexColor (not programmable)'); pass = false; }

// --- textured shader ---
const TW = 4, TH = 4;
const tex = new Uint32Array(TW * TH);
for (let i = 0; i < tex.length; i++) tex[i] = ((i * 60) & 255) | (((i * 30) & 255) << 8) | (((i * 90) & 255) << 16) | (255 << 24);
r.clear(0xff000000, 1.0);
r.drawProgram(quad, 2, shaders.texturedModulate(tex, TW, TH));
fb = r.getFramebuffer();
console.log('textured     center', px(fb, 32, 32), ' (sampled+modulated)');

console.log(pass ? '\n✅ PROGRAMMABLE SHADERS PASS' : '\n❌ FAIL');
process.exit(pass ? 0 : 1);
