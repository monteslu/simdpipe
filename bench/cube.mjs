/**
 * Spinning textured cube through the simdpipe GL surface — the end-to-end demo.
 * Exercises: vertex MVP transform → near-cull → perspective divide → viewport →
 * SIMD rasterization → programmable fragment shader → framebuffer.
 *
 * Writes a PPM of one frame (so it's verifiable) and times a spin.
 */
import { createGL } from '../lib/gl.mjs';
import { shaders } from '../lib/program.mjs';
import { writeFileSync } from 'node:fs';

const W = 256, H = 256;
const gl = await createGL({ width: W, height: H });

// checkerboard texture
const TW = 8, TH = 8;
const tex = new Uint32Array(TW * TH);
for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
  const on = (x + y) & 1;
  tex[y * TW + x] = on ? (0xfff0a020 >>> 0) : (0xff2040f0 >>> 0);
}
gl.useProgram(shaders.texturedModulate(tex, TW, TH));

// cube: 8 corners, 12 triangles, per-vertex uv + tint
// interleaved: x y z  r g b  u v  (stride 8)
const S = 1;
function quad(verts, p0, p1, p2, p3, tint) {
  const uv = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const corners = [p0, p1, p2, p3];
  const tris = [[0, 1, 2], [0, 2, 3]];
  for (const tr of tris) for (const ci of tr) {
    verts.push(...corners[ci], ...tint, ...uv[ci]);
  }
}
const v = [
  [-S, -S, -S], [S, -S, -S], [S, S, -S], [-S, S, -S], // back
  [-S, -S, S], [S, -S, S], [S, S, S], [-S, S, S],     // front
];
const arr = [];
quad(arr, v[4], v[5], v[6], v[7], [1, 0.6, 0.6]); // front
quad(arr, v[1], v[0], v[3], v[2], [0.6, 1, 0.6]); // back
quad(arr, v[0], v[4], v[7], v[3], [0.6, 0.6, 1]); // left
quad(arr, v[5], v[1], v[2], v[6], [1, 1, 0.6]);   // right
quad(arr, v[7], v[6], v[2], v[3], [1, 0.6, 1]);   // top
quad(arr, v[0], v[1], v[5], v[4], [0.6, 1, 1]);   // bottom
const attribs = new Float32Array(arr);
const stride = 8, layout = { pos: 0, color: 3, uv: 6 };
const count = attribs.length / stride;

const proj = gl.mat4.perspective(Math.PI / 4, W / H, 0.1, 100);

function renderFrame(angle) {
  gl.clear();
  let mv = gl.mat4.identity();
  mv = gl.mat4.translate(mv, 0, 0, -4);
  mv = gl.mat4.rotateY(mv, angle);
  mv = gl.mat4.rotateX(mv, angle * 0.6);
  gl.setMVP(gl.mat4.multiply(proj, mv));
  gl.drawArrays(attribs, stride, layout, count);
}

// render one frame and save PPM
renderFrame(0.7);
const fb = gl.getFramebuffer();
let ppm = `P3\n${W} ${H}\n255\n`;
const rows = [];
for (let i = 0; i < W * H; i++) rows.push(`${fb[i * 4]} ${fb[i * 4 + 1]} ${fb[i * 4 + 2]}`);
ppm += rows.join(' ');
writeFileSync(new URL('./cube.ppm', import.meta.url), ppm);
console.log('wrote cube.ppm (256x256)');
console.log('frame stats:', gl.stats());

// count non-background pixels as a sanity check the cube actually drew
let drawn = 0;
for (let i = 0; i < W * H; i++) if (fb[i * 4] || fb[i * 4 + 1] || fb[i * 4 + 2]) drawn++;
console.log(`non-background pixels: ${drawn} / ${W * H} (${(100 * drawn / (W * H)).toFixed(1)}%)`);

// time a spin
const now = () => Number(process.hrtime.bigint()) / 1e6;
for (let i = 0; i < 20; i++) renderFrame(i * 0.1);
const t0 = now();
const N = 120;
for (let i = 0; i < N; i++) renderFrame(i * 0.05);
const ms = (now() - t0) / N;
console.log(`\nspinning cube: ${ms.toFixed(3)} ms/frame → ${(1000 / ms).toFixed(0)} fps @ ${W}x${H}`);

if (drawn < 1000) { console.error('❌ cube did not render'); process.exit(1); }
console.log('\n✅ GL CUBE PASS');
