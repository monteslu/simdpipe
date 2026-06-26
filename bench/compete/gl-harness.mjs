/**
 * GL competitor harness — drives a real OpenGL ES 3.0 driver (llvmpipe when
 * LIBGL_ALWAYS_SOFTWARE=1, the AMD GPU otherwise) through native-gles, over the
 * SAME geometry simdpipe uses. Screen-space verts are converted to NDC; the
 * fragment shader does Gouraud vertex color (matching simdpipe's vertex-color
 * path). Depth test on, like simdpipe's DEPTH_TEST flag.
 *
 * We pre-upload the VBO once (so we measure rasterization, not transfer) and time
 * glDrawArrays + glFinish (glFinish forces the async GL pipeline to complete —
 * without it the timing is a lie).
 *
 * createRequire is used so this stays an ESM file while loading the CJS addon.
 */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const gl = require('/home/monteslu/code/cliemu/native-gles');

import { makeScene, WORKLOADS, STRIDE } from './scene.mjs';
import { writePNG, fbStats } from './png.mjs';

// ---- GL constants ----
const GL = {
  VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30, COMPILE_STATUS: 0x8B81,
  LINK_STATUS: 0x8B82, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x100,
  TRIANGLES: 0x0004, FLOAT: 0x1406, ARRAY_BUFFER: 0x8892, STATIC_DRAW: 0x88E4,
  RGBA: 0x1908, UNSIGNED_BYTE: 0x1401, DEPTH_TEST: 0x0B71, RENDERER: 0x1F01,
  LEQUAL: 0x0203,
};

const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SIZE = getArg('--size', '512x512');
const [W, H] = SIZE.split('x').map(Number);
const FRAMES = parseInt(getArg('--frames', '60'), 10);
const WARMUP = parseInt(getArg('--warmup', '20'), 10);
const DUMP = getArg('--dump', null); // path prefix for PNG screenshots
const TAG = getArg('--tag', 'gl');  // filename tag (e.g. 'llvmpipe' / 'gpu')

const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

if (!gl.createContext(W, H)) { console.error('FAIL: no GL context'); process.exit(1); }
const RENDERER = gl.glGetString(GL.RENDERER);

function compile(type, src) {
  const s = gl.glCreateShader(type);
  gl.glShaderSource(s, src); gl.glCompileShader(s);
  if (!gl.glGetShaderiv(s, GL.COMPILE_STATUS)) { console.error('shader:', gl.glGetShaderInfoLog(s)); process.exit(1); }
  return s;
}

// Vertex shader: screen-space (px) → NDC. Pass color through. z∈[0,1] → clip z∈[-1,1].
const VS = `#version 300 es
in vec3 aPos;     // x,y in pixels, z in [0,1]
in vec3 aColor;
uniform vec2 uRes;
out vec3 vColor;
void main() {
  vec2 ndc = vec2(aPos.x / uRes.x * 2.0 - 1.0, 1.0 - aPos.y / uRes.y * 2.0);
  gl_Position = vec4(ndc, aPos.z * 2.0 - 1.0, 1.0);
  vColor = aColor;
}`;

// Fragment shader for the vertex-color workload (matches simdpipe's flat path).
const FS_COLOR = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 fragColor;
void main() { fragColor = vec4(vColor, 1.0); }`;

// Heavy ALU fragment shader (matches the JIT shade-bound workload).
const FS_HEAVY = `#version 300 es
precision highp float;
in vec3 vColor;
uniform float uT;
out vec4 fragColor;
void main() {
  float wr = 0.5 + 0.5*sin(vColor.x*12.0 + uT);
  float wg = 0.5 + 0.5*sin(vColor.y*12.0 + uT*1.3);
  float wb = 0.5 + 0.5*sin(vColor.z*12.0 + uT*0.7);
  vec3 c = mix(vColor, vec3(wr, wg, wb), 0.5);
  fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

function makeProgram(fsSrc) {
  const vs = compile(GL.VERTEX_SHADER, VS), fs = compile(GL.FRAGMENT_SHADER, fsSrc);
  const p = gl.glCreateProgram();
  gl.glAttachShader(p, vs); gl.glAttachShader(p, fs); gl.glLinkProgram(p);
  if (!gl.glGetProgramiv(p, GL.LINK_STATUS)) { console.error('link:', gl.glGetProgramInfoLog(p)); process.exit(1); }
  return p;
}

// Interleaved VBO: aPos(3) + aColor(3) from the 8-float scene (drop u,v).
function uploadVBO(scene, ntris) {
  const n = ntris * 3;
  const data = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const s = i * STRIDE, d = i * 6;
    data[d] = scene[s]; data[d + 1] = scene[s + 1]; data[d + 2] = scene[s + 2];
    data[d + 3] = scene[s + 3]; data[d + 4] = scene[s + 4]; data[d + 5] = scene[s + 5];
  }
  const vaoArr = new Uint32Array(1); gl.glGenVertexArrays(1, vaoArr); gl.glBindVertexArray(vaoArr[0]);
  const vboArr = new Uint32Array(1); gl.glGenBuffers(1, vboArr); gl.glBindBuffer(GL.ARRAY_BUFFER, vboArr[0]);
  gl.glBufferData(GL.ARRAY_BUFFER, new Uint8Array(data.buffer), GL.STATIC_DRAW); // binding wants a byte view
  gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 3, GL.FLOAT, false, 24, 0);
  gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1, 3, GL.FLOAT, false, 24, 12);
  return n;
}

function timeFrames(drawFn) {
  for (let i = 0; i < WARMUP; i++) drawFn();
  gl.glFinish();
  const t = [];
  for (let i = 0; i < FRAMES; i++) { const a = now(); drawFn(); gl.glFinish(); t.push(now() - a); }
  return median(t);
}

// Read the framebuffer back and (optionally) dump a PNG + return a fingerprint.
// GL readback is bottom-to-top, so flipY when saving.
function snapshot(drawFn, label) {
  drawFn(); gl.glFinish();
  const px = new Uint8Array(W * H * 4);
  gl.glReadPixels(0, 0, W, H, GL.RGBA, GL.UNSIGNED_BYTE, px);
  if (DUMP) writePNG(`${DUMP}-${TAG}-${label}.png`, px, W, H, true);
  return fbStats(px, W, H);
}

gl.glViewport(0, 0, W, H);
gl.glEnable(GL.DEPTH_TEST);
gl.glDepthFunc(GL.LEQUAL);

const results = [];
for (const wl of WORKLOADS) {
  const scene = makeScene(wl.kind, wl.ntris, W, H, { px: wl.px });
  const n = uploadVBO(scene, wl.ntris); // bind VAO/VBO for this workload

  // color (light fragment) program
  const pColor = makeProgram(FS_COLOR);
  gl.glUseProgram(pColor);
  const resLoc = gl.glGetUniformLocation(pColor, 'uRes');
  gl.glUniform2f(resLoc, W, H);
  const drawColor = () => { gl.glClearColor(0.06, 0.06, 0.09, 1); gl.glClear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT); gl.glDrawArrays(GL.TRIANGLES, 0, n); };
  const msColor = timeFrames(drawColor);
  const snap = snapshot(drawColor, wl.kind);

  results.push({ name: wl.name, shader: 'color', ms: +msColor.toFixed(3), ntris: wl.ntris, ...snap });
}

// shade-bound: balanced geometry, heavy fragment shader
{
  const scene = makeScene('balanced', 2000, W, H);
  const n = uploadVBO(scene, 2000);
  const pHeavy = makeProgram(FS_HEAVY);
  gl.glUseProgram(pHeavy);
  gl.glUniform2f(gl.glGetUniformLocation(pHeavy, 'uRes'), W, H);
  gl.glUniform1f(gl.glGetUniformLocation(pHeavy, 'uT'), 0.7);
  const draw = () => { gl.glClearColor(0.06, 0.06, 0.09, 1); gl.glClear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT); gl.glDrawArrays(GL.TRIANGLES, 0, n); };
  const ms = timeFrames(draw);
  const snap = snapshot(draw, 'heavy');
  results.push({ name: 'shade-bound (heavy frag, 2k tris)', shader: 'heavy', ms: +ms.toFixed(3), ntris: 2000, ...snap });
}

gl.destroyContext();
console.log(JSON.stringify({ renderer: RENDERER, size: { W, H }, frames: FRAMES, results }));
