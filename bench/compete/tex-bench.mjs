/**
 * Textured head-to-head: simdpipe vs llvmpipe on the most common real-renderer
 * workload — sampling a bound texture. Same geometry, same texture, both reading
 * the framebuffer back so output is verifiable (fingerprints must agree).
 *
 *   LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=0 node bench/compete/tex-bench.mjs --renderer llvmpipe
 *   node bench/compete/tex-bench.mjs --renderer simdpipe
 *
 * simdpipe runs nearest+affine (its fast tier) AND bilinear+perspective (matched
 * fidelity) so you can see both the like-for-like number and the fidelity lever.
 */
import { createRequire } from 'node:module';
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../../lib/index.mjs';
import { makeScene, STRIDE } from './scene.mjs';
import { fbStats, writePNG } from './png.mjs';

const argv = process.argv.slice(2);
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SIZE = getArg('--size', '512x512');
const [W, H] = SIZE.split('x').map(Number);
const FRAMES = parseInt(getArg('--frames', '80'), 10);
const WARMUP = parseInt(getArg('--warmup', '25'), 10);
const RENDERER = getArg('--renderer', 'simdpipe');
const DUMP = getArg('--dump', null);
const now = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

// Shared 256x256 checker texture (deterministic), in both u32 (simdpipe) and rgba8 (GL).
const TW = 256, TH = 256;
const texU32 = new Uint32Array(TW * TH);
const texRGBA = new Uint8Array(TW * TH * 4);
for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
  const c = ((x >> 4) ^ (y >> 4)) & 1;
  const r = c ? 0xc0 : 0x60, g = c ? 0x80 : 0x30, b = c ? 0x40 : 0x20;
  texU32[y * TW + x] = (0xff << 24) | (b << 16) | (g << 8) | r;
  const o = (y * TW + x) * 4; texRGBA[o] = r; texRGBA[o + 1] = g; texRGBA[o + 2] = b; texRGBA[o + 3] = 255;
}

const WL = [
  ['fill (200 big tris)', 'fill', 200, undefined],
  ['dense (8k big tris, overdraw)', 'fill', 8000, undefined],
  ['balanced (2k mid tris)', 'balanced', 2000, undefined],
  ['small (20k @ 8px)', 'small', 20000, 8],
];

function timeFrames(draw, finish) {
  for (let i = 0; i < WARMUP; i++) draw();
  if (finish) finish();
  const t = [];
  for (let i = 0; i < FRAMES; i++) { const a = now(); draw(); if (finish) finish(); t.push(now() - a); }
  return median(t);
}

const out = { renderer: RENDERER, size: { W, H }, results: [] };

if (RENDERER === 'simdpipe') {
  const sp = await createRenderer({ width: W, height: H });
  sp.bindTexture(texU32, TW, TH);
  const M = sp._module;
  const toUV = (sc, nt) => { const n = nt * 3, o = new Float32Array(n * VERTEX_STRIDE);
    for (let i = 0; i < n; i++) { const s = i * STRIDE, d = i * VERTEX_STRIDE;
      o[d]=sc[s];o[d+1]=sc[s+1];o[d+2]=sc[s+2];o[d+3]=1;o[d+4]=sc[s+3];o[d+5]=sc[s+4];o[d+6]=sc[s+5];o[d+7]=1;o[d+8]=sc[s+6];o[d+9]=sc[s+7]; }
    return o; };
  for (const [tier, flags] of [['nearest+affine', FLAGS.DEPTH_TEST | FLAGS.TEXTURE],
                               ['bilinear+persp', FLAGS.DEPTH_TEST | FLAGS.TEXTURE | FLAGS.BILINEAR | FLAGS.PERSP_CORRECT]]) {
    sp.setFlags(flags);
    for (const [name, kind, nt, px] of WL) {
      const sc = makeScene(kind, nt, W, H, px ? { px } : undefined);
      const buf = toUV(sc, nt); const p = sp.alloc(buf.length * 4); M.HEAPF32.set(buf, p >> 2);
      const draw = () => { sp.clear(0xff180f10, 1.0); M._sp_draw_triangles_flat(p, nt); };
      const ms = timeFrames(draw, null);
      draw(); const snap = fbStats(sp.getFramebuffer(), W, H);
      if (DUMP) writePNG(`${DUMP}-simdpipe-${tier.split('+')[0]}-${kind}.png`, sp.getFramebuffer(), W, H, false);
      sp.free(p);
      out.results.push({ name, tier, ms: +ms.toFixed(3), ...snap });
    }
  }
} else {
  // GL (llvmpipe / GPU)
  const require = createRequire(import.meta.url);
  const gl = require('/home/monteslu/code/cliemu/native-gles');
  const G = { VERTEX_SHADER:0x8B31, FRAGMENT_SHADER:0x8B30, COMPILE_STATUS:0x8B81, COLOR_BUFFER_BIT:0x4000,
    DEPTH_BUFFER_BIT:0x100, TRIANGLES:0x4, FLOAT:0x1406, ARRAY_BUFFER:0x8892, STATIC_DRAW:0x88E4, DEPTH_TEST:0x0B71,
    LEQUAL:0x0203, TEXTURE_2D:0x0DE1, TEXTURE0:0x84C0, RGBA:0x1908, UNSIGNED_BYTE:0x1401, RENDERER:0x1F01,
    TEXTURE_MIN_FILTER:0x2801, TEXTURE_MAG_FILTER:0x2800, NEAREST:0x2600, LINEAR:0x2601,
    TEXTURE_WRAP_S:0x2802, TEXTURE_WRAP_T:0x2803, REPEAT:0x2901 };
  gl.createContext(W, H);
  out.glRenderer = gl.glGetString(G.RENDERER);
  const VS = `#version 300 es
in vec3 aPos; in vec2 aUV; uniform vec2 uRes; out vec2 vUV;
void main(){ vec2 ndc=vec2(aPos.x/uRes.x*2.0-1.0, 1.0-aPos.y/uRes.y*2.0); gl_Position=vec4(ndc, aPos.z*2.0-1.0, 1.0); vUV=aUV; }`;
  const FS = `#version 300 es
precision highp float; in vec2 vUV; uniform sampler2D uTex; out vec4 o; void main(){ o=texture(uTex,vUV); }`;
  const comp = (t, s) => { const sh = gl.glCreateShader(t); gl.glShaderSource(sh, s); gl.glCompileShader(sh);
    if (!gl.glGetShaderiv(sh, G.COMPILE_STATUS)) { console.error(gl.glGetShaderInfoLog(sh)); process.exit(1); } return sh; };
  const p = gl.glCreateProgram(); gl.glAttachShader(p, comp(G.VERTEX_SHADER, VS)); gl.glAttachShader(p, comp(G.FRAGMENT_SHADER, FS));
  gl.glBindAttribLocation(p, 0, 'aPos'); gl.glBindAttribLocation(p, 1, 'aUV'); gl.glLinkProgram(p); gl.glUseProgram(p);
  gl.glUniform2f(gl.glGetUniformLocation(p, 'uRes'), W, H); gl.glUniform1i(gl.glGetUniformLocation(p, 'uTex'), 0);
  gl.glViewport(0, 0, W, H); gl.glEnable(G.DEPTH_TEST); gl.glDepthFunc(G.LEQUAL);
  const ta = new Uint32Array(1); gl.glGenTextures(1, ta); gl.glActiveTexture(G.TEXTURE0); gl.glBindTexture(G.TEXTURE_2D, ta[0]);
  gl.glTexImage2D(G.TEXTURE_2D, 0, G.RGBA, TW, TH, 0, G.RGBA, G.UNSIGNED_BYTE, texRGBA);
  gl.glTexParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.REPEAT); gl.glTexParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.REPEAT);
  const up = (sc, nt) => { const n = nt * 3, data = new Float32Array(n * 5);
    for (let i = 0; i < n; i++) { const s = i * STRIDE, d = i * 5; data[d]=sc[s];data[d+1]=sc[s+1];data[d+2]=sc[s+2];data[d+3]=sc[s+6];data[d+4]=sc[s+7]; }
    const va = new Uint32Array(1); gl.glGenVertexArrays(1, va); gl.glBindVertexArray(va[0]);
    const vb = new Uint32Array(1); gl.glGenBuffers(1, vb); gl.glBindBuffer(G.ARRAY_BUFFER, vb[0]);
    gl.glBufferData(G.ARRAY_BUFFER, new Uint8Array(data.buffer), G.STATIC_DRAW);
    gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 3, G.FLOAT, false, 20, 0);
    gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1, 2, G.FLOAT, false, 20, 12); return n; };
  const finish = () => gl.glFinish();
  for (const [tier, filt] of [['nearest', G.NEAREST], ['bilinear', G.LINEAR]]) {
    gl.glTexParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, filt); gl.glTexParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, filt);
    for (const [name, kind, nt, px] of WL) {
      const sc = makeScene(kind, nt, W, H, px ? { px } : undefined); const n = up(sc, nt);
      const draw = () => { gl.glClearColor(0.06,0.06,0.09,1); gl.glClear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT); gl.glDrawArrays(G.TRIANGLES, 0, n); };
      const ms = timeFrames(draw, finish);
      draw(); finish();
      const px2 = new Uint8Array(W * H * 4); gl.glReadPixels(0, 0, W, H, G.RGBA, G.UNSIGNED_BYTE, px2);
      const snap = fbStats(px2, W, H);
      if (DUMP) writePNG(`${DUMP}-${out.glRenderer && out.glRenderer.includes('llvm') ? 'llvmpipe' : 'gpu'}-${tier}-${kind}.png`, px2, W, H, true);
      out.results.push({ name, tier, ms: +ms.toFixed(3), ...snap });
    }
  }
  gl.destroyContext();
}

console.log(JSON.stringify(out));
