/**
 * Render the 'fill' workload once through a GL driver (llvmpipe or GPU per env)
 * and write the framebuffer as raw RGBA (top-to-bottom, matching simdpipe's
 * orientation — GL readback is bottom-up so we flip). For compose-proof.mjs.
 * Usage: node gl-dump-raw.mjs <N> <out.rgba>
 */
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const gl = require('/home/monteslu/code/cliemu/native-gles');
import { makeScene, STRIDE } from './scene.mjs';

const N = parseInt(process.argv[2], 10);
const out = process.argv[3];

const G = { VERTEX_SHADER: 0x8B31, FRAGMENT_SHADER: 0x8B30, COMPILE_STATUS: 0x8B81, LINK_STATUS: 0x8B82, COLOR_BUFFER_BIT: 0x4000, DEPTH_BUFFER_BIT: 0x100, TRIANGLES: 4, FLOAT: 0x1406, ARRAY_BUFFER: 0x8892, STATIC_DRAW: 0x88E4, RGBA: 0x1908, UNSIGNED_BYTE: 0x1401, DEPTH_TEST: 0x0B71, LEQUAL: 0x0203 };
gl.createContext(N, N);
const sh = (t, s) => { const o = gl.glCreateShader(t); gl.glShaderSource(o, s); gl.glCompileShader(o); return o; };
const VS = `#version 300 es
in vec3 aPos; in vec3 aColor; uniform vec2 uRes; out vec3 vColor;
void main(){ vec2 ndc=vec2(aPos.x/uRes.x*2.0-1.0, 1.0-aPos.y/uRes.y*2.0); gl_Position=vec4(ndc, aPos.z*2.0-1.0, 1.0); vColor=aColor; }`;
const FS = `#version 300 es
precision highp float; in vec3 vColor; out vec4 c; void main(){ c=vec4(vColor,1.0); }`;
const p = gl.glCreateProgram(); gl.glAttachShader(p, sh(G.VERTEX_SHADER, VS)); gl.glAttachShader(p, sh(G.FRAGMENT_SHADER, FS)); gl.glLinkProgram(p); gl.glUseProgram(p);
gl.glUniform2f(gl.glGetUniformLocation(p, 'uRes'), N, N);

const scene = makeScene('fill', 200, N, N);
const data = new Float32Array(200 * 3 * 6);
for (let i = 0; i < 200 * 3; i++) { const s = i * STRIDE, d = i * 6; data[d] = scene[s]; data[d + 1] = scene[s + 1]; data[d + 2] = scene[s + 2]; data[d + 3] = scene[s + 3]; data[d + 4] = scene[s + 4]; data[d + 5] = scene[s + 5]; }
const vao = new Uint32Array(1); gl.glGenVertexArrays(1, vao); gl.glBindVertexArray(vao[0]);
const vbo = new Uint32Array(1); gl.glGenBuffers(1, vbo); gl.glBindBuffer(G.ARRAY_BUFFER, vbo[0]);
gl.glBufferData(G.ARRAY_BUFFER, new Uint8Array(data.buffer), G.STATIC_DRAW);
gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 3, G.FLOAT, false, 24, 0);
gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1, 3, G.FLOAT, false, 24, 12);
gl.glViewport(0, 0, N, N); gl.glEnable(G.DEPTH_TEST); gl.glDepthFunc(G.LEQUAL);
gl.glClearColor(0.094, 0.059, 0.063, 1); gl.glClear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT);
gl.glDrawArrays(G.TRIANGLES, 0, 600); gl.glFinish();

const px = new Uint8Array(N * N * 4);
gl.glReadPixels(0, 0, N, N, G.RGBA, G.UNSIGNED_BYTE, px);
// flip vertically to top-to-bottom
const flipped = new Uint8Array(N * N * 4);
for (let y = 0; y < N; y++) flipped.set(px.subarray((N - 1 - y) * N * 4, (N - y) * N * 4), y * N * 4);
writeFileSync(out, flipped);
gl.destroyContext();
