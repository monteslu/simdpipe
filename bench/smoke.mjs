/** Smoke test: render a couple of triangles, verify pixels are sane. */
import { createRenderer, FLAGS, VERTEX_STRIDE } from '../lib/index.mjs';

function vtx(x, y, z, r, g, b, a = 1) {
  return [x, y, z, 1.0 /*invw*/, r, g, b, a, 0, 0];
}

const W = 64, H = 64;
const r = await createRenderer({ width: W, height: H });

// vertex-color path (no texture), depth on, persp on (invw=1 so affine==persp)
r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);
r.clear(0xff000000 /*opaque black (A=ff,B=0,G=0,R=0)*/, 1.0);

// One big red triangle covering the lower-left half.
const tris = new Float32Array([
  ...vtx(2, 2, 0.5, 1, 0, 0),
  ...vtx(2, 60, 0.5, 1, 0, 0),
  ...vtx(60, 60, 0.5, 1, 0, 0),
]);
r.resetStats();
r.drawTriangles(tris, 1);

const fb = r.getFramebuffer();
const px = (x, y) => {
  const i = (y * W + x) * 4;
  return [fb[i], fb[i + 1], fb[i + 2], fb[i + 3]];
};

// Inside the triangle (lower-left): should be red.
const inside = px(10, 50);
// Outside (upper-right): should be black.
const outside = px(55, 8);

console.log('stats:', r.stats());
console.log('inside (10,50)  =', inside, '(expect ~[255,0,0,255])');
console.log('outside (55,8)  =', outside, '(expect [0,0,0,255])');

let pass = true;
if (!(inside[0] > 200 && inside[1] < 40 && inside[2] < 40)) { console.error('FAIL: inside not red'); pass = false; }
if (!(outside[0] < 40 && outside[1] < 40 && outside[2] < 40)) { console.error('FAIL: outside not black'); pass = false; }

// Depth test: draw a green triangle BEHIND (z=0.9) the red one over the same area;
// it must NOT overwrite where red (z=0.5) already is.
const green = new Float32Array([
  ...vtx(2, 2, 0.9, 0, 1, 0),
  ...vtx(2, 60, 0.9, 0, 1, 0),
  ...vtx(60, 60, 0.9, 0, 1, 0),
]);
r.drawTriangles(green, 1);
const stillRed = px(10, 50);
console.log('after green-behind (10,50) =', stillRed, '(expect still red — depth test)');
if (!(stillRed[0] > 200 && stillRed[1] < 40)) { console.error('FAIL: depth test did not reject behind tri'); pass = false; }

// Draw blue IN FRONT (z=0.1): must overwrite.
const blue = new Float32Array([
  ...vtx(2, 2, 0.1, 0, 0, 1),
  ...vtx(2, 60, 0.1, 0, 0, 1),
  ...vtx(60, 60, 0.1, 0, 0, 1),
]);
r.drawTriangles(blue, 1);
const nowBlue = px(10, 50);
console.log('after blue-in-front (10,50) =', nowBlue, '(expect blue)');
if (!(nowBlue[2] > 200 && nowBlue[0] < 40)) { console.error('FAIL: front tri did not overwrite'); pass = false; }

console.log(pass ? '\n✅ SMOKE PASS' : '\n❌ SMOKE FAIL');
process.exit(pass ? 0 : 1);
