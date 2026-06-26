/**
 * Pure-JS scalar software rasterizer — the baseline to beat.
 *
 * Same edge-function algorithm as the WASM core, but scalar (one pixel at a
 * time) and in plain JS. This isolates what WASM+SIMD actually buys vs. an
 * idiomatic JS implementation a game might otherwise use.
 *
 * Same flat vertex layout: x y z invw r g b a u v (10 floats/vertex).
 */

export const VERTEX_STRIDE = 10;

export function createBaseline({ width, height }) {
  const W = width, H = height;
  const color = new Uint8Array(W * H * 4);
  const depth = new Float32Array(W * H);
  let depthTest = true, perspCorrect = true;
  let stats = { tris: 0, fragTested: 0, fragShaded: 0 };

  function clear(rgba = 0xff000000, d = 1.0) {
    const rB = rgba & 0xff, gB = (rgba >>> 8) & 0xff, bB = (rgba >>> 16) & 0xff, aB = (rgba >>> 24) & 0xff;
    for (let i = 0; i < W * H; i++) {
      color[i * 4] = rB; color[i * 4 + 1] = gB; color[i * 4 + 2] = bB; color[i * 4 + 3] = aB;
      if (depthTest) depth[i] = d;
    }
  }

  function drawTriangles(verts, ntris) {
    for (let t = 0; t < ntris; t++) {
      tri(verts, t * 30);
    }
  }

  function tri(v, o) {
    const x0 = v[o], y0 = v[o + 1], z0 = v[o + 2];
    const x1 = v[o + 10], y1 = v[o + 11], z1 = v[o + 12];
    const x2 = v[o + 20], y2 = v[o + 21], z2 = v[o + 22];
    const r0 = v[o + 4], g0 = v[o + 5], b0 = v[o + 6];
    const r1 = v[o + 14], g1 = v[o + 15], b1 = v[o + 16];
    const r2 = v[o + 24], g2 = v[o + 25], b2 = v[o + 26];

    const area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (area2 === 0) return;
    const sgn = area2 < 0 ? -1 : 1;
    const invArea2 = 1 / area2;
    stats.tris++;

    let minx = Math.floor(Math.min(x0, x1, x2)); let maxx = Math.ceil(Math.max(x0, x1, x2));
    let miny = Math.floor(Math.min(y0, y1, y2)); let maxy = Math.ceil(Math.max(y0, y1, y2));
    if (minx < 0) minx = 0; if (miny < 0) miny = 0;
    if (maxx > W) maxx = W; if (maxy > H) maxy = H;

    const A0 = y1 - y2, B0 = x2 - x1, C0 = -(A0 * x1 + B0 * y1);
    const A1 = y2 - y0, B1 = x0 - x2, C1 = -(A1 * x2 + B1 * y2);
    const A2 = y0 - y1, B2 = x1 - x0, C2 = -(A2 * x0 + B2 * y0);

    for (let y = miny; y < maxy; y++) {
      const yc = y + 0.5;
      for (let x = minx; x < maxx; x++) {
        stats.fragTested++;
        const xc = x + 0.5;
        const e0 = (A0 * xc + B0 * yc + C0) * sgn;
        const e1 = (A1 * xc + B1 * yc + C1) * sgn;
        const e2 = (A2 * xc + B2 * yc + C2) * sgn;
        if (e0 < 0 || e1 < 0 || e2 < 0) continue;

        const w0 = (A0 * xc + B0 * yc + C0) * invArea2;
        const w1 = (A1 * xc + B1 * yc + C1) * invArea2;
        const w2 = (A2 * xc + B2 * yc + C2) * invArea2;
        const z = w0 * z0 + w1 * z1 + w2 * z2;
        const idx = y * W + x;
        if (depthTest) {
          if (z >= depth[idx]) continue;
        }
        let rr = w0 * r0 + w1 * r1 + w2 * r2;
        let gg = w0 * g0 + w1 * g1 + w2 * g2;
        let bb = w0 * b0 + w1 * b1 + w2 * b2;
        rr = rr < 0 ? 0 : rr > 1 ? 1 : rr;
        gg = gg < 0 ? 0 : gg > 1 ? 1 : gg;
        bb = bb < 0 ? 0 : bb > 1 ? 1 : bb;
        color[idx * 4] = (rr * 255) | 0;
        color[idx * 4 + 1] = (gg * 255) | 0;
        color[idx * 4 + 2] = (bb * 255) | 0;
        color[idx * 4 + 3] = 255;
        if (depthTest) depth[idx] = z;
        stats.fragShaded++;
      }
    }
  }

  return {
    width: W, height: H,
    clear, drawTriangles,
    setDepthTest(b) { depthTest = b; },
    setPerspCorrect(b) { perspCorrect = b; },
    getFramebuffer() { return color; },
    resetStats() { stats = { tris: 0, fragTested: 0, fragShaded: 0 }; },
    stats() { return { ...stats }; },
  };
}
