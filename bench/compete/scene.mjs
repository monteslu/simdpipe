/**
 * Shared scene definitions for the cross-renderer competition. Every renderer
 * (simdpipe, llvmpipe, GPU, native C) consumes the SAME geometry so the
 * comparison is apples-to-apples: identical triangle count, identical screen
 * coverage, identical per-fragment workload.
 *
 * Geometry is produced in SCREEN SPACE (pixels, origin top-left, x∈[0,W], y∈[0,H],
 * z∈[0,1]) — simdpipe's native input. The GL harness converts to NDC. Both sides
 * therefore rasterize the exact same pixels.
 *
 * Each vertex: { x, y, z, r, g, b, u, v }. RNG is deterministic (mulberry32) so
 * the buffers are bit-identical run to run.
 */

export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Returns a flat Float32Array of [x,y,z,r,g,b,u,v] * 3 * ntris (8 floats/vertex).
 * @param {string} kind 'fill' | 'balanced' | 'small'
 */
export function makeScene(kind, ntris, W, H, opts = {}) {
  const STRIDE = 8;
  const buf = new Float32Array(ntris * 3 * STRIDE);
  const seed = kind === 'fill' ? 1 : kind === 'balanced' ? 3 : 2;
  const rng = mulberry32(seed);
  const put = (i, x, y, z, r, g, b, u, v) => {
    const o = i * STRIDE;
    buf[o] = x; buf[o + 1] = y; buf[o + 2] = z;
    buf[o + 3] = r; buf[o + 4] = g; buf[o + 5] = b;
    buf[o + 6] = u; buf[o + 7] = v;
  };

  if (kind === 'fill') {
    // big overlapping triangles → fill-rate / overdraw bound
    for (let t = 0; t < ntris; t++) {
      const z = rng();
      put(t * 3 + 0, rng() * W * 0.4, rng() * H, z, rng(), rng(), rng(), rng(), rng());
      put(t * 3 + 1, W * 0.3 + rng() * W * 0.7, rng() * H * 0.4, z, rng(), rng(), rng(), rng(), rng());
      put(t * 3 + 2, rng() * W, H * 0.4 + rng() * H * 0.6, z, rng(), rng(), rng(), rng(), rng());
    }
  } else if (kind === 'balanced') {
    const s = Math.max(16, Math.min(W, H) / 8);
    for (let t = 0; t < ntris; t++) {
      const cx = rng() * (W - s), cy = rng() * (H - s), z = rng();
      put(t * 3 + 0, cx, cy, z, rng(), rng(), rng(), 0, 0);
      put(t * 3 + 1, cx + s, cy + rng() * s, z, rng(), rng(), rng(), 1, 0);
      put(t * 3 + 2, cx + rng() * s, cy + s, z, rng(), rng(), rng(), 0, 1);
    }
  } else { // 'small'
    const px = opts.px || 8;
    for (let t = 0; t < ntris; t++) {
      const cx = rng() * (W - px), cy = rng() * (H - px), z = rng();
      put(t * 3 + 0, cx, cy, z, rng(), rng(), rng(), 0, 0);
      put(t * 3 + 1, cx + px, cy, z, rng(), rng(), rng(), 1, 0);
      put(t * 3 + 2, cx, cy + px, z, rng(), rng(), rng(), 0, 1);
    }
  }
  return buf;
}

/** Workload table shared by all harnesses. */
export const WORKLOADS = [
  { name: 'fill (200 big tris, overdraw)', kind: 'fill', ntris: 200 },
  { name: 'balanced (2k mid tris)', kind: 'balanced', ntris: 2000 },
  { name: 'small (20k @ 8px)', kind: 'small', ntris: 20000, px: 8 },
];

export const STRIDE = 8;
