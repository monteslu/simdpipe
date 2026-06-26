/**
 * simdpipe GL — a minimal OpenGL-ES-shaped front end over the rasterizer.
 *
 * Not a conformant WebGL2 context (that's a large roadmap item); this is a
 * GL-flavored, programmable-pipeline API that exercises the full stack: a vertex
 * stage (MVP transform → near clip → perspective divide → viewport), a fragment
 * program (Tier-2 shader), depth, and clear. Enough to draw real 3D (a spinning
 * textured cube) through a familiar shape.
 *
 * The vertex format is interleaved floats; you describe it with attributes.
 */
import { createRenderer, FLAGS, VERTEX_STRIDE } from './index.mjs';
import * as mat4 from './mat4.mjs';

/**
 * @param {{width:number,height:number}} opts
 */
export async function createGL({ width, height }) {
  const r = await createRenderer({ width, height });
  r.setFlags(FLAGS.DEPTH_TEST | FLAGS.PERSP_CORRECT);

  const state = {
    mvp: mat4.identity(),
    viewport: [0, 0, width, height],
    program: null,
    uniforms: {},
    clearColor: 0xff000000,
    depthTest: true,
    near: 0.1,
  };

  // scratch for transformed verts (grows as needed)
  let scratch = new Float32Array(0);

  function ensureScratch(floats) {
    if (scratch.length < floats) scratch = new Float32Array(floats);
    return scratch;
  }

  /**
   * Draw indexed/array triangles.
   * @param {Float32Array} attribs interleaved vertex data
   * @param {number} stride floats per vertex
   * @param {{pos:number, color?:number, uv?:number}} layout offsets (in floats) of attributes
   * @param {number} count number of vertices (multiple of 3)
   * @param {Uint16Array|Uint32Array} [indices]
   */
  function drawArrays(attribs, stride, layout, count, indices) {
    const ntris = (indices ? indices.length : count) / 3;
    const buf = ensureScratch(ntris * 3 * VERTEX_STRIDE);
    const [vx, vy, vw, vh] = state.viewport;
    const mvp = state.mvp;
    const near = state.near;

    let out = 0; // running vertex count actually emitted (post near-cull)
    const get = (vi) => indices ? indices[vi] : vi;

    for (let t = 0; t < ntris; t++) {
      // gather 3 verts, transform
      const tri = [];
      let allBehind = true, anyBehind = false;
      for (let k = 0; k < 3; k++) {
        const vi = get(t * 3 + k);
        const o = vi * stride;
        const x = attribs[o + layout.pos], y = attribs[o + layout.pos + 1], z = attribs[o + layout.pos + 2];
        const clip = mat4.transformPoint(mvp, x, y, z);
        const w = clip[3];
        if (w > near) allBehind = false; else anyBehind = true;
        const col = layout.color != null ? [attribs[o + layout.color], attribs[o + layout.color + 1], attribs[o + layout.color + 2], 1] : [1, 1, 1, 1];
        const uv = layout.uv != null ? [attribs[o + layout.uv], attribs[o + layout.uv + 1]] : [0, 0];
        tri.push({ clip, col, uv });
      }
      // near-plane cull (whole triangle); proper clipping is a roadmap item
      if (allBehind || anyBehind) continue;

      for (let k = 0; k < 3; k++) {
        const { clip, col, uv } = tri[k];
        const invw = 1 / clip[3];
        // NDC
        const ndcx = clip[0] * invw, ndcy = clip[1] * invw, ndcz = clip[2] * invw;
        // viewport: NDC [-1,1] → screen; flip Y (screen is top-left origin)
        const sx = vx + (ndcx * 0.5 + 0.5) * vw;
        const sy = vy + (1 - (ndcy * 0.5 + 0.5)) * vh;
        const sz = ndcz * 0.5 + 0.5; // [0,1]
        const b = out * VERTEX_STRIDE;
        buf[b] = sx; buf[b + 1] = sy; buf[b + 2] = sz; buf[b + 3] = invw;
        buf[b + 4] = col[0]; buf[b + 5] = col[1]; buf[b + 6] = col[2]; buf[b + 7] = col[3];
        buf[b + 8] = uv[0]; buf[b + 9] = uv[1];
        out++;
      }
    }
    const emittedTris = out / 3;
    if (emittedTris > 0) r.drawProgram(buf, emittedTris, state.program, state.uniforms);
  }

  return {
    width: r.width, height: r.height,
    // state setters (GL-ish)
    useProgram(p) { state.program = p; },
    uniforms(u) { state.uniforms = u; },
    setMVP(m) { state.mvp = m; },
    viewport(x, y, w, h) { state.viewport = [x, y, w, h]; },
    clearColor(rgba) { state.clearColor = rgba >>> 0; },
    enableDepth(b) { state.depthTest = b; r.setFlags(b ? (r.getFlags() | FLAGS.DEPTH_TEST) : (r.getFlags() & ~FLAGS.DEPTH_TEST)); },
    setFlags(f) { r.setFlags(f); },

    clear() { r.clear(state.clearColor, 1.0); },
    drawArrays,

    getFramebuffer: () => r.getFramebuffer(),
    getImageData: () => r.getImageData(),
    stats: () => r.stats(),
    resetStats: () => r.resetStats(),
    mat4,
    _renderer: r,
  };
}
