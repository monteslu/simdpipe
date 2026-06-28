/**
 * simdpipe — portable SIMD software rasterizer (WASM + 128-bit SIMD).
 *
 * Isomorphic ESM entry. Same code in Node and the browser. Produces a
 * framebuffer (RGBA8) in WASM linear memory; the host reads it directly.
 *
 * Lineage: the Mesa "-pipe" software-renderer family (softpipe/llvmpipe/lavapipe).
 *
 * @module simdpipe
 */

import createModule from '../dist/simdpipe.mjs';
import { makeJITProgram } from './shader-jit.mjs';

/** Fidelity flags — turn expensive work off to go faster. */
export const FLAGS = Object.freeze({
  DEPTH_TEST: 1 << 0,
  BILINEAR: 1 << 1,
  PERSP_CORRECT: 1 << 2,
  BLEND: 1 << 3,
  TEXTURE: 1 << 4,
});

/** Floats per vertex in the flat draw buffer: x y z invw r g b a u v */
export const VERTEX_STRIDE = 10;

/**
 * @typedef {Object} Renderer
 * @property {number} width
 * @property {number} height
 * @property {(rgba?:number, depth?:number)=>void} clear
 * @property {(flags:number)=>void} setFlags
 * @property {()=>number} getFlags
 * @property {(pixels:Uint32Array|Uint8Array, w:number, h:number)=>void} bindTexture
 * @property {(verts:Float32Array, ntris:number)=>void} drawTriangles
 * @property {()=>Uint8Array} getFramebuffer  RGBA8 view (length w*h*4) over linear memory
 * @property {()=>Uint8ClampedArray} getImageData  Uint8ClampedArray view (for canvas ImageData)
 * @property {()=>{tris:number,fragTested:number,fragShaded:number}} stats
 * @property {()=>void} resetStats
 * @property {(n:number)=>number} alloc  alloc n bytes in linear memory, returns ptr
 * @property {(ptr:number)=>void} free
 * @property {any} _module  the raw emscripten module (escape hatch)
 */

/**
 * Create a renderer backed by a fresh WASM instance.
 * @param {{width:number,height:number}} opts
 * @returns {Promise<Renderer>}
 */
export async function createRenderer({ width, height }) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`simdpipe: invalid size ${width}x${height}`);
  }
  const Module = await createModule();

  const ok = Module._sp_init(width, height);
  if (!ok) throw new Error(`simdpipe: sp_init failed for ${width}x${height}`);

  // Adaptive tiling: the rasterizer auto-picks an 8px or 16px traversal tile per draw
  // from the mean triangle size (big-and-few → 16 amortizes per-tile setup over big
  // spans, ~1.8x on fill; dense/mid → 8 rejects empty space better). Output-identical
  // to a fixed tile (edge values recompute from the absolute row origin each scanline,
  // so a pixel's coverage never depends on where its tile begins) → zero fidelity cost,
  // on by default. Disable with SP_ADAPT_TILE=0.
  if (Module._sp_set_adapt_tile) Module._sp_set_adapt_tile(process.env.SP_ADAPT_TILE === '0' ? 0 : 1);

  const W = Module._sp_width();
  const H = Module._sp_height();

  // Helper: get a fresh typed-array view over the framebuffer. We re-fetch on
  // demand because ALLOW_MEMORY_GROWTH can detach the backing ArrayBuffer.
  const colorPtr = () => Module._sp_color_ptr();
  let gbReady = false;
  let boundTex = null;

  /** @type {Renderer} */
  const r = {
    width: W,
    height: H,

    clear(rgba = 0x00000000, depth = 1.0) {
      Module._sp_clear(rgba >>> 0, depth);
    },

    setFlags(flags) { Module._sp_set_flags(flags >>> 0); },
    getFlags() { return Module._sp_get_flags() >>> 0; },

    bindTexture(pixels, w, h) {
      // Copy texture into linear memory (so the WASM side has a stable pointer).
      const bytes = w * h * 4;
      const ptr = Module._sp_alloc(bytes);
      const src = pixels instanceof Uint8Array
        ? pixels
        : new Uint8Array(pixels.buffer, pixels.byteOffset, bytes);
      Module.HEAPU8.set(src.subarray(0, bytes), ptr);
      Module._sp_bind_texture(ptr, w, h);
      boundTex = { ptr, w, h };
      // (Phase 0: we leak texture allocations on rebind; fine for benches.)
    },

    drawTriangles(verts, ntris) {
      const floats = ntris * 3 * VERTEX_STRIDE;
      const ptr = Module._sp_alloc(floats * 4);
      Module.HEAPF32.set(verts.subarray(0, floats), ptr >> 2);
      Module._sp_draw_triangles_flat(ptr, ntris);
      Module._sp_free(ptr);
    },

    /**
     * Programmable draw: rasterize varyings into the G-buffer, then run a
     * fragment shader (Tier-2 JS program) over the covered pixels.
     * @param {Float32Array} verts
     * @param {number} ntris
     * @param {{fragShade:Function}} program
     * @param {object} [uniforms]
     */
    drawProgram(verts, ntris, program, uniforms) {
      if (!gbReady) { if (!Module._sp_gbuffer_init()) throw new Error('gbuffer init failed'); gbReady = true; }
      Module._sp_gbuffer_clear();
      // Tell the rasterizer which varying planes this program reads, so it can skip
      // interpolating + storing the rest (the shade-bound bandwidth win). A JIT
      // program declares its usage (needsUV→U/V from texture() or direct uv reads,
      // needsColor→R/G/B/A); a JS fallback reads planes opaquely in JS, so we must
      // write them all (0x3).
      if (Module._sp_gbuffer_set_varymask) {
        const VARY_UV = 0x1, VARY_COLOR = 0x2;
        const mask = program.jit
          ? ((program.needsUV ? VARY_UV : 0) | (program.needsColor !== false ? VARY_COLOR : 0))
          : (VARY_UV | VARY_COLOR);
        Module._sp_gbuffer_set_varymask(mask);
      }
      const floats = ntris * 3 * VERTEX_STRIDE;
      const ptr = Module._sp_alloc(floats * 4);
      Module.HEAPF32.set(verts.subarray(0, floats), ptr >> 2);
      Module._sp_draw_gbuffer_flat(ptr, ntris);
      Module._sp_free(ptr);
      // build SoA plane views (re-fetch each call; memory may have grown)
      const n = W * H;
      const f32 = (p) => new Float32Array(Module.HEAPU8.buffer, p, n);
      const planes = {
        u: f32(Module._sp_gb_u()), v: f32(Module._sp_gb_v()),
        r: f32(Module._sp_gb_r()), g: f32(Module._sp_gb_g()),
        b: f32(Module._sp_gb_b()), a: f32(Module._sp_gb_a()),
        cover: new Uint8Array(Module.HEAPU8.buffer, Module._sp_gb_cover(), n),
      };
      if (boundTex) {
        planes.tex = new Uint32Array(Module.HEAPU8.buffer, boundTex.ptr, boundTex.w * boundTex.h);
        planes.texW = boundTex.w; planes.texH = boundTex.h;
      }
      // raw pointers for JIT backends that operate directly on linear memory
      planes._ptrs = {
        u: Module._sp_gb_u(), v: Module._sp_gb_v(),
        r: Module._sp_gb_r(), g: Module._sp_gb_g(),
        b: Module._sp_gb_b(), a: Module._sp_gb_a(),
        cover: Module._sp_gb_cover(), color: colorPtr(),
        // texture (if bound): the JIT kernel gathers texels directly from here.
        tex: boundTex ? boundTex.ptr : 0,
        texW: boundTex ? boundTex.w : 0,
        texH: boundTex ? boundTex.h : 0,
      };
      const color = new Uint32Array(Module.HEAPU8.buffer, colorPtr(), n);
      program.fragShade(planes, color, n, uniforms);
    },

    getFramebuffer() {
      return new Uint8Array(Module.HEAPU8.buffer, colorPtr(), W * H * 4);
    },
    getImageData() {
      return new Uint8ClampedArray(Module.HEAPU8.buffer, colorPtr(), W * H * 4);
    },

    stats() {
      return {
        tris: Module._sp_stat_tris(),
        fragTested: Module._sp_stat_frag_tested(),
        fragShaded: Module._sp_stat_frag_shaded(),
      };
    },
    resetStats() { Module._sp_reset_stats(); },

    alloc(n) { return Module._sp_alloc(n); },
    free(ptr) { Module._sp_free(ptr); },

    /**
     * Compile a GLSL-ish fragment shader into a Program. Tries the Tier-1 WASM
     * JIT (generated SIMD kernel the engine compiles to native), including
     * texture() sampling; falls back to the Tier-2 JS backend only for source
     * the codegen can't yet emit (never for pure-calculation shaders).
     * @param {string} src
     * @param {{bilinear?:boolean, coarse?:boolean}} [opts]
     *   bilinear: texture() filter (4-tap) vs nearest (1-tap).
     *   coarse: 2×1 variable-rate shading — evaluate the fragment kernel once per
     *     horizontal pixel pair (half the transcendental throughput). A VISIBLE
     *     fidelity drop (half horizontal shading rate, like hardware VRS 2×1), so
     *     opt-in; also enabled globally by SP_COARSE=1. Ignored for texture shaders.
     * @returns {{fragShade:Function, jit:boolean, reason?:string, coarse?:boolean}}
     */
    createJITProgram(src, opts) { return makeJITProgram(Module, src, opts); },

    _module: Module,
  };
  return r;
}

export default { createRenderer, FLAGS, VERTEX_STRIDE };
