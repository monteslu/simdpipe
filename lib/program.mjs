/**
 * simdpipe programmable shading — Tier-2 (JS fragment shaders over SoA planes).
 *
 * The renderer rasterizes interpolated varyings (u,v,r,g,b,a) into a
 * full-framebuffer SoA G-buffer + a coverage mask (done in WASM/C). A fragment
 * shader then consumes those planes and writes the color buffer. The shader is a
 * plain JS function in Tier-2; Tier-1 will swap it for a JIT'd WASM module with
 * the same plane-in / color-out contract.
 *
 * This is what makes simdpipe a *programmable* renderer instead of fixed-function.
 *
 * Fragment shader signature:
 *   fragShade(planes, color, n, uniforms)
 *     planes  : { u, v, r, g, b, a, cover }  (Float32Array planes + Uint8Array cover, length n)
 *     color   : Uint32Array (length n) — write RGBA8 little-endian for covered pixels
 *     n       : width*height
 *     uniforms: caller-provided object
 *
 * For SoA speed, the shader processes the whole framebuffer (n pixels) in one
 * call — the same whole-tile dispatch model the JIT path uses.
 */

/** Build a Program from a fragment-shade function. */
export function createProgram({ fragShade }) {
  if (typeof fragShade !== 'function') throw new Error('createProgram: fragShade must be a function');
  return { fragShade };
}

/** A few stock fragment shaders (the bounded common set — Tier-0-ish). */
export const shaders = {
  /** flat vertex color */
  vertexColor() {
    return createProgram({
      fragShade(p, color, n) {
        const { r, g, b, a, cover } = p;
        for (let i = 0; i < n; i++) {
          if (!cover[i]) continue;
          const R = clamp255(r[i]), G = clamp255(g[i]), B = clamp255(b[i]), A = clamp255(a[i]);
          color[i] = R | (G << 8) | (B << 16) | (A << 24);
        }
      },
    });
  },

  /** sample a texture at (u,v), nearest, modulate by vertex color */
  texturedModulate(tex, tw, th) {
    return createProgram({
      fragShade(p, color, n) {
        const { u, v, r, g, b, cover } = p;
        const tmask = tw - 1, tmaskv = th - 1;
        for (let i = 0; i < n; i++) {
          if (!cover[i]) continue;
          let xi = (u[i] * tw) | 0, yi = (v[i] * th) | 0;
          xi &= tmask; yi &= tmaskv;
          const texel = tex[yi * tw + xi];
          const tr = texel & 255, tg = (texel >> 8) & 255, tb = (texel >> 16) & 255;
          const R = (tr * clampUnit(r[i])) | 0, G = (tg * clampUnit(g[i])) | 0, B = (tb * clampUnit(b[i])) | 0;
          color[i] = R | (G << 8) | (B << 16) | (255 << 24);
        }
      },
    });
  },

  /** a procedural shader: animated by a uniform `t`, shows programmability */
  procedural() {
    return createProgram({
      fragShade(p, color, n, uni) {
        const { u, v, cover } = p;
        const t = (uni && uni.t) || 0;
        for (let i = 0; i < n; i++) {
          if (!cover[i]) continue;
          const uu = u[i], vv = v[i];
          const R = clamp255(0.5 + 0.5 * Math.sin((uu * 10 + t)));
          const G = clamp255(0.5 + 0.5 * Math.sin((vv * 10 + t * 1.3)));
          const B = clamp255(0.5 + 0.5 * Math.sin(((uu + vv) * 8 + t * 0.7)));
          color[i] = R | (G << 8) | (B << 16) | (255 << 24);
        }
      },
    });
  },
};

function clamp255(x) { x = x * 255; return x < 0 ? 0 : x > 255 ? 255 : x | 0; }
function clampUnit(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
