/**
 * JS backend for the shader IR — evaluate per-pixel. Produces a Program-compatible
 * fragShade. This is the reference/Tier-2 path and the correctness oracle the
 * WASM-JIT backend is checked against.
 */
import { parseShader, swizzleIndices, BUILTINS } from './shader-ir.mjs';

/**
 * compile GLSL-ish source → { fragShade, uniforms } (a Program).
 * @param {string} src
 * @param {{bilinear?:boolean}} [opts] texture() filter (matches the JIT's option)
 */
export function compileJS(src, opts = {}) {
  const { uniforms, ast } = parseShader(src);
  const bilinear = !!opts.bilinear;

  function fragShade(planes, color, n, uni = {}) {
    const { u, v, r, g, b, a, cover } = planes;
    // texture access closure
    const tex = planes.tex, tw = planes.texW | 0, th = planes.texH | 0;
    const texel = (xi, yi) => {
      xi &= (tw - 1); yi &= (th - 1);
      const t = tex[yi * tw + xi];
      return [(t & 255) / 255, ((t >> 8) & 255) / 255, ((t >> 16) & 255) / 255, ((t >> 24) & 255) / 255];
    };
    for (let i = 0; i < n; i++) {
      if (!cover[i]) continue;
      const env = {
        uv: [u[i], v[i]],
        color: [r[i], g[i], b[i], a[i]],
        uni,
        sampleTex(uu, vv) {
          if (!tex) return [1, 1, 1, 1];
          if (!bilinear) return texel((uu * tw) | 0, (vv * th) | 0);
          // bilinear: -0.5 pixel-center shift, floor base, frac weights
          const fx = uu * tw - 0.5, fy = vv * th - 0.5;
          const flx = Math.floor(fx), fly = Math.floor(fy);
          const txw = fx - flx, tyw = fy - fly;
          const x0 = flx | 0, y0 = fly | 0, x1 = x0 + 1, y1 = y0 + 1;
          const t00 = texel(x0, y0), t10 = texel(x1, y0), t01 = texel(x0, y1), t11 = texel(x1, y1);
          const out = new Array(4);
          for (let c = 0; c < 4; c++) {
            const top = t00[c] * (1 - txw) + t10[c] * txw;
            const bot = t01[c] * (1 - txw) + t11[c] * txw;
            out[c] = top * (1 - tyw) + bot * tyw;
          }
          return out;
        },
      };
      const out = evalNode(ast, env); // length-4 expected
      const R = clamp255(out[0]), G = clamp255(out[1] ?? 0), B = clamp255(out[2] ?? 0), A = clamp255(out[3] ?? 1);
      color[i] = R | (G << 8) | (B << 16) | (A << 24);
    }
  }
  return { fragShade, uniforms };
}

function evalNode(node, env) {
  switch (node.op) {
    case 'num': return [node.v];
    case 'varying': return env[node.name].slice();
    case 'uniform': { const val = env.uni[node.name]; return Array.isArray(val) ? val.slice() : [val ?? 0]; }
    case 'tex': { const uv = evalNode(node.uv, env); return env.sampleTex(uv[0], uv[1]); }
    case 'vec': { const out = []; for (const p of node.parts) out.push(...evalNode(p, env)); return out; }
    case 'swizzle': { const e = evalNode(node.e, env); return swizzleIndices(node.sw).map((k) => e[k]); }
    case 'neg': return evalNode(node.e, env).map((x) => -x);
    case 'bin': return binop(node.o, evalNode(node.a, env), evalNode(node.b, env));
    case 'call': return callBuiltin(node.fn, node.args.map((x) => evalNode(x, env)));
    default: throw new Error('eval: bad node ' + node.op);
  }
}

function binop(o, a, b) {
  // scalar broadcast
  if (a.length === 1 && b.length > 1) a = b.map(() => a[0]);
  if (b.length === 1 && a.length > 1) b = a.map(() => b[0]);
  const n = Math.max(a.length, b.length), out = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = a[i] ?? a[0], y = b[i] ?? b[0];
    out[i] = o === '+' ? x + y : o === '-' ? x - y : o === '*' ? x * y : x / y;
  }
  return out;
}

function callBuiltin(fn, args) {
  const a = args[0], b = args[1], c = args[2];
  const map1 = (f) => a.map(f);
  switch (fn) {
    case 'sin': return map1(Math.sin); case 'cos': return map1(Math.cos);
    case 'abs': return map1(Math.abs); case 'floor': return map1(Math.floor);
    case 'sqrt': return map1(Math.sqrt);
    case 'fract': return map1((x) => x - Math.floor(x));
    case 'min': return comp2(a, b, Math.min); case 'max': return comp2(a, b, Math.max);
    case 'mod': return comp2(a, b, (x, y) => x - y * Math.floor(x / y));
    case 'step': return comp2(a, b, (edge, x) => (x < edge ? 0 : 1));
    case 'clamp': return comp3(a, b, c, (x, lo, hi) => Math.min(Math.max(x, lo), hi));
    case 'mix': return comp3(a, b, c, (x, y, t) => x * (1 - t) + y * t);
    default: throw new Error('eval: builtin ' + fn);
  }
}
function bc(x, n) { return x.length === 1 ? new Array(n).fill(x[0]) : x; }
function comp2(a, b, f) { const n = Math.max(a.length, b.length); a = bc(a, n); b = bc(b, n); return a.map((x, i) => f(x, b[i])); }
function comp3(a, b, c, f) { const n = Math.max(a.length, b.length, c.length); a = bc(a, n); b = bc(b, n); c = bc(c, n); return a.map((x, i) => f(x, b[i], c[i])); }

function clamp255(x) { x = x * 255; return x < 0 ? 0 : x > 255 ? 255 : x | 0; }
