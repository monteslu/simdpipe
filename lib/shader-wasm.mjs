/**
 * WASM-SIMD backend for the shader IR — the Tier-1 JIT.
 *
 * Walks the IR and emits a v128 kernel that shades 4 fragments per instruction
 * (SoA). The host engine (V8) compiles the generated module to native machine
 * code; we call it over the whole framebuffer. This makes authored GLSL run as
 * JIT'd SIMD, not a JS callback.
 *
 * Subset: the math IR (varyings uv/color, uniforms, +-* /, swizzle, vecN,
 * sin/cos/abs/floor/fract/sqrt/min/max/clamp/mix/step/mod). Texture sampling is
 * NOT emitted here (WASM has no gather); shaders that call texture() fall back to
 * the JS backend. Procedural / vertex-color-math shaders — the SIMD-bound ones —
 * JIT fully.
 *
 * Values during codegen are "rvalues": arrays of WASM local indices (v128), one
 * per component (length 1..4). Scalars broadcast.
 */
import { parseShader, swizzleIndices } from './shader-ir.mjs';
import { buildModule, T, OP, SIMD, simd, E } from './wasm-emit.mjs';

const V128 = 0x7b;

/** @returns {{supported:boolean, reason?:string, bytes?:Uint8Array, uniformOrder?:string[]}} */
export function compileWASM(src) {
  let parsed;
  try { parsed = parseShader(src); } catch (e) { return { supported: false, reason: 'parse: ' + e.message }; }
  const { uniforms, ast } = parsed;
  if (usesTexture(ast)) return { supported: false, reason: 'texture() not supported by JIT backend (use JS)' };

  // uniform calling order (each uniform passed as N f32 params -> splatted)
  const uniformOrder = Object.keys(uniforms);

  const g = new Gen(uniforms, uniformOrder);
  let bytes;
  try { bytes = g.build(ast); } catch (e) { return { supported: false, reason: 'codegen: ' + e.message }; }
  return { supported: true, bytes, uniformOrder, uniforms };
}

function usesTexture(node) {
  if (!node || typeof node !== 'object') return false;
  if (node.op === 'tex') return true;
  for (const k of ['e', 'a', 'b', 'uv']) if (node[k] && usesTexture(node[k])) return true;
  if (node.parts) for (const p of node.parts) if (usesTexture(p)) return true;
  if (node.args) for (const p of node.args) if (usesTexture(p)) return true;
  return false;
}

class Gen {
  constructor(uniforms, uniformOrder) {
    this.uniforms = uniforms;
    this.uniformOrder = uniformOrder;
    // params: planes ptrs (u,v,r,g,b,a -> 0..5), coverPtr 6, colorPtr 7, n4 8,
    //         then one f32 param per uniform-scalar-component starting at 9.
    this.compsOf = (t) => ({ float: 1, vec2: 2, vec3: 3, vec4: 4 }[t]);
    this.params = [T.i32, T.i32, T.i32, T.i32, T.i32, T.i32, T.i32, T.i32, T.i32];
    this.uniBase = this.params.length; // first uniform f32 param index
    this.uniLayout = {}; // name -> {param0, comps}
    let p = this.uniBase;
    for (const name of uniformOrder) {
      const c = this.compsOf(uniforms[name]);
      this.uniLayout[name] = { param0: p, comps: c };
      for (let i = 0; i < c; i++) this.params.push(T.f32);
      p += c;
    }
    this.locals = [];           // declared local types after params
    this.body = [];
    this.b = (...xs) => this.body.push(...xs.flat());
    this.localBase = this.params.length;
    // reserved working locals
    this.L_i = this.newLocal(T.i32);
    this.L_off = this.newLocal(T.i32);
    // splatted uniform v128 locals (filled at prologue)
    this.uniSplat = {}; // name -> [v128 local per component]
  }

  newLocal(t) { const idx = this.localBase + this.locals.length; this.locals.push(t); return idx; }

  build(ast) {
    // prologue: splat each uniform component into a v128 local
    for (const name of this.uniformOrder) {
      const { param0, comps } = this.uniLayout[name];
      const arr = [];
      for (let c = 0; c < comps; c++) {
        const L = this.newLocal(V128);
        this.b(OP.local_get, ...E.uleb(param0 + c)); this.b(simd(SIMD.f32x4_splat)); this.b(OP.local_set, ...E.uleb(L));
        arr.push(L);
      }
      this.uniSplat[name] = arr;
    }
    // i = 0
    this.b(OP.i32_const, ...E.sleb(0), OP.local_set, ...E.uleb(this.L_i));
    this.b(OP.block, 0x40); this.b(OP.loop, 0x40);
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.local_get, ...E.uleb(8), OP.i32_ge_s, OP.br_if, ...E.uleb(1));
    // off = i*16
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(16), OP.i32_mul, OP.local_set, ...E.uleb(this.L_off));

    // evaluate the fragment expr → up to 4 v128 component locals
    const rgba = this.emit(ast);
    const comp = (i, def) => rgba[i] != null ? rgba[i] : (def != null ? this.constV(def) : this.constV(0));
    const R = comp(0, 0), G = comp(1, 0), B = comp(2, 0), A = rgba[3] != null ? rgba[3] : this.constV(1);

    // pack: clamp [0,1], *255, trunc, shift/or
    const packed = this.packRGBA(R, G, B, A);

    // coverage mask: load 4 cover bytes (i8) at coverPtr(param6)+i*4, widen to
    // i32 lanes, build a per-lane all-ones/zero mask, and bitselect packed vs the
    // existing color so uncovered pixels keep the background.
    const coverMask = this.loadCoverMask();
    const dst = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(7), OP.local_get, ...E.uleb(this.L_off), OP.i32_add);
    this.b(simd(SIMD.v128_load, 0x04, 0x00)); this.b(OP.local_set, ...E.uleb(dst));
    const outc = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(packed), OP.local_get, ...E.uleb(dst), OP.local_get, ...E.uleb(coverMask));
    this.b(simd(SIMD.v128_bitselect)); this.b(OP.local_set, ...E.uleb(outc));

    this.b(OP.local_get, ...E.uleb(7), OP.local_get, ...E.uleb(this.L_off), OP.i32_add); // store addr
    this.b(OP.local_get, ...E.uleb(outc));
    this.b(simd(SIMD.v128_store, 0x04, 0x00));

    // i++ ; loop
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(1), OP.i32_add, OP.local_set, ...E.uleb(this.L_i));
    this.b(OP.br, ...E.uleb(0));
    this.b(OP.end); this.b(OP.end);

    return buildModule({ params: this.params, results: [], locals: this.locals, body: this.body, exportName: 'run' });
  }

  // build a per-lane coverage mask (0xFFFFFFFF where covered, else 0).
  // cover is 1 byte/pixel; the 4 pixels of this group are 4 contiguous bytes at
  // coverPtr(param6) + (off>>2)  [off is in color bytes = i*16, so cover offset = i*4].
  loadCoverMask() {
    // cover byte index = i*4 ; reuse L_i*4
    const coff = this.newLocal(T.i32);
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(4), OP.i32_mul, OP.local_set, ...E.uleb(coff));
    // load 4 cover bytes into the low 32 bits of a v128 via v128.load32_zero,
    // then i16x8.extend/i32x4 widen unsigned: bytes -> i32 lanes.
    const v = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(6), OP.local_get, ...E.uleb(coff), OP.i32_add);
    this.b(simd(0x5c /*v128.load32_zero*/, 0x02, 0x00));
    this.b(OP.local_set, ...E.uleb(v));
    // widen u8 low4 -> u16 (i16x8.extend_low_i8x16_u = 0x88), then u16 low4 -> u32 (i32x4.extend_low_i16x8_u = 0xa8)
    const w16 = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(v)); this.b(simd(0x88)); this.b(OP.local_set, ...E.uleb(w16));
    const w32 = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(w16)); this.b(simd(0xa8)); this.b(OP.local_set, ...E.uleb(w32));
    // mask = i32x4.ne(w32, 0)  -> all-ones where cover!=0  (i32x4.ne = 0x37)
    const mask = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(w32)); this.b(OP.i32_const, ...E.sleb(0)); this.b(simd(SIMD.i32x4_splat)); this.b(simd(0x37));
    this.b(OP.local_set, ...E.uleb(mask));
    return mask;
  }

  // load a plane (param index) at off into a fresh v128 local
  loadPlane(paramIdx) {
    const L = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(paramIdx), OP.local_get, ...E.uleb(this.L_off), OP.i32_add);
    this.b(simd(SIMD.v128_load, 0x04, 0x00));
    this.b(OP.local_set, ...E.uleb(L));
    return L;
  }

  constV(x) {
    const L = this.newLocal(V128);
    this.b(OP.f32_const, ...E.f32bytes(x)); this.b(simd(SIMD.f32x4_splat)); this.b(OP.local_set, ...E.uleb(L));
    return L;
  }

  // binary v128 op into a fresh local
  v2(opSimd, a, b) {
    const L = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(a), OP.local_get, ...E.uleb(b)); this.b(simd(opSimd)); this.b(OP.local_set, ...E.uleb(L));
    return L;
  }

  packRGBA(R, G, B, A) {
    const c01 = (v) => { // clamp [0,1]
      const lo = this.constV(0), hi = this.constV(1);
      const m1 = this.v2(SIMD.f32x4_min, v, hi);
      return this.v2(SIMD.f32x4_max, m1, lo);
    };
    const toI = (v) => { // (i32x4) clamp(v)*255
      const c = c01(v); const m255 = this.v2(SIMD.f32x4_mul, c, this.constV(255));
      const L = this.newLocal(V128);
      this.b(OP.local_get, ...E.uleb(m255)); this.b(simd(SIMD.i32x4_trunc_sat_f32x4_s)); this.b(OP.local_set, ...E.uleb(L));
      return L;
    };
    const ri = toI(R), gi = toI(G), bi = toI(B), ai = toI(A);
    // shift g<<8, b<<16, a<<24, OR together with r
    const shl = (v, n) => { const L = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(v), OP.i32_const, ...E.sleb(n)); this.b(simd(SIMD.i32x4_shl)); this.b(OP.local_set, ...E.uleb(L)); return L; };
    const or = (a, b) => this.v2(SIMD.v128_or, a, b);
    return or(or(ri, shl(gi, 8)), or(shl(bi, 16), shl(ai, 24)));
  }

  // emit an IR node → array of v128 component locals
  emit(node) {
    switch (node.op) {
      case 'num': return [this.constV(node.v)];
      case 'varying': {
        if (node.name === 'uv') return [this.loadPlane(0), this.loadPlane(1)];
        if (node.name === 'color') return [this.loadPlane(2), this.loadPlane(3), this.loadPlane(4), this.loadPlane(5)];
        throw new Error('varying ' + node.name);
      }
      case 'uniform': return this.uniSplat[node.name].slice();
      case 'vec': { const out = []; for (const p of node.parts) out.push(...this.emit(p)); return out; }
      case 'swizzle': { const e = this.emit(node.e); return swizzleIndices(node.sw).map((k) => e[k]); }
      case 'neg': { const e = this.emit(node.e); const z = this.constV(0); return e.map((c) => this.v2(SIMD.f32x4_sub, z, c)); }
      case 'bin': return this.emitBin(node.o, this.emit(node.a), this.emit(node.b));
      case 'call': return this.emitCall(node.fn, node.args.map((a) => this.emit(a)));
      default: throw new Error('emit ' + node.op);
    }
  }

  emitBin(o, a, b) {
    if (a.length === 1 && b.length > 1) a = b.map(() => a[0]);
    if (b.length === 1 && a.length > 1) b = a.map(() => b[0]);
    const opMap = { '+': SIMD.f32x4_add, '-': SIMD.f32x4_sub, '*': SIMD.f32x4_mul, '/': SIMD.f32x4_div };
    return a.map((c, i) => this.v2(opMap[o], c, b[i]));
  }

  emitCall(fn, args) {
    const a = args[0], b = args[1], c = args[2];
    const comp = (f) => a.map((_, i) => f(i));
    const bc = (x, n) => x.length === 1 ? new Array(n).fill(x[0]) : x;
    switch (fn) {
      case 'abs': return a.map((v) => { const L = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(v)); this.b(simd(SIMD.f32x4_abs)); this.b(OP.local_set, ...E.uleb(L)); return L; }); // f32x4.abs=0xe1
      case 'sqrt': return a.map((v) => { const L = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(v)); this.b(simd(SIMD.f32x4_sqrt)); this.b(OP.local_set, ...E.uleb(L)); return L; });
      case 'floor': return a.map((v) => { const L = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(v)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(L)); return L; }); // f32x4.floor=0x67
      case 'fract': return a.map((v) => { const fl = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(v)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(fl)); return this.v2(SIMD.f32x4_sub, v, fl); });
      case 'min': { const n = Math.max(a.length, b.length); const A = bc(a, n), B = bc(b, n); return A.map((x, i) => this.v2(SIMD.f32x4_min, x, B[i])); }
      case 'max': { const n = Math.max(a.length, b.length); const A = bc(a, n), B = bc(b, n); return A.map((x, i) => this.v2(SIMD.f32x4_max, x, B[i])); }
      case 'clamp': { const n = Math.max(a.length, b.length, c.length); const A = bc(a, n), B = bc(b, n), C = bc(c, n); return A.map((x, i) => this.v2(SIMD.f32x4_min, this.v2(SIMD.f32x4_max, x, B[i]), C[i])); }
      case 'mix': { const n = Math.max(a.length, b.length, c.length); const A = bc(a, n), B = bc(b, n), C = bc(c, n);
        // x*(1-t)+y*t
        return A.map((x, i) => { const t = C[i]; const omt = this.v2(SIMD.f32x4_sub, this.constV(1), t); return this.v2(SIMD.f32x4_add, this.v2(SIMD.f32x4_mul, x, omt), this.v2(SIMD.f32x4_mul, B[i], t)); }); }
      case 'sin': return a.map((v) => this.emitSin(v));
      case 'cos': return a.map((v) => this.emitSin(this.v2(SIMD.f32x4_add, v, this.constV(Math.PI / 2))));
      case 'step': { const n = Math.max(a.length, b.length); const E_ = bc(a, n), X = bc(b, n); // step(edge,x)= x<edge?0:1 = ge(x,edge)&1.0
        return E_.map((edge, i) => { const ge = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(X[i]), OP.local_get, ...E.uleb(edge)); this.b(simd(0x40)); this.b(OP.local_set, ...E.uleb(ge)); // f32x4.ge -> all ones mask
          return this.v2(SIMD.v128_and, ge, this.constV(1)); }); }
      case 'mod': { const n = Math.max(a.length, b.length); const A = bc(a, n), B = bc(b, n); // x - y*floor(x/y)
        return A.map((x, i) => { const y = B[i]; const d = this.v2(SIMD.f32x4_div, x, y); const fl = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(d)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(fl)); return this.v2(SIMD.f32x4_sub, x, this.v2(SIMD.f32x4_mul, y, fl)); }); }
      default: throw new Error('JIT builtin ' + fn);
    }
  }

  // polynomial sine approximation (Bhaskara-ish via range-reduced minimax),
  // component-wise on a v128. Good to ~1e-3 — fine for shading.
  emitSin(x) {
    // wrap to [-PI,PI]: x = x - 2PI*round(x/2PI); round via floor(z+0.5)
    const inv2pi = this.constV(1 / (2 * Math.PI));
    const z = this.v2(SIMD.f32x4_mul, x, inv2pi);
    const zp = this.v2(SIMD.f32x4_add, z, this.constV(0.5));
    const fl = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(zp)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(fl)); // floor
    const xr = this.v2(SIMD.f32x4_sub, x, this.v2(SIMD.f32x4_mul, fl, this.constV(2 * Math.PI)));
    // minimax: sin(x) ~ x*(0.98793 - 0.15545*x^2 + 0.00565*x^4)  (x in [-PI,PI])
    const x2 = this.v2(SIMD.f32x4_mul, xr, xr);
    const c4 = this.v2(SIMD.f32x4_add, this.v2(SIMD.f32x4_mul, x2, this.constV(0.0056549)), this.constV(-0.1554991));
    const poly = this.v2(SIMD.f32x4_add, this.v2(SIMD.f32x4_mul, x2, c4), this.constV(0.9879227));
    return this.v2(SIMD.f32x4_mul, xr, poly);
  }
}
