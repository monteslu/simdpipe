/**
 * WASM-SIMD backend for the shader IR — the Tier-1 JIT.
 *
 * Walks the IR and emits a v128 kernel that shades 4 fragments per instruction
 * (SoA). The host engine (V8) compiles the generated module to native machine
 * code; we call it over the whole framebuffer. This makes authored GLSL run as
 * JIT'd SIMD, not a JS callback.
 *
 * Subset: the math IR (varyings uv/color, uniforms, +-* /, swizzle, vecN,
 * sin/cos/abs/floor/fract/sqrt/min/max/clamp/mix/step/mod) AND texture(uv)
 * sampling (nearest or bilinear). WASM has no SIMD gather opcode, but a gather is
 * just N address computes + N scalar loads — done by hand and reassembled into
 * lanes (see emitTex/gather4). Nothing that is pure calculation falls back to JS.
 *
 * Values during codegen are "rvalues": arrays of WASM local indices (v128), one
 * per component (length 1..4). Scalars broadcast.
 */
import { parseShader, swizzleIndices } from './shader-ir.mjs';
import { buildModule, T, OP, SIMD, simd, E } from './wasm-emit.mjs';

const V128 = 0x7b;

/**
 * @param {string} src GLSL-ish source
 * @param {{bilinear?:boolean}} [opts] texture filter: bilinear (4-tap) vs nearest
 * @returns {{supported:boolean, reason?:string, bytes?:Uint8Array, uniformOrder?:string[], needsTexture?:boolean}}
 */
export function compileWASM(src, opts = {}) {
  let parsed;
  try { parsed = parseShader(src); } catch (e) { return { supported: false, reason: 'parse: ' + e.message }; }
  const { uniforms, ast } = parsed;
  const needsTexture = usesTexture(ast);  // texture() IS supported — emitted as an inline gather
  // Which varying planes the kernel actually reads — lets the G-buffer rasterizer
  // skip interpolating + storing the rest (the shade-bound bandwidth win). The U/V
  // planes feed BOTH texture() sampling AND any direct `uv.x`/`uv.y` read, so check
  // both (a shader can read uv without sampling). `color` (r/g/b/a planes) is needed
  // whenever the `color` varying appears.
  const needsUV = needsTexture || usesVarying(ast, 'uv');
  const needsColor = usesVarying(ast, 'color');

  // uniform calling order (each uniform passed as N f32 params -> splatted)
  const uniformOrder = Object.keys(uniforms);

  const g = new Gen(uniforms, uniformOrder, needsTexture, !!opts.bilinear);
  let bytes;
  try { bytes = g.build(ast); } catch (e) { return { supported: false, reason: 'codegen: ' + e.message }; }
  return { supported: true, bytes, uniformOrder, uniforms, needsTexture, needsUV, needsColor };
}

function usesTexture(node) {
  if (!node || typeof node !== 'object') return false;
  if (node.op === 'tex') return true;
  for (const k of ['e', 'a', 'b', 'uv']) if (node[k] && usesTexture(node[k])) return true;
  if (node.parts) for (const p of node.parts) if (usesTexture(p)) return true;
  if (node.args) for (const p of node.args) if (usesTexture(p)) return true;
  return false;
}

/** True if the AST references the named varying (e.g. 'color') anywhere. */
function usesVarying(node, name) {
  if (!node || typeof node !== 'object') return false;
  if (node.op === 'varying' && node.name === name) return true;
  for (const k of ['e', 'a', 'b', 'uv']) if (node[k] && usesVarying(node[k], name)) return true;
  if (node.parts) for (const p of node.parts) if (usesVarying(p, name)) return true;
  if (node.args) for (const p of node.args) if (usesVarying(p, name)) return true;
  return false;
}

class Gen {
  constructor(uniforms, uniformOrder, needsTexture, bilinear) {
    this.uniforms = uniforms;
    this.uniformOrder = uniformOrder;
    this.needsTexture = !!needsTexture;
    this.bilinear = !!bilinear;
    // params: planes ptrs (u,v,r,g,b,a -> 0..5), coverPtr 6, colorPtr 7, n4 8,
    //         then one f32 param per uniform-scalar-component, then (if texture
    //         is used) 5 i32: texBase, texW, texH, texWmask, texHmask.
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
    if (this.needsTexture) {
      this.texP = { base: p, w: p + 1, h: p + 2, wmask: p + 3, hmask: p + 4 };
      for (let i = 0; i < 5; i++) this.params.push(T.i32);
      p += 5;
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

  /* Emit one 4-pixel group's shade+pack+store, reading from the CURRENT this.L_off
   * (byte offset) and a given cover mask local. No control flow — the caller owns the
   * skip. emit(ast) allocates fresh locals per call, so two back-to-back invocations
   * (the 2-wide unroll) produce two INDEPENDENT register chains the engine can
   * interleave — that's the ILP that hides the sin-poly latency. */
  emitGroupBody(coverMask) {
    const rgba = this.emit(this.ast);
    const comp = (i, def) => rgba[i] != null ? rgba[i] : (def != null ? this.constV(def) : this.constV(0));
    const R = comp(0, 0), G = comp(1, 0), B = comp(2, 0), A = rgba[3] != null ? rgba[3] : this.constV(1);
    const packed = this.packRGBA(R, G, B, A);
    const dst = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(7), OP.local_get, ...E.uleb(this.L_off), OP.i32_add);
    this.b(simd(SIMD.v128_load, 0x04, 0x00)); this.b(OP.local_set, ...E.uleb(dst));
    const outc = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(packed), OP.local_get, ...E.uleb(dst), OP.local_get, ...E.uleb(coverMask));
    this.b(simd(SIMD.v128_bitselect)); this.b(OP.local_set, ...E.uleb(outc));
    this.b(OP.local_get, ...E.uleb(7), OP.local_get, ...E.uleb(this.L_off), OP.i32_add); // store addr
    this.b(OP.local_get, ...E.uleb(outc));
    this.b(simd(SIMD.v128_store, 0x04, 0x00));
  }

  build(ast) {
    this.ast = ast;                  // emitGroupBody re-walks it per group
    this.L_off_main = this.L_off;    // the primary offset local (group A / remainder)
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

    // ---- N-wide unrolled main loop: process groups i..i+N-1 per iteration ----
    // The fragment expression (sin/poly/mix) is a latency chain; emitting N groups'
    // independent chains as back-to-back straight-line code lets the engine overlap
    // them (later groups' FMAs fill the bubbles in earlier groups' sin), approaching a
    // 256-bit renderer's throughput via ILP rather than wider-than-128 vectors. One
    // COMBINED cover-skip (skip the block only if NONE of the N groups is covered) keeps
    // the bodies in a single basic block so control flow doesn't break the interleave.
    // 4-wide is the measured sweet spot: ILP keeps filling sin-poly latency bubbles
    // up to ~4 in-flight group chains, then register pressure cancels the gain.
    const UNROLL = Math.max(1, parseInt(process.env.SP_UNROLL || '4', 10));
    const idxL = [], offL = [];
    for (let k = 0; k < UNROLL; k++) { idxL.push(this.newLocal(T.i32)); offL.push(this.newLocal(T.i32)); }
    this.b(OP.i32_const, ...E.sleb(0), OP.local_set, ...E.uleb(this.L_i));
    this.b(OP.block, 0x40); this.b(OP.loop, 0x40);
    // while (i+UNROLL-1 < n4): all N groups in range for the unrolled path
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(UNROLL - 1), OP.i32_add, OP.local_get, ...E.uleb(8), OP.i32_ge_s, OP.br_if, ...E.uleb(1));
    // idx[k] = i+k ; off[k] = idx[k]*16
    for (let k = 0; k < UNROLL; k++) {
      this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(k), OP.i32_add, OP.local_set, ...E.uleb(idxL[k]));
      this.b(OP.local_get, ...E.uleb(idxL[k]), OP.i32_const, ...E.sleb(16), OP.i32_mul, OP.local_set, ...E.uleb(offL[k]));
    }
    const cms = idxL.map((il) => this.loadCoverMask(il));
    this.b(OP.block, 0x40);                                  // combined skip block
    // skip only if NONE covered: i32.eqz( any(cm0) | any(cm1) | ... ) → br_if 0
    this.b(OP.local_get, ...E.uleb(cms[0])); this.b(simd(SIMD.v128_any_true));
    for (let k = 1; k < UNROLL; k++) { this.b(OP.local_get, ...E.uleb(cms[k])); this.b(simd(SIMD.v128_any_true)); this.b(OP.i32_or); }
    this.b(OP.i32_eqz, OP.br_if, ...E.uleb(0));
    for (let k = 0; k < UNROLL; k++) { this.L_off = offL[k]; this.emitGroupBody(cms[k]); }
    this.L_off = this.L_off_main;
    this.b(OP.end);                                         // end combined skip block
    // i += UNROLL ; loop
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(UNROLL), OP.i32_add, OP.local_set, ...E.uleb(this.L_i));
    this.b(OP.br, ...E.uleb(0));
    this.b(OP.end); this.b(OP.end);

    // ---- remainder: trailing groups (n4 not a multiple of UNROLL), 1 at a time ----
    this.L_off = this.L_off_main;
    this.b(OP.block, 0x40); this.b(OP.loop, 0x40);
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.local_get, ...E.uleb(8), OP.i32_ge_s, OP.br_if, ...E.uleb(1));
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(16), OP.i32_mul, OP.local_set, ...E.uleb(this.L_off));
    const cmR = this.loadCoverMask(this.L_i);
    this.b(OP.block, 0x40);
    this.b(OP.local_get, ...E.uleb(cmR)); this.b(simd(SIMD.v128_any_true));
    this.b(OP.i32_eqz, OP.br_if, ...E.uleb(0));
    this.emitGroupBody(cmR);
    this.b(OP.end);
    this.b(OP.local_get, ...E.uleb(this.L_i), OP.i32_const, ...E.sleb(1), OP.i32_add, OP.local_set, ...E.uleb(this.L_i));
    this.b(OP.br, ...E.uleb(0));
    this.b(OP.end); this.b(OP.end);

    return buildModule({ params: this.params, results: [], locals: this.locals, body: this.body, exportName: 'run' });
  }

  // build a per-lane coverage mask (0xFFFFFFFF where covered, else 0).
  // cover is 1 byte/pixel; the 4 pixels of this group are 4 contiguous bytes at
  // coverPtr(param6) + (off>>2)  [off is in color bytes = i*16, so cover offset = i*4].
  loadCoverMask(iLocal = this.L_i) {
    // cover byte index = i*4 (the group-index local is a parameter so the 2-wide
    // unrolled loop can load both groups' cover from their own index locals).
    const coff = this.newLocal(T.i32);
    this.b(OP.local_get, ...E.uleb(iLocal), OP.i32_const, ...E.sleb(4), OP.i32_mul, OP.local_set, ...E.uleb(coff));
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
      case 'tex': { const uv = this.emit(node.uv); return this.emitTex(uv[0], uv[1]); }
      case 'vec': { const out = []; for (const p of node.parts) out.push(...this.emit(p)); return out; }
      case 'swizzle': { const e = this.emit(node.e); return swizzleIndices(node.sw).map((k) => e[k]); }
      case 'neg': { const e = this.emit(node.e); const z = this.constV(0); return e.map((c) => this.v2(SIMD.f32x4_sub, z, c)); }
      case 'bin': return this.emitBin(node.o, this.emit(node.a), this.emit(node.b));
      case 'call': return this.emitCall(node.fn, node.args.map((a) => this.emit(a)));
      default: throw new Error('emit ' + node.op);
    }
  }

  // texture(uv) → [Rplane, Gplane, Bplane, Aplane] (v128 each, channels in [0,1]).
  // Dispatches to nearest or bilinear. Both are PURE address arithmetic + scalar
  // loads + lane reassembly — fully in the kernel, NO JS. WASM has no SIMD gather,
  // but a gather is just N address computes + N loads, which we do by hand. There
  // is no missing primitive here.
  emitTex(u, v) {
    return this.bilinear ? this.emitTexBilinear(u, v) : this.emitTexNearest(u, v);
  }

  // Splat texW/texH as f32 into v128 locals (params are i32). Cached per build.
  texDimsF() {
    if (this._texDimsF) return this._texDimsF;
    const { w, h } = this.texP;
    const wf = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(w)); this.b(OP.f32_convert_i32_s); this.b(simd(SIMD.f32x4_splat)); this.b(OP.local_set, ...E.uleb(wf));
    const hf = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(h)); this.b(OP.f32_convert_i32_s); this.b(simd(SIMD.f32x4_splat)); this.b(OP.local_set, ...E.uleb(hf));
    return (this._texDimsF = { wf, hf });
  }
  // Splat the i32 wrap masks + texW into v128 locals. Cached per build.
  texMasks() {
    if (this._texMasks) return this._texMasks;
    const { w, wmask, hmask } = this.texP;
    const wmaskv = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(wmask)); this.b(simd(SIMD.i32x4_splat)); this.b(OP.local_set, ...E.uleb(wmaskv));
    const hmaskv = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(hmask)); this.b(simd(SIMD.i32x4_splat)); this.b(OP.local_set, ...E.uleb(hmaskv));
    const wv = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(w)); this.b(simd(SIMD.i32x4_splat)); this.b(OP.local_set, ...E.uleb(wv));
    return (this._texMasks = { wmaskv, hmaskv, wv });
  }

  // Gather 4 texels at the given i32x4 of linear texel indices → [R,G,B,A] v128
  // planes (channels in [0,1] f32). This is the "gather" WASM lacks, done by hand:
  // 4 lanes × (extract index, addr = base + idx*4, i32.load, unpack 4 bytes).
  gather4(linear) {
    const { base } = this.texP;
    const R = this.constV(0), G = this.constV(0), B = this.constV(0), A = this.constV(0);
    const inv255 = 1 / 255;
    for (let lane = 0; lane < 4; lane++) {
      const addr = this.newLocal(T.i32);
      this.b(OP.local_get, ...E.uleb(base));
      this.b(OP.local_get, ...E.uleb(linear)); this.b(simd(SIMD.i32x4_extract_lane, lane));
      this.b(OP.i32_const, ...E.sleb(4), OP.i32_mul, OP.i32_add, OP.local_set, ...E.uleb(addr));
      const texel = this.newLocal(T.i32);
      this.b(OP.local_get, ...E.uleb(addr)); this.b(OP.i32_load, 0x02, 0x00); this.b(OP.local_set, ...E.uleb(texel));
      const chan = (shift, plane) => {
        const bf = this.newLocal(T.f32);
        this.b(OP.local_get, ...E.uleb(texel));
        if (shift) { this.b(OP.i32_const, ...E.sleb(shift)); this.b(0x76 /*i32.shr_u*/); }
        this.b(OP.i32_const, ...E.sleb(0xff), OP.i32_and);
        this.b(OP.f32_convert_i32_s);
        this.b(OP.f32_const, ...E.f32bytes(inv255), OP.f32_mul);
        this.b(OP.local_set, ...E.uleb(bf));
        this.b(OP.local_get, ...E.uleb(plane), OP.local_get, ...E.uleb(bf));
        this.b(simd(SIMD.f32x4_replace_lane, lane));
        this.b(OP.local_set, ...E.uleb(plane));
      };
      chan(0, R); chan(8, G); chan(16, B); chan(24, A);
    }
    return [R, G, B, A];
  }

  // linear index from integer texel coords (already wrapped): yi*texW + xi
  linearIndex(xiM, yiM) {
    const { wv } = this.texMasks();
    return this.v2(SIMD.i32x4_add, this.v2(SIMD.i32x4_mul, yiM, wv), xiM);
  }

  emitTexNearest(u, v) {
    const { wf, hf } = this.texDimsF();
    const { wmaskv, hmaskv } = this.texMasks();
    // xi = trunc(u*wf) & wmask ; yi = trunc(v*hf) & hmask  (i32x4)
    const xi = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(this.v2(SIMD.f32x4_mul, u, wf))); this.b(simd(SIMD.i32x4_trunc_sat_f32x4_s)); this.b(OP.local_set, ...E.uleb(xi));
    const xiM = this.v2(SIMD.i32x4_and, xi, wmaskv);
    const yi = this.newLocal(V128);
    this.b(OP.local_get, ...E.uleb(this.v2(SIMD.f32x4_mul, v, hf))); this.b(simd(SIMD.i32x4_trunc_sat_f32x4_s)); this.b(OP.local_set, ...E.uleb(yi));
    const yiM = this.v2(SIMD.i32x4_and, yi, hmaskv);
    return this.gather4(this.linearIndex(xiM, yiM));
  }

  // Bilinear: sample the 4 neighboring texels and blend by the fractional part of
  // the texel-space coordinate. Matches GL convention: sample at (u*W - 0.5).
  // All pure calc — 4 gathers + float lerps. (POT repeat-wrap on both axes.)
  emitTexBilinear(u, v) {
    const { wf, hf } = this.texDimsF();
    const { wmaskv, hmaskv } = this.texMasks();
    const half = this.constV(0.5), one = this.constV(1);
    // continuous texel coord, shifted by -0.5 for pixel-center convention
    const fx = this.v2(SIMD.f32x4_sub, this.v2(SIMD.f32x4_mul, u, wf), half);
    const fy = this.v2(SIMD.f32x4_sub, this.v2(SIMD.f32x4_mul, v, hf), half);
    // floor → integer base coord; frac → blend weight
    const flx = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(fx)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(flx));
    const fly = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(fy)); this.b(simd(SIMD.f32x4_floor)); this.b(OP.local_set, ...E.uleb(fly));
    const tx = this.v2(SIMD.f32x4_sub, fx, flx); // [0,1)
    const ty = this.v2(SIMD.f32x4_sub, fy, fly);
    // integer coords x0,y0 (trunc of floored value is exact); x1=x0+1,y1=y0+1
    const toI = (vf) => { const L = this.newLocal(V128); this.b(OP.local_get, ...E.uleb(vf)); this.b(simd(SIMD.i32x4_trunc_sat_f32x4_s)); this.b(OP.local_set, ...E.uleb(L)); return L; };
    const onei = this.newLocal(V128); this.b(OP.i32_const, ...E.sleb(1)); this.b(simd(SIMD.i32x4_splat)); this.b(OP.local_set, ...E.uleb(onei));
    const x0 = this.v2(SIMD.i32x4_and, toI(flx), wmaskv);
    const y0 = this.v2(SIMD.i32x4_and, toI(fly), hmaskv);
    const x1 = this.v2(SIMD.i32x4_and, this.v2(SIMD.i32x4_add, x0, onei), wmaskv);
    const y1 = this.v2(SIMD.i32x4_and, this.v2(SIMD.i32x4_add, y0, onei), hmaskv);
    // four taps
    const t00 = this.gather4(this.linearIndex(x0, y0));
    const t10 = this.gather4(this.linearIndex(x1, y0));
    const t01 = this.gather4(this.linearIndex(x0, y1));
    const t11 = this.gather4(this.linearIndex(x1, y1));
    // lerp: top = mix(t00,t10,tx); bot = mix(t01,t11,tx); out = mix(top,bot,ty)
    const lerp = (a, b, t) => { const omt = this.v2(SIMD.f32x4_sub, one, t); return this.v2(SIMD.f32x4_add, this.v2(SIMD.f32x4_mul, a, omt), this.v2(SIMD.f32x4_mul, b, t)); };
    const out = [];
    for (let ch = 0; ch < 4; ch++) {
      const top = lerp(t00[ch], t10[ch], tx);
      const bot = lerp(t01[ch], t11[ch], tx);
      out.push(lerp(top, bot, ty));
    }
    return out;
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
